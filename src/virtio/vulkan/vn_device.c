/*
 * Copyright 2019 Google LLC
 * SPDX-License-Identifier: MIT
 *
 * based in part on anv and radv which are:
 * Copyright © 2015 Intel Corporation
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 */

#include "vn_device.h"

#include "util/disk_cache.h"
#include "util/hex.h"
#include "vn_protocol_driver_device.h"

#include "vn_android.h"
#include "vn_instance.h"
#include "vn_physical_device.h"
#include "vn_queue.h"
#include "vn_ring.h"

/* device commands */

static void
vn_queue_fini(struct vn_queue *queue)
{
   VkDevice dev_handle = vk_device_to_handle(queue->base.vk.base.device);

   if (queue->async_present.initialized) {
      mtx_lock(&queue->async_present.mutex);
      queue->async_present.join = true;
      cnd_signal(&queue->async_present.cond);
      mtx_unlock(&queue->async_present.mutex);

      thrd_join(queue->async_present.thread, NULL);

      simple_mtx_destroy(&queue->async_present.queue_mutex);
      mtx_destroy(&queue->async_present.mutex);
      cnd_destroy(&queue->async_present.cond);

      vn_DestroyFence(dev_handle, queue->async_present.fence, NULL);
   }

   if (queue->wait_fence != VK_NULL_HANDLE) {
      vn_DestroyFence(dev_handle, queue->wait_fence, NULL);
   }
   vn_cached_storage_fini(&queue->storage);
   vn_queue_base_fini(&queue->base);
}

static VkResult
vn_queue_init(struct vn_device *dev,
              struct vn_queue *queue,
              uint32_t array_index,
              const VkDeviceQueueCreateInfo *queue_info,
              uint32_t queue_index,
              struct vn_queue *shared_queue)
{
   VkResult result =
      vn_queue_base_init(&queue->base, &dev->base, queue_info, queue_index);
   if (result != VK_SUCCESS)
      return result;

   queue->index = array_index;

   vn_cached_storage_init(&queue->storage, &dev->base.vk.alloc);

   if (dev->physical_device->emulate_second_queue ==
          queue_info->queueFamilyIndex &&
       shared_queue != NULL) {
      assert(queue_index > 0);
      queue->emulated = true;
      queue->base.id = shared_queue->base.id;
      queue->can_feedback = shared_queue->can_feedback;
      queue->ring_idx = shared_queue->ring_idx;
      return VK_SUCCESS;
   }

   queue->can_feedback = vn_queue_family_can_feedback(
      dev->physical_device, queue_info->queueFamilyIndex);

   const int ring_idx = vn_instance_acquire_ring_idx(dev->instance);
   if (ring_idx < 0) {
      vn_cached_storage_fini(&queue->storage);
      vn_queue_base_fini(&queue->base);
      vn_log(dev->instance, "failed binding VkQueue to renderer timeline");
      return VK_ERROR_INITIALIZATION_FAILED;
   }
   queue->ring_idx = (uint32_t)ring_idx;

   const VkDeviceQueueTimelineInfoMESA timeline_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_TIMELINE_INFO_MESA,
      .ringIdx = queue->ring_idx,
   };
   const VkDeviceQueueInfo2 device_queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2,
      .pNext = &timeline_info,
      .flags = queue_info->flags,
      .queueFamilyIndex = queue_info->queueFamilyIndex,
      .queueIndex = queue_index,
   };

   VkQueue queue_handle = vn_queue_to_handle(queue);
   vn_call_vkGetDeviceQueue2(dev->primary_ring, vn_device_to_handle(dev),
                             &device_queue_info, &queue_handle);

   return VK_SUCCESS;
}

static VkResult
vn_device_init_queues(struct vn_device *dev,
                      const VkDeviceCreateInfo *create_info)
{
   const VkAllocationCallbacks *alloc = &dev->base.vk.alloc;

   /* allowed in maintenance9 */
   if (!create_info->queueCreateInfoCount)
      return VK_SUCCESS;

   uint32_t count = 0;
   for (uint32_t i = 0; i < create_info->queueCreateInfoCount; i++)
      count += create_info->pQueueCreateInfos[i].queueCount;

   struct vn_queue *queues =
      vk_zalloc(alloc, sizeof(*queues) * count, VN_DEFAULT_ALIGN,
                VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!queues)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   count = 0;
   struct vn_queue *shared_queue = NULL;
   for (uint32_t i = 0; i < create_info->queueCreateInfoCount; i++) {
      VkResult result;

      const VkDeviceQueueCreateInfo *queue_info =
         &create_info->pQueueCreateInfos[i];
      for (uint32_t j = 0; j < queue_info->queueCount; j++) {
         result = vn_queue_init(dev, &queues[count], count, queue_info, j,
                                shared_queue);
         if (result != VK_SUCCESS) {
            for (uint32_t k = 0; k < count; k++)
               vn_queue_fini(&queues[k]);

            /* ring_idx release is deferred until vkDestroyDevice is emitted */
            dev->queues = queues;
            dev->queue_count = count;
            return result;
         }

         shared_queue = &queues[count++];
      }
   }

   dev->queues = queues;
   dev->queue_count = count;

   return VK_SUCCESS;
}

static void
vn_device_fini_queues_deferred(struct vn_device *dev,
                               const VkAllocationCallbacks *alloc)
{
   /* We must emit vkDestroyDevice before releasing bound ring_idx. Otherwise,
    * another thread might reuse their ring_idx while they are still bound to
    * the queues in the renderer.
    */
   for (uint32_t i = 0; i < dev->queue_count; i++) {
      if (!dev->queues[i].emulated)
         vn_instance_release_ring_idx(dev->instance, dev->queues[i].ring_idx);
   }
   vk_free(alloc, dev->queues);
}

static bool
vn_device_queue_family_init(struct vn_device *dev,
                            const VkDeviceCreateInfo *create_info)
{
   const VkAllocationCallbacks *alloc = &dev->base.vk.alloc;
   uint32_t *queue_families = NULL;
   uint32_t count = 0;

   queue_families = vk_zalloc(
      alloc, sizeof(*queue_families) * create_info->queueCreateInfoCount,
      VN_DEFAULT_ALIGN, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!queue_families)
      return false;

   for (uint32_t i = 0; i < create_info->queueCreateInfoCount; i++) {
      const uint32_t index =
         create_info->pQueueCreateInfos[i].queueFamilyIndex;
      bool new_index = true;

      for (uint32_t j = 0; j < count; j++) {
         if (queue_families[j] == index) {
            new_index = false;
            break;
         }
      }
      if (new_index)
         queue_families[count++] = index;
   }

   dev->queue_families = queue_families;
   dev->queue_family_count = count;

   return true;
}

static inline void
vn_device_queue_family_fini(struct vn_device *dev)
{
   vk_free(&dev->base.vk.alloc, dev->queue_families);
}

static bool
find_extension_names(const char *const *exts,
                     uint32_t ext_count,
                     const char *name)
{
   for (uint32_t i = 0; i < ext_count; i++) {
      if (!strcmp(exts[i], name))
         return true;
   }
   return false;
}

static bool
merge_extension_names(const char *const *exts,
                      uint32_t ext_count,
                      const char *const *extra_exts,
                      uint32_t extra_count,
                      const char *const *block_exts,
                      uint32_t block_count,
                      const VkAllocationCallbacks *alloc,
                      const char *const **out_exts,
                      uint32_t *out_count)
{
   const char **merged =
      vk_alloc(alloc, sizeof(*merged) * (ext_count + extra_count),
               VN_DEFAULT_ALIGN, VK_SYSTEM_ALLOCATION_SCOPE_COMMAND);
   if (!merged)
      return false;

   uint32_t count = 0;
   for (uint32_t i = 0; i < ext_count; i++) {
      if (!find_extension_names(block_exts, block_count, exts[i]))
         merged[count++] = exts[i];
   }
   for (uint32_t i = 0; i < extra_count; i++) {
      if (!find_extension_names(exts, ext_count, extra_exts[i]))
         merged[count++] = extra_exts[i];
   }

   *out_exts = merged;
   *out_count = count;
   return true;
}

struct vn_device_create_info {
   VkDeviceCreateInfo create;
   VkDeviceGroupDeviceCreateInfo group;
   VkPhysicalDeviceFeatures2 features2;
   VkPhysicalDeviceVulkan11Features features11;
   VkPhysicalDeviceVulkan12Features features12;
   VkPhysicalDeviceVulkan13Features features13;
   VkPhysicalDeviceRobustness2FeaturesEXT robustness2;
   VkPhysicalDeviceDescriptorIndexingFeatures descriptor_indexing;
   VkPhysicalDeviceHostQueryResetFeatures host_query_reset;
   VkPhysicalDeviceShaderDrawParametersFeatures shader_draw_parameters;
   VkPhysicalDeviceShaderFloat16Int8Features shader_float16_int8;
   VkPhysicalDevice8BitStorageFeatures storage_8bit;
   VkPhysicalDevice16BitStorageFeatures storage_16bit;
   VkPhysicalDeviceUniformBufferStandardLayoutFeatures
      uniform_buffer_standard_layout;
   VkPhysicalDeviceScalarBlockLayoutFeatures scalar_block_layout;
   VkPhysicalDeviceTimelineSemaphoreFeatures timeline_semaphore;
   VkPhysicalDeviceBufferDeviceAddressFeatures buffer_device_address;
   VkPhysicalDeviceImagelessFramebufferFeatures imageless_framebuffer;
   VkPhysicalDeviceSeparateDepthStencilLayoutsFeatures
      separate_depth_stencil_layouts;
   VkPhysicalDeviceMemoryPriorityFeaturesEXT memory_priority;
   VkPhysicalDevicePageableDeviceLocalMemoryFeaturesEXT
      pageable_device_local_memory;
};

static bool
vn_device_enable_timeline_semaphores(const VkDeviceCreateInfo *create_info,
                                     struct vn_device_create_info *local_info)
{
   bool enabled = false;
   local_info->create = *create_info;
   void *cur = (void *)&local_info->create;

   vk_foreach_struct_const(sType, src, create_info->pNext) {
      void *next = NULL;
      switch (sType) {
      case VK_STRUCTURE_TYPE_DEVICE_GROUP_DEVICE_CREATE_INFO:
         memcpy(&local_info->group, src, sizeof(local_info->group));
         next = &local_info->group;
         break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2:
         memcpy(&local_info->features2, src, sizeof(local_info->features2));
         next = &local_info->features2;
         break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES:
         memcpy(&local_info->features11, src, sizeof(local_info->features11));
         next = &local_info->features11;
         break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES:
         memcpy(&local_info->features12, src, sizeof(local_info->features12));
         next = &local_info->features12;
         local_info->features12.timelineSemaphore = VK_TRUE;
         enabled = true;
         break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES:
         memcpy(&local_info->features13, src, sizeof(local_info->features13));
         next = &local_info->features13;
         break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT:
         memcpy(&local_info->robustness2, src,
                sizeof(local_info->robustness2));
         next = &local_info->robustness2;
         break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES:
         memcpy(&local_info->descriptor_indexing, src,
                sizeof(local_info->descriptor_indexing));
         next = &local_info->descriptor_indexing;
         break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES:
         memcpy(&local_info->host_query_reset, src,
                sizeof(local_info->host_query_reset));
         next = &local_info->host_query_reset;
         break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES:
         memcpy(&local_info->shader_draw_parameters, src,
                sizeof(local_info->shader_draw_parameters));
         next = &local_info->shader_draw_parameters;
         break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES:
         memcpy(&local_info->shader_float16_int8, src,
                sizeof(local_info->shader_float16_int8));
         next = &local_info->shader_float16_int8;
         break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES:
         memcpy(&local_info->storage_8bit, src,
                sizeof(local_info->storage_8bit));
         next = &local_info->storage_8bit;
         break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES:
         memcpy(&local_info->storage_16bit, src,
                sizeof(local_info->storage_16bit));
         next = &local_info->storage_16bit;
         break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFORM_BUFFER_STANDARD_LAYOUT_FEATURES:
         memcpy(&local_info->uniform_buffer_standard_layout, src,
                sizeof(local_info->uniform_buffer_standard_layout));
         next = &local_info->uniform_buffer_standard_layout;
         break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES:
         memcpy(&local_info->scalar_block_layout, src,
                sizeof(local_info->scalar_block_layout));
         next = &local_info->scalar_block_layout;
         break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES:
         memcpy(&local_info->timeline_semaphore, src,
                sizeof(local_info->timeline_semaphore));
         next = &local_info->timeline_semaphore;
         local_info->timeline_semaphore.timelineSemaphore = VK_TRUE;
         enabled = true;
         break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES:
         memcpy(&local_info->buffer_device_address, src,
                sizeof(local_info->buffer_device_address));
         next = &local_info->buffer_device_address;
         break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES:
         memcpy(&local_info->imageless_framebuffer, src,
                sizeof(local_info->imageless_framebuffer));
         next = &local_info->imageless_framebuffer;
         break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SEPARATE_DEPTH_STENCIL_LAYOUTS_FEATURES:
         memcpy(&local_info->separate_depth_stencil_layouts, src,
                sizeof(local_info->separate_depth_stencil_layouts));
         next = &local_info->separate_depth_stencil_layouts;
         break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT:
         memcpy(&local_info->memory_priority, src,
                sizeof(local_info->memory_priority));
         next = &local_info->memory_priority;
         break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PAGEABLE_DEVICE_LOCAL_MEMORY_FEATURES_EXT:
         memcpy(&local_info->pageable_device_local_memory, src,
                sizeof(local_info->pageable_device_local_memory));
         next = &local_info->pageable_device_local_memory;
         break;

      default:
         break;
      }

      if (next) {
         vk_pnext_set_next(cur, next);
         cur = next;
      }
   }

   vk_pnext_set_next(cur, NULL);

   return enabled;
}

static const VkDeviceCreateInfo *
vn_device_fix_create_info(const struct vn_device *dev,
                          const VkDeviceCreateInfo *dev_info,
                          const VkAllocationCallbacks *alloc,
                          struct vn_device_create_info *local_info)
{
   const struct vn_physical_device *physical_dev = dev->physical_device;
   const struct vk_device_extension_table *app_exts =
      &dev->base.vk.enabled_extensions;
   const struct vk_device_extension_table *renderer_exts =
      &physical_dev->renderer_extensions;
   /* extra_exts and block_exts must not overlap */
   const char *extra_exts[16];
   const char *block_exts[32];
   uint32_t extra_count = 0;
   uint32_t block_count = 0;

   /* fix for WSI (treat AHB as WSI extension for simplicity) */
   const bool has_wsi =
      app_exts->KHR_swapchain || app_exts->ANDROID_native_buffer ||
      app_exts->ANDROID_external_memory_android_hardware_buffer;
   if (has_wsi) {
      if (renderer_exts->EXT_image_drm_format_modifier &&
          !app_exts->EXT_image_drm_format_modifier) {
         extra_exts[extra_count++] =
            VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME;

         if (physical_dev->renderer_version < VK_API_VERSION_1_2 &&
             !app_exts->KHR_image_format_list) {
            extra_exts[extra_count++] =
               VK_KHR_IMAGE_FORMAT_LIST_EXTENSION_NAME;
         }
      }

      if (renderer_exts->EXT_queue_family_foreign &&
          !app_exts->EXT_queue_family_foreign) {
         extra_exts[extra_count++] =
            VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME;
      }

      if (app_exts->KHR_swapchain) {
         /* see vn_physical_device_get_native_extensions */
         block_exts[block_count++] = VK_KHR_PRESENT_ID_EXTENSION_NAME;
         block_exts[block_count++] = VK_KHR_PRESENT_ID_2_EXTENSION_NAME;
         block_exts[block_count++] = VK_KHR_PRESENT_WAIT_EXTENSION_NAME;
         block_exts[block_count++] = VK_KHR_PRESENT_WAIT_2_EXTENSION_NAME;
         block_exts[block_count++] = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
         block_exts[block_count++] =
            VK_KHR_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME;
         block_exts[block_count++] =
            VK_KHR_SWAPCHAIN_MUTABLE_FORMAT_EXTENSION_NAME;
         block_exts[block_count++] =
            VK_KHR_INCREMENTAL_PRESENT_EXTENSION_NAME;
         block_exts[block_count++] =
            VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME;
         block_exts[block_count++] = VK_EXT_HDR_METADATA_EXTENSION_NAME;
         block_exts[block_count++] = VK_EXT_DISPLAY_CONTROL_EXTENSION_NAME;
      }

      if (app_exts->ANDROID_native_buffer) {
         block_exts[block_count++] = VK_ANDROID_NATIVE_BUFFER_EXTENSION_NAME;
      }

      if (app_exts->ANDROID_external_memory_android_hardware_buffer) {
         block_exts[block_count++] =
            VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME;
      }
   }

#ifdef VK_USE_PLATFORM_WIN32_KHR
   /* Windows D3D11/DXGI WSI needs timeline semaphores */
   const VkPhysicalDeviceTimelineSemaphoreFeatures *timeline_features =
      vk_find_struct_const(dev_info->pNext,
                           PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES);
   const VkPhysicalDeviceVulkan12Features *features12 = vk_find_struct_const(
      dev_info->pNext, PHYSICAL_DEVICE_VULKAN_1_2_FEATURES);
   const bool timeline_semaphores_enabled =
      (timeline_features && timeline_features->timelineSemaphore) ||
      (features12 && features12->timelineSemaphore);
   const bool need_timeline_semaphores =
      has_wsi && !timeline_semaphores_enabled;

   if (need_timeline_semaphores &&
       physical_dev->renderer_version < VK_API_VERSION_1_2 &&
       !app_exts->KHR_timeline_semaphore) {
      extra_exts[extra_count++] =
         VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME;
   }
#else
   const bool need_timeline_semaphores = false;
#endif

   if (app_exts->KHR_external_memory_fd ||
       app_exts->EXT_external_memory_dma_buf || has_wsi) {
      if (physical_dev->external_memory.renderer_handle_type ==
          VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT) {
         if (!app_exts->EXT_external_memory_dma_buf) {
            extra_exts[extra_count++] =
               VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME;
         }
         if (!app_exts->KHR_external_memory_fd) {
            extra_exts[extra_count++] =
               VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME;
         }
      }
   }

   /* see vn_cmd_set_external_acquire_unmodified */
   if (VN_PRESENT_SRC_INTERNAL_LAYOUT != VK_IMAGE_LAYOUT_PRESENT_SRC_KHR &&
       renderer_exts->EXT_external_memory_acquire_unmodified &&
       !app_exts->EXT_external_memory_acquire_unmodified && has_wsi) {
      extra_exts[extra_count++] =
         VK_EXT_EXTERNAL_MEMORY_ACQUIRE_UNMODIFIED_EXTENSION_NAME;
   }

   if (app_exts->KHR_external_fence_fd) {
      /* see vn_physical_device_get_native_extensions */
      block_exts[block_count++] = VK_KHR_EXTERNAL_FENCE_FD_EXTENSION_NAME;
   }

   if (app_exts->KHR_external_semaphore_fd) {
      /* see vn_physical_device_get_native_extensions */
      block_exts[block_count++] = VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME;
   }

   if (app_exts->KHR_map_memory2) {
      /* see vn_physical_device_get_native_extensions */
      block_exts[block_count++] = VK_KHR_MAP_MEMORY_2_EXTENSION_NAME;
   }

   if (app_exts->EXT_map_memory_placed) {
      /* see vn_physical_device_get_native_extensions */
      block_exts[block_count++] = VK_EXT_MAP_MEMORY_PLACED_EXTENSION_NAME;
   }

   if (app_exts->EXT_device_memory_report) {
      /* see vn_physical_device_get_native_extensions */
      block_exts[block_count++] = VK_EXT_DEVICE_MEMORY_REPORT_EXTENSION_NAME;
   }

   if (app_exts->EXT_physical_device_drm) {
      /* see vn_physical_device_get_native_extensions */
      block_exts[block_count++] = VK_EXT_PHYSICAL_DEVICE_DRM_EXTENSION_NAME;
   }

   if (app_exts->EXT_tooling_info) {
      /* see vn_physical_device_get_native_extensions */
      block_exts[block_count++] = VK_EXT_TOOLING_INFO_EXTENSION_NAME;
   }

   if (app_exts->EXT_pci_bus_info) {
      /* always filter for simplicity */
      block_exts[block_count++] = VK_EXT_PCI_BUS_INFO_EXTENSION_NAME;
   }

#ifdef VK_USE_PLATFORM_WIN32_KHR
   if (app_exts->KHR_external_fence_win32) {
      /* see vn_physical_device_get_native_extensions */
      block_exts[block_count++] = VK_KHR_EXTERNAL_FENCE_WIN32_EXTENSION_NAME;
   }
   if (app_exts->KHR_external_semaphore_win32) {
      /* see vn_physical_device_get_native_extensions */
      block_exts[block_count++] =
         VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME;
   }
   if (app_exts->KHR_external_memory_win32) {
      /* see vn_physical_device_get_native_extensions */
      block_exts[block_count++] = VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME;
   }
#endif

   assert(extra_count <= ARRAY_SIZE(extra_exts));
   assert(block_count <= ARRAY_SIZE(block_exts));

   if (!extra_count && (!block_count || !dev_info->enabledExtensionCount) &&
       !need_timeline_semaphores)
      return dev_info;

   if (need_timeline_semaphores &&
       !vn_device_enable_timeline_semaphores(dev_info, local_info)) {
      __vk_append_struct(&local_info->create,
                         &local_info->timeline_semaphore);
      local_info->timeline_semaphore.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
      local_info->timeline_semaphore.timelineSemaphore = VK_TRUE;
   } else {
      local_info->create = *dev_info;
   }

   if (!merge_extension_names(dev_info->ppEnabledExtensionNames,
                              dev_info->enabledExtensionCount, extra_exts,
                              extra_count, block_exts, block_count, alloc,
                              &local_info->create.ppEnabledExtensionNames,
                              &local_info->create.enabledExtensionCount))
      return NULL;

   return &local_info->create;
}

static inline VkResult
vn_device_feedback_pool_init(struct vn_device *dev)
{
   /* The feedback pool defaults to suballocate slots of 8 bytes each. Initial
    * pool size of 4096 corresponds to a total of 512 fences, semaphores and
    * events, which well covers the common scenarios. Pool can grow anyway.
    */
   static const uint32_t pool_size = 4096;
   const VkAllocationCallbacks *alloc = &dev->base.vk.alloc;

   if (VN_PERF(NO_EVENT_FEEDBACK) && VN_PERF(NO_SEMAPHORE_FEEDBACK))
      return VK_SUCCESS;

   return vn_feedback_pool_init(dev, &dev->feedback_pool, pool_size, alloc);
}

static inline void
vn_device_feedback_pool_fini(struct vn_device *dev)
{
   if (VN_PERF(NO_EVENT_FEEDBACK) && VN_PERF(NO_SEMAPHORE_FEEDBACK))
      return;

   vn_feedback_pool_fini(&dev->feedback_pool);
}

static void
vn_device_update_shader_cache_id(struct vn_device *dev)
{
   /* venus utilizes the host side shader cache.
    * This is a WA to generate shader cache files containing headers
    * with a unique cache id that will change based on host driver
    * identifiers. This allows fossilize replay to detect if the host
    * side shader cach is no longer up to date.
    * The shader cache is destroyed after creating the necessary files
    * and not utilized by venus.
    */
#if !defined(VK_USE_PLATFORM_ANDROID_KHR) && defined(ENABLE_SHADER_CACHE)
   const uint8_t *device_uuid =
      dev->physical_device->base.vk.properties.pipelineCacheUUID;

   char uuid[VK_UUID_SIZE * 2 + 1];
   mesa_bytes_to_hex(uuid, device_uuid, VK_UUID_SIZE);

   struct disk_cache *cache = disk_cache_create("venus", uuid, 0);
   if (!cache)
      return;

   /* The entry header is what contains the cache id / timestamp so we
    * need to create a fake entry.
    */
   uint8_t key[BLAKE3_KEY_LEN];
   char data[] = "Fake Shader";

   disk_cache_compute_key(cache, data, sizeof(data), key);
   disk_cache_put(cache, key, data, sizeof(data), NULL);

   disk_cache_destroy(cache);
#endif
}

static VkResult
vn_get_timestamp(struct vk_device *device, uint64_t *timestamp)
{
   VkDevice dev_handle = vk_device_to_handle(device);
   struct vn_device *dev = vn_device_from_handle(dev_handle);
   UNUSED uint64_t device_max_deviation;

   const VkCalibratedTimestampInfoKHR ts_info = {
      .sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR,
      .timeDomain = VK_TIME_DOMAIN_DEVICE_KHR,
   };
   return vn_call_vkGetCalibratedTimestampsKHR(dev->primary_ring, dev_handle,
                                               1, &ts_info, timestamp,
                                               &device_max_deviation);
}

static VkResult
vn_device_init(struct vn_device *dev,
               struct vn_physical_device *physical_dev,
               const VkDeviceCreateInfo *create_info,
               const VkAllocationCallbacks *alloc)
{
   struct vn_instance *instance = physical_dev->instance;
   VkPhysicalDevice physical_dev_handle =
      vn_physical_device_to_handle(physical_dev);
   VkDevice dev_handle = vn_device_to_handle(dev);
   struct vn_device_create_info local_create_info;
   VkResult result;

   dev->base.vk.get_timestamp = vn_get_timestamp;
   dev->instance = instance;
   dev->physical_device = physical_dev;
   dev->device_mask = 1;
   dev->renderer = instance->renderer;
   dev->primary_ring = instance->ring.ring;

   create_info =
      vn_device_fix_create_info(dev, create_info, alloc, &local_create_info);
   if (!create_info)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   const VkDeviceGroupDeviceCreateInfo *group = vk_find_struct_const(
      create_info->pNext, DEVICE_GROUP_DEVICE_CREATE_INFO);
   if (group && group->physicalDeviceCount)
      dev->device_mask = (1 << group->physicalDeviceCount) - 1;

   VkDeviceCreateInfo final_create_info = *create_info;
   STACK_ARRAY(VkDeviceQueueCreateInfo, queue_infos,
               create_info->queueCreateInfoCount);
   for (uint32_t i = 0; i < create_info->queueCreateInfoCount; i++) {
      const VkDeviceQueueCreateInfo *queue_info =
         &create_info->pQueueCreateInfos[i];
      if (queue_info->queueFamilyIndex ==
             physical_dev->emulate_second_queue &&
          queue_info->queueCount == 2) {
         typed_memcpy(queue_infos, create_info->pQueueCreateInfos,
                      create_info->queueCreateInfoCount);
         final_create_info.pQueueCreateInfos = queue_infos;
         queue_infos[i].queueCount = 1;
         break;
      }
   }
   result = vn_call_vkCreateDevice(dev->primary_ring, physical_dev_handle,
                                   &final_create_info, NULL, &dev_handle);
   STACK_ARRAY_FINISH(queue_infos);

   /* free the fixed extensions here since no longer needed below */
   if (create_info == &local_create_info.create)
      vk_free(alloc, (void *)create_info->ppEnabledExtensionNames);

   if (result != VK_SUCCESS)
      return result;

   if (!vn_device_queue_family_init(dev, create_info)) {
      result = VK_ERROR_OUT_OF_HOST_MEMORY;
      goto out_destroy_device;
   }

   result = vn_device_feedback_pool_init(dev);
   if (result != VK_SUCCESS)
      goto out_queue_family_fini;

   result = vn_feedback_cmd_pools_init(dev);
   if (result != VK_SUCCESS)
      goto out_feedback_pool_fini;

   result = vn_device_init_queues(dev, create_info);
   if (result != VK_SUCCESS)
      goto out_feedback_cmd_pools_fini;

   vn_buffer_reqs_cache_init(dev);
   vn_image_reqs_cache_init(dev);

   /* This is a WA to allow fossilize replay to detect if the host side shader
    * cache is no longer up to date.
    */
   vn_device_update_shader_cache_id(dev);

   simple_mtx_init(&dev->mutex, mtx_plain);
   list_inithead(&dev->chains);

   return VK_SUCCESS;

out_feedback_cmd_pools_fini:
   vn_feedback_cmd_pools_fini(dev);

out_feedback_pool_fini:
   vn_device_feedback_pool_fini(dev);

out_queue_family_fini:
   vn_device_queue_family_fini(dev);

out_destroy_device:
   /* surpress -Wc23-extensions */
   {
      struct vn_ring_submit_command ring_submit;
      vn_submit_vkDestroyDevice(dev->primary_ring, 0, dev_handle, NULL,
                                &ring_submit);
      if (ring_submit.ring_seqno_valid)
         vn_ring_wait_seqno(dev->primary_ring, ring_submit.ring_seqno);
   }

   vn_device_fini_queues_deferred(dev, alloc);

   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
vn_CreateDevice(VkPhysicalDevice physicalDevice,
                const VkDeviceCreateInfo *pCreateInfo,
                const VkAllocationCallbacks *pAllocator,
                VkDevice *pDevice)
{
   VN_TRACE_FUNC();
   struct vn_physical_device *physical_dev =
      vn_physical_device_from_handle(physicalDevice);
   struct vn_instance *instance = physical_dev->instance;
   const VkAllocationCallbacks *alloc =
      pAllocator ? pAllocator : &instance->base.vk.alloc;
   struct vn_device *dev;
   VkResult result;

   dev = vk_zalloc(alloc, sizeof(*dev), VN_DEFAULT_ALIGN,
                   VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!dev)
      return vn_error(instance, VK_ERROR_OUT_OF_HOST_MEMORY);

   struct vk_device_dispatch_table dispatch_table;
   vk_device_dispatch_table_from_entrypoints(&dispatch_table,
                                             &vn_device_entrypoints, true);
   vk_device_dispatch_table_from_entrypoints(&dispatch_table,
                                             &wsi_device_entrypoints, false);
   result = vn_device_base_init(&dev->base, &physical_dev->base,
                                &dispatch_table, pCreateInfo, alloc);
   if (result != VK_SUCCESS) {
      vk_free(alloc, dev);
      return vn_error(instance, result);
   }

   result = vn_device_init(dev, physical_dev, pCreateInfo, alloc);
   if (result != VK_SUCCESS) {
      vn_device_base_fini(&dev->base);
      vk_free(alloc, dev);
      return vn_error(instance, result);
   }

   if (VN_DEBUG(LOG_CTX_INFO)) {
      vn_log(instance, "%s", physical_dev->base.vk.properties.deviceName);
      vn_log(instance, "%s", physical_dev->base.vk.properties.driverInfo);
   }

   vn_tls_set_async_pipeline_create();

   *pDevice = vn_device_to_handle(dev);

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
vn_DestroyDevice(VkDevice device, const VkAllocationCallbacks *pAllocator)
{
   VN_TRACE_FUNC();
   struct vn_device *dev = vn_device_from_handle(device);
   const VkAllocationCallbacks *alloc =
      pAllocator ? pAllocator : &dev->base.vk.alloc;

   if (!dev)
      return;

   assert(list_is_empty(&dev->chains));
   simple_mtx_destroy(&dev->mutex);

   vn_image_reqs_cache_fini(dev);
   vn_buffer_reqs_cache_fini(dev);

   for (uint32_t i = 0; i < dev->queue_count; i++)
      vn_queue_fini(&dev->queues[i]);

   vn_feedback_cmd_pools_fini(dev);

   vn_device_feedback_pool_fini(dev);

   vn_device_queue_family_fini(dev);

   vn_async_vkDestroyDevice(dev->primary_ring, device, NULL);

   vn_device_fini_queues_deferred(dev, alloc);

   vn_device_base_fini(&dev->base);
   vk_free(alloc, dev);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vn_GetDeviceProcAddr(VkDevice device, const char *pName)
{
   struct vn_device *dev = vn_device_from_handle(device);
   return vk_device_get_proc_addr(&dev->base.vk, pName);
}

VKAPI_ATTR void VKAPI_CALL
vn_GetDeviceGroupPeerMemoryFeatures(
   VkDevice device,
   uint32_t heapIndex,
   uint32_t localDeviceIndex,
   uint32_t remoteDeviceIndex,
   VkPeerMemoryFeatureFlags *pPeerMemoryFeatures)
{
   struct vn_device *dev = vn_device_from_handle(device);

   /* TODO get and cache the values in vkCreateDevice */
   vn_call_vkGetDeviceGroupPeerMemoryFeatures(
      dev->primary_ring, device, heapIndex, localDeviceIndex,
      remoteDeviceIndex, pPeerMemoryFeatures);
}
