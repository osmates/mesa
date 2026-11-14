/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "kk_buffer_view.h"

#include "kk_buffer.h"
#include "kk_device.h"
#include "kk_entrypoints.h"
#include "kk_format.h"
#include "kk_image_layout.h"
#include "kk_nir_lower_vbo.h"
#include "kk_physical_device.h"

#include "kosmickrisp/bridge/mtl_bridge.h"
#include "kosmickrisp/bridge/mtl_format.h"

#include "util/format/u_format.h"
#include "vk_format.h"

static uint64_t
kk_texel_buffer_alignment_B(struct kk_physical_device *pdev,
                            const struct kk_va_format *format)
{
   return mtl_minimum_texture_buffer_alignment_for_pixel_format(
      pdev->mtl_dev_handle, format->mtl_pixel_format);
}

/* A view starts at any multiple of its texel size, or of its component size
 * for a 3-component format. The Metal texture starts at the offset aligned
 * down to Metal's alignment, and the shader skips the texels in between,
 * which only works if they are whole texels. A 3-component format qualifies
 * only if Metal takes a texture buffer at any component.
 */
static bool
kk_texel_buffer_offsets_are_whole_texels(struct kk_physical_device *pdev,
                                         enum pipe_format p_format,
                                         const struct kk_va_format *format)
{
   const unsigned texel_B = util_format_get_blocksize(p_format);
   if (util_is_power_of_two_nonzero(texel_B))
      return true;

   const unsigned component_B =
      texel_B / util_format_get_nr_components(p_format);
   return kk_texel_buffer_alignment_B(pdev, format) <= component_B;
}

VkFormatFeatureFlags2
kk_get_buffer_format_features(struct kk_physical_device *pdev,
                              VkFormat vk_format)
{
   VkFormatFeatureFlags2 features = 0;
   enum pipe_format p_format = vk_format_to_pipe_format(vk_format);

   if (p_format == PIPE_FORMAT_NONE)
      return 0;

   const struct kk_va_format *format = kk_get_va_format(p_format);
   if (format && pdev->info.gpu_apple_family >= format->gpu_family &&
       kk_texel_buffer_offsets_are_whole_texels(pdev, p_format, format)) {
      if (format->texel_buffer.read)
         features |= VK_FORMAT_FEATURE_2_UNIFORM_TEXEL_BUFFER_BIT;

      if (format->texel_buffer.write) {
         features |= VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_BIT;
         features |= VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT;
         features |= VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT;
      }

      /* Only these formats allow atomics for texel buffers */
      if (vk_format == VK_FORMAT_R32_UINT || vk_format == VK_FORMAT_R32_SINT)
         features |= VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_ATOMIC_BIT;
   }

   if (kk_vbo_supports_format(p_format))
      features |= VK_FORMAT_FEATURE_2_VERTEX_BUFFER_BIT;

   return features;
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_CreateBufferView(VkDevice _device, const VkBufferViewCreateInfo *pCreateInfo,
                    const VkAllocationCallbacks *pAllocator,
                    VkBufferView *pBufferView)
{
   VK_FROM_HANDLE(kk_device, dev, _device);
   struct kk_physical_device *pdev = kk_device_physical(dev);
   struct kk_buffer_view *view =
      vk_buffer_view_create(&dev->vk, pCreateInfo, pAllocator, sizeof(*view));
   if (!view)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   enum pipe_format p_format = vk_format_to_pipe_format(view->vk.format);
   const struct kk_va_format *supported_format = kk_get_va_format(p_format);

   /* If we reached here, we support reading at least */
   enum mtl_texture_usage usage = MTL_TEXTURE_USAGE_SHADER_READ;
   if (supported_format->texel_buffer.write)
      usage |= MTL_TEXTURE_USAGE_SHADER_WRITE;

   /* Only these formats allow atomics for texel buffers */
   if (view->vk.format == VK_FORMAT_R32_UINT ||
       view->vk.format == VK_FORMAT_R32_SINT)
      usage |= MTL_TEXTURE_USAGE_SHADER_ATOMIC;

   struct kk_buffer *buffer =
      container_of(view->vk.buffer, struct kk_buffer, vk);
   const uint64_t offset_B = kk_buffer_mtl_offset(buffer, view->vk.offset);

   /* Metal needs a texture buffer's offset aligned to more than a texel.
    * Start the texture at the aligned offset below the view's and skip the
    * texels in between in the shader.
    */
   const uint64_t skip_B =
      offset_B % kk_texel_buffer_alignment_B(pdev, supported_format);
   const unsigned texel_B = util_format_get_blocksize(p_format);
   assert(skip_B % texel_B == 0);
   view->offset_el = skip_B / texel_B;

   struct kk_image_layout layout = {
      .width_px = view->offset_el + view->vk.elements,
      .height_px = 1u,
      .depth_px = 1u,
      .layers = 1u,
      .type = MTL_TEXTURE_TYPE_TEXTURE_BUFFER,
      .sample_count_sa = 1u,
      .levels = 1u,
      .linear = true,
      .optimized_layout = false,
      .usage = usage,
      .format = {.pipe = p_format, .mtl = supported_format->mtl_pixel_format},
      .swizzle =
         {
            .red = supported_format->unswizzle.red,
            .green = supported_format->unswizzle.green,
            .blue = supported_format->unswizzle.blue,
            .alpha = supported_format->unswizzle.alpha,
         },
      .linear_stride_B = skip_B + view->vk.range,
   };
   view->mtl_texel_buffer_handle = mtl_new_texture_with_descriptor_linear(
      buffer->metal.handle, &layout, offset_B - skip_B);
   if (!view->mtl_texel_buffer_handle) {
      vk_buffer_view_destroy(&dev->vk, pAllocator, &view->vk);
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }
   view->texel_buffer_gpu_id =
      mtl_texture_get_gpu_resource_id(view->mtl_texel_buffer_handle);

   *pBufferView = kk_buffer_view_to_handle(view);

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
kk_DestroyBufferView(VkDevice _device, VkBufferView bufferView,
                     const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(kk_device, dev, _device);
   VK_FROM_HANDLE(kk_buffer_view, view, bufferView);

   if (!view)
      return;

   mtl_release(view->mtl_texel_buffer_handle);
   vk_buffer_view_destroy(&dev->vk, pAllocator, &view->vk);
}

void
kk_buffer_view_set_label(struct kk_buffer_view *bview, const char *label)
{
   if (bview->mtl_texel_buffer_handle)
      mtl_resource_set_label(bview->mtl_texel_buffer_handle, label);
}
