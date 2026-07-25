/*
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef VN_SYNC_H
#define VN_SYNC_H

#include "vn_common.h"

#include "vn_device.h"
#include "vn_feedback.h"

enum vn_sync_type {
   /* no payload */
   VN_SYNC_TYPE_INVALID,

   /* device object */
   VN_SYNC_TYPE_DEVICE_ONLY,

   /* renderer sync object */
   VN_SYNC_TYPE_SYNC,

   /* renderer sync objects
    * - cpu sync x 1
    * - gpu sync x vn_device::queue_count
    */
   VN_SYNC_TYPE_TIMELINE_SYNC,

   /* payload is an imported sync file */
   VN_SYNC_TYPE_IMPORTED_SYNC_FD,
};

struct vn_sync_payload {
   enum vn_sync_type type;

   union {
      /* If type is VN_SYNC_TYPE_SYNC, sync is non-NULL. */
      struct vn_renderer_sync *sync;

      /* If type is VN_SYNC_TYPE_TIMELINE_SYNC, syncs are non-NULL. */
      struct vn_renderer_sync **syncs;

#ifdef VK_USE_PLATFORM_WIN32_KHR
      /* If type is VN_SYNC_TYPE_IMPORTED_SYNC_FD, fd is an async event
       * handle. */
      intptr_t fd;
#else
      /* If type is VN_SYNC_TYPE_IMPORTED_SYNC_FD, fd is a sync file. */
      int fd;
#endif
   };
};

struct vn_fence {
   struct vn_object_base base;

   struct vn_sync_payload *payload;

   struct vn_sync_payload permanent;
   struct vn_sync_payload temporary;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(vn_fence,
                               base.vk,
                               VkFence,
                               VK_OBJECT_TYPE_FENCE)

struct vn_semaphore {
   struct vn_object_base base;

   VkSemaphoreType type;
   bool sync_fd_export;
   bool syncobj_exported;

   /* last value known to vk, either current or pending */
   uint64_t last_value;

   struct vn_sync_payload *payload;

   struct vn_sync_payload permanent;
   struct vn_sync_payload temporary;

   struct vn_sync_feedback feedback;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(vn_semaphore,
                               base.vk,
                               VkSemaphore,
                               VK_OBJECT_TYPE_SEMAPHORE)

struct vn_event {
   struct vn_object_base base;

   /* non-NULL if below are satisfied:
    * - event is created without VK_EVENT_CREATE_DEVICE_ONLY_BIT
    * - VN_PERF_NO_EVENT_FEEDBACK is disabled
    */
   struct vn_feedback_slot *feedback_slot;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(vn_event,
                               base.vk,
                               VkEvent,
                               VK_OBJECT_TYPE_EVENT)

static inline bool
vn_semaphore_is_timeline(VkSemaphore sem_handle)
{
   struct vn_semaphore *sem = vn_semaphore_from_handle(sem_handle);
   return sem->type == VK_SEMAPHORE_TYPE_TIMELINE;
}

static inline bool
vn_wait_semaphore_is_sync_fd(VkSemaphore sem_handle)
{
   struct vn_semaphore *sem = vn_semaphore_from_handle(sem_handle);
   return sem->payload->type == VN_SYNC_TYPE_IMPORTED_SYNC_FD ||
          (sem->payload->type == VN_SYNC_TYPE_SYNC && sem->sync_fd_export);
}

static inline bool
vn_signal_semaphore_is_sync_fd(VkSemaphore sem_handle)
{
   struct vn_semaphore *sem = vn_semaphore_from_handle(sem_handle);
   return sem->permanent.type == VN_SYNC_TYPE_SYNC && sem->sync_fd_export;
}

static inline bool
vn_semaphore_is_exported(VkSemaphore sem_handle)
{
   struct vn_semaphore *sem = vn_semaphore_from_handle(sem_handle);
   return sem->syncobj_exported;
}

static inline uint64_t
vn_semaphore_get_last_value(VkSemaphore sem_handle)
{
   assert(vn_semaphore_is_timeline(sem_handle));
   struct vn_semaphore *sem = vn_semaphore_from_handle(sem_handle);
   return sem->last_value;
}

bool
vn_semaphore_wait_sync_fd(VkDevice dev_handle, VkSemaphore sem_handle);

void
vn_sync_payload_release(struct vn_device *dev,
                        struct vn_sync_payload *payload);

#ifdef VK_USE_PLATFORM_WIN32_KHR
VkResult
vn_submit_semaphore_signal(VkDevice dev_handle,
                           VkSemaphore sem_handle,
                           uint64_t sem_value,
                           struct vn_renderer_sync *const *syncs,
                           uint64_t *sync_values,
                           uint32_t sync_count);

static inline VkResult
vn_submit_external_timeline_signal(VkDevice dev_handle,
                                   VkSemaphore sem_handle,
                                   uint64_t sem_value)
{
   assert(vn_semaphore_is_exported(sem_handle));
   assert(vn_semaphore_is_timeline(sem_handle));

   struct vn_device *dev = vn_device_from_handle(dev_handle);
   struct vn_semaphore *sem = vn_semaphore_from_handle(sem_handle);
   struct vn_renderer_sync *wait_sync = sem->payload->syncs[dev->queue_count];

   return vn_submit_semaphore_signal(dev_handle, sem_handle, sem_value,
                                     &wait_sync, &sem_value, 1);
}
#endif

#endif /* VN_SYNC_H */
