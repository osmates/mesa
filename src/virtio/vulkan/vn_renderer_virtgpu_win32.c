#include "vn_renderer_internal.h"
#include "vn_image.h"

#include <winddk_compat.h>
#include <d3d10umddi.h>
#include <d3dkmthk.h>
#include <d3dukmdt.h>
#include <dxgiddi.h>
#include <winnt.h>
#include <winternl.h>

#include "util/u_atomic.h"
#include "util/os_file.h"
#include "util/sparse_array.h"

#include "vk_internal_d3dddi_ext.h"
#include "virtio/virtio-gpu/venus_hw.h"
#include "virtio/virtio-gpu/wddm_hw.h"

#ifndef DRM_FORMAT_MOD_INVALID
#define DRM_FORMAT_MOD_INVALID 0x00FFFFFFFFFFFFFF
#endif

#define VIRTGPU_PCI_VENDOR_ID 0x1af4
#define VIRTGPU_PCI_DEVICE_ID 0x1050

struct virtgpu;

struct virtgpu_shmem {
   struct vn_renderer_shmem base;
   D3DKMT_HANDLE alloc;
   union {
      D3DKMT_HANDLE kmt;
      HANDLE h;
   };
};

struct virtgpu_bo {
   struct vn_renderer_bo base;
   D3DKMT_HANDLE alloc;
   union {
      struct {
         D3DKMT_HANDLE local;
         D3DKMT_HANDLE global;
      } kmt;

      HANDLE h;
   } /* resource */;
   uint32_t blob_flags; /* 3d resource if 0 */
   bool external;
};

#define VIRTUAL_CONTEXT_SUPPORT

// #define INDEPENDENT_RENDER_CONTEXT

struct virtgpu_sync {
   struct vn_renderer_sync base;
   D3DKMT_HANDLE handle; /* Monitored fence */
   volatile uint64_t *value_map;
   D3DGPU_VIRTUAL_ADDRESS gpu_map;
};

#define D3DKMT_FUNCTIONS                                                     \
   X(PFND3DKMT_ENUMADAPTERS2, enum_adapters2, "D3DKMTEnumAdapters2")         \
   X(PFND3DKMT_QUERYADAPTERINFO, query_adapter_info,                         \
     "D3DKMTQueryAdapterInfo")                                               \
   X(PFND3DKMT_ESCAPE, escape, "D3DKMTEscape")                               \
   X(PFND3DKMT_RENDER, render, "D3DKMTRender")                               \
   X(PFND3DKMT_SUBMITCOMMAND, submit_command, "D3DKMTSubmitCommand")         \
   X(PFND3DKMT_MAKERESIDENT, make_resident, "D3DKMTMakeResident")            \
   X(PFND3DKMT_CREATESYNCHRONIZATIONOBJECT2, create_synchronization_object2, \
     "D3DKMTCreateSynchronizationObject2")                                   \
   X(PFND3DKMT_SIGNALSYNCHRONIZATIONOBJECT2, signal_synchronization_object2, \
     "D3DKMTSignalSynchronizationObject2")                                   \
   X(PFND3DKMT_DESTROYSYNCHRONIZATIONOBJECT,                                 \
     destroy_synchronization_object2, "D3DKMTDestroySynchronizationObject")  \
   X(PFND3DKMT_OPENSYNCOBJECTFROMNTHANDLE2,                                  \
     open_sync_object_from_nt_handle2, "D3DKMTOpenSyncObjectFromNtHandle2")  \
   X(PFND3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMGPU,                          \
     wait_for_synchronization_object_from_gpu,                               \
     "D3DKMTWaitForSynchronizationObjectFromGpu")                            \
   X(PFND3DKMT_SIGNALSYNCHRONIZATIONOBJECTFROMGPU2,                          \
     signal_synchronization_object_from_gpu2,                                \
     "D3DKMTSignalSynchronizationObjectFromGpu2")                            \
   X(PFND3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU,                          \
     wait_for_synchronization_object_from_cpu,                               \
     "D3DKMTWaitForSynchronizationObjectFromCpu")                            \
   X(PFND3DKMT_SIGNALSYNCHRONIZATIONOBJECTFROMCPU,                           \
     signal_synchronization_object_from_cpu,                                 \
     "D3DKMTSignalSynchronizationObjectFromCpu")                             \
   X(PFND3DKMT_CREATEPAGINGQUEUE, create_paging_queue,                       \
     "D3DKMTCreatePagingQueue")                                              \
   X(PFND3DKMT_CREATECONTEXT, create_context, "D3DKMTCreateContext")         \
   X(PFND3DKMT_CREATECONTEXTVIRTUAL, create_context_virtual,                 \
     "D3DKMTCreateContextVirtual")                                           \
   X(PFND3DKMT_DESTROYCONTEXT, destroy_context, "D3DKMTDestroyContext")      \
   X(PFND3DKMT_CREATEHWQUEUE, create_hw_queue, "D3DKMTCreateHwQueue")        \
   X(PFND3DKMT_DESTROYHWQUEUE, destroy_hw_queue, "D3DKMTDestroyHwQueue")     \
   X(PFND3DKMT_SUBMITCOMMANDTOHWQUEUE, submit_command_to_hw_queue,           \
      "D3DKMTSubmitCommandToHwQueue")                                        \
   X(PFND3DKMT_SUBMITSIGNALSYNCOBJECTSTOHWQUEUE,                             \
      submit_signal_sync_objects_to_hw_queue,                                \
      "D3DKMTSubmitSignalSyncObjectsToHwQueue")                              \
   X(PFND3DKMT_SUBMITWAITFORSYNCOBJECTSTOHWQUEUE,                            \
      submit_wait_for_sync_objects_to_hw_queue,                              \
      "D3DKMTSubmitWaitForSyncObjectsToHwQueue")                             \
   X(PFND3DKMT_SUBMITPRESENTTOHWQUEUE,                                       \
      submit_present_to_hw_queue,                                            \
      "D3DKMTSubmitPresentToHwQueue")                                        \
   X(PFND3DKMT_CREATEALLOCATION, create_allocation,                          \
     "D3DKMTCreateAllocation")                                               \
   X(PFND3DKMT_DESTROYALLOCATION, destroy_allocation,                        \
     "D3DKMTDestroyAllocation")                                              \
   X(PFND3DKMT_LOCK, lock, "D3DKMTLock")                                     \
   X(PFND3DKMT_UNLOCK, unlock, "D3DKMTUnlock")                               \
   X(PFND3DKMT_QUERYRESOURCEINFO, query_resource_info,                       \
     "D3DKMTQueryResourceInfo")                                              \
   X(PFND3DKMT_QUERYRESOURCEINFOFROMNTHANDLE,                                \
     query_resource_info_from_nt_handle,                                     \
     "D3DKMTQueryResourceInfoFromNtHandle")                                  \
   X(PFND3DKMT_OPENRESOURCE, open_resource, "D3DKMTOpenResource")            \
   X(PFND3DKMT_OPENRESOURCEFROMNTHANDLE, open_resource_from_nt_handle,       \
     "D3DKMTOpenResourceFromNtHandle")                                       \
   X(PFND3DKMT_SHAREOBJECTS, share_objects, "D3DKMTShareObjects")            \
   X(PFND3DKMT_CREATEDEVICE, create_device, "D3DKMTCreateDevice")            \
   X(PFND3DKMT_GETDEVICESTATE, get_device_state, "D3DKMTGetDeviceState")     \
   X(PFND3DKMT_DESTROYDEVICE, destroy_device, "D3DKMTDestroyDevice")         \
   X(PFND3DKMT_OPENADAPTERFROMHDC, open_adapter_from_hdc,                    \
     "D3DKMTOpenAdapterFromHdc")                                             \
   X(PFND3DKMT_CLOSEADAPTER, close_adapter, "D3DKMTCloseAdapter")

struct virtgpu_hw_queue {
    union {
        D3DKMT_HANDLE kmt;
        HANDLE h;
    };
    struct virtgpu_sync progress_fence;
};

struct virtgpu_exec_context_base {
   union {
      D3DKMT_HANDLE kmt;
      HANDLE h;
   };
};

#ifdef VIRTUAL_CONTEXT_SUPPORT

/* A driver with hardware queues takes every ring on one context with a
 * hardware queue.  Without them, each ring gets a context of its own on a
 * node of its own (wddm_hw.h), since dxgkrnl runs at most two submissions
 * of a context at a time.
 */
#define MAX_COMMAND_SIZE           (4096 - 32)
struct virtgpu_exec_context_virtual {
   struct virtgpu_exec_context_base base;
   // mtx_t lock;
};
#else
#define VIRTIO_WDDM_TIMELINE_COUNT 64
struct virtgpu_exec_context {
   struct virtgpu_exec_context_base base;

   void *cmd_buf;
   size_t cmd_size;

   D3DDDI_ALLOCATIONLIST *alloc_list;
   size_t alloc_size;

   D3DDDI_PATCHLOCATIONLIST *patch_list;
   size_t patch_size;
};
#endif

struct virtgpu {
   struct vn_renderer base;

   struct vn_instance *instance;

   VkD3DDDICallbacksMESA *ddicb;
   struct {
      D3DKMT_HANDLE adapter;
      D3DKMT_HANDLE device;
      LUID luid;
      HINSTANCE lib;
      struct {
#define X(type, name, fun) type name;
         D3DKMT_FUNCTIONS
#undef X
      } cb;
   } d3dkmt;

   struct virtgpu_sync *submission_fence;
   uint64_t pending_submission_value;
#ifdef VIRTUAL_CONTEXT_SUPPORT
   bool use_hw_queue;
   mtx_t hwq_submit_lock;
   struct virtgpu_exec_context_virtual hw_ctx;
   struct virtgpu_hw_queue hw_queue;
#endif
#ifndef INDEPENDENT_RENDER_CONTEXT
#if defined(VIRTUAL_CONTEXT_SUPPORT)
   struct virtgpu_exec_context_virtual ctx[VIRTIO_WDDM_RING_NODE_COUNT];
#else
   mtx_t ctx_lock;
   struct virtgpu_exec_context ctx[64];
#endif
#endif

   struct {
      D3DKMT_HANDLE queue;
      D3DKMT_HANDLE sync_object;
      void *fence_value;
   } paging;

   struct {
      uint16_t domain;
      uint8_t bus;
      uint8_t dev;
      uint8_t func;
   } pci_bus_info;

   uint32_t max_timeline_count;

   struct {
      uint32_t id;
      uint32_t version;
      struct virgl_renderer_capset_venus data;
   } capset;

   uint32_t shmem_blob_mem;
   uint32_t bo_blob_mem;

   struct util_sparse_array syncobj_array;
   /* note that we use kmt_handle instead of res_id to index because
    * res_id is monotonically increasing by default (see
    * virtio_gpu_resource_id_get)
    */
   struct util_sparse_array shmem_array;
   struct util_sparse_array bo_array;

   mtx_t win32_handle_import_mutex;

   struct vn_renderer_shmem_cache shmem_cache;

   // bool supports_cross_device;
};

static inline bool
virtgpu_uses_hw_queue(const struct virtgpu *gpu)
{
#ifdef VIRTUAL_CONTEXT_SUPPORT
   return gpu->use_hw_queue;
#else
   return false;
#endif
}

static inline NTSTATUS
hr_to_nt(struct virtgpu *gpu, HRESULT hr)
{
   switch (hr) {
   case S_OK:
      return STATUS_SUCCESS;
   case E_OUTOFMEMORY:
      return STATUS_NO_MEMORY;
   case E_INVALIDARG:
      return STATUS_INVALID_PARAMETER;
   case E_PENDING:
      return STATUS_PENDING;
   case D3DDDIERR_DEVICEREMOVED:
   case DXGI_ERROR_DEVICE_REMOVED:
   case DXGI_ERROR_DEVICE_HUNG:
   case DXGI_ERROR_DEVICE_RESET:
      return STATUS_DEVICE_REMOVED;
   default:
      vn_log(gpu->instance, "Unknown HRESULT: %lx", hr);
      return STATUS_INVALID_PARAMETER;
   }
}

#ifdef VIRTUAL_CONTEXT_SUPPORT
static NTSTATUS
virtgpu_ioctl_create_context_virtual(struct virtgpu *gpu,
                                     unsigned node,
                                     bool hw_queue,
                                     struct virtgpu_exec_context_virtual *ctx)
{
   if (gpu->ddicb != NULL) {
      D3DDDICB_CREATECONTEXTVIRTUAL context = {
         .NodeOrdinal = node,
         .Flags = {
            .HwQueueSupported = hw_queue,
         },
      };
      NTSTATUS status =
         hr_to_nt(gpu, gpu->ddicb->pKTCallbacks->pfnCreateContextVirtualCb(
                          gpu->ddicb->hRTDevice, &context));
      if (!NT_SUCCESS(status)) {
         return status;
      }

      ctx->base.h = context.hContext;

      return STATUS_SUCCESS;
   } else {
      D3DKMT_CREATECONTEXTVIRTUAL context = {
         .hDevice = gpu->d3dkmt.device,
         .NodeOrdinal = node,
         .ClientHint = D3DKMT_CLIENTHINT_VULKAN,
         .Flags = {
            .HwQueueSupported = hw_queue,
         },
      };

      NTSTATUS status = gpu->d3dkmt.cb.create_context_virtual(&context);
      if (!NT_SUCCESS(status)) {
         return status;
      }

      ctx->base.kmt = context.hContext;

      return STATUS_SUCCESS;
   }
}
#else
static NTSTATUS
virtgpu_ioctl_create_context(struct virtgpu *gpu,
                             struct virtgpu_exec_context *ctx)
{
   if (gpu->ddicb != NULL) {
      D3DDDICB_CREATECONTEXT context = {};
      NTSTATUS status =
         hr_to_nt(gpu, gpu->ddicb->pKTCallbacks->pfnCreateContextCb(
                          gpu->ddicb->hRTDevice, &context));
      if (!NT_SUCCESS(status)) {
         return status;
      }

      gpu->ddicb->hContext = context.hContext;

      ctx->base.h = context.hContext;

      ctx->cmd_buf = context.pCommandBuffer;
      ctx->cmd_size = context.CommandBufferSize;

      ctx->alloc_list = context.pAllocationList;
      ctx->alloc_size = context.AllocationListSize;

      ctx->patch_list = context.pPatchLocationList;
      ctx->patch_size = context.PatchLocationListSize;

      return STATUS_SUCCESS;
   } else {
      D3DKMT_CREATECONTEXT context = {
         .hDevice = gpu->d3dkmt.device,
         .ClientHint = D3DKMT_CLIENTHINT_VULKAN,
      };

      NTSTATUS status = gpu->d3dkmt.cb.create_context(&context);
      if (!NT_SUCCESS(status)) {
         return status;
      }

      ctx->base.kmt = context.hContext;

      ctx->cmd_buf = context.pCommandBuffer;
      ctx->cmd_size = context.CommandBufferSize;

      ctx->alloc_list = context.pAllocationList;
      ctx->alloc_size = context.AllocationListSize;

      ctx->patch_list = context.pPatchLocationList;
      ctx->patch_size = context.PatchLocationListSize;

      return STATUS_SUCCESS;
   }
}
#endif

#ifdef VIRTUAL_CONTEXT_SUPPORT
static NTSTATUS
virtgpu_ioctl_create_hw_queue(struct virtgpu *gpu,
                              struct virtgpu_exec_context_virtual *ctx,
                              struct virtgpu_hw_queue *hw_queue)
{
   if (gpu->ddicb != NULL) {
      D3DDDICB_CREATEHWQUEUE create_hw_queue = {
         .hHwContext = ctx->base.h,
      };
      NTSTATUS status =
         hr_to_nt(gpu, gpu->ddicb->pKTCallbacks->pfnCreateHwQueueCb(
                          gpu->ddicb->hRTDevice, &create_hw_queue));
      if (!NT_SUCCESS(status)) {
         return status;
      }

      hw_queue->h = create_hw_queue.hHwQueue;
      hw_queue->progress_fence.handle = create_hw_queue.hHwQueueProgressFence;
      hw_queue->progress_fence.value_map = create_hw_queue.HwQueueProgressFenceCPUVirtualAddress;
      hw_queue->progress_fence.gpu_map = create_hw_queue.HwQueueProgressFenceGPUVirtualAddress;

      return STATUS_SUCCESS;
   } else {
      D3DKMT_CREATEHWQUEUE create_hw_queue = {
         .hHwContext = ctx->base.kmt,
      };
      NTSTATUS status = gpu->d3dkmt.cb.create_hw_queue(&create_hw_queue);
      if (!NT_SUCCESS(status)) {
         return status;
      }

      hw_queue->kmt = create_hw_queue.hHwQueue;
      hw_queue->progress_fence.handle = create_hw_queue.hHwQueueProgressFence;
      hw_queue->progress_fence.value_map = create_hw_queue.HwQueueProgressFenceCPUVirtualAddress;
      hw_queue->progress_fence.gpu_map = create_hw_queue.HwQueueProgressFenceGPUVirtualAddress;

      return STATUS_SUCCESS;
   }
}

static NTSTATUS
virtgpu_ioctl_destroy_hw_queue(struct virtgpu *gpu,
                               struct virtgpu_hw_queue *hw_queue)
{
   if (gpu->ddicb != NULL) {
      D3DDDICB_DESTROYHWQUEUE destroy_hw_queue = {
         .hHwQueue = hw_queue->h,
      };
      NTSTATUS status =
         hr_to_nt(gpu, gpu->ddicb->pKTCallbacks->pfnDestroyHwQueueCb(
                          gpu->ddicb->hRTDevice, &destroy_hw_queue));
      if (!NT_SUCCESS(status)) {
         return status;
      }
   } else {
      D3DKMT_DESTROYHWQUEUE destroy_hw_queue = {
         .hHwQueue = hw_queue->kmt,
      };
      NTSTATUS status = gpu->d3dkmt.cb.destroy_hw_queue(&destroy_hw_queue);
      if (!NT_SUCCESS(status)) {
         return status;
      }
   }
   memset(hw_queue, 0, sizeof(*hw_queue));

   return STATUS_SUCCESS;
}
#endif

static NTSTATUS
virtgpu_ioctl_destroy_context(struct virtgpu *gpu,
                              struct virtgpu_exec_context_base *ctx)
{
   if (gpu->ddicb != NULL) {
      D3DDDICB_DESTROYCONTEXT destroy = {
         .hContext = ctx->h,
      };
      return hr_to_nt(gpu, gpu->ddicb->pKTCallbacks->pfnDestroyContextCb(
                              gpu->ddicb->hRTDevice, &destroy));
   } else {
      D3DKMT_DESTROYCONTEXT destroy = {
         .hContext = ctx->kmt,
      };
      return gpu->d3dkmt.cb.destroy_context(&destroy);
   }
}

static NTSTATUS
virtgpu_ioctl_create_synchronization_object2(
   struct virtgpu *gpu,
   D3DDDI_SYNCHRONIZATIONOBJECTINFO2 *info,
   D3DKMT_HANDLE *sync_object)
{
   if (gpu->ddicb != NULL) {
      D3DDDICB_CREATESYNCHRONIZATIONOBJECT2 create = {
         .Info = *info,
      };
      NTSTATUS status = hr_to_nt(
         gpu, gpu->ddicb->pKTCallbacks->pfnCreateSynchronizationObject2Cb(
                 gpu->ddicb->hRTDevice, &create));
      *sync_object = create.hSyncObject;
      *info = create.Info;
      return status;
   } else {
      D3DKMT_CREATESYNCHRONIZATIONOBJECT2 create = {
         .hDevice = gpu->d3dkmt.device,
         .Info = *info,
      };
      NTSTATUS status =
         gpu->d3dkmt.cb.create_synchronization_object2(&create);
      *sync_object = create.hSyncObject;
      *info = create.Info;
      return status;
   }
}

static NTSTATUS
virtgpu_ioctl_open_synchronization_object_from_nt_handle(
   struct virtgpu *gpu,
   HANDLE nt_handle,
   D3DKMT_HANDLE *sync_object,
   uint64_t **value_map)
{
   assert(gpu->ddicb == NULL);

   D3DKMT_OPENSYNCOBJECTFROMNTHANDLE2 open_sync = {
       .hNtHandle = nt_handle,
       .hDevice = gpu->d3dkmt.device,
       .Flags = {
         .Shared = true,
         .NtSecuritySharing = true,
         .NoGPUAccess = true,
       },
   };

   NTSTATUS status =
      gpu->d3dkmt.cb.open_sync_object_from_nt_handle2(&open_sync);
   *sync_object = open_sync.hSyncObject;
   *value_map = open_sync.MonitoredFence.FenceValueCPUVirtualAddress;
   return status;
}

static NTSTATUS
virtgpu_ioctl_destroy_synchronization_object(struct virtgpu *gpu,
                                             D3DKMT_HANDLE sync_object)
{
   if (gpu->ddicb != NULL) {
      D3DDDICB_DESTROYSYNCHRONIZATIONOBJECT destroy = {
         .hSyncObject = sync_object,
      };
      return hr_to_nt(
         gpu, gpu->ddicb->pKTCallbacks->pfnDestroySynchronizationObjectCb(
                 gpu->ddicb->hRTDevice, &destroy));
   } else {
      D3DKMT_DESTROYSYNCHRONIZATIONOBJECT destroy = {
         .hSyncObject = sync_object,
      };
      return gpu->d3dkmt.cb.destroy_synchronization_object2(&destroy);
   }
}

static NTSTATUS
virtgpu_ioctl_signal_synchronization_objects_from_gpu(
   struct virtgpu *gpu,
   struct virtgpu_exec_context_base *ctx,
   D3DKMT_HANDLE *sync_objects,
   const uint64_t *values,
   uint32_t sync_count,
   bool allow_fence_rewind,
   bool signal_at_submission)
{
   if (gpu->ddicb != NULL) {
      D3DDDICB_SIGNALSYNCHRONIZATIONOBJECTFROMGPU2 signal = {
         .BroadcastContextCount = 1,
         .BroadcastContextArray = &ctx->h,
         .ObjectCount = sync_count,
         .ObjectHandleArray = sync_objects,
         .MonitoredFenceValueArray = values,
         .Flags = {
            .AllowFenceRewind = allow_fence_rewind,
            .SignalAtSubmission = signal_at_submission,
         },
      };
      return hr_to_nt(
         gpu,
         gpu->ddicb->pKTCallbacks->pfnSignalSynchronizationObjectFromGpu2Cb(
            gpu->ddicb->hRTDevice, &signal));
   } else {
      D3DKMT_SIGNALSYNCHRONIZATIONOBJECTFROMGPU2 signal = {
         .BroadcastContextCount = 1,
         .BroadcastContextArray = &ctx->kmt,
         .ObjectCount = sync_count,
         .ObjectHandleArray = sync_objects,
         .MonitoredFenceValueArray = values,
         .Flags = {
            .AllowFenceRewind = allow_fence_rewind,
            .SignalAtSubmission = signal_at_submission,
         },
      };
      return gpu->d3dkmt.cb.signal_synchronization_object_from_gpu2(&signal);
   }
}

static NTSTATUS
virtgpu_ioctl_wait_for_synchronization_objects_from_gpu(
   struct virtgpu *gpu,
   struct virtgpu_exec_context_base *ctx,
   D3DKMT_HANDLE *sync_objects,
   const uint64_t *values,
   uint32_t sync_count)
{
   if (gpu->ddicb != NULL) {
      D3DDDICB_WAITFORSYNCHRONIZATIONOBJECTFROMGPU wait = {
         .hContext = ctx->h,
         .ObjectCount = sync_count,
         .ObjectHandleArray = sync_objects,
         .MonitoredFenceValueArray = values,
      };
      return hr_to_nt(
         gpu,
         gpu->ddicb->pKTCallbacks->pfnWaitForSynchronizationObjectFromGpuCb(
            gpu->ddicb->hRTDevice, &wait));
   } else {
      D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMGPU wait = {
         .hContext = ctx->kmt,
         .ObjectCount = sync_count,
         .ObjectHandleArray = sync_objects,
         .MonitoredFenceValueArray = values,
      };
      return gpu->d3dkmt.cb.wait_for_synchronization_object_from_gpu(&wait);
   }
}

#ifdef VIRTUAL_CONTEXT_SUPPORT
static NTSTATUS
virtgpu_ioctl_submit_signal_synchronization_objects_to_hw_queue(
   struct virtgpu *gpu,
   struct virtgpu_hw_queue *hw_queue,
   D3DKMT_HANDLE *sync_objects,
   const uint64_t *values,
   uint32_t sync_count,
   bool allow_fence_rewind,
   bool signal_at_submission)
{
   if (gpu->ddicb != NULL) {
      D3DDDICB_SUBMITSIGNALSYNCOBJECTSTOHWQUEUE signal = {
         .BroadcastHwQueueCount = 1,
         .BroadcastHwQueueArray = &hw_queue->h,
         .ObjectCount = sync_count,
         .ObjectHandleArray = sync_objects,
         .FenceValueArray = values,
         .Flags = {
            .AllowFenceRewind = allow_fence_rewind,
            .SignalAtSubmission = signal_at_submission,
         },
      };
      return hr_to_nt(gpu,
         gpu->ddicb->pKTCallbacks->pfnSubmitSignalSyncObjectsToHwQueueCb(
            gpu->ddicb->hRTDevice, &signal));
   } else {
      D3DKMT_SUBMITSIGNALSYNCOBJECTSTOHWQUEUE signal = {
         .BroadcastHwQueueCount = 1,
         .BroadcastHwQueueArray = &hw_queue->kmt,
         .ObjectCount = sync_count,
         .ObjectHandleArray = sync_objects,
         .FenceValueArray = values,
         .Flags = {
            .AllowFenceRewind = allow_fence_rewind,
            .SignalAtSubmission = signal_at_submission,
         },
      };
      return gpu->d3dkmt.cb.submit_signal_sync_objects_to_hw_queue(&signal);
   }
}

static NTSTATUS
virtgpu_ioctl_submit_wait_for_synchronization_objects_to_hw_queue(
   struct virtgpu *gpu,
   struct virtgpu_hw_queue *hw_queue,
   D3DKMT_HANDLE *sync_objects,
   const uint64_t *values,
   uint32_t sync_count)
{
   if (gpu->ddicb != NULL) {
      D3DDDICB_SUBMITWAITFORSYNCOBJECTSTOHWQUEUE wait = {
         .hHwQueue = hw_queue->h,
         .ObjectCount = sync_count,
         .ObjectHandleArray = sync_objects,
         .FenceValueArray = values,
      };
      return hr_to_nt(
         gpu,
         gpu->ddicb->pKTCallbacks->pfnSubmitWaitForSyncObjectsToHwQueueCb(
            gpu->ddicb->hRTDevice, &wait));
   } else {
      D3DKMT_SUBMITWAITFORSYNCOBJECTSTOHWQUEUE wait = {
         .hHwQueue = hw_queue->kmt,
         .ObjectCount = sync_count,
         .ObjectHandleArray = sync_objects,
         .FenceValueArray = values,
      };
      return gpu->d3dkmt.cb.submit_wait_for_sync_objects_to_hw_queue(&wait);
   }
}
#endif

static NTSTATUS
virtgpu_ioctl_signal_synchronization_object_from_cpu(struct virtgpu *gpu,
                                                     D3DKMT_HANDLE sync_object,
                                                     uint64_t val,
                                                     bool allow_rewind)
{
   if (gpu->ddicb != NULL) {
      if (allow_rewind) {
         /* This is stupid, but there doesn't seem to be another way */
         struct virtgpu_exec_context_virtual context = {};
         NTSTATUS status = virtgpu_ioctl_create_context_virtual(
            gpu, VIRTIO_WDDM_NODE_HW_QUEUE, virtgpu_uses_hw_queue(gpu),
            &context);
         if (!NT_SUCCESS(status)) {
            vn_log(gpu->instance, "could create sync context: 0x%lx", status);
            return status;
         }

         status = virtgpu_ioctl_signal_synchronization_objects_from_gpu(
            gpu, &context.base, &sync_object, &val, 1, true, false);
         if (!NT_SUCCESS(status)) {
            return status;
         }

         return virtgpu_ioctl_destroy_context(gpu, &context.base);
      }
      D3DDDICB_SIGNALSYNCHRONIZATIONOBJECTFROMCPU signal = {
         .ObjectCount = 1,
         .ObjectHandleArray = &sync_object,
         .FenceValueArray = &val,
      };
      return hr_to_nt(
         gpu,
         gpu->ddicb->pKTCallbacks->pfnSignalSynchronizationObjectFromCpuCb(
            gpu->ddicb->hRTDevice, &signal));
   } else {
      D3DKMT_SIGNALSYNCHRONIZATIONOBJECTFROMCPU signal = {
         .hDevice = gpu->d3dkmt.device,
         .ObjectCount = 1,
         .ObjectHandleArray = &sync_object,
         .FenceValueArray = &val,
         .Flags = {
            .AllowFenceRewind = allow_rewind,
         },
      };
      return gpu->d3dkmt.cb.signal_synchronization_object_from_cpu(&signal);
   }
}

static NTSTATUS
virtgpu_ioctl_wait_for_synchronization_objects_from_cpu(
   struct virtgpu *gpu,
   D3DKMT_HANDLE *sync_objects,
   const uint64_t *values,
   uint32_t sync_count,
   bool wait_any,
   HANDLE async_event)
{
   if (gpu->ddicb != NULL) {
      D3DDDICB_WAITFORSYNCHRONIZATIONOBJECTFROMCPU wait = {
         .ObjectCount = sync_count,
         .ObjectHandleArray = sync_objects,
         .FenceValueArray = values,
         .hAsyncEvent = async_event,
         .Flags = {
            .WaitAny = wait_any,
         },
      };
      return hr_to_nt(
         gpu,
         gpu->ddicb->pKTCallbacks->pfnWaitForSynchronizationObjectFromCpuCb(
            gpu->ddicb->hRTDevice, &wait));
   } else {
      D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU signal = {
         .hDevice = gpu->d3dkmt.device,
         .ObjectCount = sync_count,
         .ObjectHandleArray = sync_objects,
         .FenceValueArray = values,
         .hAsyncEvent = async_event,
         .Flags = {
            .WaitAny = wait_any,
         },
      };
      return gpu->d3dkmt.cb.wait_for_synchronization_object_from_cpu(&signal);
   }
}

static NTSTATUS
virtgpu_ioctl_share_objects(struct virtgpu *gpu,
                            uint32_t flags,
                            const D3DKMT_HANDLE *handles,
                            uint32_t num_handles,
                            HANDLE *nt_handle)
{
   assert(gpu->ddicb == NULL);
   OBJECT_ATTRIBUTES attr;
   InitializeObjectAttributes(&attr, NULL, OBJ_INHERIT, NULL, NULL);
   return gpu->d3dkmt.cb.share_objects(num_handles, handles, &attr, flags,
                                       nt_handle);
}

static NTSTATUS
virtgpu_ioctl_destroy_device(struct virtgpu *gpu)
{
   D3DKMT_DESTROYDEVICE destroy = {
      .hDevice = gpu->d3dkmt.device,
   };
   return gpu->d3dkmt.cb.destroy_device(&destroy);
}

static NTSTATUS
virtgpu_ioctl_close_adapter(struct virtgpu *gpu)
{
   D3DKMT_CLOSEADAPTER close = {
      .hAdapter = gpu->d3dkmt.adapter,
   };
   return gpu->d3dkmt.cb.close_adapter(&close);
}

static VkResult
virtgpu_sync_create(struct vn_renderer *renderer,
                    uint64_t initial_val,
                    struct vn_renderer_sync **out_sync)
{
   struct virtgpu *gpu = (struct virtgpu *)renderer;

   // TODO: this needs to be a parameter
   const bool shared = true;

   D3DDDI_SYNCHRONIZATIONOBJECTINFO2 sync_info = {
      .Type = D3DDDI_MONITORED_FENCE,
      .Flags = {
         .Shared = shared,
         .NtSecuritySharing = shared,
         .NoGPUAccess = true,
      },
      .MonitoredFence = {
         .InitialFenceValue = initial_val,
      },
   };

   D3DKMT_HANDLE sync_object = 0;
   NTSTATUS status = virtgpu_ioctl_create_synchronization_object2(
      gpu, &sync_info, &sync_object);
   if (!NT_SUCCESS(status)) {
      vn_log(gpu->instance, "failed to create monitored fence: 0x%lx",
             status);
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   }

   struct virtgpu_sync *sync = calloc(1, sizeof(*sync));
   if (!sync) {
      virtgpu_ioctl_destroy_synchronization_object(gpu, sync_object);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }

   sync->handle = sync_object;
   sync->value_map = sync_info.MonitoredFence.FenceValueCPUVirtualAddress;
   /* TODO: handle might as well live here */
   sync->base.sync_id = 0;

   *out_sync = &sync->base;

   return VK_SUCCESS;
}

static VkResult
virtgpu_sync_create_from_handle(struct vn_renderer *renderer,
                                void *handle,
                                VkExternalSemaphoreHandleTypeFlagBits type,
                                struct vn_renderer_sync **out_sync)
{
   struct virtgpu *gpu = (struct virtgpu *)renderer;

   assert(type == VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT ||
          type == VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE_BIT);

   /* We cannot really do this in UMD */
   assert(gpu->ddicb == NULL);
   uint64_t *value_map = NULL;
   D3DKMT_HANDLE sync_object = 0;
   NTSTATUS status = virtgpu_ioctl_open_synchronization_object_from_nt_handle(
      gpu, handle, &sync_object, &value_map);
   if (!NT_SUCCESS(status)) {
      vn_log(gpu->instance, "failed to create monitored fence: 0x%lx",
             status);
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   }

   struct virtgpu_sync *sync = calloc(1, sizeof(*sync));
   if (!sync) {
      virtgpu_ioctl_destroy_synchronization_object(gpu, sync_object);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }

   sync->handle = sync_object;
   sync->value_map = value_map;
   /* TODO: handle might as well live here */
   sync->base.sync_id = 0;

   *out_sync = &sync->base;

   return VK_SUCCESS;
}

static void
virtgpu_sync_destroy(struct vn_renderer *renderer,
                     struct vn_renderer_sync *_sync)
{
   struct virtgpu *gpu = (struct virtgpu *)renderer;
   struct virtgpu_sync *sync = (struct virtgpu_sync *)_sync;

   NTSTATUS status =
      virtgpu_ioctl_destroy_synchronization_object(gpu, sync->handle);
   if (!NT_SUCCESS(status)) {
      vn_log(gpu->instance, "failed to destroy monitored fence: 0x%lx",
             status);
   }

   free(sync);
}

static void *
virtgpu_sync_export_handle(struct vn_renderer *renderer,
                           struct vn_renderer_sync *_sync,
                           VkExternalSemaphoreHandleTypeFlagBits type,
                           uint64_t async_event_value)
{
   struct virtgpu *gpu = (struct virtgpu *)renderer;
   struct virtgpu_sync *sync = (struct virtgpu_sync *)_sync;

   switch (type) {
   /* This is a "sync file": event which will be signalled once the underlying
    * object reaches the specified value */
   case VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT: {
      HANDLE async_event = CreateEventA(NULL, true, false, NULL);
      if (!async_event) {
         vn_log(gpu->instance, "failed to create async event");
         return NULL;
      }

      NTSTATUS status =
         virtgpu_ioctl_wait_for_synchronization_objects_from_cpu(
            gpu, &sync->handle, &async_event_value, 1, false, async_event);
      if (!NT_SUCCESS(status)) {
         vn_log(gpu->instance, "failed to wait for monitored fence: 0x%lx",
                status);
         CloseHandle(async_event);
         return NULL;
      }
      return async_event;
   }

   case VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT:
   case VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE_BIT:
      if (gpu->ddicb != NULL) {
         /* This is currently only exposed for D3D11 UMD */
         assert(gpu->ddicb != NULL);
         return (void *)(intptr_t)sync->handle;
      } else {
         HANDLE nt_handle = NULL;
         NTSTATUS status = virtgpu_ioctl_share_objects(
            gpu, D3DDDI_SYNC_OBJECT_ALL_ACCESS, &sync->handle, 1, &nt_handle);
         if (!NT_SUCCESS(status)) {
            vn_log(gpu->instance, "failed to share syncobj: 0x%lx", status);
            return NULL;
         }
         return nt_handle;
      }

   default:
      return NULL;
   }
}

static VkResult
virtgpu_sync_reset(struct vn_renderer *renderer,
                   struct vn_renderer_sync *_sync)
{
   struct virtgpu *gpu = (struct virtgpu *)renderer;
   struct virtgpu_sync *sync = (struct virtgpu_sync *)_sync;

   if (gpu->ddicb != NULL) {
      /* rewinding is not allowed in D3D UMD, we have to create a new syncobj */
      virtgpu_ioctl_destroy_synchronization_object(gpu, sync->handle);
      D3DDDI_SYNCHRONIZATIONOBJECTINFO2 sync_info = {
         .Type = D3DDDI_MONITORED_FENCE,
         .Flags = {
            .NoGPUAccess = true,
         },
         .MonitoredFence = {
            .InitialFenceValue = 0,
         },
      };

      D3DKMT_HANDLE sync_object = 0;
      NTSTATUS status = virtgpu_ioctl_create_synchronization_object2(
         gpu, &sync_info, &sync_object);
      if (!NT_SUCCESS(status)) {
         vn_log(gpu->instance,
                "failed to create monitored fence for reset: 0x%lx", status);
         return VK_ERROR_UNKNOWN;
      }
      sync->handle = sync_object;
      sync->value_map = sync_info.MonitoredFence.FenceValueCPUVirtualAddress;

      return VK_SUCCESS;
   }

   NTSTATUS status = virtgpu_ioctl_signal_synchronization_object_from_cpu(
      gpu, sync->handle, 0, true);
   if (!NT_SUCCESS(status)) {
      vn_log(gpu->instance, "failed to reset monitored fence: 0x%lx", status);
      return VK_ERROR_UNKNOWN;
   }

   return VK_SUCCESS;
}

static VkResult
virtgpu_sync_read(struct vn_renderer *renderer,
                  struct vn_renderer_sync *_sync,
                  uint64_t *val)
{
   // struct virtgpu *gpu = (struct virtgpu *)renderer;
   struct virtgpu_sync *sync = (struct virtgpu_sync *)_sync;

   *val = p_atomic_read(sync->value_map);

   return VK_SUCCESS;
}

static VkResult
virtgpu_sync_write(struct vn_renderer *renderer,
                   struct vn_renderer_sync *_sync,
                   uint64_t val)
{
   struct virtgpu *gpu = (struct virtgpu *)renderer;
   struct virtgpu_sync *sync = (struct virtgpu_sync *)_sync;

   // Windows can't handle signal 1 if signal 2 is already in flight even when
   // it's blocked by wait 1
   NTSTATUS status = virtgpu_ioctl_signal_synchronization_object_from_cpu(
      gpu, sync->handle, val, true);
   if (!NT_SUCCESS(status)) {
      vn_log(gpu->instance,
             "failed to signal monitored fence from CPU: 0x%lx", status);
      return VK_ERROR_UNKNOWN;
   }

   // CPU signal may return before the syncobj is actually signalled to that
   // value
   status = virtgpu_ioctl_wait_for_synchronization_objects_from_cpu(
      gpu, &sync->handle, &val, 1, false, NULL);
   if (!NT_SUCCESS(status)) {
      vn_log(gpu->instance,
             "failed to signal monitored fence from CPU: 0x%lx", status);
      return VK_ERROR_UNKNOWN;
   }

   return VK_SUCCESS;
}

#ifdef VIRTUAL_CONTEXT_SUPPORT
static NTSTATUS
virtgpu_ioctl_submit(struct virtgpu *gpu,
                     struct virtgpu_exec_context_virtual *ctx,
                     uint64_t cmd_buf,
                     size_t cmd_length,
                     void *priv,
                     unsigned priv_size)
{
   if (gpu->ddicb != NULL) {
      D3DDDICB_SUBMITCOMMAND submit = {
         .Commands = cmd_buf,
         .CommandLength = cmd_length,
         .BroadcastContextCount = 1,
         .BroadcastContext = {
            ctx->base.h,
         },
         .pPrivateDriverData = priv,
         .PrivateDriverDataSize = priv_size,
      };

      return hr_to_nt(gpu, gpu->ddicb->pKTCallbacks->pfnSubmitCommandCb(
                              gpu->ddicb->hRTDevice, &submit));
   } else {
      D3DKMT_SUBMITCOMMAND submit = {
         .Commands = cmd_buf,
         .CommandLength = cmd_length,
         .BroadcastContextCount = 1,
         .BroadcastContext = {
            ctx->base.kmt,
         },
         .pPrivateDriverData = priv,
         .PrivateDriverDataSize = priv_size,
      };

      return gpu->d3dkmt.cb.submit_command(&submit);
   }
}
#else
static NTSTATUS
virtgpu_ioctl_render(struct virtgpu *gpu,
                     struct virtgpu_exec_context *ctx,
                     unsigned cmd_offset,
                     unsigned cmd_length,
                     unsigned alloc_count,
                     void *priv,
                     unsigned priv_size)
{
   if (gpu->ddicb != NULL) {
      D3DDDICB_RENDER render = {
         .hContext = ctx->base.h,
         .CommandOffset = cmd_offset,
         .CommandLength = cmd_length,
         .NumAllocations = alloc_count,
         .NumPatchLocations = alloc_count,
         .pPrivateDriverData = priv,
         .PrivateDriverDataSize = priv_size,
      };

      NTSTATUS status = hr_to_nt(gpu, gpu->ddicb->pKTCallbacks->pfnRenderCb(
                                         gpu->ddicb->hRTDevice, &render));

      ctx->cmd_buf = render.pNewCommandBuffer;
      ctx->cmd_size = render.NewCommandBufferSize;

      ctx->alloc_list = render.pNewAllocationList;
      ctx->alloc_size = render.NewAllocationListSize;

      ctx->patch_list = render.pNewPatchLocationList;
      ctx->patch_size = render.NewPatchLocationListSize;

      return status;
   } else {
      D3DKMT_RENDER render = {
         .hContext = ctx->base.kmt,
         .CommandOffset = cmd_offset,
         .CommandLength = cmd_length,
         .AllocationCount = alloc_count,
         .PatchLocationCount = alloc_count,
         .pPrivateDriverData = priv,
         .PrivateDriverDataSize = priv_size,
      };

      NTSTATUS status = gpu->d3dkmt.cb.render(&render);

      ctx->cmd_buf = render.pNewCommandBuffer;
      ctx->cmd_size = render.NewCommandBufferSize;

      ctx->alloc_list = render.pNewAllocationList;
      ctx->alloc_size = render.NewAllocationListSize;

      ctx->patch_list = render.pNewPatchLocationList;
      ctx->patch_size = render.NewPatchLocationListSize;

      return status;
   }
}
#endif

#ifdef VIRTUAL_CONTEXT_SUPPORT
static NTSTATUS
virtgpu_ioctl_submit_to_hw_queue(struct virtgpu *gpu,
                                 struct virtgpu_hw_queue *hw_queue,
                                 uint64_t cmd_buf,
                                 size_t cmd_length,
                                 void *priv,
                                 unsigned priv_size)
{
   NTSTATUS status = STATUS_SUCCESS;

   // TODO: can we get away without a lock?
   mtx_lock(&gpu->hwq_submit_lock);

   if (gpu->ddicb != NULL) {
      D3DDDICB_SUBMITCOMMANDTOHWQUEUE submit = {
         .hHwQueue = hw_queue->h,
         .HwQueueProgressFenceId = ++gpu->pending_submission_value,
         .Commands = cmd_buf,
         .CommandLength = cmd_length,
         .pPrivateDriverData = priv,
         .PrivateDriverDataSize = priv_size,
      };

      status = hr_to_nt(gpu,
         gpu->ddicb->pKTCallbacks->pfnSubmitCommandToHwQueueCb(
            gpu->ddicb->hRTDevice, &submit));
   } else {
      D3DKMT_SUBMITCOMMANDTOHWQUEUE submit = {
         .hHwQueue = hw_queue->kmt,
         .HwQueueProgressFenceId = ++gpu->pending_submission_value,
         .CommandBuffer = cmd_buf,
         .CommandLength = cmd_length,
         .pPrivateDriverData = priv,
         .PrivateDriverDataSize = priv_size,
      };

      status = gpu->d3dkmt.cb.submit_command_to_hw_queue(&submit);
   }

   mtx_unlock(&gpu->hwq_submit_lock);

   return status;
}
#endif

static NTSTATUS
virtgpu_ioctl_query(struct virtgpu *gpu, void *priv, unsigned priv_size)
{
   if (gpu->ddicb != NULL) {
      D3DDDICB_QUERYADAPTERINFO query = {
         .pPrivateDriverData = priv,
         .PrivateDriverDataSize = priv_size,
      };
      return hr_to_nt(gpu,
                      gpu->ddicb->pAdapterCallbacks->pfnQueryAdapterInfoCb(
                         gpu->ddicb->hRTAdapter, &query));
   } else {
      D3DKMT_QUERYADAPTERINFO query = {
         .hAdapter = gpu->d3dkmt.adapter,
         .Type = KMTQAITYPE_UMDRIVERPRIVATE,
         .pPrivateDriverData = priv,
         .PrivateDriverDataSize = priv_size,
      };

      return gpu->d3dkmt.cb.query_adapter_info(&query);
   }
}

static NTSTATUS
virtgpu_ioctl_pci_address(struct virtgpu *gpu,
                          uint16_t *domain,
                          uint8_t *bus,
                          uint8_t *device,
                          uint8_t *function)
{
   /* These are dummy values */
   *domain = 0;
   *bus = 0;
   *device = 0;
   *function = 0;

   if (gpu->ddicb != NULL) {
      return STATUS_SUCCESS;
   }

   D3DKMT_ADAPTERADDRESS address = {};
   D3DKMT_QUERYADAPTERINFO query = {
      .hAdapter = gpu->d3dkmt.adapter,
      .Type = KMTQAITYPE_ADAPTERADDRESS,
      .pPrivateDriverData = &address,
      .PrivateDriverDataSize = sizeof(address),
   };

   NTSTATUS status = gpu->d3dkmt.cb.query_adapter_info(&query);
   if (NT_SUCCESS(status)) {
      *bus = address.BusNumber;
      *device = address.DeviceNumber;
      *function = address.FunctionNumber;
   }
   return status;
}

static NTSTATUS
virtgpu_ioctl_escape(struct virtgpu *gpu,
                     struct virtgpu_exec_context_base *ctx,
                     void *priv,
                     size_t priv_size)
{
   if (gpu->ddicb != NULL) {
      D3DDDICB_ESCAPE escape = {
         .hDevice = gpu->ddicb->hRTDevice,
         .pPrivateDriverData = priv,
         .PrivateDriverDataSize = priv_size,
         .hContext = ctx ? ctx->h : NULL,
      };
      return hr_to_nt(gpu, gpu->ddicb->pKTCallbacks->pfnEscapeCb(
                              gpu->ddicb->hRTAdapter, &escape));
   } else {
      D3DKMT_ESCAPE escape = {
         .hAdapter = gpu->d3dkmt.adapter,
         .hDevice = gpu->d3dkmt.device,
         .hContext = ctx ? ctx->kmt : 0,
         .Type = D3DKMT_ESCAPE_DRIVERPRIVATE,
         .pPrivateDriverData = priv,
         .PrivateDriverDataSize = priv_size,
      };

      return gpu->d3dkmt.cb.escape(&escape);
   }
}

static NTSTATUS
virtgpu_ioctl_get_device_state(struct virtgpu *gpu) {
   if (gpu->ddicb != NULL) {
      D3DDDI_EXECUTIONSTATEESCAPE state = {};
      D3DDDICB_ESCAPE escape = {
         .hDevice = gpu->ddicb->hRTDevice,
         .Flags = {
             .DeviceStatusQuery = 1,
         },
         .pPrivateDriverData = &state,
         .PrivateDriverDataSize = sizeof(state),
      };

      NTSTATUS status = hr_to_nt(gpu, gpu->ddicb->pKTCallbacks->pfnEscapeCb(
                          gpu->ddicb->hRTAdapter, &escape));
      if (status == STATUS_DEVICE_REMOVED)
         return status;
      else if (state.State != D3DDDI_DEVICEEXECUTION_ACTIVE)
         return STATUS_DEVICE_REMOVED;
      else
         return status;
   } else {
      D3DKMT_GETDEVICESTATE state = {
         .hDevice = gpu->d3dkmt.device,
         .StateType = D3DKMT_DEVICESTATE_EXECUTION,
      };

      NTSTATUS status = gpu->d3dkmt.cb.get_device_state(&state);
      if (status == STATUS_DEVICE_REMOVED)
         return status;
      else if (state.ExecutionState != D3DKMT_DEVICEEXECUTION_ACTIVE)
         return STATUS_DEVICE_REMOVED;
      else
         return status;
   }
}

static NTSTATUS
virtgpu_ioctl_get_caps(struct virtgpu *gpu,
                       uint32_t id,
                       uint32_t version,
                       void *capset,
                       size_t capset_size)
{
   const size_t size = sizeof(VIRTIO_WDDM_Capset) + capset_size;
   VIRTIO_WDDM_Capset *caps = calloc(1, size);
   if (!caps)
      return STATUS_NO_MEMORY;

   *caps = (VIRTIO_WDDM_Capset){
      .tag = VIRTIO_WDDM_ESCAPE_CAPSET_TAG,
      .capset_id = id,
      .version = version,
   };

   NTSTATUS status = virtgpu_ioctl_escape(gpu, NULL, caps, size);
   if (NT_SUCCESS(status))
      memcpy(capset, caps->capset, capset_size);

   free(caps);
   return status;
}

static NTSTATUS
virtgpu_ioctl_create_paging_queue(struct virtgpu *gpu)
{
   if (gpu->ddicb != NULL) {
      D3DDDICB_CREATEPAGINGQUEUE paging_queue = {
         .Priority = D3DDDI_PAGINGQUEUE_PRIORITY_NORMAL,
         .PhysicalAdapterIndex = 0,
      };

      NTSTATUS status =
         hr_to_nt(gpu, gpu->ddicb->pKTCallbacks->pfnCreatePagingQueueCb(
                          gpu->ddicb->hRTDevice, &paging_queue));
      if (!NT_SUCCESS(status)) {
         return status;
      }

      gpu->paging.queue = paging_queue.hPagingQueue;
      gpu->paging.sync_object = paging_queue.hSyncObject;
      gpu->paging.fence_value = paging_queue.FenceValueCPUVirtualAddress;

      return STATUS_SUCCESS;
   } else {
      D3DKMT_CREATEPAGINGQUEUE paging_queue = {
         .hDevice = gpu->d3dkmt.device,
         .Priority = D3DDDI_PAGINGQUEUE_PRIORITY_NORMAL,
         .PhysicalAdapterIndex = 0,
      };

      NTSTATUS status = gpu->d3dkmt.cb.create_paging_queue(&paging_queue);
      if (!NT_SUCCESS(status)) {
         return status;
      }

      gpu->paging.queue = paging_queue.hPagingQueue;
      gpu->paging.sync_object = paging_queue.hSyncObject;
      gpu->paging.fence_value = paging_queue.FenceValueCPUVirtualAddress;

      return STATUS_SUCCESS;
   }
}

static NTSTATUS
virtgpu_ioctl_resource_make_resident(struct virtgpu *gpu,
                                     D3DKMT_HANDLE alloc_handle,
                                     uint64_t *paging_fence_value)
{
   D3DDDI_MAKERESIDENT make_resident = {
      .hPagingQueue = gpu->paging.queue,
      .NumAllocations = 1,
      .AllocationList = &alloc_handle,
      .PriorityList = NULL,
      .Flags = {
         .CantTrimFurther = 1,
         .MustSucceed = 1,
      },
   };

   NTSTATUS status = STATUS_SUCCESS;

   if (gpu->ddicb != NULL) {
      status = hr_to_nt(gpu, gpu->ddicb->pKTCallbacks->pfnMakeResidentCb(
                                gpu->ddicb->hRTDevice, &make_resident));
      if (!NT_SUCCESS(status)) {
         return status;
      }
   } else {
      status = gpu->d3dkmt.cb.make_resident(&make_resident);
      if (!NT_SUCCESS(status)) {
         return status;
      }
   }
   if (status == STATUS_PENDING) {
      *paging_fence_value = make_resident.PagingFenceValue;
   }

   return status;
}

static NTSTATUS
virtgpu_ioctl_wait_from_cpu(struct virtgpu *gpu,
                            D3DKMT_HANDLE sync_object,
                            uint64_t fence_value)
{
   if (gpu->ddicb != NULL) {
      D3DDDICB_WAITFORSYNCHRONIZATIONOBJECTFROMCPU wait = {
         .ObjectCount = 1,
         .ObjectHandleArray = &sync_object,
         .FenceValueArray = &fence_value,
      };

      return hr_to_nt(
         gpu,
         gpu->ddicb->pKTCallbacks->pfnWaitForSynchronizationObjectFromCpuCb(
            gpu->ddicb->hRTDevice, &wait));
   } else {
      D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU wait = {
         .hDevice = gpu->d3dkmt.device,
         .ObjectCount = 1,
         .ObjectHandleArray = &sync_object,
         .FenceValueArray = &fence_value,
      };

      return gpu->d3dkmt.cb.wait_for_synchronization_object_from_cpu(&wait);
   }
}

static NTSTATUS
virtgpu_ioctl_resource_create_blob(struct virtgpu *gpu,
                                   VIRTIO_WDDM_CreateAllocation *alloc_priv,
                                   size_t alloc_priv_size,
                                   const VkD3DDDICreateResourceMESA *d3d_create,
                                   bool nt_shared,
                                   struct vn_renderer_submit_batch *batch,
                                   D3DKMT_HANDLE *alloc_handle,
                                   D3DKMT_HANDLE *res_kmt_local,
                                   D3DKMT_HANDLE *res_kmt_global,
                                   HANDLE *res_h)
{
   struct {
      VIRTIO_WDDM_CreateResource resource;
      VIRTIO_WDDM_CommandHeader hdr;
      uint8_t body[MAX_COMMAND_SIZE];
   } res_priv;
   res_priv.resource.tag = VIRTIO_WDDM_CREATE_RESOURCE_TAG;

   size_t res_priv_size = sizeof(res_priv.resource);

   if (batch != NULL && batch->cs_size > 0) {
      /* TODO */
      assert(batch->sync_count == 0);
      /* Kernel does not yet support buffers bigger than
       * PAGE_SIZE - sizeof(struct virtio_gpu_cmd_submit) */
      if (batch->cs_size > MAX_COMMAND_SIZE)
         return STATUS_INVALID_PARAMETER;

      res_priv_size =
         sizeof(res_priv.resource) + sizeof(res_priv.hdr) + batch->cs_size;

      res_priv.hdr.id = VIRTIO_WDDM_COMMAND_ID_SUBMIT;
      res_priv.hdr.flags = VIRTIO_WDDM_COMMAND_FLAG_RING_IDX;
      res_priv.hdr.ring = batch->ring_idx;
      res_priv.hdr.size = batch->cs_size;
      memcpy(res_priv.body, batch->cs_data, batch->cs_size);
   }

   D3DDDI_ALLOCATIONINFO alloc_info = {
      .pPrivateDriverData = alloc_priv,
      .PrivateDriverDataSize = alloc_priv_size,
   };

   bool is_shareable =
      !!(alloc_priv->base.blob.flags & VIRTIO_WDDM_BLOB_FLAG_SHAREABLE);

   if (gpu->ddicb != NULL) {
      if (d3d_create && d3d_create->pCreateResource11->pPrimaryDesc) {
          alloc_info.VidPnSourceId = d3d_create->pCreateResource11->pPrimaryDesc->VidPnSourceId;
          alloc_info.Flags.Primary = true;
      }

      D3DDDICB_ALLOCATE alloc = {
         .pPrivateDriverData = &res_priv,
         .PrivateDriverDataSize = res_priv_size,
         .hResource = *res_h,
         .NumAllocations = 1,
         .pAllocationInfo = &alloc_info,
      };

      NTSTATUS status = hr_to_nt(gpu, gpu->ddicb->pKTCallbacks->pfnAllocateCb(
                                         gpu->ddicb->hRTDevice, &alloc));
      if (!NT_SUCCESS(status)) {
         return status;
      }
      *res_kmt_local = alloc.hKMResource;
      *alloc_handle = alloc_info.hAllocation;
   } else {
      D3DKMT_CREATEALLOCATION alloc = {
         .hDevice = gpu->d3dkmt.device,
         .pPrivateDriverData = &res_priv,
         .PrivateDriverDataSize = res_priv_size,
         .NumAllocations = 1,
         .pAllocationInfo = &alloc_info,
         .Flags = {
            .CreateResource = 1,
            .CreateShared = is_shareable,
            .NtSecuritySharing = nt_shared,
         },
      };
      NTSTATUS status = gpu->d3dkmt.cb.create_allocation(&alloc);
      if (!NT_SUCCESS(status)) {
         return status;
      }
      *res_kmt_local = alloc.hResource;
      if (res_kmt_global) {
         *res_kmt_global = alloc.hGlobalShare;
      }
      *alloc_handle = alloc_info.hAllocation;
   }

   uint64_t paging_fence_value = 0;
   NTSTATUS status = virtgpu_ioctl_resource_make_resident(
      gpu, *alloc_handle, &paging_fence_value);
   if (status == STATUS_PENDING) {
      status = virtgpu_ioctl_wait_from_cpu(gpu, gpu->paging.sync_object,
                                           paging_fence_value);
   }
   if (!NT_SUCCESS(status)) {
      return status;
   }
   assert(alloc_priv->res_id != 0);

   return STATUS_SUCCESS;
}

static NTSTATUS
virtgpu_ioctl_resource_destroy_blob(struct virtgpu *gpu,
                                    D3DKMT_HANDLE alloc_handle,
                                    D3DKMT_HANDLE res_kmt,
                                    HANDLE res_h)
{
   if (gpu->ddicb != NULL) {
      D3DDDICB_DEALLOCATE destroy = {
         .hResource = res_h,
         .NumAllocations = res_h == NULL ? 1 : 0,
         .HandleList = res_h == NULL ? &alloc_handle : NULL,
      };

      NTSTATUS status =
         hr_to_nt(gpu, gpu->ddicb->pKTCallbacks->pfnDeallocateCb(
                          gpu->ddicb->hRTDevice, &destroy));

      if (!NT_SUCCESS(status)) {
         return status;
      }
   } else {
      D3DKMT_DESTROYALLOCATION destroy = {
         .hDevice = gpu->d3dkmt.device,
         .hResource = res_kmt,
         .AllocationCount = res_kmt == 0 ? 1 : 0,
         .phAllocationList = res_kmt == 0 ? &alloc_handle : NULL,
      };

      NTSTATUS status = gpu->d3dkmt.cb.destroy_allocation(&destroy);
      if (!NT_SUCCESS(status)) {
         return status;
      }
   }

   return STATUS_SUCCESS;
}

static NTSTATUS
virtgpu_ioctl_lock(struct virtgpu *gpu, D3DKMT_HANDLE handle, void **ptr)
{
   if (gpu->ddicb != NULL) {
      D3DDDICB_LOCK lock = {
          .hAllocation = handle,
          .Flags = {
              // .IgnoreSync = 1,
              .LockEntire = 1,
          },
      };
      NTSTATUS status = hr_to_nt(gpu, gpu->ddicb->pKTCallbacks->pfnLockCb(
                                         gpu->ddicb->hRTDevice, &lock));
      if (!NT_SUCCESS(status)) {
         return status;
      }
      *ptr = lock.pData;
   } else {
      D3DKMT_LOCK lock = {
         .hDevice = gpu->d3dkmt.device,
         .Flags = {
            // .IgnoreSync = 1,
            .LockEntire = 1,
         },
         .hAllocation = handle,
      };
      NTSTATUS status = gpu->d3dkmt.cb.lock(&lock);
      if (!NT_SUCCESS(status)) {
         return status;
      }
      *ptr = lock.pData;
   }

   return STATUS_SUCCESS;
}

static NTSTATUS
virtgpu_ioctl_unlock(struct virtgpu *gpu, D3DKMT_HANDLE handle)
{
   if (gpu->ddicb != NULL) {
      D3DDDICB_UNLOCK unlock = {
         .NumAllocations = 1,
         .phAllocations = &handle,
      };
      return hr_to_nt(gpu, gpu->ddicb->pKTCallbacks->pfnUnlockCb(
                              gpu->ddicb->hRTDevice, &unlock));
   } else {
      D3DKMT_UNLOCK unlock = {
         .hDevice = gpu->d3dkmt.device,
         .NumAllocations = 1,
         .phAllocations = &handle,
      };
      return gpu->d3dkmt.cb.unlock(&unlock);
   }
}

static NTSTATUS
virtgpu_ioctl_map(struct virtgpu *gpu, D3DKMT_HANDLE handle, void **ptr)
{
   VIRTIO_WDDM_Escape blob_map = {
      .blob_map = {
         .tag = VIRTIO_WDDM_ESCAPE_BLOB_MAP_TAG,
         .handle = handle,
         .flags = 0,
      },
   };

   NTSTATUS status =
      virtgpu_ioctl_escape(gpu, NULL, &blob_map, sizeof(blob_map));
   *ptr = blob_map.blob_map.ptr.ptr;
   return status;
}

static NTSTATUS
virtgpu_ioctl_unmap(struct virtgpu *gpu, D3DKMT_HANDLE handle, void *ptr)
{
   VIRTIO_WDDM_Escape blob_map = {
      .blob_map = {
         .tag = VIRTIO_WDDM_ESCAPE_BLOB_MAP_TAG,
         .handle = handle,
         .flags = VIRTIO_WDDM_BLOB_MAP_FLAGS_UNMAP,
         .ptr = {
             .ptr = ptr,
         },
      },
   };

   return virtgpu_ioctl_escape(gpu, NULL, &blob_map, sizeof(blob_map));
}

static inline void
virtgpu_init_shmem_blob_mem(ASSERTED struct virtgpu *gpu)
{
   /* VIRTIO_WDDM_BLOB_MEM_GUEST allocates from the guest system memory.  They
    * are logically contiguous in the guest but are sglists (iovecs) in the
    * host. That makes them slower to process in the host.  With host process
    * isolation, it also becomes impossible for the host to access sglists
    * directly.
    *
    * While there are ideas (and shipped code in some cases) such as creating
    * udmabufs from sglists, or having a dedicated guest heap, it seems the
    * easiest way is to reuse VIRTIO_WDDM_BLOB_MEM_HOST3D. That is, when the
    * renderer sees a request to export a blob where
    *
    *  - blob_mem is VIRTIO_WDDM_BLOB_MEM_HOST3D
    *  - blob_flags is VIRTIO_WDDM_BLOB_FLAG_MAPPABLE
    *  - blob_id is 0
    *
    * it allocates a host shmem.
    *
    * supports_blob_id_0 has been enforced by mandated render server config.
    */
   assert(gpu->capset.data.supports_blob_id_0);
   gpu->shmem_blob_mem = VIRTIO_WDDM_BLOB_MEM_HOST3D;
}

#if defined(VIRTUAL_CONTEXT_SUPPORT) && !defined(INDEPENDENT_RENDER_CONTEXT)
/* A driver without hardware queues refuses the context or the queue, and
 * each ring then gets a context of its own.
 */
static bool
virtgpu_init_hw_queue(struct virtgpu *gpu)
{
   NTSTATUS status = virtgpu_ioctl_create_context_virtual(
      gpu, VIRTIO_WDDM_NODE_HW_QUEUE, true, &gpu->hw_ctx);
   if (!NT_SUCCESS(status)) {
      if (VN_DEBUG(INIT)) {
         vn_log(gpu->instance, "no hw context (0x%lx), using one per ring",
                status);
      }
      memset(&gpu->hw_ctx, 0, sizeof(gpu->hw_ctx));
      return false;
   }

   status = virtgpu_ioctl_create_hw_queue(gpu, &gpu->hw_ctx, &gpu->hw_queue);
   if (!NT_SUCCESS(status)) {
      if (VN_DEBUG(INIT)) {
         vn_log(gpu->instance,
                "no hw queue (0x%lx), using one context per ring", status);
      }
      virtgpu_ioctl_destroy_context(gpu, &gpu->hw_ctx.base);
      memset(&gpu->hw_ctx, 0, sizeof(gpu->hw_ctx));
      memset(&gpu->hw_queue, 0, sizeof(gpu->hw_queue));
      return false;
   }

   if (VN_DEBUG(INIT))
      vn_log(gpu->instance, "created hw queue: 0x%x", gpu->hw_queue.kmt);

   return true;
}
#endif

static VkResult
virtgpu_init_context(struct virtgpu *gpu)
{
   assert(!gpu->capset.version);

   VIRTIO_WDDM_Escape ctx_init = {
      .ctx_init = {
         .tag =
            VIRTIO_WDDM_ESCAPE_CONTEXT_INIT_TAG,
         .capset_id = gpu->capset.id,
         .num_rings = VIRTIO_WDDM_MAX_RINGS,
         .debug_name = "venus-win32",
      },
   };

   char full_path[PATH_MAX] = {};
   GetModuleFileNameA(NULL, full_path, sizeof(full_path));
   char *proc = strrchr(full_path, '\\');
   if (proc != NULL) {
      proc++;
   }

   if (gpu->ddicb) {
      snprintf((char *)ctx_init.ctx_init.debug_name,
               sizeof(ctx_init.ctx_init.debug_name), "venus-win32-%llu-%s-%p",
               (uintptr_t)GetCurrentProcessId(), proc, gpu->ddicb->hRTDevice);
   } else {
      snprintf((char *)ctx_init.ctx_init.debug_name,
               sizeof(ctx_init.ctx_init.debug_name), "venus-win32-%llu-%s-%u",
               (uintptr_t)GetCurrentProcessId(), proc, gpu->d3dkmt.device);
   }

   NTSTATUS status =
      virtgpu_ioctl_escape(gpu, NULL, &ctx_init, sizeof(ctx_init));
   if (!NT_SUCCESS(status)) {
      if (VN_DEBUG(INIT)) {
         vn_log(gpu->instance, "failed to create virtio context: 0x%lx",
                status);
      }
      return VK_ERROR_INITIALIZATION_FAILED;
   }

#ifndef INDEPENDENT_RENDER_CONTEXT
#if defined(VIRTUAL_CONTEXT_SUPPORT)
   gpu->use_hw_queue = virtgpu_init_hw_queue(gpu);
   if (gpu->use_hw_queue) {
      gpu->max_timeline_count = VIRTIO_WDDM_MAX_RINGS;
   } else {
      STATIC_ASSERT(VIRTIO_WDDM_RING_NODE_COUNT <= VIRTIO_WDDM_MAX_RINGS);
      gpu->max_timeline_count = VIRTIO_WDDM_RING_NODE_COUNT;
      for (size_t i = 0; i < ARRAY_SIZE(gpu->ctx); i++) {
         status = virtgpu_ioctl_create_context_virtual(
            gpu, VIRTIO_WDDM_NODE_FIRST_RING + i, false, &gpu->ctx[i]);
         if (!NT_SUCCESS(status)) {
            if (VN_DEBUG(INIT)) {
               vn_log(gpu->instance,
                      "failed to create virtual context: 0x%lx", status);
            }
            return VK_ERROR_INITIALIZATION_FAILED;
         }
      }
   }
#else
   for (size_t i = 0; i < ARRAY_SIZE(gpu->ctx); i++) {
      status = virtgpu_ioctl_create_context(gpu, &gpu->ctx[i]);
      if (!NT_SUCCESS(status)) {
         if (VN_DEBUG(INIT)) {
            vn_log(gpu->instance, "failed to create physical context: 0x%lx",
                   status);
         }
         return VK_ERROR_INITIALIZATION_FAILED;
      }
   }
#endif
#endif

   status = virtgpu_ioctl_create_paging_queue(gpu);
   if (!NT_SUCCESS(status)) {
      if (VN_DEBUG(INIT)) {
         vn_log(gpu->instance, "failed to create paging queue: 0x%lx",
                status);
      }
      return VK_ERROR_INITIALIZATION_FAILED;
   }

   return VK_SUCCESS;
}

static VkResult
virtgpu_init_capset(struct virtgpu *gpu)
{
   gpu->capset.id = VIRTIO_WDDM_CAPSET_ID_VENUS;
   gpu->capset.version = 0;

   NTSTATUS status =
      virtgpu_ioctl_get_caps(gpu, gpu->capset.id, gpu->capset.version,
                             &gpu->capset.data, sizeof(gpu->capset.data));
   if (!NT_SUCCESS(status)) {
      if (VN_DEBUG(INIT)) {
         vn_log(gpu->instance, "failed to get venus v%d capset: 0x%lx",
                gpu->capset.version, status);
      }
      return VK_ERROR_INITIALIZATION_FAILED;
   }

   if (gpu->capset.data.wire_format_version == 0) {
      if (VN_DEBUG(INIT)) {
         vn_log(gpu->instance, "Unsupported wire format version %u",
                gpu->capset.data.wire_format_version);
      }
      return VK_ERROR_INITIALIZATION_FAILED;
   }

   return VK_SUCCESS;
}

static VkResult
virtgpu_init_params(struct virtgpu *gpu)
{
   VIRTIO_WDDM_AdapterInfo info = { 0 };

   NTSTATUS status = virtgpu_ioctl_query(gpu, &info, sizeof(info));
   if (!NT_SUCCESS(status)) {
      if (VN_DEBUG(INIT)) {
         vn_log(gpu->instance,
                "failed to get adapter info from kernel: 0x%lx", status);
      }
      return VK_ERROR_INITIALIZATION_FAILED;
   }

   if (info.tag != VIRTIO_WDDM_ADAPTER_INFO_TAG || !info.supports_3d ||
       !info.has_shmem) {
      if (VN_DEBUG(INIT)) {
         vn_log(gpu->instance, "no venus support in this driver");
      }
      return VK_ERROR_INITIALIZATION_FAILED;
   }

   /* Don't care for VIRTGPU_BLOB_MEM_GUEST_VRAM since this driver is mainly
    * developed for QEMU, but whoever needs it may feel free to implement this
    */
   if (info.has_shmem) {
      gpu->bo_blob_mem = VIRTIO_WDDM_BLOB_MEM_HOST3D;
   } else {
      if (VN_DEBUG(INIT)) {
         vn_log(
            gpu->instance,
            "driver does not support the required host-visible shmem region");
      }
      return VK_ERROR_INITIALIZATION_FAILED;
   }

   /* Don't care about cross-device */
   // gpu->supports_cross_device = false;

#if defined(VIRTUAL_CONTEXT_SUPPORT) && defined(INDEPENDENT_RENDER_CONTEXT)
   gpu->max_timeline_count = VIRTIO_WDDM_RING_NODE_COUNT;
#elif !defined(VIRTUAL_CONTEXT_SUPPORT)
   gpu->max_timeline_count = VIRTIO_WDDM_TIMELINE_COUNT;
#endif

   status = virtgpu_ioctl_pci_address(
      gpu, &gpu->pci_bus_info.domain, &gpu->pci_bus_info.bus,
      &gpu->pci_bus_info.dev, &gpu->pci_bus_info.func);
   if (!NT_SUCCESS(status)) {
      if (VN_DEBUG(INIT)) {
         vn_log(gpu->instance, "failed to get device pci info from kernel");
      }
      return VK_ERROR_INITIALIZATION_FAILED;
   }

   return VK_SUCCESS;
}

static VkResult
virtgpu_find_adapter(struct virtgpu *gpu)
{
   D3DKMT_ENUMADAPTERS2 adapters = { 0 };
   NTSTATUS status = gpu->d3dkmt.cb.enum_adapters2(&adapters);
   if (!NT_SUCCESS(status)) {
      vn_log(gpu->instance, "failed to enumerate adapters: 0x%lx", status);
      return VK_ERROR_INCOMPATIBLE_DRIVER;
   }
   adapters.pAdapters =
      calloc(sizeof(*adapters.pAdapters), adapters.NumAdapters);
   status = gpu->d3dkmt.cb.enum_adapters2(&adapters);
   if (!NT_SUCCESS(status)) {
      vn_log(gpu->instance, "failed to enumerate adapters: 0x%lx", status);
      return VK_ERROR_INCOMPATIBLE_DRIVER;
   }

   ssize_t virtio_index = -1;
   for (size_t i = 0; i < adapters.NumAdapters; i++) {
      D3DKMT_QUERY_DEVICE_IDS device_ids = {};
      D3DKMT_QUERYADAPTERINFO query = {
         .hAdapter = adapters.pAdapters[i].hAdapter,
         .Type = KMTQAITYPE_PHYSICALADAPTERDEVICEIDS,
         .pPrivateDriverData = &device_ids,
         .PrivateDriverDataSize = sizeof(device_ids),
      };
      status = gpu->d3dkmt.cb.query_adapter_info(&query);
      if (!NT_SUCCESS(status)) {
         vn_log(gpu->instance, "failed to query device ids: 0x%lx", status);
         continue;
      }

      if (device_ids.DeviceIds.VendorID == VIRTGPU_PCI_VENDOR_ID &&
          device_ids.DeviceIds.DeviceID == VIRTGPU_PCI_DEVICE_ID) {
         virtio_index = i;
      } else {
         D3DKMT_CLOSEADAPTER close = {
            .hAdapter = adapters.pAdapters[i].hAdapter,
         };
         status = gpu->d3dkmt.cb.close_adapter(&close);
         if (!NT_SUCCESS(status)) {
            vn_log(gpu->instance, "failed to close adapter: 0x%lx", status);
            continue;
         }
      }
   }

   if (virtio_index < 0) {
      free(adapters.pAdapters);
      return VK_ERROR_INCOMPATIBLE_DRIVER;
   }

   gpu->d3dkmt.adapter = adapters.pAdapters[virtio_index].hAdapter;
   gpu->d3dkmt.luid = adapters.pAdapters[virtio_index].AdapterLuid;
   free(adapters.pAdapters);

   if (VN_DEBUG(INIT)) {
      vn_log(gpu->instance, "using adapter %zd (LUID %lx-%lx)", virtio_index,
             gpu->d3dkmt.luid.HighPart, gpu->d3dkmt.luid.LowPart);
   }

   return VK_SUCCESS;
}

static NTSTATUS
virtgpu_ioctl_create_device(struct virtgpu *gpu)
{
   if (gpu->ddicb != NULL) {
      /* Nothing to do here, device was already created before */
      return STATUS_SUCCESS;
   } else {
      D3DKMT_CREATEDEVICE create_device = {
         .hAdapter = gpu->d3dkmt.adapter,
      };
      NTSTATUS status = gpu->d3dkmt.cb.create_device(&create_device);
      if (!NT_SUCCESS(status)) {
         return status;
      }

      gpu->d3dkmt.device = create_device.hDevice;
      return STATUS_SUCCESS;
   }
}

static VkResult
virtgpu_open(struct virtgpu *gpu, void *info)
{
   VkD3DDDICallbacksMESA *callbacks =
      vk_find_struct(info, D3DDDI_CALLBACKS_MESA);
   if (callbacks != NULL) {
      /* D3D1X UMD */
      gpu->ddicb = callbacks;
   } else {
      /* Standalone Vulkan ICD, using D3DKMT */
      HINSTANCE gdi32lib = LoadLibraryA("GDI32.dll");
      gpu->d3dkmt.lib = gdi32lib;

#define X(type, name, fun)                                                   \
   gpu->d3dkmt.cb.name = (type)GetProcAddress(gdi32lib, fun);
      D3DKMT_FUNCTIONS
#undef X

#define X(type, name, fun) assert(gpu->d3dkmt.cb.name != NULL);
      D3DKMT_FUNCTIONS
#undef X

      NTSTATUS status = virtgpu_find_adapter(gpu);
      if (!NT_SUCCESS(status)) {
         return VK_ERROR_DEVICE_LOST;
      }
   }

   NTSTATUS status = virtgpu_ioctl_create_device(gpu);
   if (!NT_SUCCESS(status)) {
      return VK_ERROR_DEVICE_LOST;
   }

   return VK_SUCCESS;
}

static VIRTIO_WDDM_BlobFlag
virtgpu_bo_blob_flags(struct virtgpu *gpu,
                      VkMemoryPropertyFlags flags,
                      VkExternalMemoryHandleTypeFlags external_handles)
{
   VIRTIO_WDDM_BlobFlag blob_flags = 0;
   if (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
      blob_flags |= VIRTIO_WDDM_BLOB_FLAG_MAPPABLE;
   if (external_handles)
      blob_flags |= VIRTIO_WDDM_BLOB_FLAG_SHAREABLE;

   // if (gpu->supports_cross_device && external_handles &
   // VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT) {
   //    blob_flags |= VIRTIO_WDDM_BLOB_FLAG_CROSS_DEVICE;

   return blob_flags;
}

static NTSTATUS
virtgpu_ioctl_open_resource(struct virtgpu *gpu,
                            D3DKMT_HANDLE global,
                            D3DKMT_HANDLE *alloc,
                            D3DKMT_HANDLE *resource,
                            VIRTIO_WDDM_ResourceInfo *info)
{
   assert(gpu->ddicb == NULL);
   D3DKMT_QUERYRESOURCEINFO query = {
      .hDevice = gpu->d3dkmt.device,
      .hGlobalShare = global,
   };

   NTSTATUS status = gpu->d3dkmt.cb.query_resource_info(&query);
   if (!NT_SUCCESS(status)) {
      return status;
   }

   assert(query.ResourcePrivateDriverDataSize >=
          sizeof(VIRTIO_WDDM_CreateResource));
   assert(query.TotalPrivateDriverDataSize >=
          sizeof(VIRTIO_WDDM_CreateAllocation) * query.NumAllocations);

   size_t runtime_data_off = 0;
   size_t res_priv_off =
      runtime_data_off + align64(query.PrivateRuntimeDataSize, 8);
   size_t alloc_priv_off =
      res_priv_off + align64(query.ResourcePrivateDriverDataSize, 8);
   size_t alloc_list_off =
      alloc_priv_off + align64(query.TotalPrivateDriverDataSize, 8);

   size_t total_size = alloc_list_off + sizeof(D3DDDI_OPENALLOCATIONINFO) *
                                           query.NumAllocations;
   void *data = calloc(total_size, 1);
   uintptr_t p = (uintptr_t)data;

   void *runtime = (void *)(p + runtime_data_off);
   VIRTIO_WDDM_CreateResource *resource_priv = (void *)(p + res_priv_off);

   VIRTIO_WDDM_CreateAllocation *full_alloc_priv =
      (void *)(p + alloc_priv_off);

   D3DDDI_OPENALLOCATIONINFO *alloc_list = (void *)(p + alloc_list_off);
   alloc_list->pPrivateDriverData = full_alloc_priv;
   alloc_list->PrivateDriverDataSize = query.TotalPrivateDriverDataSize / query.NumAllocations;

   D3DKMT_OPENRESOURCE open = {
      .hDevice = gpu->d3dkmt.device,
      .hGlobalShare = global,
      .NumAllocations = query.NumAllocations,
      .pOpenAllocationInfo = alloc_list,
      .pResourcePrivateDriverData = resource_priv,
      .ResourcePrivateDriverDataSize = query.ResourcePrivateDriverDataSize,
      .pPrivateRuntimeData = runtime,
      .PrivateRuntimeDataSize = query.PrivateRuntimeDataSize,
      .pTotalPrivateDriverDataBuffer = full_alloc_priv,
      .TotalPrivateDriverDataBufferSize = query.TotalPrivateDriverDataSize,
   };

   status = gpu->d3dkmt.cb.open_resource(&open);
   if (!NT_SUCCESS(status)) {
      goto end;
   }

   const VIRTIO_WDDM_CreateAllocation *alloc_priv =
      alloc_list[0].pPrivateDriverData;
   // assert(alloc_priv->tag == VIRTIO_WDDM_ALLOCATE_BLOB_TAG);
   assert(alloc_priv->res_id != 0);

   *alloc = alloc_list[0].hAllocation;
   *resource = open.hResource;
   *info = (VIRTIO_WDDM_ResourceInfo) {
      .tag = VIRTIO_WDDM_ESCAPE_RESOURCE_INFO_TAG,
      .handle = *alloc,
      .info = *alloc_priv,
   };

end:
   free(data);
   return status;
}

static NTSTATUS
virtgpu_ioctl_open_resource_from_nt_handle(struct virtgpu *gpu,
                                           HANDLE nt_handle,
                                           D3DKMT_HANDLE *alloc,
                                           D3DKMT_HANDLE *resource,
                                           VIRTIO_WDDM_ResourceInfo *info)
{
   assert(gpu->ddicb == NULL);
   D3DKMT_QUERYRESOURCEINFOFROMNTHANDLE query = {
      .hDevice = gpu->d3dkmt.device,
      .hNtHandle = nt_handle,
   };

   NTSTATUS status =
      gpu->d3dkmt.cb.query_resource_info_from_nt_handle(&query);
   if (!NT_SUCCESS(status)) {
      return status;
   }

   assert(query.NumAllocations == 1);
   assert(query.ResourcePrivateDriverDataSize >=
          sizeof(VIRTIO_WDDM_CreateResource));
   assert(query.TotalPrivateDriverDataSize >=
          sizeof(VIRTIO_WDDM_CreateAllocation) * query.NumAllocations);

   size_t runtime_priv_off = 0;
   size_t res_priv_off =
      runtime_priv_off + align64(query.PrivateRuntimeDataSize, 8);
   size_t alloc_priv_off =
      res_priv_off + align64(query.ResourcePrivateDriverDataSize, 8);
   size_t alloc_list_off =
      alloc_priv_off + align64(query.TotalPrivateDriverDataSize, 8);

   size_t total_size = alloc_list_off + sizeof(D3DDDI_OPENALLOCATIONINFO2) *
                                           query.NumAllocations;
   void *data = calloc(total_size, 1);
   uintptr_t p = (uintptr_t)data;

   void *runtime = (void *)(p + runtime_priv_off);
   VIRTIO_WDDM_CreateResource *resource_priv = (void *)(p + res_priv_off);

   VIRTIO_WDDM_CreateAllocation *full_alloc_priv =
      (void *)(p + alloc_priv_off);

   D3DDDI_OPENALLOCATIONINFO2 *alloc_list = (void *)(p + alloc_list_off);
   alloc_list->pPrivateDriverData = full_alloc_priv;
   alloc_list->PrivateDriverDataSize = query.TotalPrivateDriverDataSize / query.NumAllocations;

   D3DKMT_OPENRESOURCEFROMNTHANDLE open = {
      .hDevice = gpu->d3dkmt.device,
      .hNtHandle = nt_handle,
      .NumAllocations = query.NumAllocations,
      .pOpenAllocationInfo2 = alloc_list,
      .pResourcePrivateDriverData = resource_priv,
      .ResourcePrivateDriverDataSize = query.ResourcePrivateDriverDataSize,
      .pPrivateRuntimeData = runtime,
      .PrivateRuntimeDataSize = query.PrivateRuntimeDataSize,
      .pTotalPrivateDriverDataBuffer = full_alloc_priv,
      .TotalPrivateDriverDataBufferSize = query.TotalPrivateDriverDataSize,
   };

   status = gpu->d3dkmt.cb.open_resource_from_nt_handle(&open);
   if (!NT_SUCCESS(status)) {
      goto end;
   }

   const VIRTIO_WDDM_CreateAllocation *alloc_priv =
      alloc_list[0].pPrivateDriverData;

   /* TODO: we'd need syncobj import for this and I don't see a reasonable way
    * to implement that... Maybe we could get away with CPU waits?
    * Or maybe create host timeline semaphore and submit vkSignalSemaphore via
    * vq with guest hSyncObject wait?
    *
    * hSyncObject is a legacy fence, so we can't really handle it as a monitored
    * fence
    */
   assert(open.hKeyedMutex == 0 && open.hSyncObject == 0);
   assert(alloc_priv->res_id != 0);

   *alloc = alloc_list[0].hAllocation;
   *resource = open.hResource;
   *info = (VIRTIO_WDDM_ResourceInfo) {
      .tag = VIRTIO_WDDM_ESCAPE_RESOURCE_INFO_TAG,
      .handle = *alloc,
      .info = *alloc_priv,
   };

end:
   free(data);
   return status;
}

static VkResult
virtgpu_bo_create_from_handle(struct vn_renderer *renderer,
                              VkDeviceSize size,
                              vn_object_id mem_id,
                              bool is_kmt,
                              void *handle,
                              VkMemoryPropertyFlags flags,
                              const VkMemoryAllocateInfo *alloc_info,
                              struct vn_renderer_bo **out_bo)
{
   struct virtgpu *gpu = (struct virtgpu *)renderer;
   struct virtgpu_bo *bo = NULL;

   VIRTIO_WDDM_ResourceInfo res_info = {};

   mtx_lock(&gpu->win32_handle_import_mutex);

   D3DKMT_HANDLE alloc = 0, kmt_local = 0;
   HANDLE res_h = NULL;

   const VkD3DDDIOpenResourceMESA *d3d_open =
      vk_find_struct_const(alloc_info, D3DDDI_OPEN_RESOURCE_MESA);

   NTSTATUS status;
   if (gpu->ddicb != NULL) {
      assert(d3d_open != NULL);
      alloc = d3d_open->pOpenResource->pOpenAllocationInfo[0].hAllocation;
      res_h = d3d_open->hRTResource;
      res_info = *(VIRTIO_WDDM_ResourceInfo *)d3d_open->pResourceInfo;
   } else if (is_kmt) {
      status =
         virtgpu_ioctl_open_resource(gpu, (D3DKMT_HANDLE)(uintptr_t)handle,
                                     &alloc, &kmt_local, &res_info);
      if (!NT_SUCCESS(status)) {
         vn_log(gpu->instance, "failed to open resource: 0x%lx", status);
         return VK_ERROR_INVALID_EXTERNAL_HANDLE;
      }
   } else {
      status = virtgpu_ioctl_open_resource_from_nt_handle(
         gpu, handle, &alloc, &kmt_local, &res_info);
      if (!NT_SUCCESS(status)) {
         vn_log(gpu->instance,
                "failed to open resource from nt handle: 0x%lx", status);
         return VK_ERROR_INVALID_EXTERNAL_HANDLE;
      }
   }

   uint64_t paging_fence_value = 0;
   status =
      virtgpu_ioctl_resource_make_resident(gpu, alloc, &paging_fence_value);
   if (status == STATUS_PENDING) {
      status = virtgpu_ioctl_wait_from_cpu(gpu, gpu->paging.sync_object,
                                           paging_fence_value);
   }
   if (!NT_SUCCESS(status)) {
      vn_log(gpu->instance, "failed to make opened resource resident: 0x%lx",
             status);
      // return VK_ERROR_DEVICE_LOST;
   }

   if (!alloc)
      goto fail;
   bo = util_sparse_array_get(&gpu->bo_array, alloc);

   /* Upon import, blob_flags is not passed to the kernel and is only for
    * internal use. Set it to what works best for us.
    * - blob mem: SHAREABLE + conditional MAPPABLE per VkMemoryPropertyFlags
    * - classic 3d: SHAREABLE only for export and to fail the map
    */
   VIRTIO_WDDM_BlobFlag blob_flags = VIRTIO_WDDM_BLOB_FLAG_SHAREABLE;
   size_t mmap_size = 0;
   if (res_info.info.base.tag == VIRTIO_WDDM_ALLOCATE_BLOB_TAG) {
      /* must be VIRTIO_WDDM_BLOB_MEM_HOST3D */
      if (res_info.info.base.blob.mem != gpu->bo_blob_mem) {
         vn_log(gpu->instance,
                "NT/KMT handle import failed: info.blob_mem(%u) != "
                "gpu->bo_blob_mem(%u)",
                res_info.info.base.blob.mem, gpu->bo_blob_mem);
         goto fail;
      }

      blob_flags |= virtgpu_bo_blob_flags(gpu, flags, 0);

      /* mmap_size is only used when mappable */
      if (blob_flags & VIRTIO_WDDM_BLOB_FLAG_MAPPABLE) {
         if (res_info.info.base.blob.size < size) {
            /* If queried blob size is smaller than requested allocation size,
             * we drop the mappable flag to defer the mapping failure till the
             * app attempts to map the imported memory.
             */
            blob_flags &= ~VIRTIO_WDDM_BLOB_FLAG_MAPPABLE;
         } else {
            /* Similar to virtgpu_bo_create_from_device_memory, the app can
             * do multiple imports with different sizes for suballocation. So
             * on the initial import, the mapping size has to be initialized
             * with the real size of the backing blob resource.
             */
            mmap_size = res_info.info.base.blob.size;
         }
      }
   } else {
      assert(res_info.info.base.tag == VIRTIO_WDDM_ALLOCATE_3D_TAG);
   }

   /* we check bo->alloc instead of bo->refcount because bo->refcount
    * might only be memset to 0 and is not considered initialized in theory
    */
   if (bo->alloc == alloc) {
      if (bo->base.mmap_size < mmap_size) {
         vn_log(gpu->instance,
                "NT/KMT handle import failed: bo->base.mmap_size(%zu) < "
                "mmap_size(%zu)",
                bo->base.mmap_size, mmap_size);
         goto fail;
      }
      if (blob_flags & ~bo->blob_flags) {
         vn_log(gpu->instance,
                "NT/KMT handle import failed: blob_flags(%u) & "
                "~bo->blob_flags(%u)",
                blob_flags, bo->blob_flags);
         goto fail;
      }

      /* we can't use vn_renderer_bo_ref as the refcount may drop to 0
       * temporarily before virtgpu_bo_destroy grabs the lock
       */
      vn_refcount_fetch_add_relaxed(&bo->base.refcount, 1);
   } else {
      *bo = (struct virtgpu_bo){
         .base = {
            .refcount = VN_REFCOUNT_INIT(1),
            .res_id = res_info.info.res_id,
            .mmap_size = mmap_size,
         },
         .alloc = alloc,
         .blob_flags = blob_flags,
         .external = true,
      };
   }
   if (gpu->ddicb != NULL) {
      bo->h = res_h;
   } else if (is_kmt) {
      bo->kmt.local = kmt_local;
      bo->kmt.global = (D3DKMT_HANDLE)(uintptr_t)handle;
   } else {
      bo->kmt.local = kmt_local;
   }

   mtx_unlock(&gpu->win32_handle_import_mutex);

   *out_bo = &bo->base;

   return VK_SUCCESS;

fail:
   mtx_unlock(&gpu->win32_handle_import_mutex);
   return VK_ERROR_INVALID_EXTERNAL_HANDLE;
}

static VkResult
virtgpu_bo_query_info(struct vn_renderer *renderer,
                      struct vn_renderer_bo *bo_,
                      VkImageDrmFormatModifierExplicitCreateInfoEXT *out_mod,
                      VkSubresourceLayout out_layouts[VN_MAX_PLANES])
{
   struct virtgpu *gpu = (struct virtgpu *)renderer;
   struct virtgpu_bo *bo = (struct virtgpu_bo *)bo_;

   VIRTIO_WDDM_Escape escape_priv = {
      .res_info = {
         .tag = VIRTIO_WDDM_ESCAPE_RESOURCE_INFO_TAG,
         .handle = bo->alloc,
         .info = {
            .modifier = DRM_FORMAT_MOD_INVALID,
         },
      },
   };

   NTSTATUS status =
      virtgpu_ioctl_escape(gpu, NULL, &escape_priv, sizeof(escape_priv));
   if (!NT_SUCCESS(status)) {
      vn_log(gpu->instance, "Resource info query failed: %lx", status);
      return VK_ERROR_UNKNOWN;
   }

   *out_mod = (VkImageDrmFormatModifierExplicitCreateInfoEXT){
      .sType =
         VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
      .drmFormatModifier = escape_priv.res_info.info.modifier,
      .drmFormatModifierPlaneCount =
         escape_priv.res_info.info.num_planes,
      .pPlaneLayouts = out_layouts,
   };

   for (size_t i = 0; i < escape_priv.res_info.info.num_planes; i++) {
      out_layouts[i] = (VkSubresourceLayout) {
         .offset = escape_priv.res_info.info.planes[i].offset,
         // .size = escape_priv.res_info.info.planes[i].size[i],
         .rowPitch = escape_priv.res_info.info.planes[i].stride,
      };
   }

   return VK_SUCCESS;
}

static uint32_t
virtgpu_format_from_vk(VkFormat format)
{
   /* TODO: do we need to handle SNORM/USCALED/SSCALED/UINT/SINT/SRGB here?
    * After all, these seem to have the same underlying byte representation */
   switch (format) {
   case VK_FORMAT_B8G8R8A8_UNORM:
      return 1; /* VIRGL_FORMAT_B8G8R8A8_UNORM */
   case VK_FORMAT_R8G8B8_UNORM:
      return 66; /* VIRGL_FORMAT_R8G8B8_UNORM */
   case VK_FORMAT_R8G8B8A8_UNORM:
      return 67; /* VIRGL_FORMAT_R8G8B8A8_UNORM */
   case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
      return 121; /* VIRGL_FORMAT_A8B8G8R8_UNORM */
   case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
      return 8; /* VIRGL_FORMAT_R10G10B10A2_UNORM */
   case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
      return 131; /* VIRGL_FORMAT_B10G10R10A2_UNORM */
   case VK_FORMAT_R8_UNORM:
      return 64; /* VIRGL_FORMAT_R8_UNORM */
   case VK_FORMAT_A8_UNORM:
      return 10; /* VIRGL_FORMAT_A8_UNORM */
   default:
      return 0;
   }
}

static VkFormat
vk_format_from_virtgpu(uint32_t format)
{
   switch (format) {
   case 1: /* VIRGL_FORMAT_B8G8R8A8_UNORM */
      return VK_FORMAT_B8G8R8A8_UNORM;
   case 66: /* VIRGL_FORMAT_R8G8B8_UNORM */
      return VK_FORMAT_R8G8B8_UNORM;
   case 67: /* VIRGL_FORMAT_R8G8B8A8_UNORM */
      return VK_FORMAT_R8G8B8A8_UNORM;
   case 121: /* VIRGL_FORMAT_A8B8G8R8_UNORM */
      return VK_FORMAT_A8B8G8R8_UNORM_PACK32;
   case 8: /* VIRGL_FORMAT_R10G10B10A2_UNORM */
      return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
   case 131: /* VIRGL_FORMAT_B10G10R10A2_UNORM */
      return VK_FORMAT_A2R10G10B10_UNORM_PACK32;
   case 64: /* VIRGL_FORMAT_R8_UNORM */
      return VK_FORMAT_R8_UNORM;
   case 10: /* VIRGL_FORMAT_A8_UNORM */
      return VK_FORMAT_A8_UNORM;
   default:
      return VK_FORMAT_UNDEFINED;
   }
}

static VkResult
virtgpu_bo_create_from_device_memory(
   struct vn_renderer *renderer,
   struct vn_renderer_submit_batch *batch,
   VkDevice device,
   VkDeviceSize size,
   vn_object_id mem_id,
   VkMemoryPropertyFlags flags,
   VkExternalMemoryHandleTypeFlags external_handles,
   const VkMemoryAllocateInfo *alloc_info,
   struct vn_renderer_bo **out_bo)
{
   struct virtgpu *gpu = (struct virtgpu *)renderer;
   const uint32_t blob_flags =
      virtgpu_bo_blob_flags(gpu, flags, external_handles);

   const bool nt_shared =
      external_handles & (VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT |
                          VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT);

   HANDLE h = NULL;
   D3DKMT_HANDLE alloc, kmt_local, kmt_global;

   struct {
      VIRTIO_WDDM_CreateAllocation create;
      uint8_t extra_data[MAX_COMMAND_SIZE];
   } alloc_priv = {
      .create = {
         .base.blob = {
            .tag = VIRTIO_WDDM_ALLOCATE_BLOB_TAG,
            .id = mem_id,
            .mem = gpu->bo_blob_mem,
            .flags = blob_flags,
            .size = align64(size, 4096),
         },
         .modifier = DRM_FORMAT_MOD_INVALID,
      },
   };
   size_t alloc_priv_size = sizeof(alloc_priv.create);

   const VkD3DDDICreateResourceMESA *d3d_create =
      vk_find_struct_const(alloc_info, D3DDDI_CREATE_RESOURCE_MESA);
   if (gpu->ddicb != NULL && d3d_create != NULL) {
      h = d3d_create->hRTResource;
      if (d3d_create->ExtraDataSize > 0) {
         memcpy(alloc_priv.extra_data, d3d_create->pExtraData,
            d3d_create->ExtraDataSize);
         alloc_priv_size += d3d_create->ExtraDataSize;
      }
   }

   const VkMemoryDedicatedAllocateInfo *dedicated =
      vk_find_struct_const(alloc_info, MEMORY_DEDICATED_ALLOCATE_INFO);
   if (dedicated != NULL) {
      struct vn_image *img = vn_image_from_handle(dedicated->image);

      const bool is_image_2d = img->base.vk.image_type == VK_IMAGE_TYPE_2D;
      const bool is_linear = img->base.vk.tiling == VK_IMAGE_TILING_LINEAR;
      const bool has_mod =
         img->base.vk.tiling == VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;

      uint32_t format = virtgpu_format_from_vk(img->base.vk.format);

      if (is_image_2d && format != 0) {
         if (is_linear || has_mod) {
         /* Assume single plane */
            VkImageSubresource sub_res = {
               .aspectMask = is_linear ? VK_IMAGE_ASPECT_COLOR_BIT
                                       : VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT,
            };
            VkSubresourceLayout sub_res_layout = { 0 };
            vn_GetImageSubresourceLayout(device, dedicated->image, &sub_res,
                                         &sub_res_layout);

            VkImageDrmFormatModifierPropertiesEXT mod_props = {
               .sType =
                  VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_PROPERTIES_EXT,
               .drmFormatModifier = DRM_FORMAT_MOD_INVALID,
            };

            if (img->base.vk.tiling == VK_IMAGE_TILING_LINEAR) {
               mod_props.drmFormatModifier = 0;
            } else if (img->base.vk.tiling ==
                       VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT) {
               ASSERTED VkResult result =
                  vn_GetImageDrmFormatModifierPropertiesEXT(
                     device, dedicated->image, &mod_props);
               assert(result == VK_SUCCESS);
            }

            alloc_priv.create.num_planes = 1;
            alloc_priv.create.modifier = mod_props.drmFormatModifier;

            alloc_priv.create.planes[0] = (VIRTIO_WDDM_PlaneLayout) {
               .offset = sub_res_layout.offset,
               .stride = sub_res_layout.rowPitch,
               .size = sub_res_layout.size,
            };

            vn_log(gpu->instance,
                   "BLOB INFO: type=%u, flags=%u, size=%zu, "
                   "id=%" PRIu64 ", format=%u, w=%u, h=%u, mod=0x%016llx",
                   gpu->bo_blob_mem, blob_flags, size, mem_id, format,
                   img->base.vk.extent.width, img->base.vk.extent.height,
                   mod_props.drmFormatModifier);
         }

         alloc_priv.create.base.blob.info = (VIRTIO_WDDM_BlobInfo) {
            .format = format,
            .width = img->base.vk.extent.width,
            .height = img->base.vk.extent.height,
            .format = format,
            .bind = (1u << 20), /* VIRGL_BIND_SHARED */
         };

         if (img->base.vk.usage & VK_IMAGE_USAGE_2_DEPTH_STENCIL_ATTACHMENT_BIT_KHR) {
             /* VIRGL_BIND_DEPTH_STENCIL */
             alloc_priv.create.base.blob.info.bind |= 1u << 0;
         }
         if (img->base.vk.usage & VK_IMAGE_USAGE_2_COLOR_ATTACHMENT_BIT_KHR) {
             /* VIRGL_BIND_RENDER_TARGET */
             alloc_priv.create.base.blob.info.bind |= 1u << 1;
         }
         if (img->base.vk.usage & VK_IMAGE_USAGE_2_SAMPLED_BIT_KHR) {
             /* VIRGL_BIND_SAMPLER_VIEW */
             alloc_priv.create.base.blob.info.bind |= 1u << 3;
         }
      }
   }

   NTSTATUS status = virtgpu_ioctl_resource_create_blob(
      gpu, &alloc_priv.create, alloc_priv_size, d3d_create, nt_shared, batch, &alloc,
      &kmt_local, &kmt_global, &h);
   if (!NT_SUCCESS(status)) {
      vn_log(gpu->instance,
             "RESOURCE_CREATE_BLOB failed: type=%u, flags=%u, size=%zu, "
             "id=%" PRIu64 ", err=0x%lx",
             gpu->bo_blob_mem, blob_flags, size, mem_id, status);
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   }

   size_t real_size = alloc_priv.create.base.blob.size;

   /* There's a single underlying bo mapping shared by the initial alloc here
    * and the later import of the same. The mapping size has to be initialized
    * with the real size of the created blob resource, since the app can query
    * the exported native handle size for re-import. e.g. lseek dma-buf size
    */
   const uint32_t mappable_and_shareable =
      VIRTIO_WDDM_BLOB_FLAG_MAPPABLE | VIRTIO_WDDM_BLOB_FLAG_SHAREABLE;
   if ((blob_flags & mappable_and_shareable) == mappable_and_shareable) {
      if (real_size < size) {
         virtgpu_ioctl_resource_destroy_blob(gpu, alloc, kmt_local, h);
         vn_log(gpu->instance,
                "blob mem create failed: real_size(%zu) < size(%" PRIu64 ")",
                real_size, size);
         return VK_ERROR_INVALID_EXTERNAL_HANDLE;
      }

      size = real_size;
   }

   struct virtgpu_bo *bo = util_sparse_array_get(&gpu->bo_array, alloc);
   *bo = (struct virtgpu_bo){
      .base = {
         .refcount = VN_REFCOUNT_INIT(1),
         .res_id = alloc_priv.create.res_id,
         .mmap_size = size,
      },
      .alloc = alloc,
      .blob_flags = blob_flags,
   };

   if (gpu->ddicb != NULL) {
      bo->h = h;
   } else {
      bo->kmt.local = kmt_local;
      bo->kmt.global = kmt_global;
   }

   *out_bo = &bo->base;

   return VK_SUCCESS;
}

static void
virtgpu_bo_invalidate(struct vn_renderer *renderer,
                      struct vn_renderer_bo *bo,
                      VkDeviceSize offset,
                      VkDeviceSize size)
{
   /* nop because kernel makes every mapping coherent */
   // TODO: check if this is true
}

static void
virtgpu_bo_flush(struct vn_renderer *renderer,
                 struct vn_renderer_bo *bo,
                 VkDeviceSize offset,
                 VkDeviceSize size)
{
   /* nop because kernel makes every mapping coherent */
   // TODO: check if this is true
}

static void *
virtgpu_bo_map(struct vn_renderer *renderer,
               struct vn_renderer_bo *_bo,
               void *placed_addr)
{
   assert(placed_addr == NULL);
   struct virtgpu *gpu = (struct virtgpu *)renderer;
   struct virtgpu_bo *bo = (struct virtgpu_bo *)_bo;
   const bool mappable = bo->blob_flags & VIRTIO_WDDM_BLOB_FLAG_MAPPABLE;

   /* not thread-safe but is fine */
   if (!bo->base.mmap_ptr && mappable) {
      NTSTATUS status = virtgpu_ioctl_map(gpu, bo->alloc, &bo->base.mmap_ptr);
      //   virtgpu_ioctl_lock(gpu, bo->alloc, &bo->base.mmap_ptr);
      if (!NT_SUCCESS(status)) {
         vn_log(gpu->instance, "failed to map blob resource: 0x%lx", status);
      }
   }

   return bo->base.mmap_ptr;
}

static void *
virtgpu_bo_export_handle(struct vn_renderer *renderer,
                         struct vn_renderer_bo *_bo,
                         bool is_kmt)
{
   struct virtgpu *gpu = (struct virtgpu *)renderer;
   struct virtgpu_bo *bo = (struct virtgpu_bo *)_bo;
   const bool shareable = bo->blob_flags & VIRTIO_WDDM_BLOB_FLAG_SHAREABLE;

   if (gpu->ddicb != NULL) {
      assert(is_kmt);
      /* Special case for DXGI DDI */
      return (void *)(uintptr_t)bo->alloc;
   }

   if (!shareable)
      return NULL;

   if (is_kmt)
      return (void *)(uintptr_t)bo->kmt.global;

   HANDLE nt_handle = NULL;
   NTSTATUS status = virtgpu_ioctl_share_objects(
      gpu, SHARED_ALLOCATION_ALL_ACCESS, &bo->kmt.local, 1, &nt_handle);
   if (!NT_SUCCESS(status)) {
      vn_log(gpu->instance, "failed to share bo: 0x%lx", status);
      return NULL;
   }
   return nt_handle;
}

static bool
virtgpu_bo_destroy(struct vn_renderer *renderer, struct vn_renderer_bo *_bo)
{
   struct virtgpu *gpu = (struct virtgpu *)renderer;
   struct virtgpu_bo *bo = (struct virtgpu_bo *)_bo;

   mtx_lock(&gpu->win32_handle_import_mutex);

   /* Check the refcount again after the import lock is grabbed.  Yes, we use
    * the double-checked locking anti-pattern.
    */
   if (vn_refcount_is_valid(&bo->base.refcount)) {
      mtx_unlock(&gpu->win32_handle_import_mutex);
      return false;
   }

   if (bo->base.mmap_ptr) {
      // virtgpu_ioctl_unlock(gpu, bo->alloc);
      // virtgpu_ioctl_destroy_map(gpu, bo->alloc, bo->base.res_id);
      virtgpu_ioctl_unmap(gpu, bo->alloc, bo->base.mmap_ptr);
      bo->base.mmap_ptr = NULL;
   }

   /* Set alloc and res to 0 to indicate that the bo is invalid. Must be set
    * before closing the handles. Otherwise the same handles can be reused
    * by another newly created bo and unexpectedly gotten zero'ed out the
    * tracked handles.
    */
   const D3DKMT_HANDLE alloc = bo->alloc, kmt = bo->kmt.local;
   const HANDLE h = bo->h;
   bo->alloc = 0;
   bo->kmt.local = 0;
   bo->h = NULL;
   if (!bo->external) {
      virtgpu_ioctl_resource_destroy_blob(gpu, alloc, kmt, h);
   }

   mtx_unlock(&gpu->win32_handle_import_mutex);

   return true;
}

static void
virtgpu_shmem_destroy_now(struct vn_renderer *renderer,
                          struct vn_renderer_shmem *_shmem)
{
   struct virtgpu *gpu = (struct virtgpu *)renderer;
   struct virtgpu_shmem *shmem = (struct virtgpu_shmem *)_shmem;

   // virtgpu_ioctl_unlock(gpu, shmem->alloc);
   // virtgpu_ioctl_destroy_map(gpu, shmem->alloc, shmem->base.res_id);
   virtgpu_ioctl_unmap(gpu, shmem->alloc, shmem->base.mmap_ptr);
   virtgpu_ioctl_resource_destroy_blob(gpu, shmem->alloc, shmem->kmt,
                                       shmem->h);
}

static void
virtgpu_shmem_destroy(struct vn_renderer *renderer,
                      struct vn_renderer_shmem *shmem)
{
   struct virtgpu *gpu = (struct virtgpu *)renderer;

   if (vn_renderer_shmem_cache_add(&gpu->shmem_cache, shmem))
      return;

   virtgpu_shmem_destroy_now(&gpu->base, shmem);
}

static struct vn_renderer_shmem *
virtgpu_shmem_create(struct vn_renderer *renderer, size_t size)
{
   struct virtgpu *gpu = (struct virtgpu *)renderer;

   struct vn_renderer_shmem *cached_shmem =
      vn_renderer_shmem_cache_get(&gpu->shmem_cache, size);
   if (cached_shmem) {
      cached_shmem->refcount = VN_REFCOUNT_INIT(1);
      return cached_shmem;
   }

   HANDLE h = NULL; /* This is a device allocation */
   D3DKMT_HANDLE alloc, kmt;

   VIRTIO_WDDM_CreateAllocation alloc_priv = {
      .base.blob = {
         .tag = VIRTIO_WDDM_ALLOCATE_BLOB_TAG,
         .id = 0,
         .mem = gpu->shmem_blob_mem,
         .flags = VIRTIO_WDDM_BLOB_FLAG_MAPPABLE,
         .size = align64(size, 4096),
      },
      .modifier = DRM_FORMAT_MOD_INVALID,
   };

   NTSTATUS status = virtgpu_ioctl_resource_create_blob(
      gpu, &alloc_priv, sizeof(alloc_priv), NULL /* d3d_create */,
      false /* nt_shared */, NULL /* batch */, &alloc, &kmt, NULL, &h);
   if (!NT_SUCCESS(status))
      return NULL;

   void *ptr = NULL;
   status = virtgpu_ioctl_map(gpu, alloc, &ptr);
   if (!NT_SUCCESS(status)) {
      virtgpu_ioctl_resource_destroy_blob(gpu, alloc, kmt, h);
      vn_log(gpu->instance, "failed to map blob resource: 0x%lx", status);
      return NULL;
   }

   struct virtgpu_shmem *shmem =
      util_sparse_array_get(&gpu->shmem_array, alloc);
   *shmem = (struct virtgpu_shmem){
      .base = {
         .refcount = VN_REFCOUNT_INIT(1),
         .res_id = alloc_priv.res_id,
         .mmap_size = alloc_priv.base.blob.size,
         .mmap_ptr = ptr,
      },
      .alloc = alloc,
   };

   if (gpu->ddicb != NULL) {
      shmem->h = h;
   } else {
      shmem->kmt = kmt;
   }

   return &shmem->base;
}

static VkResult
sync_wait(HANDLE event, int timeout)
{
   int status = WaitForSingleObject(event, timeout);
   switch (status) {
   case WAIT_OBJECT_0:
      return VK_SUCCESS;
   case WAIT_TIMEOUT:
      return VK_TIMEOUT;
   default:
      vn_log(NULL, "failed to wait for async event: 0x%x", status);
      return VK_ERROR_DEVICE_LOST;
   }
}

static VkResult
virtgpu_wait(struct vn_renderer *renderer,
             const struct vn_renderer_wait *wait)
{
   struct virtgpu *gpu = (struct virtgpu *)renderer;
   const int timeout = vn_timeout_to_poll_timeout(wait->timeout);
   HANDLE async_event =
      timeout == -1 ? NULL : CreateEventA(NULL, true, false, NULL);

   VkResult res = VK_SUCCESS;
   STACK_ARRAY(D3DKMT_HANDLE, sync_objects, wait->sync_count);
   if (!sync_objects)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   for (size_t i = 0; i < wait->sync_count; i++) {
      struct virtgpu_sync *sync = (struct virtgpu_sync *)wait->syncs[i];
      sync_objects[i] = sync->handle;
   }

   NTSTATUS status = virtgpu_ioctl_wait_for_synchronization_objects_from_cpu(
      gpu, sync_objects, wait->sync_values, wait->sync_count, wait->wait_any,
      async_event);
   if (!NT_SUCCESS(status)) {
      vn_log(gpu->instance, "failed to wait for monitored fence: 0x%lx",
             status);
      res = VK_ERROR_DEVICE_LOST;
      goto end;
   } else if (async_event) {
      res = sync_wait(async_event, timeout);
   }
end:
   STACK_ARRAY_FINISH(sync_objects);
   return res;
}

/* With a hardware queue, ctx is NULL and the batch goes to hw_queue. */
static VkResult
virtgpu_submit_batch(struct virtgpu *gpu,
#ifdef VIRTUAL_CONTEXT_SUPPORT
                     struct virtgpu_exec_context_virtual *ctx,
                     struct virtgpu_hw_queue *hw_queue,
#else
                     struct virtgpu_exec_context *ctx,
#endif
                     const struct vn_renderer_submit_batch *batch,
                     bool escape,
                     bool syncobj_wait)
{
   VkResult ret = VK_SUCCESS;
   NTSTATUS status = STATUS_SUCCESS;
#ifndef VIRTUAL_CONTEXT_SUPPORT
   const void *hw_queue = NULL;
#endif

   STACK_ARRAY(D3DKMT_HANDLE, sync_objects, batch->sync_count);
   if (!sync_objects)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   for (size_t i = 0; i < batch->sync_count; i++) {
      assert(batch->sync_values[i] != 0);
      struct virtgpu_sync *sync = (struct virtgpu_sync *)batch->syncs[i];
      sync_objects[i] = sync->handle;
   }

   if (syncobj_wait && batch->sync_count > 0) {
#ifdef VIRTUAL_CONTEXT_SUPPORT
      if (hw_queue) {
         status =
            virtgpu_ioctl_submit_wait_for_synchronization_objects_to_hw_queue(
               gpu, hw_queue, sync_objects, batch->sync_values,
               batch->sync_count);
      } else
#endif
      {
         status = virtgpu_ioctl_wait_for_synchronization_objects_from_gpu(
            gpu, &ctx->base, sync_objects, batch->sync_values,
            batch->sync_count);
      }
      if (!NT_SUCCESS(status)) {
         vn_log(gpu->instance, "failed to wait: 0x%lx", status);
         ret = VK_ERROR_DEVICE_LOST;
         goto end;
      }
   }

   /* Contexts of their own may run in any order: keep the order of the
    * submissions with a fence they wait on and signal in turn.
    */
   const bool ordered = !hw_queue && !escape;
   uint64_t next_submission_value = 0, current_submission_value;
   if (ordered) {
      next_submission_value = p_atomic_inc_return(&gpu->pending_submission_value);
      current_submission_value = next_submission_value - 1;

      status = virtgpu_ioctl_wait_for_synchronization_objects_from_gpu(
         gpu, &ctx->base, &gpu->submission_fence->handle,
         &current_submission_value, 1);
      if (!NT_SUCCESS(status)) {
         vn_log(gpu->instance, "failed to wait for submission: 0x%lx", status);
         ret = VK_ERROR_DEVICE_LOST;
         goto end;
      }
   }

#ifdef VIRTUAL_CONTEXT_SUPPORT
   assert(batch->cs_size <= MAX_COMMAND_SIZE);
   // FIXME: empty command is somehow valid?
   // assert(batch->cs_size > 0);

   if (hw_queue) {
      /* dxgkrnl will ignore zero-sized submissions and break KMD assumptions */
      if (batch->cs_size > 0) {
         /* Submit via SubmitCommandToHwQueue */
         struct {
            VIRTIO_WDDM_SubmitCommand submit;
            VIRTIO_WDDM_CommandHeader hdr;
            uint8_t command[MAX_COMMAND_SIZE];
         } submit_priv;

         submit_priv.submit.tag = VIRTIO_WDDM_SUBMIT_COMMAND_VIRTUAL_TAG;
         submit_priv.hdr.id = VIRTIO_WDDM_COMMAND_ID_SUBMIT;
         submit_priv.hdr.flags = VIRTIO_WDDM_COMMAND_FLAG_RING_IDX;
         submit_priv.hdr.ring = batch->ring_idx;
         submit_priv.hdr.size = batch->cs_size;

         memcpy(submit_priv.command, batch->cs_data, batch->cs_size);

         const size_t submit_priv_size = sizeof(submit_priv.submit) +
                                         sizeof(submit_priv.hdr) +
                                         batch->cs_size;
         status = virtgpu_ioctl_submit_to_hw_queue(
            gpu, hw_queue, 0, batch->cs_size, &submit_priv, submit_priv_size);
         if (!NT_SUCCESS(status)) {
            vn_log(gpu->instance, "failed to submit command: 0x%lx", status);
            ret = VK_ERROR_DEVICE_LOST;
            goto end;
         }
      }
   } else if (escape) {
      /* Submit via vendor Escape */
      struct {
         VIRTIO_WDDM_ExecBuffer exec_buf;
         VIRTIO_WDDM_CommandHeader hdr;
         uint8_t command[MAX_COMMAND_SIZE];
      } escape_priv;

      escape_priv.exec_buf.tag = VIRTIO_WDDM_ESCAPE_EXEC_BUF_TAG;
      escape_priv.hdr.id = VIRTIO_WDDM_COMMAND_ID_SUBMIT;
      escape_priv.hdr.flags = VIRTIO_WDDM_COMMAND_FLAG_RING_IDX;
      escape_priv.hdr.ring = batch->ring_idx;
      escape_priv.hdr.size = batch->cs_size;

      memcpy(escape_priv.command, batch->cs_data, batch->cs_size);

      status =
         virtgpu_ioctl_escape(gpu, &ctx->base, &escape_priv,
                              sizeof(escape_priv.exec_buf) +
                                 sizeof(escape_priv.hdr) + batch->cs_size);
      if (!NT_SUCCESS(status)) {
         vn_log(gpu->instance, "failed to submit fenced command: 0x%lx",
                status);
         ret = VK_ERROR_DEVICE_LOST;
         goto end;
      }

      struct {
         VIRTIO_WDDM_SubmitCommand submit;
         VIRTIO_WDDM_CommandHeader hdr;
         uint64_t fence_id;
      } submit_priv = {
         .submit = {
            .tag = VIRTIO_WDDM_SUBMIT_COMMAND_VIRTUAL_TAG,
         },
         .hdr = {
            .id = VIRTIO_WDDM_COMMAND_ID_FENCE,
            .size = sizeof(submit_priv.fence_id),
         },
         .fence_id = escape_priv.exec_buf.fence_id,
      };

      status = virtgpu_ioctl_submit(gpu, ctx, 0, batch->cs_size, &submit_priv,
                                    sizeof(submit_priv));
      if (!NT_SUCCESS(status)) {
         vn_log(gpu->instance, "failed to submit command: 0x%lx", status);
         ret = VK_ERROR_DEVICE_LOST;
         goto end;
      }
   } else {
      /* Submit via SubmitCommand */
      struct {
         VIRTIO_WDDM_SubmitCommand submit;
         VIRTIO_WDDM_CommandHeader hdr;
         uint8_t command[MAX_COMMAND_SIZE];
      } submit_priv;

      submit_priv.submit.tag = VIRTIO_WDDM_SUBMIT_COMMAND_VIRTUAL_TAG;
      submit_priv.hdr.id = VIRTIO_WDDM_COMMAND_ID_SUBMIT;
      submit_priv.hdr.flags = VIRTIO_WDDM_COMMAND_FLAG_RING_IDX;
      submit_priv.hdr.ring = batch->ring_idx;
      submit_priv.hdr.size = batch->cs_size;

      memcpy(submit_priv.command, batch->cs_data, batch->cs_size);

      const size_t submit_priv_size =
         sizeof(submit_priv.submit) + sizeof(submit_priv.hdr) + batch->cs_size;
      status =
         virtgpu_ioctl_submit(gpu, ctx, 0, batch->cs_size, &submit_priv,
                              submit_priv_size);
      if (!NT_SUCCESS(status)) {
         vn_log(gpu->instance, "failed to submit command: 0x%lx", status);
         ret = VK_ERROR_DEVICE_LOST;
         goto end;
      }
   }
#else
   VIRTIO_WDDM_CommandHeader *hdr = ctx->cmd_buf;
   hdr->id = VIRTIO_WDDM_COMMAND_ID_SUBMIT;
   hdr->flags = VIRTIO_WDDM_COMMAND_FLAG_RING_IDX;
   hdr->ring = batch->ring_idx;
   hdr->size = batch->cs_size;

   assert(batch->cs_size + sizeof(*hdr) <= ctx->cmd_size);
   memcpy(ctx->cmd_buf + sizeof(*hdr), batch->cs_data, batch->cs_size);

   status = virtgpu_ioctl_render(gpu, ctx, 0, sizeof(*hdr) + batch->cs_size,
                                 0, NULL, 0);
   if (!NT_SUCCESS(status)) {
      vn_log(gpu->instance, "failed to render: 0x%lx", status);
      ret = VK_ERROR_DEVICE_LOST;
      goto end;
   }
#endif
   if (ordered) {
      status = virtgpu_ioctl_signal_synchronization_objects_from_gpu(gpu,
         &ctx->base, &gpu->submission_fence->handle, &next_submission_value, 1,
         /* rewind */ false, /* signal at submission */ true);
      if (!NT_SUCCESS(status)) {
         vn_log(gpu->instance, "failed to signal for submission: 0x%lx", status);
         ret = VK_ERROR_DEVICE_LOST;
         goto end;
      }
   }

   if (!syncobj_wait && batch->sync_count > 0) {
      // TODO: handle more syncs
      assert(batch->sync_count <= 32);

#ifdef VIRTUAL_CONTEXT_SUPPORT
      if (hw_queue) {
         status =
            virtgpu_ioctl_submit_signal_synchronization_objects_to_hw_queue(
               gpu, hw_queue, sync_objects, batch->sync_values,
               batch->sync_count, /* rewind */ false,
               /* signal at submission */ false);
      } else
#endif
      {
         status = virtgpu_ioctl_signal_synchronization_objects_from_gpu(
            gpu, &ctx->base, sync_objects, batch->sync_values,
            batch->sync_count, /* rewind */ false,
            /* signal at submission */ false);
      }
      if (!NT_SUCCESS(status)) {
         vn_log(gpu->instance, "failed to signal: 0x%lx", status);
         ret = VK_ERROR_DEVICE_LOST;
         goto end;
      }
   }

end:
   STACK_ARRAY_FINISH(sync_objects);
   return ret;
}

static VkResult
virtgpu_submit(struct vn_renderer *renderer,
               const struct vn_renderer_submit_batch *batch)
{
   struct virtgpu *gpu = (struct virtgpu *)renderer;
   VkResult ret = VK_SUCCESS;

#if defined(VIRTUAL_CONTEXT_SUPPORT) && defined(INDEPENDENT_RENDER_CONTEXT)
   struct virtgpu_exec_context_virtual ctx = {};
   NTSTATUS status = virtgpu_ioctl_create_context_virtual(
      gpu, VIRTIO_WDDM_NODE_FIRST_RING + batch->ring_idx, false, &ctx);
   if (!NT_SUCCESS(status)) {
      vn_log(gpu->instance, "failed to create independent context: 0x%lx",
             status);
      ret = VK_ERROR_DEVICE_LOST;
      goto end;
   }

   ret = virtgpu_submit_batch(gpu, &ctx, NULL, batch,
                              !VN_PERF(NO_ASYNC_QUEUE_SUBMIT), false);

   status = virtgpu_ioctl_destroy_context(gpu, &ctx.base);
   if (!NT_SUCCESS(status)) {
      vn_log(gpu->instance, "failed to destroy context: 0x%lx", status);
      ret = VK_ERROR_DEVICE_LOST;
      goto end;
   }
end:
#elif defined(VIRTUAL_CONTEXT_SUPPORT)
   if (gpu->use_hw_queue) {
      ret =
         virtgpu_submit_batch(gpu, NULL, &gpu->hw_queue, batch, false, false);
   } else {
      assert(batch->ring_idx < ARRAY_SIZE(gpu->ctx));
      ret = virtgpu_submit_batch(gpu, &gpu->ctx[batch->ring_idx], NULL, batch,
                                 !VN_PERF(NO_ASYNC_QUEUE_SUBMIT), false);
   }
#elif defined(INDEPENDENT_RENDER_CONTEXT)
   struct virtgpu_exec_context ctx = {};
   NTSTATUS status = virtgpu_ioctl_create_context(gpu, &ctx);
   if (!NT_SUCCESS(status)) {
      vn_log(gpu->instance, "failed to create independent context: 0x%lx",
             status);
      ret = VK_ERROR_DEVICE_LOST;
      goto end;
   }

   ret = virtgpu_submit_batch(gpu, &ctx, batch, !VN_PERF(NO_ASYNC_QUEUE_SUBMIT),
                              false);

   status = virtgpu_ioctl_destroy_context(gpu, &ctx);
   if (!NT_SUCCESS(status)) {
      vn_log(gpu->instance, "failed to destroy context: 0x%lx", status);
      ret = VK_ERROR_DEVICE_LOST;
      goto end;
   }
end:
#else
   mtx_lock(&gpu->ctx_lock);
   size_t c = batch->ring_idx;
   assert(c <= ARRAY_SIZE(gpu->ctx));
   ret = virtgpu_submit_batch(gpu, &gpu->ctx[c], batch,
                              !VN_PERF(NO_ASYNC_QUEUE_SUBMIT), false);
   mtx_unlock(&gpu->ctx_lock);
#endif
   return ret;
}

static VkResult
virtgpu_get_device_state(struct vn_renderer *renderer)
{
   struct virtgpu *gpu = (struct virtgpu *)renderer;

   NTSTATUS status = virtgpu_ioctl_get_device_state(gpu);
   switch (status) {
   case STATUS_SUCCESS:
      return VK_SUCCESS;
   case STATUS_DEVICE_REMOVED:
      return VK_ERROR_DEVICE_LOST;
   default:
      vn_log(gpu->instance, "Unexpected device state: 0x%08lx", status);
      return VK_ERROR_DEVICE_LOST;
   }
}

static VkResult
virtgpu_submit_wait(struct vn_renderer *renderer,
                    const struct vn_renderer_submit_batch *batch)
{
   struct virtgpu *gpu = (struct virtgpu *)renderer;
   VkResult ret = VK_SUCCESS;

#if defined(VIRTUAL_CONTEXT_SUPPORT) && defined(INDEPENDENT_RENDER_CONTEXT)
   struct virtgpu_exec_context_virtual ctx = {};
   NTSTATUS status = virtgpu_ioctl_create_context_virtual(
      gpu, VIRTIO_WDDM_NODE_FIRST_RING + batch->ring_idx, false, &ctx);
   if (!NT_SUCCESS(status)) {
      vn_log(gpu->instance, "failed to create independent context: 0x%lx",
             status);
      ret = VK_ERROR_DEVICE_LOST;
      goto end;
   }

   ret = virtgpu_submit_batch(gpu, &ctx, NULL, batch, false, true);

   status = virtgpu_ioctl_destroy_context(gpu, &ctx.base);
   if (!NT_SUCCESS(status)) {
      vn_log(gpu->instance, "failed to destroy context: 0x%lx", status);
      ret = VK_ERROR_DEVICE_LOST;
      goto end;
   }
end:
#elif defined(VIRTUAL_CONTEXT_SUPPORT)
   if (gpu->use_hw_queue) {
      ret =
         virtgpu_submit_batch(gpu, NULL, &gpu->hw_queue, batch, false, true);
   } else {
      assert(batch->ring_idx < ARRAY_SIZE(gpu->ctx));
      ret = virtgpu_submit_batch(gpu, &gpu->ctx[batch->ring_idx], NULL, batch,
                                 false, true);
   }
#elif defined(INDEPENDENT_RENDER_CONTEXT)
   struct virtgpu_exec_context ctx = {};
   NTSTATUS status = virtgpu_ioctl_create_context(gpu, &ctx);
   if (!NT_SUCCESS(status)) {
      vn_log(gpu->instance, "failed to create independent context: 0x%lx",
             status);
      ret = VK_ERROR_DEVICE_LOST;
      goto end;
   }

   ret = virtgpu_submit_batch(gpu, &ctx, batch, false, true);

   status = virtgpu_ioctl_destroy_context(gpu, &ctx);
   if (!NT_SUCCESS(status)) {
      vn_log(gpu->instance, "failed to destroy context: 0x%lx", status);
      ret = VK_ERROR_DEVICE_LOST;
      goto end;
   }
end:
#else
   mtx_lock(&gpu->ctx_lock);
   size_t c = batch->ring_idx;
   assert(c <= ARRAY_SIZE(gpu->ctx));
   ret = virtgpu_submit_batch(gpu, &gpu->ctx[c], batch, false, true);
   mtx_unlock(&gpu->ctx_lock);
#endif
   return ret;
}

static void
virtgpu_init_renderer_info(struct virtgpu *gpu)
{
   struct vn_renderer_info *info = &gpu->base.info;

   info->pci.vendor_id = VIRTGPU_PCI_VENDOR_ID;
   info->pci.device_id = VIRTGPU_PCI_DEVICE_ID;

   info->pci.has_bus_info = true;
   info->pci.props = (VkPhysicalDevicePCIBusInfoPropertiesEXT){
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT,
      .pciDomain = gpu->pci_bus_info.domain,
      .pciBus = gpu->pci_bus_info.bus,
      .pciDevice = gpu->pci_bus_info.dev,
      .pciFunction = gpu->pci_bus_info.func,
   };

   info->has_dma_buf_import = true;
   info->has_external_sync = true;
   info->has_timeline_sync = !VN_PERF(NO_TIMELINE_SYNC);
   info->has_implicit_fencing = false;

   const struct virgl_renderer_capset_venus *capset = &gpu->capset.data;
   info->wire_format_version = capset->wire_format_version;
   info->vk_xml_version = capset->vk_xml_version;
   info->vk_ext_command_serialization_spec_version =
      capset->vk_ext_command_serialization_spec_version;
   info->vk_mesa_venus_protocol_spec_version =
      capset->vk_mesa_venus_protocol_spec_version;
   assert(capset->supports_blob_id_0);

   /* ensure vk_extension_mask is large enough to hold all capset masks */
   STATIC_ASSERT(sizeof(info->vk_extension_mask) >=
                 sizeof(capset->vk_extension_mask1));
   memcpy(info->vk_extension_mask, capset->vk_extension_mask1,
          sizeof(capset->vk_extension_mask1));

   assert(capset->allow_vk_wait_syncs);

   assert(capset->supports_multiple_timelines);
   info->max_timeline_count = gpu->max_timeline_count;

   info->has_guest_vram = false;

   if (gpu->ddicb != NULL) {
      /*
       * D3D UMD does not need WSI
       */
      info->needs_sw_wsi = true;
      /* D3D runtime will insert waits for IDXGIKeyedMutex_AcquireSync or
       * ID3D11DeviceContext4_Wait, so we need to roundtrip
       */
      info->needs_implicit_queue_sync = true;
      info->id.has_luid = true;
      info->id.node_mask = 1; /* TODO D3D12 interop*/
      memcpy(info->id.luid, &gpu->ddicb->AdapterLuid, VK_LUID_SIZE);
   } else {
      info->needs_sw_wsi = false;
      info->needs_implicit_queue_sync = false;
      info->id.has_luid = true;
      info->id.node_mask = 1; /* TODO D3D12 interop*/
      static_assert(
         sizeof(gpu->d3dkmt.luid) == VK_LUID_SIZE,
         "LUID and VkPhysicalDeviceIDProperties::deviceLUID differ");
      memcpy(info->id.luid, &gpu->d3dkmt.luid, VK_LUID_SIZE);
   }
}

static void
virtgpu_destroy(struct vn_renderer *renderer,
                const VkAllocationCallbacks *alloc)
{
   struct virtgpu *gpu = (struct virtgpu *)renderer;

   if (gpu->submission_fence)
     virtgpu_sync_destroy(&gpu->base, &gpu->submission_fence->base);

   vn_renderer_shmem_cache_fini(&gpu->shmem_cache);

#ifndef INDEPENDENT_RENDER_CONTEXT
#if defined(VIRTUAL_CONTEXT_SUPPORT)
   for (size_t i = 0; i < ARRAY_SIZE(gpu->ctx); i++) {
      if (gpu->ctx[i].base.h)
         virtgpu_ioctl_destroy_context(gpu, &gpu->ctx[i].base);
   }
   if (gpu->hw_queue.h)
      virtgpu_ioctl_destroy_hw_queue(gpu, &gpu->hw_queue);
   if (gpu->hw_ctx.base.h)
      virtgpu_ioctl_destroy_context(gpu, &gpu->hw_ctx.base);
#else
   for (size_t i = 0; i < ARRAY_SIZE(gpu->ctx); i++) {
      if (gpu->ctx[i].base.h)
         virtgpu_ioctl_destroy_context(gpu, &gpu->ctx[i].base);
   }
#endif
#endif
   if (gpu->d3dkmt.device)
      virtgpu_ioctl_destroy_device(gpu);
   if (gpu->d3dkmt.adapter)
      virtgpu_ioctl_close_adapter(gpu);
   if (gpu->d3dkmt.lib)
      FreeLibrary(gpu->d3dkmt.lib);

   mtx_destroy(&gpu->win32_handle_import_mutex);
#ifdef VIRTUAL_CONTEXT_SUPPORT
   mtx_destroy(&gpu->hwq_submit_lock);
#endif
#ifndef VIRTUAL_CONTEXT_SUPPORT
#ifndef INDEPENDENT_RENDER_CONTEXT
   mtx_destroy(&gpu->ctx_lock);
#endif
#endif

   util_sparse_array_finish(&gpu->shmem_array);
   util_sparse_array_finish(&gpu->bo_array);

   vk_free(alloc, gpu);
}

static VkResult
virtgpu_init(struct virtgpu *gpu, void *info)
{
   util_sparse_array_init(&gpu->syncobj_array, sizeof(struct virtgpu_sync),
                          1024);

   util_sparse_array_init(&gpu->shmem_array, sizeof(struct virtgpu_shmem),
                          1024);
   util_sparse_array_init(&gpu->bo_array, sizeof(struct virtgpu_bo), 1024);

   mtx_init(&gpu->win32_handle_import_mutex, mtx_plain);

#if !defined(VIRTUAL_CONTEXT_SUPPORT) && !defined(INDEPENDENT_RENDER_CONTEXT)
   mtx_init(&gpu->ctx_lock, mtx_plain);
#elif defined(VIRTUAL_CONTEXT_SUPPORT)
   mtx_init(&gpu->hwq_submit_lock, mtx_plain);
#endif

   struct vn_renderer_sync *submission_fence;
   VkResult result = virtgpu_open(gpu, info);
   if (result == VK_SUCCESS)
      result = virtgpu_init_params(gpu);
   if (result == VK_SUCCESS)
      result = virtgpu_init_capset(gpu);
   if (result == VK_SUCCESS)
      result = virtgpu_init_context(gpu);
   if (result == VK_SUCCESS)
      result = virtgpu_sync_create(&gpu->base, 0, &submission_fence);
   if (result != VK_SUCCESS)
      return result;

   gpu->submission_fence = (struct virtgpu_sync *) submission_fence;

   virtgpu_init_shmem_blob_mem(gpu);

   vn_renderer_shmem_cache_init(&gpu->shmem_cache, &gpu->base,
                                virtgpu_shmem_destroy_now);

   virtgpu_init_renderer_info(gpu);

   gpu->base.ops.destroy = virtgpu_destroy;
   gpu->base.ops.submit = virtgpu_submit;
   gpu->base.ops.submit_wait = virtgpu_submit_wait;
   gpu->base.ops.get_device_state = virtgpu_get_device_state;
   gpu->base.ops.wait = virtgpu_wait;

   gpu->base.shmem_ops.create = virtgpu_shmem_create;
   gpu->base.shmem_ops.destroy = virtgpu_shmem_destroy;

   gpu->base.bo_ops.create_from_device_memory =
      virtgpu_bo_create_from_device_memory;
   gpu->base.bo_ops.destroy = virtgpu_bo_destroy;
   gpu->base.bo_ops.create_from_handle = virtgpu_bo_create_from_handle;
   gpu->base.bo_ops.query_info = virtgpu_bo_query_info;
   gpu->base.bo_ops.export_handle = virtgpu_bo_export_handle;
   gpu->base.bo_ops.map = virtgpu_bo_map;
   gpu->base.bo_ops.flush = virtgpu_bo_flush;
   gpu->base.bo_ops.invalidate = virtgpu_bo_invalidate;

   gpu->base.sync_ops.create = virtgpu_sync_create;
   gpu->base.sync_ops.create_from_handle = virtgpu_sync_create_from_handle;
   gpu->base.sync_ops.destroy = virtgpu_sync_destroy;
   gpu->base.sync_ops.export_handle = virtgpu_sync_export_handle;
   gpu->base.sync_ops.reset = virtgpu_sync_reset;
   gpu->base.sync_ops.read = virtgpu_sync_read;
   gpu->base.sync_ops.write = virtgpu_sync_write;
   return VK_SUCCESS;
}

VkResult
vn_renderer_create_virtgpu_win32(struct vn_instance *instance,
                                 const VkAllocationCallbacks *alloc,
                                 const VkInstanceCreateInfo *pCreateInfo,
                                 struct vn_renderer **renderer)
{
   struct virtgpu *gpu = vk_zalloc(alloc, sizeof(*gpu), VN_DEFAULT_ALIGN,
                                   VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
   if (!gpu)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   gpu->instance = instance;

   VkResult result = virtgpu_init(gpu, (void *)pCreateInfo->pNext);
   if (result != VK_SUCCESS) {
      virtgpu_destroy(&gpu->base, alloc);
      return result;
   }

   *renderer = &gpu->base;

   return VK_SUCCESS;
}
