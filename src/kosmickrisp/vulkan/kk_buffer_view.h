/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef KK_BUFFER_VIEW_H
#define KK_BUFFER_VIEW_H 1

#include "kk_private.h"

#include "kosmickrisp/bridge/mtl_types.h"

#include "vk_buffer_view.h"

#include "util/list.h"

struct kk_physical_device;

VkFormatFeatureFlags2
kk_get_buffer_format_features(struct kk_physical_device *pdev, VkFormat format);

struct kk_buffer_view {
   struct vk_buffer_view vk;
   mtl_texture *mtl_texel_buffer_handle;
   uint64_t texel_buffer_gpu_id;

   /* Texels the Metal texture starts before the view's offset */
   uint32_t offset_el;

   /* A view of a sparse buffer mirrors these tiles of the buffer */
   struct {
      struct kk_buffer *buffer;
      struct list_head link;
      uint64_t first_tile;
      uint64_t tile_count;
   } sparse;

   /* Where loads with residency and stores find the view's tiles in the
    * buffer's residency mirror, how many there are and the texel size. 0 for
    * views without a mirror, which are resident. */
   uint64_t residency_addr;
   uint32_t residency_tiles;
   uint32_t texel_size_B;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(kk_buffer_view, vk.base, VkBufferView,
                               VK_OBJECT_TYPE_BUFFER_VIEW)

void kk_buffer_view_set_label(struct kk_buffer_view *bview, const char *label);

#endif
