/*
 * Copyright 2026 Ngo Quoc Dat
 * SPDX-License-Identifier: MIT
 */

#ifndef KK_SPARSE_H
#define KK_SPARSE_H 1

#include "kk_private.h"

#include "kosmickrisp/bridge/mtl_types.h"

#include "util/rb_tree.h"
#include "util/simple_mtx.h"
#include "util/u_dynarray.h"

struct kk_bo;
struct kk_buffer;
struct kk_buffer_view;
struct kk_device;
struct kk_queue;
struct vk_queue_submit;

struct kk_sparse_device {
   bool enabled;

   /* Serializes mapping updates, the tiles of every sparse resource and the
    * sparse references of memory */
   simple_mtx_t lock;

   /* Mapping this tile again ends every bind batch that unmaps. Without a
    * later map, work that already read through an unmapped tile keeps seeing
    * its old page.
    */
   mtl_buffer *remap_buffer;
   mtl_heap *remap_heap;

   /* Orders the work after a bind batch behind its mapping updates */
   mtl_event *event;
   uint64_t event_value;

   /* Freed memory whose last tile unmapped, released once the unmaps are
    * queued */
   struct util_dynarray released;
};

/* What the tiles of a sparse resource map from, as ranges of consecutive
 * tiles that map consecutive tiles of one memory object. Each mapped tile
 * holds a reference on its memory, so freeing the memory leaves its heap in
 * place until the tile unmaps or the resource goes.
 */
struct kk_sparse_tiles {
   struct rb_tree ranges;
};

struct kk_sparse_range {
   struct rb_node node;
   uint64_t tile;
   uint64_t count;
   struct kk_bo *bo;
   uint64_t heap_tile;
};

VkResult kk_device_init_sparse(struct kk_device *dev);
void kk_device_finish_sparse(struct kk_device *dev);

VkResult kk_queue_bind_sparse(struct kk_queue *queue,
                              struct vk_queue_submit *submit);

void kk_sparse_tiles_init(struct kk_sparse_tiles *tiles);
void kk_sparse_tiles_finish(struct kk_device *dev,
                            struct kk_sparse_tiles *tiles);

/* False when memory can go now, true when sparse tiles still map it and it
 * goes with the last of them */
bool kk_sparse_keep_memory(struct kk_device *dev, struct kk_bo *bo);

void kk_sparse_buffer_init(struct kk_buffer *buffer);
void kk_sparse_buffer_finish(struct kk_device *dev, struct kk_buffer *buffer);

void kk_sparse_buffer_add_view(struct kk_device *dev, struct kk_buffer *buffer,
                               struct kk_buffer_view *view);
void kk_sparse_buffer_remove_view(struct kk_device *dev,
                                  struct kk_buffer_view *view);

#endif /* KK_SPARSE_H */
