/*
 * Copyright 2026 Ngo Quoc Dat
 * SPDX-License-Identifier: MIT
 */

#include "kk_sparse.h"

#include "kk_bo.h"
#include "kk_buffer.h"
#include "kk_buffer_view.h"
#include "kk_device.h"
#include "kk_device_memory.h"
#include "kk_image.h"
#include "kk_queue.h"

#include "kosmickrisp/bridge/mtl_bridge.h"

#include "vk_alloc.h"
#include "vk_log.h"
#include "vk_queue.h"

static void
kk_sparse_fill_ones(struct kk_bo *bo, enum kk_sparse_ones ones)
{
   /* An LDR void-extent block: constant (1, 1, 1, 1) for any block size */
   static const uint8_t astc_white[16] = {
      0xfc, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
      0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
   };
   const float one[4] = {1.0f, 1.0f, 1.0f, 1.0f};
   uint8_t *cpu = bo->cpu;

   for (uint64_t i = 0u; i < bo->size_B; i += 16u) {
      switch (ones) {
      case KK_SPARSE_ONES_UNORM:
         memset(cpu + i, 0xff, 16u);
         break;
      case KK_SPARSE_ONES_FLOAT:
         memcpy(cpu + i, one, 16u);
         break;
      case KK_SPARSE_ONES_ASTC:
         memcpy(cpu + i, astc_white, 16u);
         break;
      default:
         UNREACHABLE("Invalid heap of ones");
      }
   }
}

VkResult
kk_device_init_sparse(struct kk_device *dev)
{
   struct kk_sparse_device *sparse = &dev->sparse;

   simple_mtx_init(&sparse->lock, mtx_plain);
   sparse->released = UTIL_DYNARRAY_INIT;

   if (!dev->vk.enabled_features.sparseBinding)
      return VK_SUCCESS;

   sparse->enabled = true;
   sparse->remap_heap = mtl_new_heap(dev->mtl_handle, KK_SPARSE_PAGE_SIZE_B,
                                     KK_MTL_RESOURCE_OPTIONS, true);
   sparse->remap_buffer =
      mtl_new_placement_sparse_buffer(dev->mtl_handle, KK_SPARSE_PAGE_SIZE_B);
   sparse->event = mtl_new_event(dev->mtl_handle);
   if (!sparse->remap_heap || !sparse->remap_buffer || !sparse->event) {
      kk_device_finish_sparse(dev);
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }

   kk_device_add_heap_to_residency_set(dev, sparse->remap_heap);
   kk_device_add_buffer_to_residency_set(dev, sparse->remap_buffer);

   if (dev->vk.enabled_features.sparseResidencyBuffer ||
       dev->vk.enabled_features.sparseResidencyImage2D) {
      for (unsigned i = 0; i < KK_SPARSE_ONES_COUNT; i++) {
         VkResult result = kk_alloc_bo_flags(
            dev, &dev->vk.base, KK_SPARSE_ONES_TILES * KK_SPARSE_PAGE_SIZE_B,
            KK_SPARSE_PAGE_SIZE_B, KK_BO_SPARSE_TILES, &sparse->ones[i]);
         if (result != VK_SUCCESS) {
            kk_device_finish_sparse(dev);
            return result;
         }
         kk_sparse_fill_ones(sparse->ones[i], i);
      }
   }

   return VK_SUCCESS;
}

static void
kk_sparse_release_memory(struct kk_device *dev)
{
   util_dynarray_foreach(&dev->sparse.released, struct kk_bo *, bo)
      kk_destroy_bo(dev, *bo);
   util_dynarray_clear(&dev->sparse.released);
}

void
kk_device_finish_sparse(struct kk_device *dev)
{
   struct kk_sparse_device *sparse = &dev->sparse;

   if (sparse->remap_buffer) {
      kk_device_remove_buffer_from_residency_set(dev, sparse->remap_buffer);
      mtl_release(sparse->remap_buffer);
   }
   if (sparse->remap_heap) {
      kk_device_remove_heap_from_residency_set(dev, sparse->remap_heap);
      mtl_release(sparse->remap_heap);
   }
   if (sparse->event)
      mtl_release(sparse->event);
   for (unsigned i = 0; i < KK_SPARSE_ONES_COUNT; i++) {
      if (sparse->ones[i])
         kk_destroy_bo(dev, sparse->ones[i]);
   }

   kk_sparse_release_memory(dev);
   util_dynarray_fini(&sparse->released);
   simple_mtx_destroy(&sparse->lock);

   *sparse = (struct kk_sparse_device){0};
}

static void
kk_sparse_unref_memory(struct kk_device *dev, struct kk_bo *bo,
                       uint64_t tiles)
{
   assert(bo->sparse_refs >= tiles);
   bo->sparse_refs -= tiles;
   if (bo->sparse_refs == 0u && bo->sparse_freed)
      util_dynarray_append(&dev->sparse.released, bo);
}

bool
kk_sparse_keep_memory(struct kk_device *dev, struct kk_bo *bo)
{
   simple_mtx_lock(&dev->sparse.lock);
   const bool keep = bo->sparse_refs > 0u;
   bo->sparse_freed = keep;
   simple_mtx_unlock(&dev->sparse.lock);

   return keep;
}

static struct kk_sparse_range *
kk_sparse_range(const struct rb_node *node)
{
   return node ? rb_node_data(struct kk_sparse_range, node, node) : NULL;
}

/* rb_tree goes left when this is negative */
static int
kk_sparse_range_cmp(const struct rb_node *a, const struct rb_node *b)
{
   return kk_sparse_range(b)->tile < kk_sparse_range(a)->tile ? -1 : 1;
}

/* The first range that ends after the tile */
static struct kk_sparse_range *
kk_sparse_tiles_find(const struct kk_sparse_tiles *tiles, uint64_t tile)
{
   struct rb_node *found = NULL;
   struct rb_node *node = tiles->ranges.root;
   while (node != NULL) {
      const struct kk_sparse_range *range = kk_sparse_range(node);
      if (range->tile + range->count > tile) {
         found = node;
         node = node->left;
      } else {
         node = node->right;
      }
   }

   return kk_sparse_range(found);
}

void
kk_sparse_tiles_init(struct kk_sparse_tiles *tiles)
{
   rb_tree_init(&tiles->ranges);
}

void
kk_sparse_tiles_finish(struct kk_device *dev, struct kk_sparse_tiles *tiles)
{
   if (rb_tree_is_empty(&tiles->ranges))
      return;

   simple_mtx_lock(&dev->sparse.lock);
   while (!rb_tree_is_empty(&tiles->ranges)) {
      struct kk_sparse_range *range =
         kk_sparse_range(rb_tree_first(&tiles->ranges));
      kk_sparse_unref_memory(dev, range->bo, range->count);
      rb_tree_remove(&tiles->ranges, &range->node);
      vk_free(&dev->vk.alloc, range);
   }
   kk_sparse_release_memory(dev);
   simple_mtx_unlock(&dev->sparse.lock);
}

static struct kk_sparse_range *
kk_sparse_range_alloc(struct kk_device *dev)
{
   return vk_zalloc(&dev->vk.alloc, sizeof(struct kk_sparse_range), 8,
                    VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
}

/* Records that tiles [tile, tile + count) map heap tiles of bo from
 * heap_tile on, or nothing when bo is NULL. Nothing changes when it fails.
 */
static bool
kk_sparse_tiles_bind(struct kk_device *dev, struct kk_sparse_tiles *tiles,
                     uint64_t tile, uint64_t count, struct kk_bo *bo,
                     uint64_t heap_tile)
{
   const uint64_t end = tile + count;

   /* A bind splits at most one range, and adds at most one */
   struct kk_sparse_range *spare[2] = {kk_sparse_range_alloc(dev),
                                       kk_sparse_range_alloc(dev)};
   if (spare[0] == NULL || spare[1] == NULL) {
      vk_free(&dev->vk.alloc, spare[0]);
      vk_free(&dev->vk.alloc, spare[1]);
      return false;
   }

   struct kk_sparse_range *range = kk_sparse_tiles_find(tiles, tile);
   while (range != NULL && range->tile < end) {
      struct kk_sparse_range *next =
         kk_sparse_range(rb_node_next(&range->node));
      const uint64_t range_end = range->tile + range->count;

      kk_sparse_unref_memory(dev, range->bo,
                             MIN2(range_end, end) - MAX2(range->tile, tile));

      if (range->tile < tile && range_end > end) {
         struct kk_sparse_range *right = spare[0];
         spare[0] = NULL;
         *right = (struct kk_sparse_range){
            .tile = end,
            .count = range_end - end,
            .bo = range->bo,
            .heap_tile = range->heap_tile + (end - range->tile),
         };
         range->count = tile - range->tile;
         rb_tree_insert(&tiles->ranges, &right->node, kk_sparse_range_cmp);
         break;
      } else if (range->tile < tile) {
         range->count = tile - range->tile;
      } else if (range_end > end) {
         range->heap_tile += end - range->tile;
         range->count = range_end - end;
         range->tile = end;
      } else {
         rb_tree_remove(&tiles->ranges, &range->node);
         vk_free(&dev->vk.alloc, range);
      }

      range = next;
   }

   if (bo != NULL) {
      bo->sparse_refs += count;

      /* Join neighbours that map the tiles next to these in the heap */
      struct kk_sparse_range *next = kk_sparse_tiles_find(tiles, tile);
      struct kk_sparse_range *prev = kk_sparse_range(
         next ? rb_node_prev(&next->node) : rb_tree_last(&tiles->ranges));
      const bool join_prev = prev != NULL && prev->bo == bo &&
                             prev->tile + prev->count == tile &&
                             prev->heap_tile + prev->count == heap_tile;
      const bool join_next = next != NULL && next->bo == bo &&
                             next->tile == end &&
                             heap_tile + count == next->heap_tile;

      if (join_prev && join_next) {
         prev->count += count + next->count;
         rb_tree_remove(&tiles->ranges, &next->node);
         vk_free(&dev->vk.alloc, next);
      } else if (join_prev) {
         prev->count += count;
      } else if (join_next) {
         next->tile = tile;
         next->heap_tile = heap_tile;
         next->count += count;
      } else {
         struct kk_sparse_range *added = spare[1];
         spare[1] = NULL;
         *added = (struct kk_sparse_range){
            .tile = tile,
            .count = count,
            .bo = bo,
            .heap_tile = heap_tile,
         };
         rb_tree_insert(&tiles->ranges, &added->node, kk_sparse_range_cmp);
      }
   }

   vk_free(&dev->vk.alloc, spare[0]);
   vk_free(&dev->vk.alloc, spare[1]);
   return true;
}

/* Mapping updates of one resource from one heap, sent in one call */
struct kk_bind_batch {
   struct kk_device *dev;
   struct kk_queue *queue;

   void *resource;
   bool texture;
   mtl_heap *heap;
   struct util_dynarray ops;

   bool unmapped;

   /* Layers whose tail an opaque bind touched */
   struct util_dynarray tails;

   /* Updates of residency shadows and residency mirrors, sent apart so they
    * do not split the updates of the resources they follow */
   struct kk_bind_batch *shadow;
};

static void
kk_bind_batch_flush(struct kk_bind_batch *b)
{
   if (b->resource == NULL)
      return;

   if (b->texture) {
      mtl_command_queue_update_texture_mappings(
         b->queue->mtl_handle, b->resource, b->heap,
         util_dynarray_begin(&b->ops),
         util_dynarray_num_elements(&b->ops, struct mtl_texture_mapping));
   } else {
      mtl_command_queue_update_buffer_mappings(
         b->queue->mtl_handle, b->resource, b->heap,
         util_dynarray_begin(&b->ops),
         util_dynarray_num_elements(&b->ops, struct mtl_buffer_mapping));
   }

   util_dynarray_clear(&b->ops);
   b->resource = NULL;
   b->heap = NULL;
}

static void
kk_bind_batch_begin(struct kk_bind_batch *b, void *resource, bool texture,
                    mtl_heap *heap)
{
   if (b->resource == resource && b->heap == heap)
      return;

   kk_bind_batch_flush(b);
   b->resource = resource;
   b->texture = texture;
   b->heap = heap;
}

static void
kk_bind_buffer_tiles(struct kk_bind_batch *b, mtl_buffer *buffer,
                     mtl_heap *heap, uint64_t tile, uint64_t count,
                     uint64_t heap_tile)
{
   kk_bind_batch_begin(b, buffer, false, heap);

   const bool map = heap != NULL;
   b->unmapped |= !map;

   if (util_dynarray_num_elements(&b->ops, struct mtl_buffer_mapping) > 0) {
      struct mtl_buffer_mapping *last =
         util_dynarray_top_ptr(&b->ops, struct mtl_buffer_mapping);
      if (last->map == map && last->tile + last->tile_count == tile &&
          (!map || last->heap_tile + last->tile_count == heap_tile)) {
         last->tile_count += count;
         return;
      }
   }

   util_dynarray_append(&b->ops, ((struct mtl_buffer_mapping){
                                    .tile = tile,
                                    .tile_count = count,
                                    .heap_tile = map ? heap_tile : 0u,
                                    .map = map,
                                 }));
}

static void
kk_bind_texture_region(struct kk_bind_batch *b, mtl_texture *texture,
                       mtl_heap *heap, struct mtl_origin origin,
                       struct mtl_size size, uint32_t level, uint32_t slice,
                       uint64_t heap_tile)
{
   kk_bind_batch_begin(b, texture, true, heap);

   const bool map = heap != NULL;
   b->unmapped |= !map;

   /* Rows of tiles extend the previous row */
   if (util_dynarray_num_elements(&b->ops, struct mtl_texture_mapping) > 0 &&
       size.y == 1u && size.z == 1u) {
      struct mtl_texture_mapping *last =
         util_dynarray_top_ptr(&b->ops, struct mtl_texture_mapping);
      if (last->map == map && last->level == level && last->slice == slice &&
          last->size.y == 1u && last->size.z == 1u &&
          last->origin.y == origin.y && last->origin.z == origin.z &&
          last->origin.x + last->size.x == origin.x &&
          (!map || last->heap_tile + last->size.x == heap_tile)) {
         last->size.x += size.x;
         return;
      }
   }

   util_dynarray_append(&b->ops, ((struct mtl_texture_mapping){
                                    .origin = origin,
                                    .size = size,
                                    .level = level,
                                    .slice = slice,
                                    .heap_tile = map ? heap_tile : 0u,
                                    .map = map,
                                 }));
}

/* Later work on the queue starts after the mapping updates */
static void
kk_sparse_queue_barrier(struct kk_device *dev, struct kk_queue *queue)
{
   struct kk_sparse_device *sparse = &dev->sparse;

   sparse->event_value++;
   mtl_signal_event(queue->mtl_handle, sparse->event, sparse->event_value);
   mtl_wait_for_event(queue->mtl_handle, sparse->event, sparse->event_value);
}

static void
kk_bind_batch_end(struct kk_bind_batch *b)
{
   struct kk_sparse_device *sparse = &b->dev->sparse;

   kk_bind_batch_flush(b);
   if (b->shadow != NULL) {
      kk_bind_batch_flush(b->shadow);
      b->unmapped |= b->shadow->unmapped;
      util_dynarray_fini(&b->shadow->ops);
      util_dynarray_fini(&b->shadow->tails);
   }

   /* Work that read through a tile before a lone unmap keeps reading its old
    * page until the queue maps something. */
   if (b->unmapped) {
      kk_bind_buffer_tiles(b, sparse->remap_buffer, sparse->remap_heap, 0u, 1u,
                           0u);
      kk_bind_batch_flush(b);
   }

   util_dynarray_fini(&b->ops);
   util_dynarray_fini(&b->tails);

   kk_sparse_queue_barrier(b->dev, b->queue);

   /* The unmaps of freed memory are queued, so its heaps can go */
   kk_sparse_release_memory(b->dev);
}

/* A residency shadow maps a heap of ones wherever its image maps memory, at
 * most the heap's tiles in one operation, each from the heap's first tile. */
static void
kk_bind_shadow_row(struct kk_bind_batch *b,
                   const struct kk_image_sparse *sparse, bool map,
                   struct mtl_origin origin, uint64_t count, uint32_t level,
                   uint32_t slice)
{
   if (sparse->shadow == NULL)
      return;

   struct kk_bind_batch *sb = b->shadow != NULL ? b->shadow : b;
   mtl_heap *ones =
      map ? b->dev->sparse.ones[sparse->shadow_ones]->mtl_handle : NULL;
   const uint64_t step = map ? KK_SPARSE_ONES_TILES : count;

   for (uint64_t x = 0u; x < count; x += step) {
      kk_bind_texture_region(
         sb, sparse->shadow, ones,
         (struct mtl_origin){origin.x + x, origin.y, origin.z},
         (struct mtl_size){MIN2(step, count - x), 1u, 1u}, level, slice, 0u);
   }
}

static void
kk_bind_shadow_level_tiles(struct kk_bind_batch *b,
                           const struct kk_image_sparse *sparse, bool map,
                           struct mtl_size tiles, uint32_t level,
                           uint32_t slice, uint64_t first, uint64_t count)
{
   const uint64_t row = tiles.x;
   const uint64_t plane = tiles.x * tiles.y;
   const uint64_t end = first + count;

   for (uint64_t i = first; i < end;) {
      const struct mtl_origin origin = {
         .x = i % row,
         .y = (i / row) % tiles.y,
         .z = i / plane,
      };
      const uint64_t n = MIN2(row - origin.x, end - i);
      kk_bind_shadow_row(b, sparse, map, origin, n, level, slice);
      i += n;
   }
}

/* Tiles [first, first + count) of a level of tiles laid out in X, Y, then Z
 * order, as boxes Metal walks in the same order.
 */
static void
kk_bind_level_tiles(struct kk_bind_batch *b, mtl_texture *texture,
                    mtl_heap *heap, struct mtl_size tiles, uint32_t level,
                    uint32_t slice, uint64_t first, uint64_t count,
                    uint64_t heap_tile)
{
   const uint64_t row = tiles.x;
   const uint64_t plane = tiles.x * tiles.y;
   uint64_t i = first;
   const uint64_t end = first + count;

   while (i < end) {
      const struct mtl_origin origin = {
         .x = i % row,
         .y = (i / row) % tiles.y,
         .z = i / plane,
      };
      struct mtl_size size;

      if (origin.x != 0 || end - i < row) {
         size = (struct mtl_size){MIN2(row - origin.x, end - i), 1, 1};
      } else if (origin.y != 0 || end - i < plane) {
         size = (struct mtl_size){row,
                                  MIN2(tiles.y - origin.y, (end - i) / row), 1};
      } else {
         size = (struct mtl_size){row, tiles.y, (end - i) / plane};
      }

      kk_bind_texture_region(b, texture, heap, origin, size, level, slice,
                             heap_tile);

      const uint64_t n = size.x * size.y * size.z;
      i += n;
      heap_tile += n;
   }
}

/* Maps a layer's tail from the memory its tiles are bound to. Metal maps each
 * tile of a 2D tail from a heap tile of its own. It places a 3D tail across
 * all of its heap tiles at once, which can be more than its region has, and
 * misplaces or faults on one whose tiles come from several heaps. A 3D tail
 * maps from the bound memory when it is bound whole to consecutive tiles of
 * one memory object, and otherwise, while any of it is bound, from a heap of
 * the image's own.
 */
static VkResult
kk_bind_image_tail(struct kk_bind_batch *b, struct kk_image *image,
                   uint32_t layer)
{
   struct kk_image_plane *plane = &image->planes[0];
   struct kk_image_sparse *sparse = &plane->sparse;
   const uint64_t first =
      layer * sparse->layer_tiles + sparse->level_tile[sparse->tail_level];
   const uint64_t end = first + sparse->tail_tiles;

   if (plane->layout.type != MTL_TEXTURE_TYPE_3D) {
      struct kk_sparse_range *range =
         kk_sparse_tiles_find(&sparse->tiles, first);
      uint64_t tile = first;
      while (tile < end) {
         /* Unbound up to the next range, then mapped through it */
         const uint64_t next =
            range != NULL ? MIN2(MAX2(range->tile, tile), end) : end;
         if (tile < next) {
            kk_bind_texture_region(b, plane->mtl_handle, NULL,
                                   (struct mtl_origin){tile - first, 0u, 0u},
                                   (struct mtl_size){next - tile, 1u, 1u},
                                   sparse->tail_level, layer, 0u);
            kk_bind_shadow_row(b, sparse, false,
                               (struct mtl_origin){tile - first, 0u, 0u},
                               next - tile, sparse->tail_level, layer);
            tile = next;
            continue;
         }

         const uint64_t last = MIN2(range->tile + range->count, end);
         kk_bind_texture_region(
            b, plane->mtl_handle, range->bo->mtl_handle,
            (struct mtl_origin){tile - first, 0u, 0u},
            (struct mtl_size){last - tile, 1u, 1u}, sparse->tail_level, layer,
            range->heap_tile + (tile - range->tile));
         kk_bind_shadow_row(b, sparse, true,
                            (struct mtl_origin){tile - first, 0u, 0u},
                            last - tile, sparse->tail_level, layer);
         tile = last;
         range = kk_sparse_range(rb_node_next(&range->node));
      }
      return VK_SUCCESS;
   }

   const struct kk_sparse_range *range =
      kk_sparse_tiles_find(&sparse->tiles, first);
   mtl_heap *heap = NULL;
   uint64_t heap_tile = 0u;
   if (range != NULL && range->tile <= first &&
       range->tile + range->count >= end) {
      heap = range->bo->mtl_handle;
      heap_tile = range->heap_tile + (first - range->tile);
   } else if (range != NULL && range->tile < end) {
      if (sparse->tail_heap == NULL) {
         sparse->tail_heap = mtl_new_heap(
            b->dev->mtl_handle,
            (uint64_t)sparse->tail_tiles * plane->layout.layers *
               KK_SPARSE_PAGE_SIZE_B,
            KK_MTL_RESOURCE_OPTIONS, true);
         if (sparse->tail_heap == NULL)
            return vk_error(b->dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         kk_device_add_heap_to_residency_set(b->dev, sparse->tail_heap);
      }
      heap = sparse->tail_heap;
      heap_tile = (uint64_t)layer * sparse->tail_tiles;
   }

   kk_bind_texture_region(b, plane->mtl_handle, heap,
                          (struct mtl_origin){0u, 0u, 0u},
                          (struct mtl_size){sparse->tail_width, 1u, 1u},
                          sparse->tail_level, layer, heap_tile);
   return VK_SUCCESS;
}

/* Binds a range of a sparse image's opaque memory, in tiles. Tails map once
 * the image's binds are all recorded. */
static void
kk_bind_image_opaque_tiles(struct kk_bind_batch *b, struct kk_image *image,
                           mtl_heap *heap, uint64_t tile, uint64_t count,
                           uint64_t heap_tile)
{
   const struct kk_image_plane *plane = &image->planes[0];
   const struct kk_image_sparse *sparse = &plane->sparse;
   const uint64_t tail_first = sparse->level_tile[sparse->tail_level];

   while (count > 0) {
      const uint32_t layer = tile / sparse->layer_tiles;
      const uint64_t in_layer = tile % sparse->layer_tiles;

      uint64_t n;
      if (in_layer >= tail_first) {
         n = MIN2(count, sparse->tail_tiles - (in_layer - tail_first));
         if (util_dynarray_num_elements(&b->tails, uint32_t) == 0 ||
             *util_dynarray_top_ptr(&b->tails, uint32_t) != layer)
            util_dynarray_append(&b->tails, layer);
      } else {
         uint32_t level = 0;
         while (sparse->level_tile[level + 1] <= in_layer)
            level++;

         const uint64_t first = in_layer - sparse->level_tile[level];
         const uint64_t level_count =
            sparse->level_tile[level + 1] - sparse->level_tile[level];
         n = MIN2(count, level_count - first);
         kk_bind_level_tiles(b, plane->mtl_handle, heap,
                             kk_image_sparse_level_tiles(plane, level), level,
                             layer, first, n, heap_tile);
         kk_bind_shadow_level_tiles(b, sparse, heap != NULL,
                                    kk_image_sparse_level_tiles(plane, level),
                                    level, layer, first, n);
      }

      tile += n;
      count -= n;
      heap_tile += n;
   }
}

/* Tiles [*tile, *tile + *count) of a range of bytes of a resource's memory,
 * or false when the range does not start on a tile or reaches outside the
 * memory. Only a range that ends with the memory may end inside a tile.
 */
static bool
kk_sparse_bind_range(VkDeviceSize offset_B, VkDeviceSize size_B,
                     VkDeviceSize resource_B, uint64_t *tile, uint64_t *count)
{
   if (offset_B % KK_SPARSE_PAGE_SIZE_B != 0u || offset_B >= resource_B ||
       size_B > resource_B - offset_B ||
       (size_B % KK_SPARSE_PAGE_SIZE_B != 0u &&
        offset_B + size_B != resource_B))
      return false;

   *tile = offset_B / KK_SPARSE_PAGE_SIZE_B;
   *count = DIV_ROUND_UP(size_B, KK_SPARSE_PAGE_SIZE_B);
   return true;
}

/* Memory and first heap tile for a bind of count tiles, or NULL memory to
 * unbind. Memory whose heap cannot back tiles, as imported memory, binds as
 * unbound. False when the tiles do not start on a tile of the memory or reach
 * past its end.
 */
static bool
kk_sparse_bind_memory(VkDeviceMemory memory, VkDeviceSize offset_B,
                      uint64_t count, struct kk_bo **bo_out,
                      uint64_t *heap_tile_out)
{
   VK_FROM_HANDLE(kk_device_memory, mem, memory);

   *bo_out = NULL;
   *heap_tile_out = 0u;
   if (mem == NULL)
      return true;

   const uint64_t heap_tiles = mem->bo->sparse_tiles;
   if (heap_tiles == 0u) {
      vk_loge(VK_LOG_OBJS(&mem->vk.base),
              "Memory that cannot back sparse tiles is bound as unbound");
      return true;
   }

   const uint64_t heap_tile = offset_B / KK_SPARSE_PAGE_SIZE_B;
   if (offset_B % KK_SPARSE_PAGE_SIZE_B != 0u || heap_tile > heap_tiles ||
       count > heap_tiles - heap_tile)
      return false;

   *bo_out = mem->bo;
   *heap_tile_out = heap_tile;
   return true;
}

static mtl_heap *
kk_bo_heap(struct kk_bo *bo)
{
   return bo ? bo->mtl_handle : NULL;
}

/* Maps the mirror tiles of the tiles a bind maps to ones, and unmaps those of
 * the tiles it unmaps */
static void
kk_bind_mirror_tiles(struct kk_bind_batch *b, struct kk_buffer *buffer,
                     uint64_t tile, uint64_t count, bool map)
{
   struct kk_bind_batch *sb = b->shadow != NULL ? b->shadow : b;
   const uint64_t first = buffer->sparse.mirror_tile + tile;

   if (!map) {
      kk_bind_buffer_tiles(sb, buffer->metal.handle, NULL, first, count, 0u);
      return;
   }

   mtl_heap *ones = b->dev->sparse.ones[KK_SPARSE_ONES_UNORM]->mtl_handle;
   for (uint64_t i = 0u; i < count; i += KK_SPARSE_ONES_TILES) {
      kk_bind_buffer_tiles(sb, buffer->metal.handle, ones, first + i,
                           MIN2(KK_SPARSE_ONES_TILES, count - i), 0u);
   }
}

/* Mirrors a bind of the buffer into its texel buffer views */
static void
kk_bind_buffer_views(struct kk_bind_batch *b, struct kk_buffer *buffer,
                     mtl_heap *heap, uint64_t tile, uint64_t count,
                     uint64_t heap_tile)
{
   const uint64_t end = tile + count;

   list_for_each_entry(struct kk_buffer_view, view, &buffer->sparse.views,
                       sparse.link) {
      const uint64_t first = MAX2(tile, view->sparse.first_tile);
      const uint64_t last =
         MIN2(end, view->sparse.first_tile + view->sparse.tile_count);
      if (first >= last)
         continue;

      kk_bind_texture_region(
         b, view->mtl_texel_buffer_handle, heap,
         (struct mtl_origin){first - view->sparse.first_tile, 0u, 0u},
         (struct mtl_size){last - first, 1u, 1u}, 0u, 0u,
         heap_tile + (first - tile));
   }
}

static VkResult
kk_bind_sparse_buffer(struct kk_bind_batch *b,
                      const VkSparseBufferMemoryBindInfo *info)
{
   VK_FROM_HANDLE(kk_buffer, buffer, info->buffer);
   VkResult result = VK_SUCCESS;

   for (uint32_t i = 0; i < info->bindCount; i++) {
      const VkSparseMemoryBind *bind = &info->pBinds[i];
      if (bind->size == 0)
         continue;

      uint64_t tile, count, heap_tile;
      struct kk_bo *bo;
      /* The buffer's memory is whole tiles */
      if (!kk_sparse_bind_range(bind->resourceOffset, bind->size,
                                align64(buffer->vk.size, KK_SPARSE_PAGE_SIZE_B),
                                &tile, &count) ||
          !kk_sparse_bind_memory(bind->memory, bind->memoryOffset, count, &bo,
                                 &heap_tile)) {
         vk_loge(VK_LOG_OBJS(&buffer->vk.base),
                 "Sparse bind outside the buffer or its memory is dropped");
         continue;
      }

      if (!kk_sparse_tiles_bind(b->dev, &buffer->sparse.tiles, tile, count,
                                bo, heap_tile)) {
         result = vk_error(b->dev, VK_ERROR_OUT_OF_HOST_MEMORY);
         continue;
      }

      kk_bind_buffer_tiles(b, buffer->metal.handle, kk_bo_heap(bo), tile,
                           count, heap_tile);
      kk_bind_buffer_views(b, buffer, kk_bo_heap(bo), tile, count, heap_tile);
      if (buffer->sparse.mirror_tile != 0u)
         kk_bind_mirror_tiles(b, buffer, tile, count, bo != NULL);
   }

   return result;
}

static VkResult
kk_bind_sparse_image_opaque(struct kk_bind_batch *b,
                            const VkSparseImageOpaqueMemoryBindInfo *info)
{
   VK_FROM_HANDLE(kk_image, image, info->image);
   struct kk_image_plane *plane = &image->planes[0];
   VkResult result = VK_SUCCESS;

   for (uint32_t i = 0; i < info->bindCount; i++) {
      const VkSparseMemoryBind *bind = &info->pBinds[i];

      /* No sparse image has a metadata aspect */
      if (bind->size == 0 || (bind->flags & VK_SPARSE_MEMORY_BIND_METADATA_BIT))
         continue;

      uint64_t tile, count, heap_tile;
      struct kk_bo *bo;
      if (!kk_sparse_bind_range(bind->resourceOffset, bind->size,
                                plane->layout.size_B, &tile, &count) ||
          !kk_sparse_bind_memory(bind->memory, bind->memoryOffset, count, &bo,
                                 &heap_tile)) {
         vk_loge(VK_LOG_OBJS(&image->vk.base),
                 "Sparse bind outside the image or its memory is dropped");
         continue;
      }

      if (!kk_sparse_tiles_bind(b->dev, &plane->sparse.tiles, tile, count, bo,
                                heap_tile)) {
         result = vk_error(b->dev, VK_ERROR_OUT_OF_HOST_MEMORY);
         continue;
      }

      kk_bind_image_opaque_tiles(b, image, kk_bo_heap(bo), tile, count,
                                 heap_tile);
   }

   util_dynarray_foreach(&b->tails, uint32_t, layer) {
      VkResult r = kk_bind_image_tail(b, image, *layer);
      if (result == VK_SUCCESS)
         result = r;
   }
   util_dynarray_clear(&b->tails);

   return result;
}

static VkResult
kk_bind_sparse_image(struct kk_bind_batch *b,
                     const VkSparseImageMemoryBindInfo *info)
{
   VK_FROM_HANDLE(kk_image, image, info->image);
   struct kk_image_plane *plane = &image->planes[0];
   struct kk_image_sparse *sparse = &plane->sparse;
   const struct mtl_size tile_px = sparse->tile_px;
   VkResult result = VK_SUCCESS;

   for (uint32_t i = 0; i < info->bindCount; i++) {
      const VkSparseImageMemoryBind *bind = &info->pBinds[i];
      const uint32_t level = bind->subresource.mipLevel;
      const uint32_t layer = bind->subresource.arrayLayer;

      /* Offsets are whole tiles inside the subresource, which is before the
       * tail, as the tail binds through opaque binds only. Extents are whole
       * tiles or reach the level's edge. */
      if (level >= sparse->tail_level || layer >= plane->layout.layers ||
          bind->offset.x < 0 || bind->offset.y < 0 || bind->offset.z < 0 ||
          bind->offset.x % tile_px.x != 0 || bind->offset.y % tile_px.y != 0 ||
          bind->offset.z % tile_px.z != 0) {
         vk_loge(VK_LOG_OBJS(&image->vk.base),
                 "Sparse bind outside the image's tiles is dropped");
         continue;
      }

      const struct mtl_size level_tiles =
         kk_image_sparse_level_tiles(plane, level);
      const struct mtl_origin origin = {
         .x = bind->offset.x / tile_px.x,
         .y = bind->offset.y / tile_px.y,
         .z = bind->offset.z / tile_px.z,
      };
      if (origin.x >= level_tiles.x || origin.y >= level_tiles.y ||
          origin.z >= level_tiles.z)
         continue;

      const struct mtl_size size = {
         .x = MIN2(DIV_ROUND_UP(bind->extent.width, tile_px.x),
                   level_tiles.x - origin.x),
         .y = MIN2(DIV_ROUND_UP(bind->extent.height, tile_px.y),
                   level_tiles.y - origin.y),
         .z = MIN2(DIV_ROUND_UP(bind->extent.depth, tile_px.z),
                   level_tiles.z - origin.z),
      };
      if (size.x == 0 || size.y == 0 || size.z == 0)
         continue;

      uint64_t heap_tile;
      struct kk_bo *bo;
      if (!kk_sparse_bind_memory(bind->memory, bind->memoryOffset,
                                 size.x * size.y * size.z, &bo, &heap_tile)) {
         vk_loge(VK_LOG_OBJS(&image->vk.base),
                 "Sparse bind outside its memory is dropped");
         continue;
      }

      /* Row by row, so what is recorded is what is mapped */
      const uint64_t level_first =
         layer * sparse->layer_tiles + sparse->level_tile[level];
      for (uint32_t z = 0; z < size.z; z++) {
         for (uint32_t y = 0; y < size.y; y++) {
            const struct mtl_origin row = {
               origin.x,
               origin.y + y,
               origin.z + z,
            };
            const uint64_t tile =
               level_first +
               (row.z * level_tiles.y + row.y) * level_tiles.x + row.x;
            const uint64_t row_heap_tile =
               heap_tile + ((uint64_t)z * size.y + y) * size.x;

            if (!kk_sparse_tiles_bind(b->dev, &sparse->tiles, tile, size.x,
                                      bo, row_heap_tile)) {
               result = vk_error(b->dev, VK_ERROR_OUT_OF_HOST_MEMORY);
               continue;
            }

            kk_bind_texture_region(b, plane->mtl_handle, kk_bo_heap(bo), row,
                                   (struct mtl_size){size.x, 1u, 1u}, level,
                                   layer, row_heap_tile);
            kk_bind_shadow_row(b, sparse, bo != NULL, row, size.x, level,
                               layer);
         }
      }
   }

   return result;
}

VkResult
kk_queue_bind_sparse(struct kk_queue *queue, struct vk_queue_submit *submit)
{
   struct kk_device *dev = kk_queue_device(queue);
   struct kk_sparse_device *sparse = &dev->sparse;
   VkResult result = VK_SUCCESS;

   if (!sparse->enabled)
      return VK_SUCCESS;

   struct kk_bind_batch shadow = {
      .dev = dev,
      .queue = queue,
      .ops = UTIL_DYNARRAY_INIT,
      .tails = UTIL_DYNARRAY_INIT,
   };
   struct kk_bind_batch b = {
      .dev = dev,
      .queue = queue,
      .ops = UTIL_DYNARRAY_INIT,
      .tails = UTIL_DYNARRAY_INIT,
      .shadow = &shadow,
   };

   simple_mtx_lock(&sparse->lock);

   for (uint32_t i = 0; i < submit->buffer_bind_count; i++) {
      VkResult r = kk_bind_sparse_buffer(&b, &submit->buffer_binds[i]);
      if (result == VK_SUCCESS)
         result = r;
   }

   for (uint32_t i = 0; i < submit->image_opaque_bind_count; i++) {
      VkResult r =
         kk_bind_sparse_image_opaque(&b, &submit->image_opaque_binds[i]);
      if (result == VK_SUCCESS)
         result = r;
   }

   for (uint32_t i = 0; i < submit->image_bind_count; i++) {
      VkResult r = kk_bind_sparse_image(&b, &submit->image_binds[i]);
      if (result == VK_SUCCESS)
         result = r;
   }

   kk_bind_batch_end(&b);

   simple_mtx_unlock(&sparse->lock);

   return result;
}

/* Shaders write sparse residency buffers through storage descriptors, buffer
 * device addresses and storage texel buffers, and query residency through
 * texel buffers. Those get a mirror, a power of two of bytes after the start
 * of the Metal buffer, so a few bits of the address say where it is. 0 for
 * none.
 */
uint64_t
kk_sparse_buffer_mirror_tile(const struct kk_device *dev,
                             const struct kk_buffer *buffer)
{
   const VkBufferUsageFlags2KHR usage =
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
      VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT |
      VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT;

   if (!(buffer->vk.create_flags & VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT) ||
       !(buffer->vk.usage & usage) ||
       dev->sparse.ones[KK_SPARSE_ONES_UNORM] == NULL)
      return 0u;

   const uint64_t tiles = DIV_ROUND_UP(buffer->vk.size, KK_SPARSE_PAGE_SIZE_B);
   return util_next_power_of_two64(MAX2(tiles, 1u));
}

VkResult
kk_sparse_buffer_init(struct kk_device *dev, struct kk_buffer *buffer,
                      uint64_t mirror_tile)
{
   kk_sparse_tiles_init(&buffer->sparse.tiles);
   list_inithead(&buffer->sparse.views);
   buffer->sparse.mirror_tile = mirror_tile;
   buffer->sparse.addr_tag = 0u;

   if (mirror_tile == 0u)
      return VK_SUCCESS;

   /* The tag goes above every address of the buffer and its mirror */
   const uint64_t tiles = DIV_ROUND_UP(buffer->vk.size, KK_SPARSE_PAGE_SIZE_B);
   const uint64_t end = mtl_buffer_get_gpu_address(buffer->metal.handle) +
                        (mirror_tile + tiles) * KK_SPARSE_PAGE_SIZE_B;
   if (end > BITFIELD64_BIT(KK_SPARSE_TAG_SHIFT))
      return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                       "Sparse buffer address above the residency tag");

   const uint64_t tag = util_logbase2_64(mirror_tile) + 1u;
   assert(tag < BITFIELD_BIT(KK_SPARSE_TAG_BITS));
   buffer->sparse.addr_tag = tag << KK_SPARSE_TAG_SHIFT;

   return VK_SUCCESS;
}

void
kk_sparse_buffer_finish(struct kk_device *dev, struct kk_buffer *buffer)
{
   if (list_is_empty(&buffer->sparse.views))
      return;

   simple_mtx_lock(&dev->sparse.lock);

   /* A view can outlive its buffer, but not the buffer's tiles */
   if (dev->sparse.enabled && dev->has_queue) {
      struct kk_bind_batch b = {
         .dev = dev,
         .queue = &dev->queue,
         .ops = UTIL_DYNARRAY_INIT,
         .tails = UTIL_DYNARRAY_INIT,
      };
      list_for_each_entry(struct kk_buffer_view, view, &buffer->sparse.views,
                          sparse.link) {
         kk_bind_texture_region(
            &b, view->mtl_texel_buffer_handle, NULL,
            (struct mtl_origin){0u, 0u, 0u},
            (struct mtl_size){view->sparse.tile_count, 1u, 1u}, 0u, 0u, 0u);
      }
      kk_bind_batch_end(&b);
   }

   list_for_each_entry_safe(struct kk_buffer_view, view, &buffer->sparse.views,
                            sparse.link) {
      list_del(&view->sparse.link);
      view->sparse.buffer = NULL;
   }

   simple_mtx_unlock(&dev->sparse.lock);
}

void
kk_sparse_buffer_add_view(struct kk_device *dev, struct kk_buffer *buffer,
                          struct kk_buffer_view *view)
{
   simple_mtx_lock(&dev->sparse.lock);

   view->sparse.buffer = buffer;
   list_addtail(&view->sparse.link, &buffer->sparse.views);

   if (buffer->sparse.mirror_tile != 0u) {
      view->residency_addr =
         mtl_buffer_get_gpu_address(buffer->metal.handle) +
         (buffer->sparse.mirror_tile + view->sparse.first_tile) *
            KK_SPARSE_PAGE_SIZE_B;
      view->residency_tiles = view->sparse.tile_count;
   }

   /* Map the tiles the buffer has now. Binds after this map them too. */
   if (!rb_tree_is_empty(&buffer->sparse.tiles.ranges)) {
      struct kk_bind_batch b = {
         .dev = dev,
         .queue = &dev->queue,
         .ops = UTIL_DYNARRAY_INIT,
         .tails = UTIL_DYNARRAY_INIT,
      };

      const uint64_t first = view->sparse.first_tile;
      const uint64_t end = first + view->sparse.tile_count;
      for (struct kk_sparse_range *range =
              kk_sparse_tiles_find(&buffer->sparse.tiles, first);
           range != NULL && range->tile < end;
           range = kk_sparse_range(rb_node_next(&range->node))) {
         const uint64_t lo = MAX2(range->tile, first);
         const uint64_t hi = MIN2(range->tile + range->count, end);
         kk_bind_texture_region(&b, view->mtl_texel_buffer_handle,
                                range->bo->mtl_handle,
                                (struct mtl_origin){lo - first, 0u, 0u},
                                (struct mtl_size){hi - lo, 1u, 1u}, 0u, 0u,
                                range->heap_tile + (lo - range->tile));
      }

      kk_bind_batch_end(&b);
   }

   simple_mtx_unlock(&dev->sparse.lock);
}

void
kk_sparse_buffer_remove_view(struct kk_device *dev, struct kk_buffer_view *view)
{
   simple_mtx_lock(&dev->sparse.lock);

   if (view->sparse.buffer != NULL) {
      list_del(&view->sparse.link);
      view->sparse.buffer = NULL;
   }

   simple_mtx_unlock(&dev->sparse.lock);
}
