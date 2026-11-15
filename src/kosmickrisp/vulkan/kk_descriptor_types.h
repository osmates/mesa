/*
 * Copyright © 2024 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */
#ifndef KK_DESCRIPTOR_TYPES
#define KK_DESCRIPTOR_TYPES 1

#include "kk_private.h"

/* kk_sampled_image_descriptor::reduction, non-zero when the shader emulates
 * min/max reduction for the sampler
 */
#define KK_REDUCTION_MODE_MASK    0x3
#define KK_REDUCTION_MODE_MIN     0x1
#define KK_REDUCTION_MODE_MAX     0x2
#define KK_REDUCTION_MAG_LINEAR   (1 << 2)
#define KK_REDUCTION_MIN_LINEAR   (1 << 3)
#define KK_REDUCTION_MIP_LINEAR   (1 << 4)
#define KK_REDUCTION_UNNORMALIZED (1 << 5)

/* TODO_KOSMICKRISP Reduce size to 32 bytes by moving border to a heap. */
struct kk_sampled_image_descriptor {
   uint64_t image_gpu_resource_id;
   uint16_t sampler_index;

   /* Negative if there is no border color, else the clamp=0 sampler index used
    * for custom border color emulation.
    */
   int16_t clamp_0_sampler_index_or_negative;

   uint16_t sampler_lod_bias_fp16;
   uint16_t image_min_lod_fp16;

   /* Loaded together by nir_texop_sampler_reduction_kk. The bias is the part
    * of the sampler LOD bias the shader has not applied yet.
    */
   uint16_t reduction;
   uint16_t reduction_lod_bias_fp16;
   uint16_t sampler_lod_min_fp16;
   uint16_t sampler_lod_max_fp16;

   uint16_t image_min_lod_uint16;

   /* Nearest samplers for the min/max emulation, clamp=1 and clamp=0 */
   uint16_t reduction_sampler_index;
   int16_t reduction_clamp_0_sampler_index_or_negative;

   /* The sampler loads with residency sample the residency shadow with:
    * white borders, no comparison */
   uint16_t residency_sampler_index;

   uint32_t border[4];

   /* The view of the image's residency shadow, 0 when the image has none and
    * is resident */
   uint64_t residency_gpu_resource_id;
   uint64_t pad_to_power_2;
};

static_assert(sizeof(struct kk_sampled_image_descriptor) == 64,
              "kk_sampled_image_descriptor has no holes");
static_assert(offsetof(struct kk_sampled_image_descriptor, reduction) % 8 ==
                    0 &&
                 offsetof(struct kk_sampled_image_descriptor,
                          sampler_lod_max_fp16) ==
                    offsetof(struct kk_sampled_image_descriptor, reduction) + 6,
              "reduction state is one aligned 4x16-bit load");

struct kk_storage_image_descriptor {
   uint64_t image_gpu_resource_id;

   /* As in kk_sampled_image_descriptor */
   uint64_t residency_gpu_resource_id;
};

static_assert(sizeof(struct kk_storage_image_descriptor) == 16,
              "kk_storage_image_descriptor has no holes");

struct kk_texel_buffer_descriptor {
   uint64_t image_gpu_resource_id;

   /* Texels between the aligned start of the Metal texture and the view's
    * first texel, added to every coordinate in the shader.
    */
   uint32_t offset_el;

   /* For loads with residency, stores and atomics: the texel size, and where
    * the buffer's residency mirror holds the Metal texture's 64 KB tiles and
    * how many. A zero address is resident. */
   uint32_t texel_size_B;
   uint64_t residency_addr;
   uint32_t residency_tiles;
   uint32_t pad;
};

static_assert(sizeof(struct kk_texel_buffer_descriptor) == 32,
              "kk_texel_buffer_descriptor has no holes");

/* Texel buffers are read through both the sampled and the storage image
 * paths, which load the resource ID from the start of the descriptor.
 */
static_assert(offsetof(struct kk_texel_buffer_descriptor,
                       image_gpu_resource_id) ==
                 offsetof(struct kk_sampled_image_descriptor,
                          image_gpu_resource_id),
              "texel buffer and sampled image resource IDs match");
static_assert(offsetof(struct kk_texel_buffer_descriptor,
                       image_gpu_resource_id) ==
                 offsetof(struct kk_storage_image_descriptor,
                          image_gpu_resource_id),
              "texel buffer and storage image resource IDs match");

/* This has to match nir_address_format_64bit_bounded_global */
struct kk_buffer_address {
   uint64_t base_addr;
   uint32_t size;
   uint32_t zero; /* Must be zero! */
};

static_assert(sizeof(struct kk_buffer_address) == 16,
              "kk_buffer_address has no holes");

#endif /* KK_DESCRIPTOR_TYPES */
