/*
 * Copyright © 2024 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */
#ifndef KK_DESCRIPTOR_TYPES
#define KK_DESCRIPTOR_TYPES 1

#include "kk_private.h"

/* TODO_KOSMICKRISP Reduce size to 32 bytes by moving border to a heap. */
struct kk_sampled_image_descriptor {
   uint64_t image_gpu_resource_id;
   uint16_t sampler_index;

   /* Negative if there is no border color, else the clamp=0 sampler index used
    * for custom border color emulation.
    */
   int16_t clamp_0_sampler_index_or_negative;

   uint16_t sampler_lod_bias_fp16;
   uint16_t sampler_lod_min_fp16;
   uint16_t sampler_lod_max_fp16;

   uint16_t image_min_lod_fp16;
   uint16_t image_min_lod_uint16;

   uint16_t pad_to_64_bits;
   uint32_t border[4];
   uint64_t pad_to_power_2[3];
};

static_assert(sizeof(struct kk_sampled_image_descriptor) == 64,
              "kk_sampled_image_descriptor has no holes");

struct kk_storage_image_descriptor {
   uint64_t image_gpu_resource_id;
};

static_assert(sizeof(struct kk_storage_image_descriptor) == 8,
              "kk_storage_image_descriptor has no holes");

struct kk_texel_buffer_descriptor {
   uint64_t image_gpu_resource_id;

   /* Texels between the aligned start of the Metal texture and the view's
    * first texel, added to every coordinate in the shader.
    */
   uint32_t offset_el;
   uint32_t pad;
};

static_assert(sizeof(struct kk_texel_buffer_descriptor) == 16,
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
