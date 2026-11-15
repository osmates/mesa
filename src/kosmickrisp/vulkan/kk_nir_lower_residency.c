/*
 * Copyright 2026 Ngo Quoc Dat
 * SPDX-License-Identifier: MIT
 */
#include "kk_shader.h"

#include "nir.h"
#include "nir_builder.h"

/* A texture instruction with residency returns its texels as one without
 * does, plus a residency code from the same instruction on the image's
 * residency shadow: 1 when every texel it read was resident, else 0.
 * Descriptor lowering points these shadow reads at the shadow, and answers
 * resident when there is none. Image loads with residency are split there
 * too.
 */

static bool
has_shadow(const nir_tex_instr *tex)
{
   switch (tex->sampler_dim) {
   case GLSL_SAMPLER_DIM_2D:
   case GLSL_SAMPLER_DIM_CUBE:
   case GLSL_SAMPLER_DIM_BUF:
      return true;
   default:
      /* No sparse residency for 1D, 3D or multisampled images */
      return false;
   }
}

static nir_tex_instr *
copy_tex(nir_builder *b, nir_tex_instr *tex, unsigned num_components,
         unsigned bit_size)
{
   nir_tex_instr *copy =
      nir_instr_as_tex(nir_instr_clone(b->shader, &tex->instr));
   copy->is_sparse = false;
   copy->def.num_components = num_components;
   copy->def.bit_size = bit_size;
   return copy;
}

static nir_def *
shadow_resident(nir_builder *b, nir_tex_instr *tex)
{
   if (!has_shadow(tex))
      return nir_imm_true(b);

   /* Typed as the texture until descriptor lowering points it at the shadow,
    * which reads float */
   nir_tex_instr *shadow = copy_tex(b, tex, 4, 32);
   shadow->dest_type = nir_alu_type_get_base_type(tex->dest_type) | 32;
   shadow->is_shadow = false;
   shadow->is_new_style_shadow = false;
   shadow->backend_flags |= KK_TEXTURE_FLAG_RESIDENCY;
   if (tex->op == nir_texop_tg4)
      shadow->component = 0;

   /* Sources join their uses only once the clone is inserted */
   nir_builder_instr_insert(b, &shadow->instr);

   int comparator = nir_tex_instr_src_index(shadow, nir_tex_src_comparator);
   if (comparator >= 0)
      nir_tex_instr_remove_src(shadow, comparator);

   nir_def *value = &shadow->def;
   if (tex->op == nir_texop_tg4) {
      value = nir_fmin(
         b, nir_fmin(b, nir_channel(b, value, 0), nir_channel(b, value, 1)),
         nir_fmin(b, nir_channel(b, value, 2), nir_channel(b, value, 3)));
   } else {
      value = nir_channel(b, value, 0);
   }

   return nir_feq_imm(b, value, 1.0);
}

static bool
lower_tex(nir_builder *b, nir_tex_instr *tex)
{
   if (!tex->is_sparse)
      return false;

   b->cursor = nir_before_instr(&tex->instr);

   const unsigned n = nir_tex_instr_result_size(tex);
   nir_tex_instr *texels = copy_tex(b, tex, n, tex->def.bit_size);
   nir_builder_instr_insert(b, &texels->instr);

   nir_def *code =
      nir_u2uN(b, nir_b2i32(b, shadow_resident(b, tex)), tex->def.bit_size);

   nir_def *comps[NIR_MAX_VEC_COMPONENTS];
   for (unsigned i = 0; i < n; i++)
      comps[i] = nir_channel(b, &texels->def, i);
   comps[n] = code;

   nir_def_replace(&tex->def, nir_vec(b, comps, n + 1));
   return true;
}

static bool
lower_intrinsic(nir_builder *b, nir_intrinsic_instr *intr)
{
   b->cursor = nir_before_instr(&intr->instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_is_sparse_texels_resident:
      nir_def_replace(&intr->def, nir_ine_imm(b, intr->src[0].ssa, 0));
      return true;
   case nir_intrinsic_sparse_residency_code_and:
      nir_def_replace(&intr->def,
                      nir_iand(b, intr->src[0].ssa, intr->src[1].ssa));
      return true;
   default:
      return false;
   }
}

static bool
lower(nir_builder *b, nir_instr *instr, UNUSED void *data)
{
   switch (instr->type) {
   case nir_instr_type_tex:
      return lower_tex(b, nir_instr_as_tex(instr));
   case nir_instr_type_intrinsic:
      return lower_intrinsic(b, nir_instr_as_intrinsic(instr));
   default:
      return false;
   }
}

/* Metal keeps a write to an unmapped tile of a sparse resource in its caches,
 * where the writing thread and other reads before a resource alias barrier
 * see it. So writes and atomics through an address with a residency tag
 * happen only where the buffer's residency mirror reads non-zero, atomics
 * elsewhere return 0, and every access drops the tag. Runs on the accesses of
 * storage buffers and buffer device addresses, before anything else makes
 * global accesses.
 */
static nir_def *
address_tag(nir_builder *b, nir_def *addr)
{
   return nir_ubitfield_extract_imm(b, nir_unpack_64_2x32_split_y(b, addr),
                                    KK_SPARSE_TAG_SHIFT - 32u,
                                    KK_SPARSE_TAG_BITS);
}

static nir_def *
untag_address(nir_builder *b, nir_def *addr)
{
   return nir_pack_64_2x32_split(
      b, nir_unpack_64_2x32_split_x(b, addr),
      nir_iand_imm(b, nir_unpack_64_2x32_split_y(b, addr),
                   BITFIELD_MASK(KK_SPARSE_TAG_SHIFT - 32u)));
}

/* The mirror is 2^(tag + 15) bytes past the buffer */
static nir_def *
mirror_resident(nir_builder *b, nir_def *addr, nir_def *tag)
{
   nir_def *distance =
      nir_ishl(b, nir_imm_int64(b, 1), nir_iadd_imm(b, tag, 15));
   nir_def *mirror = nir_iadd(b, nir_iand_imm(b, addr, ~3ull), distance);
   return nir_ine_imm(b, nir_load_global_constant(b, 1, 32, mirror), 0);
}

/* A vector store across a tile boundary, whose tiles can differ in
 * residency, goes component by component */
static void
store_components(nir_builder *b, nir_intrinsic_instr *store, nir_def *addr,
                 nir_def *tag)
{
   nir_def *value = store->src[0].ssa;
   const unsigned comp_B = value->bit_size / 8u;

   u_foreach_bit(c, nir_intrinsic_write_mask(store)) {
      nir_def *comp_addr = nir_iadd_imm(b, addr, c * comp_B);
      nir_push_if(b, mirror_resident(b, comp_addr, tag));
      nir_store_global(b, nir_channel(b, value, c), comp_addr,
                       .align_mul = comp_B,
                       .access = nir_intrinsic_access(store));
      nir_pop_if(b, NULL);
   }
}

static nir_intrinsic_instr *
clone_with_address(nir_builder *b, nir_intrinsic_instr *intr, unsigned addr_src,
                   nir_def *addr)
{
   nir_intrinsic_instr *clone =
      nir_instr_as_intrinsic(nir_instr_clone(b->shader, &intr->instr));
   nir_builder_instr_insert(b, &clone->instr);
   nir_src_rewrite(&clone->src[addr_src], addr);
   return clone;
}

static bool
lower_global_access(nir_builder *b, nir_intrinsic_instr *intr,
                    UNUSED void *data)
{
   unsigned addr_src;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_global:
   case nir_intrinsic_load_global_constant:
   case nir_intrinsic_global_atomic:
   case nir_intrinsic_global_atomic_swap:
      addr_src = 0;
      break;
   case nir_intrinsic_store_global:
      addr_src = 1;
      break;
   default:
      return false;
   }

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *tagged = intr->src[addr_src].ssa;

   if (intr->intrinsic == nir_intrinsic_load_global ||
       intr->intrinsic == nir_intrinsic_load_global_constant) {
      nir_src_rewrite(&intr->src[addr_src], untag_address(b, tagged));
      return true;
   }

   /* Plain memory keeps the write as it is, behind one branch */
   const bool has_dest = nir_intrinsic_infos[intr->intrinsic].has_dest;
   nir_def *zero = has_dest ? nir_imm_zero(b, 1, intr->def.bit_size) : NULL;
   nir_def *tag = address_tag(b, tagged);
   nir_def *sparse_def = NULL;
   nir_push_if(b, nir_ine_imm(b, tag, 0));
   nir_def *addr = untag_address(b, tagged);
   if (intr->intrinsic == nir_intrinsic_store_global &&
       intr->num_components > 1u) {
      const unsigned size_B =
         intr->num_components * intr->src[0].ssa->bit_size / 8u;
      nir_def *last = nir_iadd_imm(b, addr, size_B - 1u);
      nir_push_if(
         b, nir_ine(b, nir_ushr_imm(b, addr, 16), nir_ushr_imm(b, last, 16)));
      store_components(b, intr, addr, tag);
      nir_push_else(b, NULL);
      nir_push_if(b, mirror_resident(b, addr, tag));
      clone_with_address(b, intr, addr_src, addr);
      nir_pop_if(b, NULL);
      nir_pop_if(b, NULL);
   } else {
      nir_push_if(b, mirror_resident(b, addr, tag));
      nir_intrinsic_instr *clone = clone_with_address(b, intr, addr_src, addr);
      nir_pop_if(b, NULL);
      if (has_dest)
         sparse_def = nir_if_phi(b, &clone->def, zero);
   }
   nir_push_else(b, NULL);
   nir_instr_remove(&intr->instr);
   nir_builder_instr_insert(b, &intr->instr);
   nir_pop_if(b, NULL);

   if (has_dest) {
      nir_def *phi = nir_if_phi(b, sparse_def, &intr->def);
      nir_def_rewrite_uses(&intr->def, phi);
      nir_src_rewrite(
         &nir_phi_get_src_from_block(nir_instr_as_phi(nir_def_instr(phi)),
                                     intr->instr.block)
             ->src,
         &intr->def);
   }

   return true;
}

bool
kk_nir_lower_sparse_buffer_access(nir_shader *nir)
{
   return nir_shader_intrinsics_pass(nir, lower_global_access,
                                     nir_metadata_none, NULL);
}

bool
kk_nir_lower_residency(nir_shader *nir)
{
   bool progress =
      nir_shader_instructions_pass(nir, lower, nir_metadata_control_flow, NULL);

   /* Texels and codes are read out of the vectors built above, which have
    * more components than MSL vectors */
   if (progress) {
      NIR_PASS(_, nir, nir_opt_copy_prop);
      NIR_PASS(_, nir, nir_opt_dce);
   }

   return progress;
}
