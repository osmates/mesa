/*
 * Copyright 2026 Ngo Quoc Dat
 * SPDX-License-Identifier: MIT
 */
#include "kk_private.h"

#include "kk_descriptor_set_layout.h"
#include "kk_descriptor_types.h"
#include "kk_sampler.h"
#include "kk_shader.h"

#include "nir.h"
#include "nir_builder.h"
#include "nir_builtin_builder.h"

#include "util/u_dynarray.h"

/* Metal reduces min/max only from Apple10 and only when every filter is
 * linear. Otherwise the shader takes the min or max of the texels in the
 * filter footprint itself: each one read at its texel centre through a
 * nearest sampler at an explicit level, so Metal still applies the address
 * modes and borders. A footprint is the whole 2x2 (1D: 2, 3D: 2x2x2) quad of
 * a linear filter, or one texel, on one level or on both levels of a linear
 * mipmap filter. On cube maps the texels outside the face are read in the
 * direction of their centres, which lands on the neighbouring face.
 */

struct lower_minmax_ctx {
   uint32_t set_layout_count;
   struct vk_descriptor_set_layout *const *set_layouts;
   bool view_min_lod;
};

static bool
is_immutable_average(nir_tex_instr *tex, const struct lower_minmax_ctx *ctx)
{
   nir_deref_instr *deref = nir_get_tex_deref(tex, nir_tex_src_sampler_deref);
   nir_variable *var = nir_deref_instr_get_variable(deref);
   if (var == NULL || var->data.descriptor_set >= ctx->set_layout_count ||
       ctx->set_layouts[var->data.descriptor_set] == NULL)
      return false;

   const struct kk_descriptor_set_layout *layout =
      vk_to_kk_descriptor_set_layout(
         ctx->set_layouts[var->data.descriptor_set]);
   if (var->data.binding >= layout->binding_count)
      return false;

   const struct kk_descriptor_set_binding_layout *binding =
      &layout->binding[var->data.binding];
   if (binding->immutable_samplers == NULL)
      return false;

   for (uint32_t i = 0; i < binding->array_size; i++) {
      const struct kk_sampler *sampler = binding->immutable_samplers[i];
      if (sampler == NULL || sampler->reduction)
         return false;
   }

   return true;
}

/* 1D sampling is 2D by now, so look at the type of the texture itself */
static bool
is_1d(nir_tex_instr *tex)
{
   nir_deref_instr *deref = nir_get_tex_deref(tex, nir_tex_src_texture_deref);
   return glsl_get_sampler_dim(glsl_without_array(deref->type)) ==
          GLSL_SAMPLER_DIM_1D;
}

static nir_tex_instr *
build_tex(nir_builder *b, nir_tex_instr *tex, nir_texop op,
          nir_alu_type dest_type, unsigned bit_size, nir_def *coord,
          nir_def *lod)
{
   bool sampler = op == nir_texop_txl || op == nir_texop_lod ||
                  op == nir_texop_sampler_reduction_kk;
   unsigned num_srcs = 1 + sampler + (coord != NULL) + (lod != NULL);

   nir_tex_instr *t = nir_tex_instr_create(b->shader, num_srcs);
   t->op = op;
   t->sampler_dim = tex->sampler_dim;
   t->is_array = tex->is_array;
   t->dest_type = dest_type;
   t->texture_non_uniform = tex->texture_non_uniform;
   t->sampler_non_uniform = tex->sampler_non_uniform;
   t->can_speculate = tex->can_speculate;

   unsigned s = 0;
   t->src[s++] =
      nir_tex_src_for_ssa(nir_tex_src_texture_deref,
                          nir_get_tex_src(tex, nir_tex_src_texture_deref));
   if (sampler) {
      t->src[s++] =
         nir_tex_src_for_ssa(nir_tex_src_sampler_deref,
                             nir_get_tex_src(tex, nir_tex_src_sampler_deref));
   }
   if (coord) {
      t->coord_components = coord->num_components;
      t->src[s++] = nir_tex_src_for_ssa(nir_tex_src_coord, coord);
   }
   if (lod)
      t->src[s++] = nir_tex_src_for_ssa(nir_tex_src_lod, lod);

   nir_def_init(&t->instr, &t->def, nir_tex_instr_dest_size(t), bit_size);
   nir_builder_instr_insert(b, &t->instr);
   return t;
}

static nir_def *
reduce(nir_builder *b, nir_def *is_min, nir_def *x, nir_def *y)
{
   return nir_bcsel(b, is_min, nir_fmin(b, x, y), nir_fmax(b, x, y));
}

struct footprint {
   nir_tex_instr *tex;
   nir_def *is_min;
   nir_def *linear;
   nir_def *unnormalized;
   nir_def *pos;
   nir_def *layer;
   nir_def *offset;
   unsigned dims;
   unsigned filtered_dims;
};

static nir_def *
read_texel(nir_builder *b, const struct footprint *fp, nir_def *pos,
           nir_def *level)
{
   nir_tex_instr *tex = fp->tex;
   nir_def *coord = pos;
   if (fp->layer)
      coord = nir_vector_insert_imm(b, nir_pad_vector(b, pos, fp->dims + 1),
                                    fp->layer, fp->dims);

   coord =
      nir_f2fN(b, coord, nir_get_tex_src(tex, nir_tex_src_coord)->bit_size);

   nir_tex_instr *t = build_tex(b, tex, nir_texop_txl, tex->dest_type,
                                tex->def.bit_size, coord, level);
   t->backend_flags = tex->backend_flags | KK_TEXTURE_FLAG_REDUCTION_TAP;
   return &t->def;
}

/* Integer coordinates of the first and last texel of the footprint on each
 * axis, from coordinates in texels.
 */
static void
footprint_bounds(nir_builder *b, const struct footprint *fp, nir_def *u,
                 nir_def **first, nir_def **last)
{
   *first = nir_bcsel(b, fp->linear, nir_ffloor(b, nir_fadd_imm(b, u, -0.5)),
                      nir_ffloor(b, u));
   *last = nir_fadd(b, *first, nir_b2f32(b, fp->linear));
}

static nir_def *
reduce_level(nir_builder *b, const struct footprint *fp, nir_def *level)
{
   nir_tex_instr *tex = fp->tex;
   bool cube = tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE;

   nir_def *size = &build_tex(b, tex, nir_texop_txs, nir_type_int32, 32, NULL,
                              nir_f2i32(b, level))
                       ->def;
   size = nir_i2f32(b, nir_trim_vector(b, size, cube ? 1 : fp->dims));

   nir_def *res = NULL;

   if (cube) {
      /* Face coordinates as Vulkan selects the face. Which way s and t run
       * on a face does not matter: the texel centres are the same.
       */
      nir_def *r = fp->pos;
      nir_def *a = nir_fabs(b, r);
      nir_def *ax = nir_channel(b, a, 0), *ay = nir_channel(b, a, 1),
              *az = nir_channel(b, a, 2);
      nir_def *x_major = nir_iand(b, nir_fge(b, ax, ay), nir_fge(b, ax, az));
      nir_def *y_major = nir_iand(b, nir_inot(b, x_major), nir_fge(b, ay, az));
      nir_def *z_major = nir_inot(b, nir_ior(b, x_major, y_major));

      nir_def *ma = nir_bcsel(
         b, x_major, nir_channel(b, r, 0),
         nir_bcsel(b, y_major, nir_channel(b, r, 1), nir_channel(b, r, 2)));
      nir_def *sc =
         nir_bcsel(b, x_major, nir_channel(b, r, 2), nir_channel(b, r, 0));
      nir_def *tc =
         nir_bcsel(b, y_major, nir_channel(b, r, 2), nir_channel(b, r, 1));
      nir_def *st = nir_fmul_imm(
         b,
         nir_fadd_imm(b, nir_fdiv(b, nir_vec2(b, sc, tc), nir_fabs(b, ma)),
                      1.0),
         0.5);
      nir_def *sign = nir_fsign(b, ma);

      nir_def *first, *last;
      footprint_bounds(b, fp, nir_fmul(b, st, size), &first, &last);

      /* Nearest filtering clamps to the face */
      first = nir_bcsel(b, fp->linear, first,
                        nir_fmin(b, first, nir_fadd_imm(b, size, -1.0)));

      for (unsigned i = 0; i < 4; i++) {
         nir_def *texel = nir_vec2(b, nir_channel(b, (i & 1) ? last : first, 0),
                                   nir_channel(b, (i & 2) ? last : first, 1));

         /* -1 to 1 across the face, beyond it for texels outside */
         nir_def *f = nir_fadd_imm(
            b,
            nir_fdiv(b, nir_fmul_imm(b, nir_fadd_imm(b, texel, 0.5), 2.0),
                     size),
            -1.0);
         nir_def *fs = nir_channel(b, f, 0), *ft = nir_channel(b, f, 1);
         nir_def *dir = nir_vec3(
            b, nir_bcsel(b, x_major, sign, fs), nir_bcsel(b, y_major, sign, ft),
            nir_bcsel(b, z_major, sign, nir_bcsel(b, x_major, fs, ft)));

         nir_def *v = read_texel(b, fp, dir, level);
         res = res ? reduce(b, fp->is_min, res, v) : v;
      }

      return res;
   }

   nir_def *scale =
      nir_bcsel(b, fp->unnormalized, nir_imm_floatN_t(b, 1.0, 32), size);
   nir_def *u = nir_fmul(b, fp->pos, scale);
   if (fp->offset)
      u =
         nir_fadd(b, u, nir_i2f32(b, nir_trim_vector(b, fp->offset, fp->dims)));

   nir_def *first, *last;
   footprint_bounds(b, fp, u, &first, &last);
   nir_def *first_pos = nir_fdiv(b, nir_fadd_imm(b, first, 0.5), scale);
   nir_def *last_pos = nir_fdiv(b, nir_fadd_imm(b, last, 0.5), scale);

   for (unsigned i = 0; i < (1u << fp->filtered_dims); i++) {
      nir_def *comps[3];
      for (unsigned c = 0; c < fp->dims; c++) {
         if (c >= fp->filtered_dims)
            comps[c] = nir_channel(b, fp->pos, c);
         else
            comps[c] = nir_channel(b, (i & (1 << c)) ? last_pos : first_pos, c);
      }

      nir_def *v = read_texel(b, fp, nir_vec(b, comps, fp->dims), level);
      res = res ? reduce(b, fp->is_min, res, v) : v;
   }

   return res;
}

static nir_def *
emulate(nir_builder *b, nir_tex_instr *tex, nir_def *params,
        const struct lower_minmax_ctx *ctx)
{
   nir_def *flags = nir_u2u32(b, nir_channel(b, params, 0));
   nir_def *bias = nir_f2f32(b, nir_channel(b, params, 1));
   nir_def *lod_min = nir_f2f32(b, nir_channel(b, params, 2));
   nir_def *lod_max = nir_f2f32(b, nir_channel(b, params, 3));

   const unsigned dims = tex->coord_components - tex->is_array;
   nir_def *coord = nir_f2f32(b, nir_get_tex_src(tex, nir_tex_src_coord));

   struct footprint fp = {
      .tex = tex,
      .is_min = nir_ieq_imm(b, nir_iand_imm(b, flags, KK_REDUCTION_MODE_MASK),
                            KK_REDUCTION_MODE_MIN),
      .unnormalized =
         nir_ine_imm(b, nir_iand_imm(b, flags, KK_REDUCTION_UNNORMALIZED), 0),
      .pos = nir_trim_vector(b, coord, dims),
      .layer = tex->is_array ? nir_channel(b, coord, dims) : NULL,
      .offset = nir_get_tex_src(tex, nir_tex_src_offset),
      .dims = dims,
      .filtered_dims = is_1d(tex) ? 1 : dims,
   };

   /* Vulkan's LOD: lambda_base, then the biases, then the clamps */
   nir_def *lambda;
   nir_def *lod = nir_get_tex_src(tex, nir_tex_src_lod);
   nir_def *ddx = nir_get_tex_src(tex, nir_tex_src_ddx);
   if (lod) {
      lambda = nir_f2f32(b, lod);
   } else if (ddx) {
      nir_def *ddy = nir_get_tex_src(tex, nir_tex_src_ddy);
      nir_def *size = &build_tex(b, tex, nir_texop_txs, nir_type_int32, 32,
                                 NULL, nir_imm_int(b, 0))
                          ->def;
      size = nir_i2f32(b, nir_trim_vector(b, size, dims));
      nir_def *rx = nir_fast_length(b, nir_fmul(b, nir_f2f32(b, ddx), size));
      nir_def *ry = nir_fast_length(b, nir_fmul(b, nir_f2f32(b, ddy), size));
      lambda = nir_flog2(b, nir_fmax(b, rx, ry));
   } else if (nir_shader_supports_implicit_lod(b->shader)) {
      /* Unclamped, unbiased: the y of OpImageQueryLod is lambda_base */
      nir_def *query =
         &build_tex(
             b, tex, nir_texop_lod, nir_type_float32, 32,
             nir_trim_vector(b, nir_get_tex_src(tex, nir_tex_src_coord), dims),
             NULL)
             ->def;
      lambda = nir_channel(b, query, 1);
   } else {
      lambda = nir_imm_float(b, 0.0);
   }

   lambda = nir_fadd(b, lambda, bias);
   nir_def *shader_bias = nir_get_tex_src(tex, nir_tex_src_bias);
   if (shader_bias)
      lambda = nir_fadd(b, lambda, nir_f2f32(b, shader_bias));

   nir_def *shader_min_lod = nir_get_tex_src(tex, nir_tex_src_min_lod);
   if (shader_min_lod)
      lod_min = nir_fmax(b, lod_min, nir_f2f32(b, shader_min_lod));
   lambda = nir_fmin(b, nir_fmax(b, lambda, lod_min), lod_max);

   nir_def *filter_bit = nir_bcsel(b, nir_fle_imm(b, lambda, 0.0),
                                   nir_imm_int(b, KK_REDUCTION_MAG_LINEAR),
                                   nir_imm_int(b, KK_REDUCTION_MIN_LINEAR));
   fp.linear = nir_ine_imm(b, nir_iand(b, flags, filter_bit), 0);

   /* Levels relative to the view's base level */
   nir_def *levels =
      &build_tex(b, tex, nir_texop_query_levels, nir_type_int32, 32, NULL, NULL)
          ->def;
   nir_def *q = nir_i2f32(b, nir_iadd_imm(b, levels, -1));
   nir_def *d = nir_fmin(b, nir_fmax(b, lambda, nir_imm_float(b, 0.0)), q);
   if (ctx->view_min_lod) {
      nir_def *view_min_lod = &build_tex(b, tex, nir_texop_image_min_lod_agx,
                                         nir_type_float16, 16, NULL, NULL)
                                  ->def;
      d = nir_fmax(b, d, nir_f2f32(b, view_min_lod));
   }

   nir_def *mip_linear =
      nir_ine_imm(b, nir_iand_imm(b, flags, KK_REDUCTION_MIP_LINEAR), 0);
   nir_def *d_hi =
      nir_bcsel(b, mip_linear, nir_ffloor(b, d),
                nir_fadd_imm(b, nir_fceil(b, nir_fadd_imm(b, d, 0.5)), -1.0));
   nir_def *d_lo = nir_fmin(b, nir_fadd_imm(b, d_hi, 1.0), q);
   nir_def *second = nir_iand(b, mip_linear, nir_fneu(b, d, d_hi));

   nir_def *res = reduce_level(b, &fp, d_hi);
   nir_def *both;
   nir_push_if(b, second);
   {
      both = reduce(b, fp.is_min, res, reduce_level(b, &fp, d_lo));
   }
   nir_pop_if(b, NULL);

   return nir_if_phi(b, both, res);
}

static bool
needs_lowering(nir_tex_instr *tex, const struct lower_minmax_ctx *ctx)
{
   /* Residency samplers average */
   if (tex->backend_flags & KK_TEXTURE_FLAG_RESIDENCY)
      return false;

   switch (tex->op) {
   case nir_texop_tex:
   case nir_texop_txb:
   case nir_texop_txl:
   case nir_texop_txd:
      break;
   default:
      return false;
   }

   switch (tex->sampler_dim) {
   case GLSL_SAMPLER_DIM_1D:
   case GLSL_SAMPLER_DIM_2D:
   case GLSL_SAMPLER_DIM_3D:
   case GLSL_SAMPLER_DIM_CUBE:
      break;
   default:
      return false;
   }

   /* Comparison, integer formats and Y'CbCr take no min/max reduction */
   return !tex->is_shadow &&
          nir_alu_type_get_base_type(tex->dest_type) == nir_type_float &&
          nir_tex_instr_src_index(tex, nir_tex_src_plane) < 0 &&
          nir_get_tex_deref(tex, nir_tex_src_sampler_deref) != NULL &&
          !is_immutable_average(tex, ctx);
}

static void
lower(nir_builder *b, nir_tex_instr *tex, const struct lower_minmax_ctx *ctx)
{
   b->cursor = nir_after_instr(&tex->instr);

   nir_instr *hw_instr = nir_instr_clone(b->shader, &tex->instr);
   nir_builder_instr_insert(b, hw_instr);
   nir_def *hw = &nir_instr_as_tex(hw_instr)->def;

   nir_def *params = &build_tex(b, tex, nir_texop_sampler_reduction_kk,
                                nir_type_uint16, 16, NULL, NULL)
                         ->def;
   nir_def *mode = nir_iand_imm(b, nir_u2u32(b, nir_channel(b, params, 0)),
                                KK_REDUCTION_MODE_MASK);

   nir_def *reduced;
   nir_push_if(b, nir_ine_imm(b, mode, 0));
   {
      reduced = emulate(b, tex, params, ctx);
   }
   nir_pop_if(b, NULL);

   nir_def_replace(&tex->def, nir_if_phi(b, reduced, hw));
}

bool
kk_nir_lower_sampler_minmax(nir_shader *nir, uint32_t set_layout_count,
                            struct vk_descriptor_set_layout *const *set_layouts,
                            bool view_min_lod)
{
   const struct lower_minmax_ctx ctx = {
      .set_layout_count = set_layout_count,
      .set_layouts = set_layouts,
      .view_min_lod = view_min_lod,
   };
   bool progress = false;

   nir_foreach_function_impl(impl, nir) {
      struct util_dynarray texs;
      util_dynarray_init(&texs, NULL);

      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;

            nir_tex_instr *tex = nir_instr_as_tex(instr);
            if (needs_lowering(tex, &ctx))
               util_dynarray_append(&texs, tex);
         }
      }

      nir_builder b = nir_builder_create(impl);
      util_dynarray_foreach(&texs, nir_tex_instr *, tex)
         lower(&b, *tex, &ctx);

      progress |= nir_progress(texs.size > 0, impl, nir_metadata_none);
      util_dynarray_fini(&texs);
   }

   return progress;
}
