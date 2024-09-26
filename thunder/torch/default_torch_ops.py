import torch

torch_auto_registered_ops = {
    torch: [
        torch.lobpcg,
        torch.unravel_index,
        torch.absolute,
        torch.adaptive_avg_pool1d,
        torch.adaptive_max_pool1d,
        torch.addbmm,
        torch.addmm,
        torch.addmv,
        torch.addr,
        torch.adjoint,
        torch.affine_grid_generator,
        torch.alpha_dropout,
        torch.aminmax,
        torch.angle,
        torch.arccos,
        torch.arccosh,
        torch.arcsin,
        torch.arcsinh,
        torch.arctan,
        torch.arctan2,
        torch.arctanh,
        torch.argsort,
        torch.argwhere,
        torch.atleast_1d,
        torch.atleast_2d,
        torch.atleast_3d,
        torch.batch_norm_backward_elemt,
        torch.batch_norm_backward_reduce,
        torch.batch_norm_elemt,
        torch.batch_norm_gather_stats,
        torch.batch_norm_gather_stats_with_counts,
        torch.batch_norm_stats,
        torch.batch_norm_update_stats,
        torch.bilinear,
        torch.binary_cross_entropy_with_logits,
        torch.bincount,
        torch.binomial,
        torch.bitwise_left_shift,
        torch.bitwise_right_shift,
        torch.block_diag,
        torch.broadcast_tensors,
        torch.broadcast_to,
        torch.bucketize,
        torch.cartesian_prod,
        torch.cdist,
        torch.celu,
        torch.chain_matmul,
        torch.channel_shuffle,
        torch.cholesky,
        torch.cholesky_inverse,
        torch.cholesky_solve,
        torch.choose_qparams_optimized,
        torch.clamp_max,
        torch.clamp_min,
        torch.clip,
        torch.column_stack,
        torch.combinations,
        torch.complex,
        torch.concat,
        torch.concatenate,
        torch.conj,
        torch.conj_physical,
        torch.constant_pad_nd,
        torch.conv_tbc,
        torch.conv_transpose1d,
        torch.conv_transpose2d,
        torch.conv_transpose3d,
        torch.corrcoef,
        torch.cosine_embedding_loss,
        torch.cosine_similarity,
        torch.count_nonzero,
        torch.cov,
        torch.cross,
        torch.crow_indices_copy,
        torch.ctc_loss,
        torch.cummax,
        torch.cummin,
        torch.cumprod,
        torch.cumulative_trapezoid,
        torch.deg2rad,
        torch.det,
        torch.detach,
        torch.detach_copy,
        torch.diag,
        torch.diag_embed,
        torch.diagflat,
        torch.diagonal_copy,
        torch.diagonal_scatter,
        torch.diff,
        torch.dist,
        torch.divide,
        torch.dot,
        torch.dropout,
        torch.dsmm,
        torch.dsplit,
        torch.dstack,
        torch.embedding,
        torch.embedding_bag,
        torch.empty_like,
        torch.equal,
        torch.expand_copy,
        torch.fake_quantize_per_channel_affine,
        torch.fake_quantize_per_tensor_affine,
        torch.fbgemm_linear_fp16_weight,
        torch.fbgemm_linear_fp16_weight_fp32_activation,
        torch.fbgemm_linear_int8_weight,
        torch.fbgemm_linear_int8_weight_fp32_activation,
        torch.fbgemm_linear_quantize_weight,
        torch.fbgemm_pack_gemm_matrix_fp16,
        torch.fbgemm_pack_quantized_matrix,
        torch.feature_alpha_dropout,
        torch.feature_dropout,
        torch.fix,
        torch.fliplr,
        torch.flipud,
        torch.float_power,
        torch.fmax,
        torch.fmin,
        torch.frac,
        torch.frexp,
        torch.frobenius_norm,
        torch.fused_moving_avg_obs_fake_quant,
        torch.gcd,
        torch.geqrf,
        torch.ger,
        torch.greater,
        torch.greater_equal,
        torch.grid_sampler,
        torch.grid_sampler_2d,
        torch.grid_sampler_3d,
        torch.gru,
        torch.gru_cell,
        torch.hardshrink,
        torch.heaviside,
        torch.hinge_embedding_loss,
        torch.histc,
        torch.histogram,
        torch.hsmm,
        torch.hsplit,
        torch.hspmm,
        torch.hstack,
        torch.hypot,
        torch.i0,
        torch.igamma,
        torch.igammac,
        torch.imag,
        torch.index_fill,
        torch.index_reduce,
        torch.indices_copy,
        torch.inner,
        torch.instance_norm,
        torch.int_repr,
        torch.inverse,
        torch.is_conj,
        torch.is_distributed,
        torch.is_inference,
        torch.is_neg,
        torch.is_nonzero,
        torch.is_same_size,
        torch.is_signed,
        torch.isclose,
        torch.isin,
        torch.isinf,
        torch.isnan,
        torch.isneginf,
        torch.isposinf,
        torch.isreal,
        torch.istft,
        torch.kl_div,
        torch.kron,
        torch.kthvalue,
        torch.lcm,
        torch.ldexp,
        torch.less,
        torch.less_equal,
        torch.logaddexp,
        torch.logaddexp2,
        torch.logcumsumexp,
        torch.logdet,
        torch.logical_or,
        torch.logical_xor,
        torch.logit,
        torch.lstm,
        torch.lstm_cell,
        torch.lu_solve,
        torch.lu_unpack,
        torch.margin_ranking_loss,
        torch.masked_scatter,
        torch.masked_select,
        torch.matrix_exp,
        torch.matrix_power,
        torch.max_pool1d_with_indices,
        torch.median,
        torch.meshgrid,
        torch.min,
        torch.mm,
        torch.mode,
        torch.moveaxis,
        torch.msort,
        torch.multiply,
        torch.mv,
        torch.mvlgamma,
        torch.nanmean,
        torch.nanmedian,
        torch.nanquantile,
        torch.nansum,
        torch.narrow,
        torch.narrow_copy,
        torch.native_batch_norm,
        torch.native_channel_shuffle,
        torch.native_dropout,
        torch.native_group_norm,
        torch.native_layer_norm,
        torch.native_norm,
        torch.negative,
        torch.nonzero,
        torch.nonzero_static,
        torch.norm,
        torch.norm_except_dim,
        torch.not_equal,
        torch.nuclear_norm,
        torch.orgqr,
        torch.ormqr,
        torch.pairwise_distance,
        torch.pdist,
        torch.permute_copy,
        torch.pinverse,
        torch.pixel_shuffle,
        torch.pixel_unshuffle,
        torch.poisson,
        torch.poisson_nll_loss,
        torch.polar,
        torch.positive,
        torch.prelu,
        torch.put,
        torch.qr,
        torch.quantile,
        torch.rad2deg,
        torch.rand_like,
        torch.randint_like,
        torch.ravel,
        torch.renorm,
        torch.repeat_interleave,
        torch.resolve_conj,
        torch.resolve_neg,
        # torch.rms_norm, # only in torch>=2.4
        torch.rnn_relu,
        torch.rnn_relu_cell,
        torch.rnn_tanh,
        torch.rnn_tanh_cell,
        torch.roll,
        torch.rot90,
        torch.row_indices_copy,
        torch.row_stack,
        torch.rrelu,
        torch.rsub,
        torch.saddmm,
        torch.scatter_reduce,
        torch.searchsorted,
        torch.select_copy,
        torch.select_scatter,
        torch.sgn,
        torch.sinc,
        torch.slice_copy,
        torch.slice_inverse,
        torch.slice_scatter,
        torch.slogdet,
        torch.smm,
        torch.split_copy,
        torch.split_with_sizes,
        torch.split_with_sizes_copy,
        torch.spmm,
        torch.square,
        torch.squeeze_copy,
        torch.sspaddmm,
        torch.std,
        torch.std_mean,
        torch.stft,
        torch.subtract,
        torch.svd,
        torch.swapaxes,
        torch.swapdims,
        torch.t_copy,
        torch.take,
        torch.tensordot,
        torch.threshold,
        torch.tile,
        torch.trace,
        torch.transpose_copy,
        torch.trapezoid,
        torch.trapz,
        torch.triangular_solve,
        torch.triplet_margin_loss,
        torch.triu,
        torch.unbind_copy,
        torch.unfold_copy,
        torch.unique_consecutive,
        torch.unsafe_chunk,
        torch.unsafe_split,
        torch.unsafe_split_with_sizes,
        torch.unsqueeze_copy,
        torch.values_copy,
        torch.vdot,
        torch.view_as_complex,
        torch.view_as_complex_copy,
        torch.view_as_real,
        torch.view_as_real_copy,
        torch.view_copy,
        torch.vsplit,
        torch.vstack,
        torch.xlogy,
    ],
    torch.nn.functional: [
        torch.nn.functional.adaptive_avg_pool1d,
        torch.nn.functional.adaptive_avg_pool3d,
        torch.nn.functional.adaptive_max_pool1d,
        torch.nn.functional.adaptive_max_pool1d_with_indices,
        torch.nn.functional.adaptive_max_pool2d,
        torch.nn.functional.adaptive_max_pool2d_with_indices,
        torch.nn.functional.adaptive_max_pool3d,
        torch.nn.functional.adaptive_max_pool3d_with_indices,
        torch.nn.functional.affine_grid,
        torch.nn.functional.alpha_dropout,
        torch.nn.functional.bilinear,
        torch.nn.functional.binary_cross_entropy,
        torch.nn.functional.binary_cross_entropy_with_logits,
        torch.nn.functional.celu,
        torch.nn.functional.channel_shuffle,
        torch.nn.functional.conv_tbc,
        torch.nn.functional.conv_transpose1d,
        torch.nn.functional.conv_transpose2d,
        torch.nn.functional.conv_transpose3d,
        torch.nn.functional.cosine_embedding_loss,
        torch.nn.functional.cosine_similarity,
        torch.nn.functional.ctc_loss,
        torch.nn.functional.dropout1d,
        torch.nn.functional.dropout2d,
        torch.nn.functional.dropout3d,
        torch.nn.functional.elu,
        torch.nn.functional.embedding_bag,
        torch.nn.functional.feature_alpha_dropout,
        torch.nn.functional.fold,
        torch.nn.functional.fractional_max_pool2d,
        torch.nn.functional.fractional_max_pool2d_with_indices,
        torch.nn.functional.fractional_max_pool3d,
        torch.nn.functional.fractional_max_pool3d_with_indices,
        torch.nn.functional.gaussian_nll_loss,
        torch.nn.functional.glu,
        torch.nn.functional.grid_sample,
        torch.nn.functional.gumbel_softmax,
        torch.nn.functional.hardshrink,
        torch.nn.functional.hardtanh,
        torch.nn.functional.hinge_embedding_loss,
        torch.nn.functional.huber_loss,
        torch.nn.functional.instance_norm,
        torch.nn.functional.kl_div,
        torch.nn.functional.l1_loss,
        torch.nn.functional.leaky_relu,
        torch.nn.functional.local_response_norm,
        torch.nn.functional.logsigmoid,
        torch.nn.functional.lp_pool1d,
        torch.nn.functional.lp_pool2d,
        torch.nn.functional.lp_pool3d,
        torch.nn.functional.margin_ranking_loss,
        torch.nn.functional.max_pool1d_with_indices,
        torch.nn.functional.max_pool2d_with_indices,
        torch.nn.functional.max_pool3d_with_indices,
        torch.nn.functional.max_unpool1d,
        torch.nn.functional.max_unpool2d,
        torch.nn.functional.max_unpool3d,
        torch.nn.functional.mish,
        torch.nn.functional.multi_head_attention_forward,
        torch.nn.functional.multi_margin_loss,
        torch.nn.functional.multilabel_margin_loss,
        torch.nn.functional.multilabel_soft_margin_loss,
        torch.nn.functional.native_channel_shuffle,
        torch.nn.functional.pairwise_distance,
        torch.nn.functional.pdist,
        torch.nn.functional.pixel_shuffle,
        torch.nn.functional.pixel_unshuffle,
        torch.nn.functional.poisson_nll_loss,
        torch.nn.functional.prelu,
        # torch.nn.functional.rms_norm,
        torch.nn.functional.rrelu,
        torch.nn.functional.smooth_l1_loss,
        torch.nn.functional.soft_margin_loss,
        torch.nn.functional.softmin,
        torch.nn.functional.softplus,
        torch.nn.functional.softshrink,
        torch.nn.functional.softsign,
        torch.nn.functional.tanhshrink,
        torch.nn.functional.triplet_margin_loss,
        torch.nn.functional.triplet_margin_with_distance_loss,
        torch.nn.functional.unfold,
    ],
    torch.Tensor: [
        torch.Tensor.absolute,
        torch.Tensor.addbmm,
        torch.Tensor.addmm,
        torch.Tensor.addmv,
        torch.Tensor.addr,
        torch.Tensor.adjoint,
        torch.Tensor.align_as,
        torch.Tensor.align_to,
        # torch.Tensor.all,
        torch.Tensor.aminmax,
        torch.Tensor.angle,
        # torch.Tensor.any,
        torch.Tensor.arccos,
        torch.Tensor.arccosh,
        torch.Tensor.arcsin,
        torch.Tensor.arcsinh,
        torch.Tensor.arctan,
        torch.Tensor.arctan2,
        torch.Tensor.arctanh,
        torch.Tensor.argsort,
        torch.Tensor.argwhere,
        torch.Tensor.as_strided,
        torch.Tensor.as_strided_scatter,
        torch.Tensor.bfloat16,
        torch.Tensor.bincount,
        torch.Tensor.bitwise_left_shift,
        torch.Tensor.bitwise_right_shift,
        torch.Tensor.bool,
        torch.Tensor.broadcast_to,
        torch.Tensor.byte,
        torch.Tensor.ccol_indices,
        torch.Tensor.cdouble,
        torch.Tensor.cfloat,
        torch.Tensor.chalf,
        torch.Tensor.char,
        torch.Tensor.cholesky,
        torch.Tensor.cholesky_inverse,
        torch.Tensor.cholesky_solve,
        torch.Tensor.clamp_max,
        torch.Tensor.clamp_min,
        torch.Tensor.clip,
        torch.Tensor.coalesce,
        torch.Tensor.col_indices,
        torch.Tensor.conj,
        torch.Tensor.conj_physical,
        torch.Tensor.corrcoef,
        torch.Tensor.count_nonzero,
        torch.Tensor.cov,
        torch.Tensor.cpu,
        torch.Tensor.cross,
        torch.Tensor.crow_indices,
        torch.Tensor.cummax,
        torch.Tensor.cummin,
        torch.Tensor.cumprod,
        torch.Tensor.data_ptr,
        torch.Tensor.deg2rad,
        torch.Tensor.dense_dim,
        torch.Tensor.det,
        torch.Tensor.detach,
        torch.Tensor.diag,
        torch.Tensor.diag_embed,
        torch.Tensor.diagflat,
        torch.Tensor.diagonal_scatter,
        torch.Tensor.diff,
        torch.Tensor.dim_order,
        torch.Tensor.dist,
        torch.Tensor.divide,
        torch.Tensor.dot,
        torch.Tensor.double,
        torch.Tensor.dsplit,
        torch.Tensor.element_size,
        torch.Tensor.equal,
        torch.Tensor.fix,
        torch.Tensor.fliplr,
        torch.Tensor.flipud,
        torch.Tensor.float_power,
        torch.Tensor.fmax,
        torch.Tensor.fmin,
        torch.Tensor.frac,
        torch.Tensor.frexp,
        torch.Tensor.gcd,
        torch.Tensor.geqrf,
        torch.Tensor.ger,
        torch.Tensor.get_device,
        torch.Tensor.greater,
        torch.Tensor.greater_equal,
        torch.Tensor.half,
        torch.Tensor.hardshrink,
        torch.Tensor.has_names,
        torch.Tensor.heaviside,
        torch.Tensor.histc,
        torch.Tensor.histogram,
        torch.Tensor.hsplit,
        torch.Tensor.hypot,
        torch.Tensor.i0,
        torch.Tensor.igamma,
        torch.Tensor.igammac,
        torch.Tensor.index_fill,
        torch.Tensor.index_reduce,
        torch.Tensor.indices,
        torch.Tensor.inner,
        torch.Tensor.int,
        torch.Tensor.int_repr,
        torch.Tensor.inverse,
        torch.Tensor.is_coalesced,
        torch.Tensor.is_conj,
        torch.Tensor.is_contiguous,
        torch.Tensor.is_distributed,
        torch.Tensor.is_inference,
        torch.Tensor.is_neg,
        torch.Tensor.is_nonzero,
        torch.Tensor.is_pinned,
        torch.Tensor.is_same_size,
        torch.Tensor.is_set_to,
        torch.Tensor.is_shared,
        torch.Tensor.is_signed,
        torch.Tensor.isclose,
        torch.Tensor.isinf,
        torch.Tensor.isnan,
        torch.Tensor.isneginf,
        torch.Tensor.isposinf,
        torch.Tensor.isreal,
        torch.Tensor.istft,
        torch.Tensor.kron,
        torch.Tensor.kthvalue,
        torch.Tensor.lcm,
        torch.Tensor.ldexp,
        torch.Tensor.less,
        torch.Tensor.less_equal,
        torch.Tensor.logaddexp,
        torch.Tensor.logaddexp2,
        torch.Tensor.logcumsumexp,
        torch.Tensor.logdet,
        torch.Tensor.logical_or,
        torch.Tensor.logical_xor,
        torch.Tensor.logit,
        torch.Tensor.lu,
        torch.Tensor.lu_solve,
        torch.Tensor.masked_scatter,
        torch.Tensor.masked_select,
        torch.Tensor.matrix_exp,
        torch.Tensor.matrix_power,
        # torch.Tensor.max,
        torch.Tensor.median,
        torch.Tensor.min,
        torch.Tensor.mm,
        torch.Tensor.mode,
        torch.Tensor.module_load,
        torch.Tensor.moveaxis,
        torch.Tensor.msort,
        torch.Tensor.multiply,
        torch.Tensor.mv,
        torch.Tensor.mvlgamma,
        torch.Tensor.nanmean,
        torch.Tensor.nanmedian,
        torch.Tensor.nanquantile,
        torch.Tensor.nansum,
        torch.Tensor.narrow,
        torch.Tensor.narrow_copy,
        torch.Tensor.ndimension,
        torch.Tensor.negative,
        torch.Tensor.nelement,
        torch.Tensor.new_ones,
        torch.Tensor.new_full,
        torch.Tensor.new_zeros,
        torch.Tensor.new_empty,
        torch.Tensor.new_tensor,
        torch.Tensor.nonzero,
        torch.Tensor.nonzero_static,
        torch.Tensor.norm,
        torch.Tensor.not_equal,
        torch.Tensor.numpy,
        torch.Tensor.orgqr,
        torch.Tensor.ormqr,
        # torch.Tensor.outer,
        torch.Tensor.pin_memory,
        torch.Tensor.pinverse,
        torch.Tensor.positive,
        torch.Tensor.prelu,
        torch.Tensor.put,
        torch.Tensor.qr,
        torch.Tensor.quantile,
        torch.Tensor.rad2deg,
        torch.Tensor.ravel,
        torch.Tensor.refine_names,
        torch.Tensor.rename,
        torch.Tensor.renorm,
        torch.Tensor.repeat_interleave,
        torch.Tensor.reshape_as,
        torch.Tensor.resize,
        torch.Tensor.resize_as,
        torch.Tensor.resolve_conj,
        torch.Tensor.resolve_neg,
        torch.Tensor.retain_grad,
        torch.Tensor.roll,
        torch.Tensor.rot90,
        torch.Tensor.row_indices,
        torch.Tensor.scatter_reduce,
        torch.Tensor.select_scatter,
        torch.Tensor.sgn,
        torch.Tensor.short,
        torch.Tensor.sinc,
        torch.Tensor.slice_inverse,
        torch.Tensor.slice_scatter,
        torch.Tensor.slogdet,
        torch.Tensor.smm,
        torch.Tensor.sparse_dim,
        torch.Tensor.sparse_mask,
        torch.Tensor.split_with_sizes,
        torch.Tensor.square,
        torch.Tensor.sspaddmm,
        torch.Tensor.std,
        torch.Tensor.stft,
        torch.Tensor.subtract,
        torch.Tensor.sum_to_size,
        torch.Tensor.svd,
        torch.Tensor.swapaxes,
        torch.Tensor.swapdims,
        torch.Tensor.take,
        # torch.Tensor.take_along_dim,
        torch.Tensor.tile,
        torch.Tensor.to_dense,
        torch.Tensor.to_sparse,
        torch.Tensor.tolist,
        torch.Tensor.trace,
        torch.Tensor.triangular_solve,
        torch.Tensor.triu,
        torch.Tensor.unique,
        torch.Tensor.unique_consecutive,
        torch.Tensor.unsafe_chunk,
        torch.Tensor.unsafe_split,
        torch.Tensor.unsafe_split_with_sizes,
        torch.Tensor.values,
        torch.Tensor.vdot,
        torch.Tensor.vsplit,
        torch.Tensor.xlogy,
    ],
    torch.special: [
        torch.special.airy_ai,
        torch.special.bessel_j0,
        torch.special.bessel_j1,
        torch.special.bessel_y0,
        torch.special.bessel_y1,
        torch.special.chebyshev_polynomial_t,
        torch.special.chebyshev_polynomial_u,
        torch.special.chebyshev_polynomial_v,
        torch.special.chebyshev_polynomial_w,
        torch.special.entr,
        torch.special.erf,
        torch.special.erfc,
        torch.special.erfcx,
        torch.special.erfinv,
        torch.special.exp2,
        torch.special.expm1,
        torch.special.gammainc,
        torch.special.gammaincc,
        torch.special.gammaln,
        torch.special.hermite_polynomial_h,
        torch.special.hermite_polynomial_he,
        torch.special.i0,
        torch.special.i0e,
        torch.special.i1,
        torch.special.i1e,
        torch.special.laguerre_polynomial_l,
        torch.special.legendre_polynomial_p,
        torch.special.log1p,
        torch.special.log_ndtr,
        torch.special.logit,
        torch.special.logsumexp,
        torch.special.modified_bessel_i0,
        torch.special.modified_bessel_i1,
        torch.special.modified_bessel_k0,
        torch.special.modified_bessel_k1,
        torch.special.multigammaln,
        torch.special.ndtr,
        torch.special.ndtri,
        torch.special.psi,
        torch.special.round,
        torch.special.scaled_modified_bessel_k0,
        torch.special.scaled_modified_bessel_k1,
        torch.special.shifted_chebyshev_polynomial_t,
        torch.special.shifted_chebyshev_polynomial_u,
        torch.special.shifted_chebyshev_polynomial_v,
        torch.special.shifted_chebyshev_polynomial_w,
        torch.special.sinc,
        torch.special.softmax,
        torch.special.spherical_bessel_j0,
        torch.special.xlog1py,
        torch.special.xlogy,
    ],
    torch.linalg: [
        torch.linalg.cholesky,
        torch.linalg.cholesky_ex,
        torch.linalg.cond,
        torch.linalg.cross,
        torch.linalg.det,
        torch.linalg.diagonal,
        torch.linalg.eig,
        torch.linalg.eigh,
        torch.linalg.eigvals,
        torch.linalg.eigvalsh,
        torch.linalg.householder_product,
        torch.linalg.inv,
        torch.linalg.inv_ex,
        torch.linalg.ldl_factor,
        torch.linalg.ldl_factor_ex,
        torch.linalg.ldl_solve,
        torch.linalg.lstsq,
        torch.linalg.lu,
        torch.linalg.lu_factor,
        torch.linalg.lu_factor_ex,
        torch.linalg.lu_solve,
        torch.linalg.matmul,
        torch.linalg.matrix_exp,
        torch.linalg.matrix_norm,
        torch.linalg.matrix_power,
        torch.linalg.matrix_rank,
        torch.linalg.multi_dot,
        torch.linalg.norm,
        torch.linalg.pinv,
        torch.linalg.qr,
        torch.linalg.slogdet,
        torch.linalg.solve,
        torch.linalg.solve_ex,
        torch.linalg.solve_triangular,
        torch.linalg.svd,
        torch.linalg.svdvals,
        torch.linalg.tensorinv,
        torch.linalg.tensorsolve,
        torch.linalg.vander,
        torch.linalg.vecdot,
        torch.linalg.vector_norm,
    ],
    torch.fft: [
        torch.fft.fft,
        torch.fft.fft2,
        torch.fft.fftn,
        torch.fft.fftshift,
        torch.fft.hfft,
        torch.fft.hfft2,
        torch.fft.hfftn,
        torch.fft.ifft,
        torch.fft.ifft2,
        torch.fft.ifftn,
        torch.fft.ifftshift,
        torch.fft.ihfft,
        torch.fft.ihfft2,
        torch.fft.ihfftn,
        torch.fft.irfft,
        torch.fft.irfft2,
        torch.fft.irfftn,
        torch.fft.rfft,
        torch.fft.rfft2,
        torch.fft.rfftn,
    ],
}

# Records all the auto-registered Torch operators that return tensor views
# Ref: https://pytorch.org/docs/stable/tensor_view.html
# NOTE this list is used to update the `_syms_returning_views`, so that the symbol returning tensor views can be processed correctly when they interact with in-place operators.
# See :func:`thunder.core.functionalization.check_inplace_to_views` for the details.
_auto_registered_operators_returning_views = [
    torch.adjoint,
    torch.Tensor.adjoint,
    torch.Tensor.as_strided,
    torch.detach,
    torch.Tensor.detach,
    torch.narrow,
    torch.Tensor.narrow,
    torch.imag,
    torch.view_as_real,
    torch.nn.functional.unfold,
    torch.Tensor.hsplit,
    torch.hsplit,
    torch.Tensor.vsplit,
    torch.vsplit,
    torch.Tensor.split_with_sizes,
    torch.split_with_sizes,
    torch.Tensor.swapaxes,
    torch.swapaxes,
    torch.Tensor.swapdims,
    torch.swapdims,
    torch.Tensor.indices,
    torch.Tensor.values,
]
