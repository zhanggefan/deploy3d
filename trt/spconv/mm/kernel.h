#pragma once
#include "cutlass/cutlass.h"

#include "cutlass/arch/arch.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/semaphore.h"

namespace spconv {
namespace kernel {

template <typename Mma_, typename Epilogue_, typename ThreadblockSwizzle_, bool SplitKSerial> struct SpConv {
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using OutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static bool const kSplitKSerial = SplitKSerial;
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  struct Params {
    cutlass::gemm::GemmCoord problem_size;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    int swizzle_log_tile;
    typename Mma::IteratorA::Params params_feature_iter;
    typename Mma::IteratorA::TensorRef ref_feature;
    typename Mma::IteratorB::Params params_weight_iter;
    typename Mma::IteratorB::TensorRef ref_weight;
    typename Epilogue::OutputTileIterator::Params params_output_iter;
    typename Epilogue::OutputTileIterator::TensorRef ref_output;
    typename OutputOp::Params output_op;
    int* semaphore;
    int gemm_k_size;
    int kernel_stride;
    int const* indices_num;
    int const* indices_gather_in;
    int const* indices_kernel_offset;
    int const* indices_scatter_out;

    CUTLASS_HOST_DEVICE
    Params() : swizzle_log_tile(0), semaphore(0), gemm_k_size(0) {}

    CUTLASS_HOST_DEVICE
    Params(int max_indices_num,
           int in_channels,
           int out_channels,
           typename Mma::IteratorA::TensorRef::Element* in,
           typename Mma::IteratorB::TensorRef::Element* weight,
           typename Epilogue::OutputTileIterator::TensorRef::Element* out,
           int const* indices_gather_in = nullptr,
           int const* indices_kernel_offset = nullptr,
           int const* indices_scatter_out = nullptr,
           int const* act_indices_num = nullptr,
           int kernel_stride = -1,
           int* workspace = nullptr)
        : problem_size({max_indices_num, out_channels, in_channels}), params_feature_iter(in_channels),
          ref_feature({in, in_channels}), params_weight_iter(out_channels), ref_weight({weight, out_channels}),
          params_output_iter(out_channels), ref_output({out, out_channels}), indices_gather_in(indices_gather_in),
          indices_kernel_offset(indices_kernel_offset), indices_scatter_out(indices_scatter_out), output_op(),
          kernel_stride(kernel_stride < 0 ? in_channels * out_channels : 0), indices_num(act_indices_num) {
      ThreadblockSwizzle threadblock_swizzle;
      grid_tiled_shape = threadblock_swizzle.get_tiled_shape({max_indices_num, out_channels, in_channels},
                                                             {Mma::Shape::kM, Mma::Shape::kN, Mma::Shape::kK}, 1);
      swizzle_log_tile = threadblock_swizzle.get_log_tile(grid_tiled_shape);
      int total_gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;
      int gemm_k_iterations = (total_gemm_k_iterations + grid_tiled_shape.k() - 1) / grid_tiled_shape.k();
      gemm_k_size = gemm_k_iterations * Mma::Shape::kK;
      semaphore = workspace;
    }

    inline dim3 block_size() { return {kThreadCount, 1, 1}; };
    inline dim3 grid_size() { return ThreadblockSwizzle().get_grid_shape(grid_tiled_shape); };
  };

  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

  CUTLASS_HOST_DEVICE
  SpConv() {}

  CUTLASS_DEVICE
  void operator()(Params const& params, SharedStorage& shared_storage) {
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord threadblock_tile_offset = threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
        params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {
      return;
    }

    cutlass::MatrixCoord tb_offset_feature{
        threadblock_tile_offset.m() * Mma::Shape::kM,
        threadblock_tile_offset.k() * params.gemm_k_size,
    };

    if (params.indices_num && (*params.indices_num) <= tb_offset_feature.row()) return;
    int indices_kernel_offset = params.indices_kernel_offset[tb_offset_feature.row()];
    if (indices_kernel_offset < 0) return;

    cutlass::MatrixCoord tb_offset_weight{threadblock_tile_offset.k() * params.gemm_k_size,
                                          threadblock_tile_offset.n() * Mma::Shape::kN};

    int problem_size_k = min(params.problem_size.k(), (threadblock_tile_offset.k() + 1) * params.gemm_k_size);

    int gemm_k_iterations = (problem_size_k - tb_offset_feature.column() + Mma::Shape::kK - 1) / Mma::Shape::kK;

    int thread_idx = threadIdx.x;

    typename Mma::IteratorA feature_iter(params.params_feature_iter, params.ref_feature.data(),
                                         {params.problem_size.m(), problem_size_k}, thread_idx, tb_offset_feature,
                                         params.indices_gather_in);

    auto* weight_ptr = params.ref_weight.data() + indices_kernel_offset * params.kernel_stride;

    typename Mma::IteratorB weight_iter(params.params_weight_iter, weight_ptr,
                                        {problem_size_k, params.problem_size.n()}, thread_idx, tb_offset_weight,
                                        nullptr);

    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    int lane_idx = threadIdx.x % 32;

    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    typename Mma::FragmentC accumulators;

    accumulators.clear();

    if (!kSplitKSerial || gemm_k_iterations > 0) {
      mma(gemm_k_iterations, accumulators, feature_iter, weight_iter, accumulators);
    }

    /// -------------- Epilogue --------------

    OutputOp output_op(params.output_op);

    threadblock_tile_offset = threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    cutlass::MatrixCoord threadblock_offset(threadblock_tile_offset.m() * Mma::Shape::kM,
                                            threadblock_tile_offset.n() * Mma::Shape::kN);

    int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

    cutlass::Semaphore semaphore(params.semaphore + block_idx, thread_idx);

    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
      semaphore.fetch();
      output_op.set_k_partition(threadblock_tile_offset.k(), params.grid_tiled_shape.k());
    }

    typename Epilogue::OutputTileIterator iterator_C(params.params_output_iter, params.ref_output.data(),
                                                     params.problem_size.mn(), thread_idx, threadblock_offset,
                                                     params.indices_scatter_out);

    typename Epilogue::OutputTileIterator iterator_D(params.params_output_iter, params.ref_output.data(),
                                                     params.problem_size.mn(), thread_idx, threadblock_offset,
                                                     params.indices_scatter_out);

    Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);

    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
      if (threadblock_tile_offset.k()) { iterator_C = iterator_D; }
      semaphore.wait(threadblock_tile_offset.k());
    }

    epilogue(output_op, iterator_D, accumulators, iterator_C);

    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
      int lock = 0;
      if (params.grid_tiled_shape.k() == threadblock_tile_offset.k() + 1) {
        lock = 0;
      } else {
        lock = threadblock_tile_offset.k() + 1;
      }
      semaphore.release(lock);
    }
  }

  CUTLASS_HOST_DEVICE
  static bool can_implement(int in_channels,
                            int out_channels,
                            typename Mma::IteratorA::TensorRef::Element* in,
                            typename Mma::IteratorB::TensorRef::Element* weight,
                            typename Epilogue::OutputTileIterator::TensorRef::Element* out,
                            int kernel_stride = -1) {
    static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
    static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
    static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

    if (!cutlass::TensorRef_aligned(typename Mma::IteratorA::TensorRef(in, in_channels), kAlignmentA)) return false;

    if (!cutlass::TensorRef_aligned(typename Mma::IteratorB::TensorRef(weight, out_channels), kAlignmentB))
      return false;

    if (kernel_stride >= 0 && kernel_stride % kAlignmentB != 0) return false;

    if (!cutlass::TensorRef_aligned(typename Epilogue::OutputTileIterator::TensorRef(out, out_channels), kAlignmentC))
      return false;

    return true;
  }
};

}  // namespace kernel
}  // namespace spconv