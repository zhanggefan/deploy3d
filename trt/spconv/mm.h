#pragma once
#include "mm/config_simt.h"
#include "mm/config_tensorop.h"

#include "common/launch.cuh"
#include "common/refnd.h"

using utils::GPU;
using utils::launch::CUDA_NUM_THREADS;
using utils::launch::getBlocks;
using utils::launch::KernelLoopX;
using utils::nd::Ref1D;
using utils::nd::Ref2D;
using utils::nd::Ref3D;

namespace spconv {
namespace kernel {
template <class T> __global__ void biasInit(Ref2D<T> outFeats, const Ref1D<const T> initVec) {
  size_t numOuts = outFeats.size(0);
  size_t numFeats = outFeats.size(1);
  size_t numOutElem = numOuts * numFeats;
  for (size_t ix : KernelLoopX<size_t>(numOutElem)) {
    if (ix < numOutElem) outFeats[ix] = initVec[ix % numFeats];
  }
}
}  // namespace kernel

namespace func {

using utils::GPU;
using utils::nd::Ref1D;
using utils::nd::Ref2D;
using utils::nd::Ref3D;

template <typename Config>
void indexedSpConv(const GPU& d,
                   Ref2D<typename Config::Element>& outFeats,
                   const Ref2D<const typename Config::Element>& inFeats,
                   const Ref3D<const typename Config::Element>& filters,
                   const Ref1D<const typename Config::Element>& bias,
                   const Ref1D<const int32_t>& gatherIn,
                   const Ref1D<const int32_t>& scatterOut,
                   const Ref1D<const int32_t>& kernelOffset,
                   const int32_t* numIndexPtr) {
  using Element = typename Config::Element;
  ssize_t numThreads;
  if (bias.empty()) {
    cudaMemsetAsync(outFeats.data(), 0, outFeats.numel() * sizeof(Element), d.getStream());
  } else {
    numThreads = (outFeats.numel() + 3) / 4;
    kernel::biasInit<Element><<<getBlocks(numThreads), CUDA_NUM_THREADS, 0, d.getStream()>>>(outFeats, bias);
  }
  ssize_t numIndicesMax = gatherIn.numel();
  ssize_t numFeatsIn = inFeats.size(1);
  ssize_t numFeatsOut = filters.size(2);
  typename Config::Params params(numIndicesMax, numFeatsIn, numFeatsOut, (typename Config::Element*)inFeats.data(),
                                 (typename Config::Element*)filters.data(), (typename Config::Element*)outFeats.data(),
                                 gatherIn.data(), kernelOffset.data(), scatterOut.data(), numIndexPtr);

  constexpr int smem_size = int(sizeof(typename Config::SharedStorage));
  if (smem_size >= (48 << 10)) {
    auto result = cudaFuncSetAttribute(cutlass::Kernel<Config>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (result != cudaSuccess) { std::cout << cudaGetErrorString(result) << std::endl; }
  }
  cutlass::Kernel<Config><<<params.grid_size(), params.block_size(), smem_size, d.getStream()>>>(params);
}

}  // namespace func
}  // namespace spconv