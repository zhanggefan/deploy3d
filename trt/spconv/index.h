#pragma once
#include "common/refnd.h"
#include <limits>

namespace spconv {
namespace func {

using utils::GPU;
using utils::nd::Ref1D;
using utils::nd::Ref2D;
using utils::nd::Ref3D;
using utils::nd::Size;
using utils::nd::Vec;

template <size_t NDim, class Index>
size_t createSparseConvIndexMalloc(const GPU& d,
                                   const size_t numInput,
                                   const Vec<NDim, Index>& kernelSize,
                                   const bool oneTimeMalloc = false);

template <size_t NDim, class Index>
void createSparseConvIndex(const GPU& d,
                           Ref1D<uint8_t>& workingStorage,
                           Ref1D<Index>& bufferFromIn,
                           Ref1D<Index>& bufferToOut,
                           Ref1D<Index>& bufferOffset,
                           Ref1D<Index>& bufferKernelNum,
                           Ref2D<Index>& coorsOut,
                           Index* numBufPtr,
                           Index* numOutPtr,
                           const Ref2D<Index>& coorsIn,
                           const Vec<NDim, Index>& kernelSize,
                           const Vec<NDim, Index>& stride,
                           const Vec<NDim, Index>& padding,
                           const Vec<NDim, Index>& dilation,
                           const Size<NDim + 1>& outSpatialShape,
                           const bool transpose = false,
                           const Index numSample = std::numeric_limits<Index>::max());

template <size_t NDim, class Index>
size_t createSparseSubMIndexMalloc(const GPU& d,
                                   const size_t numInput,
                                   const Vec<NDim, Index>& kernelSize,
                                   const bool oneTimeMalloc = false);

template <size_t NDim, class Index>
void createSparseSubMIndex(const GPU& d,
                           Ref1D<uint8_t>& workingStorage,
                           Ref1D<Index>& bufferFromIn,
                           Ref1D<Index>& bufferToOut,
                           Ref1D<Index>& bufferOffset,
                           Ref1D<Index>& bufferKernelNum,
                           Index* numBufPtr,
                           const Ref2D<Index>& coorsIn,
                           const Vec<NDim, Index>& kernelSize,
                           const Vec<NDim, Index>& stride,
                           const Vec<NDim, Index>& padding,
                           const Vec<NDim, Index>& dilation,
                           const Size<NDim + 1>& outSpatialShape);

}  // namespace func
}  // namespace spconv