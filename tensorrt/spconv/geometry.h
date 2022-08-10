#pragma once
#include "common/refnd.h"
namespace spconv {

namespace detail {

namespace geometry {

using utils::nd::Ref1D;
using utils::nd::Ref2D;
using utils::nd::Ref3D;
using utils::nd::Size;
using utils::nd::Vec;

template <bool Transpose, size_t RDim> struct DimLoopDispatchTag
{
  using tag = DimLoopDispatchTag<Transpose, RDim>;
};
template <bool Transpose> struct DoWrite
{
};

template <bool Transpose> struct DimLoopDispatchTag<Transpose, 0>
{
  using tag = DoWrite<Transpose>;
};

template <size_t NDim, size_t CurDim, class Index, class Writer>
HOST_DEVICE_INLINE void dimLoop(size_t inIndex,
                                size_t kernelOffset,
                                Index* coorOut,
                                const Index* coorIn,
                                const Index* kernelOffsetStride,
                                const Vec<NDim, Index>& kernelSize,
                                const Vec<NDim, Index>& stride,
                                const Vec<NDim, Index>& padding,
                                const Vec<NDim, Index>& dilation,
                                Writer& outputWriter,
                                DimLoopDispatchTag<false, NDim - CurDim>)
{
  Index dimCoor = coorIn[CurDim + 1] + padding[CurDim];
  for (size_t i = 0; i < kernelSize[CurDim]; i++) {
    if (dimCoor % stride[CurDim] == 0) {
      coorOut[CurDim + 1] = dimCoor / stride[CurDim];
      dimLoop<NDim, CurDim + 1>(inIndex, kernelOffset, coorOut, coorIn, kernelOffsetStride, kernelSize, stride, padding,
                                dilation, outputWriter, DimLoopDispatchTag<false, NDim - CurDim - 1>::tag());
    }
    dimCoor -= dilation[CurDim];
    kernelOffset += kernelOffsetStride[CurDim];
  }
}

template <size_t NDim, size_t CurDim, class Index, class Writer>
HOST_DEVICE_INLINE void dimLoop(size_t inIndex,
                                size_t kernelOffset,
                                Index* coorOut,
                                const Index* coorIn,
                                const Index* kernelOffsetStride,
                                const Vec<NDim, Index>& kernelSize,
                                const Vec<NDim, Index>& stride,
                                const Vec<NDim, Index>& padding,
                                const Vec<NDim, Index>& dilation,
                                Writer& outputWriter,
                                DimLoopDispatchTag<true, NDim - CurDim>)
{
  Index dimCoor = coorIn[CurDim + 1] * stride[CurDim] - padding[CurDim];
  for (size_t i = 0; i < kernelSize[CurDim]; i++) {
    coorOut[CurDim + 1] = dimCoor;
    dimLoop<NDim, CurDim + 1>(inIndex, kernelOffset, coorOut, coorIn, kernelOffsetStride, kernelSize, stride, padding,
                              dilation, outputWriter, DimLoopDispatchTag<true, NDim - CurDim - 1>::tag());
    dimCoor += dilation[CurDim];
    kernelOffset += kernelOffsetStride[CurDim];
  }
}

template <size_t NDim, size_t CurDim, class Index, class Writer, bool Transpose>
HOST_DEVICE_INLINE void dimLoop(size_t inIndex,
                                size_t kernelOffset,
                                Index* coorOut,
                                const Index* coorIn,
                                const Index* kernelOffsetStride,
                                const Vec<NDim, Index>& kernelSize,
                                const Vec<NDim, Index>& stride,
                                const Vec<NDim, Index>& padding,
                                const Vec<NDim, Index>& dilation,
                                Writer& outputWriter,
                                DoWrite<Transpose>)
{
  outputWriter(inIndex, kernelOffset, coorOut);
}

template <size_t NDim, class Index, class Writer>
HOST_DEVICE void getOutPosLoop(size_t inIndex,
                               const Ref2D<Index>& coorsIn,
                               const Vec<NDim, Index>& kernelSize,
                               const Vec<NDim, Index>& stride,
                               const Vec<NDim, Index>& padding,
                               const Vec<NDim, Index>& dilation,
                               Writer& outputWriter,
                               const bool transpose)
{
  Index kernelOffsetStride[NDim], coorOut[NDim + 1];
  Index kernelOffset = 0;
  const Index* coorIn = &coorsIn(inIndex, 0);
  coorOut[0] = coorIn[0];
  kernelOffsetStride[NDim - 1] = 1;
  for (ssize_t i = ssize_t(NDim - 2); i >= 0; i--) {
    kernelOffsetStride[i] = kernelOffsetStride[i + 1] * kernelSize[i + 1];
  }
  if (transpose)
    return dimLoop<NDim, 0, Index, Writer>(inIndex, kernelOffset, coorOut, coorIn, kernelOffsetStride, kernelSize,
                                           stride, padding, dilation, outputWriter,
                                           DimLoopDispatchTag<true, NDim>::tag());
  return dimLoop<NDim, 0, Index, Writer>(inIndex, kernelOffset, coorOut, coorIn, kernelOffsetStride, kernelSize, stride,
                                         padding, dilation, outputWriter, DimLoopDispatchTag<false, NDim>::tag());
}

template <size_t NDim, class Index>
Size<NDim + 1> outSize(const Size<NDim + 1>& inputSize,
                       const Vec<NDim, Index>& kernelSize,
                       const Vec<NDim, Index>& stride,
                       const Vec<NDim, Index>& padding,
                       const Vec<NDim, Index>& dilation)
{
  typename Size<NDim + 1>::index_vec_t outputSize;
  outputSize[0] = inputSize[0];
#pragma unroll
  for (int i = 0; i < NDim; i++) {
    Index size = (inputSize[i + 1] + 2 * padding[i] - dilation[i] * (kernelSize[i] - 1) - 1) / stride[i] + 1;
    if (kernelSize[i] == -1) { outputSize[i + 1] = 1; }
    else {
      outputSize[i + 1] = size;
    }
  }
  return Size<NDim + 1>(outputSize);
}

template <size_t NDim, class Index>
Size<NDim + 1> transposeOutSize(const Size<NDim + 1>& inputSize,
                                const Vec<NDim, Index>& kernelSize,
                                const Vec<NDim, Index>& stride,
                                const Vec<NDim, Index>& padding,
                                const Vec<NDim, Index>& dilation,
                                const Vec<NDim, Index>& outputPadding)
{
  typename Size<NDim + 1>::index_vec_t outputSize;
  outputSize[0] = inputSize[0];
#pragma unroll
  for (int i = 0; i < NDim; i++) {
    ASSERT_RT_ERR(kernelSize[i] >= 0, "transpose op does not support kernel_size < 0");
    Index size =
        (inputSize[i + 1] - 1) * stride[i] - 2 * padding[i] + (kernelSize[i] - 1) * dilation[i] + outputPadding[i] + 1;
    outputSize[i + 1] = size;
  }
  return Size<NDim + 1>(outputSize);
}

template <size_t NDim, class Index>
Size<NDim + 1> getOutSize(const Size<NDim + 1>& inputSize,
                          const Vec<NDim, Index>& kernelSize,
                          const Vec<NDim, Index>& stride,
                          const Vec<NDim, Index>& padding,
                          const Vec<NDim, Index>& dilation,
                          const Vec<NDim, Index>& outputPadding,
                          const bool transpose)
{
  if (transpose) return transposeOutSize(inputSize, kernelSize, stride, padding, dilation, outputPadding);
  return outSize(inputSize, kernelSize, stride, padding, dilation);
}

}  // namespace geometry
}  // namespace detail
}  // namespace spconv