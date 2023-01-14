#include "common/cub.cuh"
#include "common/hash32.h"
#include "common/launch.cuh"
#include "common/macros.h"
#include "common/refnd.h"
#include "geometry.h"

using utils::GPU;
using utils::launch::CUDA_NUM_THREADS;
using utils::launch::getBlocks;
using utils::launch::KernelLoopX;
using utils::nd::Ref1D;
using utils::nd::Ref2D;
using utils::nd::Ref3D;
using utils::nd::RefND;
using utils::nd::Size;
using utils::nd::Vec;

namespace spconv {

static constexpr int TBSegment = 128;

template <class T> using mkS = typename std::make_signed_t<T>;
template <class T> using mkU = typename std::make_unsigned_t<T>;

namespace kernel {

template <size_t NDim, class Index> class ConvHashWriter {
 public:
  using HashTable = hash::LinearHashTable<Index, Index>;
  DEVICE_INLINE
  ConvHashWriter(Ref2D<mkU<Index>>& hashOut, HashTable& hash, const Size<NDim + 1>& outSpatialShape)
      : hashOut(hashOut), hash(hash), outSpatialShape(outSpatialShape) {}
  DEVICE_INLINE
  void operator()(size_t inIndex, size_t kernelOffset, Index* outCoor) {
    if (outSpatialShape.is_valid(outCoor)) {
      Index outSpatialOffset = outSpatialShape.offset(outCoor);
      hashOut(kernelOffset, inIndex) = hash.insert(outSpatialOffset, 1);
    }
  }

 private:
  Ref2D<mkU<Index>>& hashOut;
  HashTable& hash;
  const Size<NDim + 1>& outSpatialShape;
};

template <size_t NDim, class Index> class SubMHashWriter {
 public:
  using HashTable = hash::LinearHashTable<Index, Index>;
  DEVICE_INLINE
  SubMHashWriter(Ref2D<mkU<Index>>& hashOut, const HashTable& hash, const Size<NDim + 1>& outSpatialShape)
      : hashOut(hashOut), hash(hash), outSpatialShape(outSpatialShape) {}
  DEVICE_INLINE
  void operator()(size_t inIndex, size_t kernelOffset, Index* outCoor) {
    if (outSpatialShape.is_valid(outCoor)) {
      Index outSpatialOffset = outSpatialShape.offset(outCoor);
      Index outIndex;
      if (hash.lookup(outSpatialOffset, outIndex)) { hashOut(kernelOffset, inIndex) = outIndex; }
    }
  }

 private:
  Ref2D<mkU<Index>>& hashOut;
  const HashTable& hash;
  const Size<NDim + 1>& outSpatialShape;
};

template <size_t NDim, class Index>
__global__ void subMGeometryKernel(Ref2D<mkU<Index>> hashOut,
                                   const typename SubMHashWriter<NDim, Index>::HashTable hash,
                                   const Ref2D<Index> coorsIn,
                                   const Vec<NDim, Index> kernelSize,
                                   const Vec<NDim, Index> stride,
                                   const Vec<NDim, Index> padding,
                                   const Vec<NDim, Index> dilation,
                                   const Size<NDim + 1> outSpatialShape) {
  auto numActIn = coorsIn.size(0);
  SubMHashWriter<NDim, Index> writer(hashOut, hash, outSpatialShape);
  for (size_t ix : KernelLoopX(numActIn)) {
    detail::geometry::getOutPosLoop<NDim, Index, SubMHashWriter<NDim, Index>>(ix, coorsIn, kernelSize, stride, padding,
                                                                              dilation, writer, false);
  }
}

template <size_t NDim, class Index>
__global__ void convGeometryKernel(Ref2D<mkU<Index>> hashOut,
                                   typename ConvHashWriter<NDim, Index>::HashTable hash,
                                   const Ref2D<Index> coorsIn,
                                   const Vec<NDim, Index> kernelSize,
                                   const Vec<NDim, Index> stride,
                                   const Vec<NDim, Index> padding,
                                   const Vec<NDim, Index> dilation,
                                   const Size<NDim + 1> outSpatialShape,
                                   const bool transpose) {
  auto numActIn = coorsIn.size(0);
  ConvHashWriter<NDim, Index> writer(hashOut, hash, outSpatialShape);
  for (size_t ix : KernelLoopX(numActIn)) {
    detail::geometry::getOutPosLoop<NDim, Index, ConvHashWriter<NDim, Index>>(ix, coorsIn, kernelSize, stride, padding,
                                                                              dilation, writer, transpose);
  }
}

template <size_t NDim, class Index>
__global__ void coorsOutOrganizeKernel(Ref2D<Index> coorsOut,
                                       Index* numActOut,
                                       const typename kernel::ConvHashWriter<NDim, Index>::HashTable hash,
                                       const Ref1D<Index> uniqueIndex,
                                       const Size<NDim + 1> outSpatialShape,
                                       const Index numSample) {
  auto numElem = hash.size();
  for (size_t ix : KernelLoopX(hash.size())) {
    auto spatialOffset = hash.keys()[ix];
    Index outIndex = uniqueIndex[ix];
    if (spatialOffset != hash.EMPTY && outIndex < numSample) {
      outSpatialShape.deserialize(&coorsOut(outIndex, 0), spatialOffset);
      outIndex++;
    }
    if (ix == numElem - 1) *numActOut = min(outIndex, numSample);
  }
}

template <class Index>
__global__ void indexOrganizeKernel(Ref1D<Index> GatherIn,
                                    Ref1D<Index> ScatterOut,
                                    Ref1D<Index> KernelOffset,
                                    Index* numIndexPtr,
                                    const Ref2D<Index> outputIndex,
                                    const Ref1D<Index> outputIndexPadCumSum,
                                    const Ref2D<mkU<Index>> hashOut) {
  auto numActIn = hashOut.size(1);
  for (size_t ix : KernelLoopX(hashOut.numel())) {
    auto scatterTo = static_cast<Index>(hashOut[ix]);
    if (scatterTo >= 0) {
      auto outIndex = outputIndex[ix] + outputIndexPadCumSum[ix / numActIn];
      GatherIn[outIndex] = ix % numActIn;
      KernelOffset[outIndex] = ix / numActIn;
      ScatterOut[outIndex] = scatterTo;
    }
    if (ix == hashOut.numel() - 1) {
      auto outIndex = outputIndex[ix] + outputIndexPadCumSum[ix / numActIn];
      outIndex += scatterTo >= 0 ? 1 : 0;
      *numIndexPtr = outIndex + ((TBSegment - (outIndex & (TBSegment - 1))) & (TBSegment - 1));
    }
  }
}

template <class Index>
__global__ void outputIndexPadOrganizeKernel(Ref1D<Index> outputIndexPad, const Ref2D<Index> outputIndex) {
  auto kVol = outputIndex.size(0);
  auto numActIn = outputIndex.size(1);
  for (size_t ix : KernelLoopX(kVol)) {
    Index numActKernel = ix ? outputIndex(ix, 0) - outputIndex(ix - 1, 0) : 0;
    outputIndexPad[ix] = ((TBSegment - (numActKernel & (TBSegment - 1))) & (TBSegment - 1));
  }
}

template <class Index>
__global__ void hashOutIndexKernel(Ref2D<mkU<Index>> hashOut, const Ref1D<Index> uniqueIndex, const Index numSample) {
  size_t numElem = hashOut.numel();
  for (size_t ix : KernelLoopX(numElem)) {
    auto slot = hashOut[ix];
    if (slot < uniqueIndex.numel()) {
      auto uid = uniqueIndex[slot];
      if (uid < numSample) {
        hashOut[ix] = uid;
      } else {
        hashOut[ix] = ~mkU<Index>(0);
      }
    }
  }
}

template <size_t NDim, class Index>
__global__ void setSubMHashKernel(typename SubMHashWriter<NDim, Index>::HashTable hash,
                                  const Ref2D<Index> coorsIn,
                                  const Size<NDim + 1> outSpatialShape) {
  for (size_t ix : KernelLoopX(coorsIn.size(0))) {
    Index index = outSpatialShape.offset(&coorsIn(ix, 0));
    hash.insert(index, ix);
  }
}

template <class Hash> __global__ void resetHashKernel(Hash hash) {
  for (size_t ix : KernelLoopX(hash.size())) { hash.keys()[ix] = hash.EMPTY; }
}

}  // namespace kernel

namespace func {

constexpr int hashSpace = 2;

template <class Index> HOST_DEVICE_INLINE Index FF() {
  const mkU<Index> ff = ~mkU<Index>(0);
  return reinterpret_cast<const Index&>(ff);
}

struct Valid {
  template <class Index> DEVICE_INLINE bool operator()(const Index& a) const { return a != FF<Index>(); }
};

struct Valid_Counter {
  template <class Index> DEVICE_INLINE Index operator()(const Index& a) const {
    return a != FF<Index>() ? Index(1) : Index(0);
  }
};

size_t getMinPower2(size_t x) {
  size_t y;
  for (y = 1; y < x; y = y << 1) {}
  return y;
}

template <size_t NDim, class Index>
size_t createSparseConvIndexMalloc(const GPU& d,
                                   const size_t numInput,
                                   const Vec<NDim, Index>& kernelSize,
                                   const bool oneTimeMalloc) {
  size_t reqBytes_, reqBytes = 0;
  size_t kVol = 1;
#pragma unroll
  for (size_t i = 0; i < NDim; i++) { kVol *= kernelSize[i]; }
  size_t outElemNumMaxPower2 = getMinPower2(numInput * kVol);
  size_t numElemHash = getMinPower2(numInput * kVol * hashSpace);

  cudaError_t status;

  for (size_t scale = oneTimeMalloc ? 1 : outElemNumMaxPower2; scale <= outElemNumMaxPower2; scale = scale << 1) {
    status = DEPLOY3D_CUB_NS_QUALIFIER::cub::DeviceScan::ExclusiveSum<
        DEPLOY3D_CUB_NS_QUALIFIER::cub::TransformInputIterator<Index, Valid_Counter, mkU<Index>*>, Index*>(
        nullptr, reqBytes_, {nullptr, {}}, nullptr, scale, d.getStream());
    CHECK_RETURN_STATUS(status);
    reqBytes = reqBytes > reqBytes_ ? reqBytes : reqBytes_;
  }

  for (size_t scale = oneTimeMalloc ? 1 : numElemHash; scale <= numElemHash; scale = scale << 1) {
    status = DEPLOY3D_CUB_NS_QUALIFIER::cub::DeviceScan::ExclusiveSum<Index*, Index*>(
        nullptr, reqBytes_, nullptr, nullptr, numElemHash, d.getStream());
    CHECK_RETURN_STATUS(status);
    reqBytes = reqBytes > reqBytes_ ? reqBytes : reqBytes_;
  }

  status = DEPLOY3D_CUB_NS_QUALIFIER::cub::DeviceScan::InclusiveSum<Index*, Index*>(nullptr, reqBytes_, nullptr,
                                                                                    nullptr, kVol, d.getStream());
  CHECK_RETURN_STATUS(status);
  reqBytes = reqBytes > reqBytes_ ? reqBytes : reqBytes_;

  reqBytes += (3 * numElemHash + 4 * outElemNumMaxPower2) * sizeof(Index);
  return reqBytes;
}

template <size_t NDim, class Index>
void createSparseConvIndex(const GPU& d,
                           Ref1D<uint8_t>& workingStorage,
                           Ref1D<Index>& GatherIn,
                           Ref1D<Index>& ScatterOut,
                           Ref1D<Index>& KernelOffset,
                           Ref2D<Index>& coorsOut,
                           Index* numIndexPtr,
                           Index* numOutPtr,
                           const Ref2D<Index>& coorsIn,
                           const Vec<NDim, Index>& kernelSize,
                           const Vec<NDim, Index>& stride,
                           const Vec<NDim, Index>& padding,
                           const Vec<NDim, Index>& dilation,
                           const Size<NDim + 1>& outSpatialShape,
                           const bool transpose,
                           const Index numSample) {
  Index numActIn = coorsIn.size(0);
  if (numActIn == 0) {
    cudaMemsetAsync(numIndexPtr, 0, sizeof(Index), d.getStream());
    cudaMemsetAsync(numOutPtr, 0, sizeof(Index), d.getStream());
    return;
  }

  ssize_t kVol = 1;
#pragma unroll
  for (size_t i = 0; i < NDim; i++) { kVol *= kernelSize[i]; }
  ssize_t outElemNumMaxPower2 = getMinPower2(kVol * numActIn);
  ssize_t numElemHash = getMinPower2(kVol * numActIn * hashSpace);

  uint8_t* workingStoragePtr = workingStorage.data();

  Ref1D<Index> hashKeys(reinterpret_cast<Index*>(workingStoragePtr), {numElemHash});
  workingStoragePtr += hashKeys.numby();
  Ref1D<Index> hashValues(reinterpret_cast<Index*>(workingStoragePtr), {numElemHash});
  workingStoragePtr += hashValues.numby();
  Ref1D<Index> uniqueIndex(reinterpret_cast<Index*>(workingStoragePtr), {numElemHash});
  workingStoragePtr += uniqueIndex.numby();

  typename kernel::ConvHashWriter<NDim, Index>::HashTable hash(hashKeys.data(), hashValues.data(), numElemHash);

  Ref2D<mkU<Index>> hashOut(reinterpret_cast<mkU<Index>*>(workingStoragePtr), {kVol, numActIn});
  workingStoragePtr += outElemNumMaxPower2 * sizeof(mkU<Index>);

  Ref2D<Index> outputIndex(reinterpret_cast<Index*>(workingStoragePtr), {kVol, numActIn});
  workingStoragePtr += outElemNumMaxPower2 * sizeof(Index);

  Ref1D<Index> outputIndexPad(reinterpret_cast<Index*>(workingStoragePtr), {kVol});
  workingStoragePtr += outElemNumMaxPower2 * sizeof(Index);

  Ref1D<Index> outputIndexPadCumSum(reinterpret_cast<Index*>(workingStoragePtr), {kVol});
  workingStoragePtr += outElemNumMaxPower2 * sizeof(Index);

  size_t workingStorageBytes = workingStorage.size(0) - (workingStoragePtr - workingStorage.data());

  {  // conv I-O geometry
    cudaMemsetAsync(hashOut.data(), 0xFF, outElemNumMaxPower2 * sizeof(mkU<Index>), d.getStream());
    cudaMemsetAsync(hashValues.data(), 0x00, hashValues.numby(), d.getStream());
    kernel::resetHashKernel<typename kernel::ConvHashWriter<NDim, Index>::HashTable>
        <<<getBlocks(hash.size()), CUDA_NUM_THREADS, 0, d.getStream()>>>(hash);
    kernel::convGeometryKernel<NDim, Index><<<getBlocks(numActIn), CUDA_NUM_THREADS, 0, d.getStream()>>>(
        hashOut, hash, coorsIn, kernelSize, stride, padding, dilation, outSpatialShape, transpose);
    CHECK_CUDA_ERR();
  }

  {  // make output unique index
    // input:
    //   hashOut: geometric io lookup table:
    //      ({input idx}, {filter idx})
    //      -> {output coors hash slot} or -1 (output invalid)
    //
    // output:
    //   hashOut: geometric io lookup table:
    //      ({input idx}, {filter idx})
    //      -> {output idx}
    //   numOutPtr: number of output coors
    //   coorsOut: output coors
    DEPLOY3D_CUB_NS_QUALIFIER::cub::DeviceScan::ExclusiveSum<Index*, Index*>(
        workingStoragePtr, workingStorageBytes, hashValues.data(), uniqueIndex.data(), hash.size(), d.getStream());
    kernel::coorsOutOrganizeKernel<NDim, Index><<<getBlocks(hash.size()), CUDA_NUM_THREADS, 0, d.getStream()>>>(
        coorsOut, numOutPtr, hash, uniqueIndex, outSpatialShape, numSample);
    kernel::hashOutIndexKernel<Index>
        <<<getBlocks(hashOut.numel()), CUDA_NUM_THREADS, 0, d.getStream()>>>(hashOut, uniqueIndex, numSample);
    CHECK_CUDA_ERR();
  }

  {  // organize output
    // input:
    //   hashOut: geometric io lookup table:
    //      ({input idx}, {filter idx})
    //      -> {output idx} or -1 (output invalid)
    //
    // output:
    //   numIndexPtr: length of GatherIn/ScatterOut/KernelOffset
    //   GatherIn: index conv gather source. calculate by filtering out -1
    //             from hashOut and then decoding the input index
    //   ScatterOut: index conv scatter target. calculate by filtering out -1
    //               from hashOut and then looking up the hashOut
    //   kernelOffset: index conv kernel offset index
    cudaMemsetAsync(GatherIn.data(), 0xFF, GatherIn.numby(), d.getStream());
    cudaMemsetAsync(ScatterOut.data(), 0xFF, ScatterOut.numby(), d.getStream());
    cudaMemsetAsync(KernelOffset.data(), 0xFF, KernelOffset.numby(), d.getStream());
    DEPLOY3D_CUB_NS_QUALIFIER::cub::DeviceScan::ExclusiveSum(
        workingStoragePtr, workingStorageBytes,
        DEPLOY3D_CUB_NS_QUALIFIER::cub::TransformInputIterator<Index, Valid_Counter, mkU<Index>*>(hashOut.data(), {}),
        outputIndex.data(), outElemNumMaxPower2, d.getStream());
    kernel::outputIndexPadOrganizeKernel<Index>
        <<<getBlocks(kVol), CUDA_NUM_THREADS, 0, d.getStream()>>>(outputIndexPad, outputIndex);
    DEPLOY3D_CUB_NS_QUALIFIER::cub::DeviceScan::InclusiveSum<Index*, Index*>(
        workingStoragePtr, workingStorageBytes, outputIndexPad.data(), outputIndexPadCumSum.data(), kVol,
        d.getStream());
    kernel::indexOrganizeKernel<Index><<<getBlocks(hashOut.numel()), CUDA_NUM_THREADS, 0, d.getStream()>>>(
        GatherIn, ScatterOut, KernelOffset, numIndexPtr, outputIndex, outputIndexPadCumSum, hashOut);
    CHECK_CUDA_ERR();
  }
}

template <size_t NDim, class Index>
size_t createSparseSubMIndexMalloc(const GPU& d,
                                   const size_t numInput,
                                   const Vec<NDim, Index>& kernelSize,
                                   const bool oneTimeMalloc) {
  size_t reqBytes_, reqBytes = 0;
  size_t kVol = 1;
#pragma unroll
  for (size_t i = 0; i < NDim; i++) { kVol *= kernelSize[i]; }
  size_t outElemNumMaxPower2 = getMinPower2(numInput * kVol);
  size_t numElemHash = getMinPower2(numInput * hashSpace);

  cudaError_t status;
  for (size_t scale = oneTimeMalloc ? 1 : outElemNumMaxPower2; scale <= outElemNumMaxPower2; scale = scale << 1) {
    status = DEPLOY3D_CUB_NS_QUALIFIER::cub::DeviceScan::ExclusiveSum<
        DEPLOY3D_CUB_NS_QUALIFIER::cub::TransformInputIterator<Index, Valid_Counter, mkU<Index>*>, Index*>(
        nullptr, reqBytes_, {nullptr, {}}, nullptr, scale, d.getStream());
    CHECK_RETURN_STATUS(status);
    reqBytes = reqBytes > reqBytes_ ? reqBytes : reqBytes_;
  }

  status = DEPLOY3D_CUB_NS_QUALIFIER::cub::DeviceScan::InclusiveSum<Index*, Index*>(nullptr, reqBytes_, nullptr,
                                                                                    nullptr, kVol, d.getStream());
  CHECK_RETURN_STATUS(status);
  reqBytes = reqBytes > reqBytes_ ? reqBytes : reqBytes_;

  reqBytes += (2 * numElemHash + 4 * outElemNumMaxPower2) * sizeof(Index);
  return reqBytes;
}

template <size_t NDim, class Index>
void createSparseSubMIndex(const GPU& d,
                           Ref1D<uint8_t>& workingStorage,
                           Ref1D<Index>& GatherIn,
                           Ref1D<Index>& ScatterOut,
                           Ref1D<Index>& KernelOffset,
                           Index* numIndexPtr,
                           const Ref2D<Index>& coorsIn,
                           const Vec<NDim, Index>& kernelSize,
                           const Vec<NDim, Index>& stride,
                           const Vec<NDim, Index>& padding,
                           const Vec<NDim, Index>& dilation,
                           const Size<NDim + 1>& outSpatialShape) {
  Index numActIn = coorsIn.size(0);
  if (numActIn == 0) {
    cudaMemsetAsync(numIndexPtr, 0, sizeof(Index), d.getStream());
    return;
  }

  ssize_t kVol = 1;
#pragma unroll
  for (size_t i = 0; i < NDim; i++) { kVol *= kernelSize[i]; }
  ssize_t outElemNumMaxPower2 = getMinPower2(kVol * numActIn);
  ssize_t numElemHash = getMinPower2(numActIn * hashSpace);

  uint8_t* workingStoragePtr = workingStorage.data();

  Ref1D<Index> hashKeys(reinterpret_cast<Index*>(workingStoragePtr), {numElemHash});
  workingStoragePtr += hashKeys.numby();
  Ref1D<Index> hashValues(reinterpret_cast<Index*>(workingStoragePtr), {numElemHash});
  workingStoragePtr += hashValues.numby();

  typename kernel::SubMHashWriter<NDim, Index>::HashTable hash(hashKeys.data(), hashValues.data(), numElemHash);

  Ref2D<mkU<Index>> hashOut(reinterpret_cast<mkU<Index>*>(workingStoragePtr), {kVol, numActIn});
  workingStoragePtr += outElemNumMaxPower2 * sizeof(mkU<Index>);

  Ref2D<Index> outputIndex(reinterpret_cast<Index*>(workingStoragePtr), {kVol, numActIn});
  workingStoragePtr += outElemNumMaxPower2 * sizeof(Index);

  Ref1D<Index> outputIndexPad(reinterpret_cast<Index*>(workingStoragePtr), {kVol});
  workingStoragePtr += outElemNumMaxPower2 * sizeof(Index);

  Ref1D<Index> outputIndexPadCumSum(reinterpret_cast<Index*>(workingStoragePtr), {kVol});
  workingStoragePtr += outElemNumMaxPower2 * sizeof(Index);

  size_t workingStorageBytes = workingStorage.size(0) - (workingStoragePtr - workingStorage.data());

  {  // subM conv I-O geometry
    cudaMemsetAsync(hashOut.data(), 0xFF, outElemNumMaxPower2 * sizeof(mkU<Index>), d.getStream());
    kernel::resetHashKernel<typename kernel::SubMHashWriter<NDim, Index>::HashTable>
        <<<getBlocks(hash.size()), CUDA_NUM_THREADS, 0, d.getStream()>>>(hash);
    kernel::setSubMHashKernel<NDim, Index>
        <<<getBlocks(numActIn), CUDA_NUM_THREADS, 0, d.getStream()>>>(hash, coorsIn, outSpatialShape);
    kernel::subMGeometryKernel<NDim, Index><<<getBlocks(numActIn), CUDA_NUM_THREADS, 0, d.getStream()>>>(
        hashOut, hash, coorsIn, kernelSize, stride, padding, dilation, outSpatialShape);
    CHECK_CUDA_ERR();
  }
  {  // make bufferFromIn and bufferOffset
    cudaMemsetAsync(GatherIn.data(), 0xFF, GatherIn.numby(), d.getStream());
    cudaMemsetAsync(ScatterOut.data(), 0xFF, ScatterOut.numby(), d.getStream());
    cudaMemsetAsync(KernelOffset.data(), 0xFF, KernelOffset.numby(), d.getStream());
    DEPLOY3D_CUB_NS_QUALIFIER::cub::DeviceScan::ExclusiveSum(
        workingStoragePtr, workingStorageBytes,
        DEPLOY3D_CUB_NS_QUALIFIER::cub::TransformInputIterator<Index, Valid_Counter, mkU<Index>*>(hashOut.data(), {}),
        outputIndex.data(), outElemNumMaxPower2, d.getStream());
    kernel::outputIndexPadOrganizeKernel<Index>
        <<<getBlocks(kVol), CUDA_NUM_THREADS, 0, d.getStream()>>>(outputIndexPad, outputIndex);
    DEPLOY3D_CUB_NS_QUALIFIER::cub::DeviceScan::InclusiveSum<Index*, Index*>(
        workingStoragePtr, workingStorageBytes, outputIndexPad.data(), outputIndexPadCumSum.data(), kVol,
        d.getStream());
    kernel::indexOrganizeKernel<Index><<<getBlocks(hashOut.numel()), CUDA_NUM_THREADS, 0, d.getStream()>>>(
        GatherIn, ScatterOut, KernelOffset, numIndexPtr, outputIndex, outputIndexPadCumSum, hashOut);
    CHECK_CUDA_ERR();
  }
  CHECK_CUDA_ERR();
};

}  // namespace func
}  // namespace spconv

#define SPECIALIZE(NDim, Index)                                                                                        \
  template size_t spconv::func::createSparseConvIndexMalloc<NDim, Index>(                                              \
      const GPU& d, const size_t numInput, const Vec<NDim, Index>& kernelSize, const bool oneTimeMalloc);              \
                                                                                                                       \
  template void spconv::func::createSparseConvIndex<NDim, Index>(                                                      \
      const GPU& d, Ref1D<uint8_t>& workingStorage, Ref1D<Index>& GatherIn, Ref1D<Index>& ScatterOut,                  \
      Ref1D<Index>& KernelOffset, Ref2D<Index>& coorsOut, Index* numIndexPtr, Index* numOutPtr,                        \
      const Ref2D<Index>& coorsIn, const Vec<NDim, Index>& kernelSize, const Vec<NDim, Index>& stride,                 \
      const Vec<NDim, Index>& padding, const Vec<NDim, Index>& dilation, const Size<NDim + 1>& outSpatialShape,        \
      const bool transpose, const Index numSample);                                                                    \
                                                                                                                       \
  template size_t spconv::func::createSparseSubMIndexMalloc<NDim, Index>(                                              \
      const GPU& d, const size_t numInput, const Vec<NDim, Index>& kernelSize, const bool oneTimeMalloc);              \
                                                                                                                       \
  template void spconv::func::createSparseSubMIndex<NDim, Index>(                                                      \
      const GPU& d, Ref1D<uint8_t>& workingStorage, Ref1D<Index>& GatherIn, Ref1D<Index>& ScatterOut,                  \
      Ref1D<Index>& KernelOffset, Index* numIndexPtr, const Ref2D<Index>& coorsIn, const Vec<NDim, Index>& kernelSize, \
      const Vec<NDim, Index>& stride, const Vec<NDim, Index>& padding, const Vec<NDim, Index>& dilation,               \
      const Size<NDim + 1>& outSpatialShape);

SPECIALIZE(1, int);
SPECIALIZE(2, int);
SPECIALIZE(3, int);
SPECIALIZE(4, int);