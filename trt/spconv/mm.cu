#include "common/launch.cuh"
#include "common/refnd.h"
#include <cublas_v2.h>
#include <cuda_fp16.hpp>

using utils::GPU;
using utils::launch::CUDA_NUM_THREADS;
using utils::launch::getBlocks;
using utils::launch::KernelLoopX;
using utils::nd::Ref1D;
using utils::nd::Ref2D;
using utils::nd::Ref3D;

namespace spconv {
namespace kernel {
template <class T, class Index>
__global__ void
gatherFeats(Ref2D<T> outFeats, const Ref2D<T> inFeats, const Ref1D<Index> bufferFromIn, const Ref1D<Index> bufferOffset)
{
  size_t numFeats = outFeats.size(1);
  size_t numOutElem = bufferFromIn.size(0) * numFeats;
  for (size_t ix : KernelLoopX<size_t>(numOutElem)) {
    auto rowId = ix / numFeats;
    auto colId = ix % numFeats;
    outFeats(bufferOffset(rowId), colId) = inFeats(bufferFromIn(rowId), colId);
  }
}

template <class T, class Index>
__global__ void gatherFeats(Ref2D<T> outFeats,
                            const Index numBufToSkip,
                            const Index numBufBeforeCenter,
                            const Ref2D<T> inFeats,
                            const Ref1D<Index> bufferFromIn,
                            const Ref1D<Index> bufferOffset)
{
  size_t numFeats = outFeats.size(1);
  size_t numOutElem = (bufferFromIn.size(0) - numBufToSkip) * numFeats;
  for (size_t ix : KernelLoopX<size_t>(numOutElem)) {
    auto rowId = ix / numFeats;
    auto colId = ix % numFeats;
    if (rowId >= numBufBeforeCenter) rowId += numBufToSkip;
    outFeats(bufferOffset(rowId), colId) = inFeats(bufferFromIn(rowId), colId);
  }
}

template <class T> __global__ void reduceInitFeats(Ref2D<T> outFeats, const Ref1D<T> initVec)
{
  size_t numOuts = outFeats.size(0);
  size_t numFeats = outFeats.size(1);
  size_t numOutElem = numOuts * numFeats;
  for (size_t ix : KernelLoopX<size_t>(numOutElem)) {
    if (ix < numOutElem) outFeats[ix] = initVec[ix % numFeats];
  }
}

template <class T> __global__ void reduceInitFeats(Ref2D<T> outFeats, const T initVal)
{
  size_t numOuts = outFeats.size(0);
  size_t numFeats = outFeats.size(1);
  size_t numOutElem = numOuts * numFeats;
  for (size_t ix : KernelLoopX<size_t>(numOutElem)) { outFeats[ix] = initVal; }
}

template <class T, class Index, class OP>
__global__ void atomicReduceFeats(Ref2D<T> outFeats,
                                  const Ref2D<T> inFeats,
                                  const Ref1D<Index> bufferToOut,
                                  const Ref1D<Index> bufferOffset)
{
  size_t numFeats = outFeats.size(1);
  size_t numInElem = bufferToOut.size(0) * numFeats;
  for (size_t ix : KernelLoopX<size_t>(numInElem)) {
    auto rowId = ix / numFeats;
    auto colId = ix % numFeats;
    OP()
    (&outFeats(bufferToOut(rowId), colId), inFeats(bufferOffset(rowId), colId));
  }
}

template <class T, class Index, class OP>
__global__ void atomicReduceFeats(Ref2D<T> outFeats,
                                  const Index numBufToSkip,
                                  const Index numBufBeforeCenter,
                                  const Ref2D<T> inFeats,
                                  const Ref1D<Index> bufferToOut,
                                  const Ref1D<Index> bufferOffset)
{
  size_t numFeats = outFeats.size(1);
  size_t numInElem = (bufferToOut.size(0) - numBufToSkip) * numFeats;
  for (size_t ix : KernelLoopX<size_t>(numInElem)) {
    auto rowId = ix / numFeats;
    auto colId = ix % numFeats;
    if (rowId >= numBufBeforeCenter) rowId += numBufToSkip;
    OP()
    (&outFeats(bufferToOut(rowId), colId), inFeats(bufferOffset(rowId), colId));
  }
}

template <class T, class Index, class OP>
__global__ void segmentReduceFeats(Ref2D<T> outFeats,
                                   const Index segmentBegin,
                                   const Index segmentLen,
                                   const Ref2D<T> inFeats,
                                   const Ref1D<Index> bufferToOut,
                                   const Ref1D<Index> bufferOffset)
{
  size_t numFeats = outFeats.size(1);
  for (size_t ix : KernelLoopX<size_t>(segmentLen * numFeats)) {
    auto rowId = segmentBegin + ix / numFeats;
    auto colId = ix % numFeats;
    OP()
    (&outFeats(bufferToOut(rowId), colId), inFeats(bufferOffset(rowId), colId));
  }
}
}  // namespace kernel

namespace reduce {

template <class T> struct NoAtomSum
{
  HOST_DEVICE_INLINE void operator()(T* addr, T val) const { *addr += val; }
};

template <class T> struct NoAtomMax
{
  HOST_DEVICE_INLINE void operator()(T* addr, T val) const
  {
    auto ori = *addr;
    *addr = max(ori, val);
  }
};

template <class T> struct AtomSum
{
  HOST_DEVICE_INLINE void operator()(T* addr, T val) const { atomicAdd(addr, val); }
};

template <class T> struct AtomMax;

template <> struct AtomMax<double>
{
  DEVICE_INLINE void operator()(double* addr, double val) const
  {
    auto* address_as_ull = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old = *address_as_ull, assumed;
    do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed,
                      static_cast<unsigned long long>(
                          __double_as_longlong(fmax(val, __longlong_as_double(static_cast<long long>(assumed))))));
    } while (assumed != old || __longlong_as_double(static_cast<long long>(old)) < val);
  }
};

template <> struct AtomMax<float>
{
  DEVICE_INLINE void operator()(float* addr, float val) const
  {
    auto* address_as_i = reinterpret_cast<int32_t*>(addr);
    int32_t old = *address_as_i, assumed;
    do {
      assumed = old;
      old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old || __int_as_float(old) < val);
  }
};

template <> struct AtomMax<half>
{
  DEVICE_INLINE void operator()(half* addr, half val) const
  {
    auto* address_as_s = reinterpret_cast<uint16_t*>(addr);
    uint16_t old = *address_as_s, assumed;
    do {
      assumed = old;
      half hassumed = __ushort_as_half(assumed);
      old = atomicCAS(address_as_s, assumed, __half_as_ushort(__hgt(val, hassumed) ? val : hassumed));
    } while (assumed != old || __ushort_as_half(old) < val);
  }
};
}  // namespace reduce
namespace blas {

template <class T> struct DeviceZeroOne
{
  T d[2];
  T* devPtr;
  DeviceZeroOne()
  {
    d[0] = T(0.0);
    d[1] = T(1.0);
    cudaMalloc(&devPtr, sizeof(d));
    cudaMemcpy(devPtr, d, sizeof(d), cudaMemcpyHostToDevice);
  }
  ~DeviceZeroOne() { cudaFree(devPtr); }
  constexpr T* zeroPtr() { return devPtr; }
  constexpr T* onePtr() { return devPtr + 1; }
};

template <class T> struct cublasConfigs;
template <> struct cublasConfigs<double>
{
  static constexpr cublasComputeType_t computeType = CUBLAS_COMPUTE_64F;
  static constexpr cudaDataType dataType = CUDA_R_64F;
  static constexpr cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
};
template <> struct cublasConfigs<float>
{
  static constexpr cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
  static constexpr cudaDataType dataType = CUDA_R_32F;
  static constexpr cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
};
template <> struct cublasConfigs<half>
{
  static constexpr cublasComputeType_t computeType = CUBLAS_COMPUTE_16F;
  static constexpr cudaDataType dataType = CUDA_R_16F;
  static constexpr cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
};

template <class T>
__inline__ cublasStatus_t MM(cublasHandle_t handle,
                             cublasOperation_t transa,
                             cublasOperation_t transb,
                             int m,
                             int n,
                             int k,
                             const T* alpha,
                             const T* A,
                             int lda,
                             const T* B,
                             int ldb,
                             const T* beta,
                             T* C,
                             int ldc)
{
  return cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, cublasConfigs<T>::dataType, lda, B,
                      cublasConfigs<T>::dataType, ldb, beta, C, cublasConfigs<T>::dataType, ldc,
                      cublasConfigs<T>::computeType, cublasConfigs<T>::algo);
}

template <class T>
__inline__ cublasStatus_t bMM(cublasHandle_t handle,
                              cublasOperation_t transa,
                              cublasOperation_t transb,
                              int m,
                              int n,
                              int k,
                              const T* alpha,
                              const T* A,
                              int lda,
                              ssize_t strideA,
                              const T* B,
                              int ldb,
                              ssize_t strideB,
                              const T* beta,
                              T* C,
                              int ldc,
                              ssize_t strideC,
                              size_t batchCount)
{
  return cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, alpha, A, cublasConfigs<T>::dataType, lda, strideA,
                                    B, cublasConfigs<T>::dataType, ldb, strideB, beta, C, cublasConfigs<T>::dataType,
                                    ldc, strideC, batchCount, cublasConfigs<T>::computeType, cublasConfigs<T>::algo);
}
}  // namespace blas

namespace func {
template <size_t FeatsPerThread, class T, class Index>
void indexConv(const GPU& d,
               Ref2D<T>& bufMMIn,
               Ref2D<T>& bufMMOut,
               Ref2D<T>& outFeats,
               const Ref2D<T>& inFeats,
               const Ref3D<T>& filters,
               const Ref1D<T>& bias,
               const Ref1D<Index>& bufferFromIn,
               const Ref1D<Index>& bufferToOut,
               const Ref1D<Index>& bufferOffset,
               const Ref1D<Index>& bufferKernelNumHost)
{
  static blas::DeviceZeroOne<T> Consts;
  size_t kVol = filters.size(0);
  size_t inNum = inFeats.size(0);
  size_t inNumFeats = inFeats.size(1);
  size_t outNumFeats = outFeats.size(1);
  size_t validBufNum = bufferFromIn.size(0);

  if (validBufNum == 0) return;

  ssize_t numThreads;

  if (bias.empty()) { cudaMemsetAsync(outFeats.data(), 0, outFeats.numel() * sizeof(T), d.getStream()); }
  else {
    numThreads = (outFeats.numel() + FeatsPerThread - 1) / FeatsPerThread;
    kernel::reduceInitFeats<T><<<getBlocks(numThreads), CUDA_NUM_THREADS, 0, d.getStream()>>>(outFeats, bias);
  }

  numThreads = (validBufNum * inNumFeats + FeatsPerThread - 1) / FeatsPerThread;
  kernel::gatherFeats<T, Index>
      <<<getBlocks(numThreads), CUDA_NUM_THREADS, 0, d.getStream()>>>(bufMMIn, inFeats, bufferFromIn, bufferOffset);
  CHECK_CUDA_ERR();

  size_t strideBufMMIn = inNumFeats * inNum;
  size_t strideBufMMOut = outNumFeats * inNum;
  size_t strideFilter = filters.stride(0);

  size_t convInNumBest = 0;
  for (int i = 0; i < kVol; i++) {
    auto numBufferKernel = bufferKernelNumHost[i];
    if (numBufferKernel > convInNumBest) convInNumBest = numBufferKernel;
  }

  cudaStream_t oldStream;
  cublasHandle_t cublasHandle = d.getCublasHandle();
  cublasGetStream(cublasHandle, &oldStream);
  cublasSetStream(cublasHandle, d.getStream());
  cublasPointerMode_t oldPtrMode;
  cublasGetPointerMode(cublasHandle, &oldPtrMode);
  cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_DEVICE);

  blas::bMM(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, outNumFeats, convInNumBest, inNumFeats, Consts.onePtr(),
            filters.data(), outNumFeats, strideFilter, bufMMIn.data(), inNumFeats, strideBufMMIn, Consts.zeroPtr(),
            bufMMOut.data(), outNumFeats, strideBufMMOut, kVol);

  cublasSetPointerMode(cublasHandle, oldPtrMode);
  cublasSetStream(cublasHandle, oldStream);

  numThreads = (validBufNum * outNumFeats + FeatsPerThread - 1) / FeatsPerThread;
  kernel::atomicReduceFeats<T, Index, reduce::AtomSum<T>>
      <<<getBlocks(numThreads), CUDA_NUM_THREADS, 0, d.getStream()>>>(outFeats, bufMMOut, bufferToOut, bufferOffset);
  CHECK_CUDA_ERR();
}

template <size_t FeatsPerThread, class T, class Index>
void indexSubM(const GPU& d,
               Ref2D<T>& bufMMIn,
               Ref2D<T>& bufMMOut,
               Ref2D<T>& outFeats,
               const Ref2D<T>& inFeats,
               const Ref3D<T>& filters,
               const Ref1D<T>& bias,
               const Ref1D<Index>& bufferFromIn,
               const Ref1D<Index>& bufferToOut,
               const Ref1D<Index>& bufferOffset,
               const Ref1D<Index>& bufferKernelNumHost)
{
  static blas::DeviceZeroOne<T> Consts;
  size_t kVol = filters.size(0);
  size_t inNum = inFeats.size(0);
  size_t inNumFeats = inFeats.size(1);
  size_t outNumFeats = outFeats.size(1);
  size_t validBufNum = bufferFromIn.size(0);

  if (validBufNum == 0) return;

  ssize_t numThreads;

  if (bias.empty()) { cudaMemsetAsync(outFeats.data(), 0, outFeats.numel() * sizeof(T), d.getStream()); }
  else {
    numThreads = (outFeats.numel() + FeatsPerThread - 1) / FeatsPerThread;
    kernel::reduceInitFeats<T><<<getBlocks(numThreads), CUDA_NUM_THREADS, 0, d.getStream()>>>(outFeats, bias);
  }

  size_t strideBufMMIn = inNumFeats * inNum;
  size_t strideBufMMOut = outNumFeats * inNum;
  size_t strideFilter = filters.stride(0);

  size_t center = kVol / 2;
  size_t convInNumBest = 0;
  size_t numBufBeforeCenter = 0;
  for (int i = 0; i < kVol; i++) {
    auto numBufferKernel = bufferKernelNumHost[i];
    if (i < center) numBufBeforeCenter += numBufferKernel;
    if ((i != center) && (numBufferKernel > convInNumBest)) convInNumBest = numBufferKernel;
  }

  numThreads = ((validBufNum - inNum) * inNumFeats + FeatsPerThread - 1) / FeatsPerThread;
  if (numThreads) {
    kernel::gatherFeats<T, Index><<<getBlocks(numThreads), CUDA_NUM_THREADS, 0, d.getStream()>>>(
        bufMMIn, bufferKernelNumHost[center], numBufBeforeCenter, inFeats, bufferFromIn, bufferOffset);
    CHECK_CUDA_ERR();
  }

  cudaStream_t oldStream;
  cublasHandle_t cublasHandle = d.getCublasHandle();
  cublasGetStream(cublasHandle, &oldStream);
  cublasSetStream(cublasHandle, d.getStream());
  cublasPointerMode_t oldPtrMode;
  cublasGetPointerMode(cublasHandle, &oldPtrMode);
  cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_DEVICE);

  blas::bMM(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, outNumFeats, convInNumBest, inNumFeats, Consts.onePtr(),
            filters.data(), outNumFeats, strideFilter, bufMMIn.data(), inNumFeats, strideBufMMIn, Consts.zeroPtr(),
            bufMMOut.data(), outNumFeats, strideBufMMOut, kVol);

  blas::MM(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, outNumFeats, bufferKernelNumHost[center], inNumFeats,
           Consts.onePtr(), &filters(center, 0, 0), outNumFeats, inFeats.data(), inNumFeats, Consts.onePtr(),
           outFeats.data(), outNumFeats);

  cublasSetPointerMode(cublasHandle, oldPtrMode);
  cublasSetStream(cublasHandle, oldStream);

  numThreads = ((validBufNum - inNum) * outNumFeats + FeatsPerThread - 1) / FeatsPerThread;
  if (numThreads) {
    kernel::atomicReduceFeats<T, Index, reduce::AtomSum<T>>
        <<<getBlocks(numThreads), CUDA_NUM_THREADS, 0, d.getStream()>>>(outFeats, inNum, numBufBeforeCenter, bufMMOut,
                                                                        bufferToOut, bufferOffset);
    CHECK_CUDA_ERR();
  }
}

template <size_t FeatsPerThread, class T, class Index>
void indexConvBP(const GPU& d,
                 Ref2D<T>& bufMMIn,
                 Ref2D<T>& bufMMOut,
                 Ref2D<T>& inFeatsGrad,
                 Ref3D<T>& filtersGrad,
                 Ref1D<T>& biasGrad,
                 const Ref2D<T>& inFeats,
                 const Ref3D<T>& filters,
                 const Ref2D<T>& outFeatsGrad,
                 const Ref1D<Index>& bufferFromIn,
                 const Ref1D<Index>& bufferToOut,
                 const Ref1D<Index>& bufferOffset,
                 const Ref1D<Index>& bufferKernelNumHost)
{
  static blas::DeviceZeroOne<T> Consts;
  size_t kVol = filters.size(0);
  size_t inNum = inFeats.size(0);
  size_t inNumFeats = inFeats.size(1);
  size_t outNumFeats = outFeatsGrad.size(1);
  size_t validBufNum = bufferFromIn.size(0);

  if (inFeatsGrad.numel()) cudaMemsetAsync(inFeatsGrad.data(), 0, inFeatsGrad.numel() * sizeof(T), d.getStream());

  if (validBufNum == 0) {
    if (filtersGrad.numel()) cudaMemsetAsync(filtersGrad.data(), 0, filtersGrad.numel() * sizeof(T), d.getStream());
    return;
  }

  size_t strideBufMMIn = inNumFeats * inNum;
  size_t strideBufMMOut = outNumFeats * inNum;
  size_t strideFilter = filters.stride(0);

  size_t convInNumBest = 0;
  for (int i = 0; i < kVol; i++) {
    auto numBufferKernel = bufferKernelNumHost[i];
    if (numBufferKernel > convInNumBest) convInNumBest = numBufferKernel;
  }

  cudaMemset2DAsync(bufMMIn.data(), strideBufMMIn * sizeof(T), 0, inNumFeats * convInNumBest * sizeof(T), kVol,
                    d.getStream());

  cudaMemset2DAsync(bufMMOut.data(), strideBufMMOut * sizeof(T), 0, outNumFeats * convInNumBest * sizeof(T), kVol,
                    d.getStream());

  ssize_t numThreads;

  if (!biasGrad.empty()) {
    // todo : sum over the columns of outFeatsGrad to get the biasGrad. it is
    //  not calculate currently.
  }

  numThreads = (validBufNum * outNumFeats + FeatsPerThread - 1) / FeatsPerThread;
  kernel::gatherFeats<T, Index><<<getBlocks(numThreads), CUDA_NUM_THREADS, 0, d.getStream()>>>(
      bufMMOut, outFeatsGrad, bufferToOut, bufferOffset);
  numThreads = (validBufNum * inNumFeats + FeatsPerThread - 1) / FeatsPerThread;
  kernel::gatherFeats<T, Index>
      <<<getBlocks(numThreads), CUDA_NUM_THREADS, 0, d.getStream()>>>(bufMMIn, inFeats, bufferFromIn, bufferOffset);
  CHECK_CUDA_ERR();

  cudaStream_t oldStream;
  cublasHandle_t cublasHandle = d.getCublasHandle();
  cublasGetStream(cublasHandle, &oldStream);
  cublasSetStream(cublasHandle, d.getStream());
  cublasPointerMode_t oldPtrMode;
  cublasGetPointerMode(cublasHandle, &oldPtrMode);
  cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_DEVICE);

  blas::bMM(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, outNumFeats, inNumFeats, convInNumBest, Consts.onePtr(),
            bufMMOut.data(), outNumFeats, strideBufMMOut, bufMMIn.data(), inNumFeats, strideBufMMIn, Consts.zeroPtr(),
            filtersGrad.data(), outNumFeats, strideFilter, kVol);

  blas::bMM(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, inNumFeats, convInNumBest, outNumFeats, Consts.onePtr(),
            filters.data(), outNumFeats, strideFilter, bufMMOut.data(), outNumFeats, strideBufMMOut, Consts.zeroPtr(),
            bufMMIn.data(), inNumFeats, strideBufMMIn, kVol);

  cublasSetPointerMode(cublasHandle, oldPtrMode);
  cublasSetStream(cublasHandle, oldStream);

  numThreads = (validBufNum * outNumFeats + FeatsPerThread - 1) / FeatsPerThread;
  kernel::atomicReduceFeats<T, Index, reduce::AtomSum<T>>
      <<<getBlocks(numThreads), CUDA_NUM_THREADS, 0, d.getStream()>>>(inFeatsGrad, bufMMIn, bufferFromIn, bufferOffset);
  CHECK_CUDA_ERR();
}

template <size_t FeatsPerThread, class T, class Index>
void indexSubMBP(const GPU& d,
                 Ref2D<T>& bufMMIn,
                 Ref2D<T>& bufMMOut,
                 Ref2D<T>& inFeatsGrad,
                 Ref3D<T>& filtersGrad,
                 Ref1D<T>& biasGrad,
                 const Ref2D<T>& inFeats,
                 const Ref3D<T>& filters,
                 const Ref2D<T>& outFeatsGrad,
                 const Ref1D<Index>& bufferFromIn,
                 const Ref1D<Index>& bufferToOut,
                 const Ref1D<Index>& bufferOffset,
                 const Ref1D<Index>& bufferKernelNumHost)
{
  static blas::DeviceZeroOne<T> Consts;
  size_t kVol = filters.size(0);
  size_t inNum = inFeats.size(0);
  size_t inNumFeats = inFeats.size(1);
  size_t outNumFeats = outFeatsGrad.size(1);
  size_t validBufNum = bufferFromIn.size(0);

  if (validBufNum == 0) {
    if (inFeatsGrad.numel()) cudaMemsetAsync(inFeatsGrad.data(), 0, inFeatsGrad.numel() * sizeof(T), d.getStream());
    if (filtersGrad.numel()) cudaMemsetAsync(filtersGrad.data(), 0, filtersGrad.numel() * sizeof(T), d.getStream());
    return;
  }

  size_t strideBufMMIn = inNumFeats * inNum;
  size_t strideBufMMOut = outNumFeats * inNum;
  size_t strideFilter = filters.stride(0);

  size_t convInNumBest = 0;
  size_t numBufBeforeCenter = 0;
  for (int i = 0; i < kVol; i++) {
    auto numBufferKernel = bufferKernelNumHost[i];
    if (i < kVol / 2) numBufBeforeCenter += numBufferKernel;
    if ((i != kVol / 2) && (numBufferKernel > convInNumBest)) convInNumBest = numBufferKernel;
  }

  if (inNumFeats < outNumFeats) {
    cudaMemset2DAsync(bufMMIn.data(), strideBufMMIn * sizeof(T), 0, inNumFeats * convInNumBest * sizeof(T), kVol,
                      d.getStream());
  }
  else {
    cudaMemset2DAsync(bufMMOut.data(), strideBufMMOut * sizeof(T), 0, outNumFeats * convInNumBest * sizeof(T), kVol,
                      d.getStream());
  }

  ssize_t numThreads;

  if (!biasGrad.empty()) {
    // todo : sum over the columns of outFeatsGrad to get the biasGrad. it is
    //  not calculate currently.
  }

  numThreads = ((validBufNum - inNum) * outNumFeats + FeatsPerThread - 1) / FeatsPerThread;
  if (numThreads) {
    kernel::gatherFeats<T, Index><<<getBlocks(numThreads), CUDA_NUM_THREADS, 0, d.getStream()>>>(
        bufMMOut, inNum, numBufBeforeCenter, outFeatsGrad, bufferToOut, bufferOffset);
  }
  numThreads = ((validBufNum - inNum) * inNumFeats + FeatsPerThread - 1) / FeatsPerThread;
  if (numThreads) {
    kernel::gatherFeats<T, Index><<<getBlocks(numThreads), CUDA_NUM_THREADS, 0, d.getStream()>>>(
        bufMMIn, inNum, numBufBeforeCenter, inFeats, bufferFromIn, bufferOffset);
  }
  CHECK_CUDA_ERR();

  cudaStream_t oldStream;
  cublasHandle_t cublasHandle = d.getCublasHandle();
  cublasGetStream(cublasHandle, &oldStream);
  cublasSetStream(cublasHandle, d.getStream());
  cublasPointerMode_t oldPtrMode;
  cublasGetPointerMode(cublasHandle, &oldPtrMode);
  cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_DEVICE);

  blas::bMM(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, outNumFeats, inNumFeats, convInNumBest, Consts.onePtr(),
            bufMMOut.data(), outNumFeats, strideBufMMOut, bufMMIn.data(), inNumFeats, strideBufMMIn, Consts.zeroPtr(),
            filtersGrad.data(), outNumFeats, strideFilter, kVol);

  size_t center = kVol / 2;
  blas::MM(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, outNumFeats, inNumFeats, bufferKernelNumHost[center],
           Consts.onePtr(), outFeatsGrad.data(), outNumFeats, inFeats.data(), inNumFeats, Consts.zeroPtr(),
           &filtersGrad(center, 0, 0), outNumFeats);

  blas::bMM(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, inNumFeats, convInNumBest, outNumFeats, Consts.onePtr(),
            filters.data(), outNumFeats, strideFilter, bufMMOut.data(), outNumFeats, strideBufMMOut, Consts.zeroPtr(),
            bufMMIn.data(), inNumFeats, strideBufMMIn, kVol);

  blas::MM(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, inNumFeats, bufferKernelNumHost[center], outNumFeats,
           Consts.onePtr(), &filters(center, 0, 0), outNumFeats, outFeatsGrad.data(), outNumFeats, Consts.zeroPtr(),
           inFeatsGrad.data(), inNumFeats);

  cublasSetPointerMode(cublasHandle, oldPtrMode);
  cublasSetStream(cublasHandle, oldStream);

  numThreads = ((validBufNum - inNum) * outNumFeats + FeatsPerThread - 1) / FeatsPerThread;
  if (numThreads) {
    kernel::atomicReduceFeats<T, Index, reduce::AtomSum<T>>
        <<<getBlocks(numThreads), CUDA_NUM_THREADS, 0, d.getStream()>>>(inFeatsGrad, inNum, numBufBeforeCenter, bufMMIn,
                                                                        bufferFromIn, bufferOffset);
  }
  CHECK_CUDA_ERR();
}

}  // namespace func
}  // namespace spconv

#define SPECIALIZE_CONV(FeatsPerThread, T, Index)                                                                      \
  template void spconv::func::indexConv<FeatsPerThread, T, Index>(                                                     \
      const GPU& d, Ref2D<T>& bufMMIn, Ref2D<T>& bufMMOut, Ref2D<T>& outFeats, const Ref2D<T>& inFeats,                \
      const Ref3D<T>& filters, const Ref1D<T>& bias, const Ref1D<Index>& bufferFromIn,                                 \
      const Ref1D<Index>& bufferToOut, const Ref1D<Index>& bufferOffset, const Ref1D<Index>& bufferKernelNumHost);     \
  template void spconv::func::indexSubM<FeatsPerThread, T, Index>(                                                     \
      const GPU& d, Ref2D<T>& bufMMIn, Ref2D<T>& bufMMOut, Ref2D<T>& outFeats, const Ref2D<T>& inFeats,                \
      const Ref3D<T>& filters, const Ref1D<T>& bias, const Ref1D<Index>& bufferFromIn,                                 \
      const Ref1D<Index>& bufferToOut, const Ref1D<Index>& bufferOffset, const Ref1D<Index>& bufferKernelNumHost);     \
  template void spconv::func::indexConvBP<FeatsPerThread, T, Index>(                                                   \
      const GPU& d, Ref2D<T>& bufMMIn, Ref2D<T>& bufMMOut, Ref2D<T>& inFeatsGrad, Ref3D<T>& filtersGrad,               \
      Ref1D<T>& biasGrad, const Ref2D<T>& inFeats, const Ref3D<T>& filters, const Ref2D<T>& outFeatsGrad,              \
      const Ref1D<Index>& bufferFromIn, const Ref1D<Index>& bufferToOut, const Ref1D<Index>& bufferOffset,             \
      const Ref1D<Index>& bufferKernelNumHost);                                                                        \
  template void spconv::func::indexSubMBP<FeatsPerThread, T, Index>(                                                   \
      const GPU& d, Ref2D<T>& bufMMIn, Ref2D<T>& bufMMOut, Ref2D<T>& inFeatsGrad, Ref3D<T>& filtersGrad,               \
      Ref1D<T>& biasGrad, const Ref2D<T>& inFeats, const Ref3D<T>& filters, const Ref2D<T>& outFeatsGrad,              \
      const Ref1D<Index>& bufferFromIn, const Ref1D<Index>& bufferToOut, const Ref1D<Index>& bufferOffset,             \
      const Ref1D<Index>& bufferKernelNumHost)

SPECIALIZE_CONV(4, double, int);
SPECIALIZE_CONV(4, float, int);
SPECIALIZE_CONV(4, half, int);
