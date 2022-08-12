#pragma once
#include <cuda_runtime.h>

namespace utils {
namespace launch {

template <typename T1, typename T2> inline ssize_t DivUp(const T1 a, const T2 b)
{
  return (a + b - 1) / b;
}

template <typename T1, typename T2> inline ssize_t Min(const T1 a, const T2 b)
{
  return a < b ? a : b;
}

constexpr ssize_t CUDA_NUM_THREADS = 128;
constexpr ssize_t CUDA_NUM_BLOCKS = 50000;

inline ssize_t getBlocks(const ssize_t N)
{
  return Min(CUDA_NUM_BLOCKS, DivUp(N, CUDA_NUM_THREADS));
}

template <typename T> class KernelLoop {
  struct Iterator
  {
    __forceinline__ __device__ Iterator(T index, T delta) : index_(index), delta_(delta) {}
    __forceinline__ __device__ T operator*() const { return index_; }
    __forceinline__ __device__ Iterator& operator++()
    {
      index_ += delta_;
      return *this;
    }
    __forceinline__ __device__ bool operator!=(const Iterator& other) const
    {
      bool greater = index_ > other.index_;
      bool less = index_ < other.index_;
      // Anything past an end iterator (delta_ == 0) is equal.
      // In range-based for loops, this optimizes to 'return less'.
      if (!other.delta_) { return less; }
      if (!delta_) { return greater; }
      return less || greater;
    }

   private:
    T index_;
    const T delta_;
  };

 public:
  __forceinline__ __device__ KernelLoop(T begin, T delta, T end) : begin_(begin), delta_(delta), end_(end) {}

  __forceinline__ __device__ Iterator begin() const { return Iterator{begin_, delta_}; }
  __forceinline__ __device__ Iterator end() const { return Iterator{end_, 0}; }

 private:
  T begin_;
  T delta_;
  T end_;
};

template <typename T> __forceinline__ __device__ KernelLoop<T> KernelLoopX(T count)
{
  return KernelLoop<T>(blockIdx.x * blockDim.x + threadIdx.x, gridDim.x * blockDim.x, count);
}

template <typename T> __forceinline__ __device__ KernelLoop<T> KernelLoopY(T count)
{
  return KernelLoop<T>(blockIdx.y * blockDim.y + threadIdx.y, gridDim.y * blockDim.y, count);
}

template <typename T> __forceinline__ __device__ KernelLoop<T> KernelLoopZ(T count)
{
  return KernelLoop<T>(blockIdx.z * blockDim.z + threadIdx.z, gridDim.z * blockDim.z, count);
}
}  // namespace launch
}  // namespace utils