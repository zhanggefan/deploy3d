#pragma once
#include "macros.h"
#include <NvInfer.h>
#include <cutlass/half.h>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <memory>
#include <vector>

namespace utils {
namespace nd {

template <size_t Rank, class T> struct Vec {
  static_assert(Rank > 0, "error");
  static constexpr size_t rank = Rank;
  using value_type = T;
  HOST_DEVICE_INLINE T& operator[](ssize_t idx) { return array_[idx]; }
  HOST_DEVICE_INLINE const T& operator[](ssize_t idx) const { return array_[idx]; }
  HOST_DEVICE_INLINE ssize_t size() const { return Rank; };
  HOST_DEVICE_INLINE const T* data() const { return array_; };
  HOST_DEVICE_INLINE T* data() { return array_; };
  HOST_DEVICE_INLINE const T* cbegin() const { return array_; };
  HOST_DEVICE_INLINE const T* cend() const { return array_ + Rank; };
  HOST_DEVICE_INLINE T* begin() { return array_; };
  HOST_DEVICE_INLINE T* end() { return array_ + Rank; };

  T array_[Rank];
};

namespace indexing {
template <size_t Rank, class Index1, class Index2>
HOST_DEVICE_INLINE ssize_t is_valid(const Index1* const coor, const Index2* const size) {
#pragma unroll
  for (ssize_t i = 0; i < Rank; ++i) {
    if (coor[i] < 0 || (coor[i] > size[i] - 1)) return false;
  }
  return true;
}

HOST_DEVICE_INLINE ssize_t offset_args(const ssize_t* const stride, ssize_t index) {
  return index * stride[0];
}
template <class... Ts> HOST_DEVICE_INLINE ssize_t offset_args(const ssize_t* const stride, ssize_t index, Ts... inds) {
  return index * stride[0] + offset_args(stride + 1, inds...);
}

template <size_t Rank, class Index1, class Index2>
HOST_DEVICE_INLINE ssize_t offset(const Index1* const coor, const Index2* const size) {
  ssize_t off(coor[0]);
#pragma unroll
  for (ssize_t i = 1; i < Rank; ++i) { off = off * size[i] + coor[i]; }
  return off;
}

template <size_t Rank, class Index1, class Index2>
HOST_DEVICE_INLINE void deserialize(Index1* const coor, ssize_t offset, const Index2* const size) {
#pragma unroll
  for (ssize_t i = Rank - 1; i >= 0; --i) {
    auto s = size[i];
    coor[i] = offset % s;
    offset /= s;
  }
}

}  // namespace indexing

template <size_t Rank> class Size {
  static_assert(Rank > 0, "error");

 public:
  static constexpr size_t rank = Rank;
  using index_vec_t = Vec<Rank, ssize_t>;
  HOST_DEVICE_INLINE explicit Size(const index_vec_t& size) : size_(size) {
    stride_[Rank - 1] = 1;
#pragma unroll
    for (ssize_t i = Rank - 1; i >= 1; --i) stride_[i - 1] = size[i] * stride_[i];
  };
  Size(const Size<Rank>& size) = default;
  HOST_DEVICE_INLINE Size() = delete;
  HOST_DEVICE_INLINE const index_vec_t& sizes() const { return size_; }
  HOST_DEVICE_INLINE ssize_t size(ssize_t idx) const { return size_[idx]; }
  HOST_DEVICE_INLINE ssize_t stride(ssize_t idx) const { return stride_[idx]; }
  HOST_DEVICE_INLINE ssize_t numel() const { return size(0) * stride(0); }
  HOST_DEVICE_INLINE ssize_t& operator[](ssize_t idx) { return size_[idx]; }
  HOST_DEVICE_INLINE const ssize_t& operator[](ssize_t idx) const { return size_[idx]; }

  template <class Index> HOST_DEVICE_INLINE ssize_t is_valid(const Index* const coor) const {
    return indexing::is_valid<Rank>(coor, size_.data());
  }

  template <class Index> HOST_DEVICE_INLINE ssize_t is_valid(const Vec<Rank, Index>& coor) const {
    return indexing::is_valid<Rank>(coor.data(), size_.data());
  }

  template <class... Coors> HOST_DEVICE_INLINE ssize_t operator()(const Coors... coors) const {
    return indexing::offset_args(stride_.data(), coors...);
  }

  template <class Index> HOST_DEVICE_INLINE ssize_t offset(const Index* const coor) const {
#if defined(__CUDACC_DEBUG__) || defined(DEBUG)
#  pragma unroll
    for (ssize_t i = 0; i < Rank; i++)
      if (coor[i] >= size_[i] || coor[i] < 0)
        printf("index out of range! coor[%d] = %d but the valid range is 0 ~ %d\n", (int)i, (int)coor[i],
               (int)size_[i]);
#endif
    return indexing::offset<Rank>(coor, size_.data());
  }

  template <class Index> HOST_DEVICE_INLINE ssize_t offset(const Vec<Rank, Index>& coor) const {
#if defined(__CUDACC_DEBUG__) || defined(DEBUG)
#  pragma unroll
    for (ssize_t i = 0; i < Rank; i++)
      if (coor[i] >= size_[i] || coor[i] < 0)
        printf("index out of range! coor[%d] = %d but the valid range is 0 ~ %d\n", (int)i, (int)coor[i],
               (int)size_[i]);
#endif
    return indexing::offset<Rank>(coor.data(), size_.data());
  }

  template <class Index> HOST_DEVICE_INLINE void deserialize(Index* const coor, ssize_t offset) const {
    return indexing::deserialize<Rank>(coor, offset, size_.data());
  }

  template <class Index> HOST_DEVICE_INLINE void deserialize(Vec<Rank, Index>& coor, ssize_t offset) const {
    return indexing::deserialize<Rank>(coor.data(), offset, size_.data());
  }

 private:
  index_vec_t size_;
  index_vec_t stride_;
};

template <size_t Rank, class T> class RefND {
  using ptr_t = T*;
  using index_vec_t = Vec<Rank, ssize_t>;
  static_assert(Rank > 0, "error");
  static constexpr size_t rank = Rank;

 public:
  using value_type = T;
  HOST_DEVICE_INLINE RefND(ptr_t data, const index_vec_t& size) : data_(data), size_and_stride_(size){};
  HOST_DEVICE_INLINE RefND(ptr_t data, const Size<Rank>& size) : data_(data), size_and_stride_(size){};

  HOST_DEVICE_INLINE bool empty() const { return data_ == nullptr; }
  HOST_DEVICE_INLINE ptr_t data() { return data_; }
  HOST_DEVICE_INLINE ptr_t data() const { return data_; }
  HOST_DEVICE_INLINE const index_vec_t& sizes() const { return size_and_stride_.sizes(); }
  HOST_DEVICE_INLINE ssize_t size(ssize_t idx) const { return size_and_stride_.size(idx); }
  HOST_DEVICE_INLINE ssize_t stride(ssize_t idx) const { return size_and_stride_.stride(idx); }
  HOST_DEVICE_INLINE ssize_t numel() const { return size_and_stride_.size(0) * size_and_stride_.stride(0); }
  HOST_DEVICE_INLINE ssize_t numby() const { return size_and_stride_.size(0) * size_and_stride_.stride(0) * sizeof(T); }
  HOST_DEVICE_INLINE T& operator[](ssize_t idx) { return data_[idx]; }
  HOST_DEVICE_INLINE const T& operator[](ssize_t idx) const { return data_[idx]; }
  template <class... Coors> HOST_DEVICE_INLINE T& operator()(Coors... coors) {
    static_assert(sizeof...(coors) == Rank, "error! wrong index dim");
    return data_[size_and_stride_(coors...)];
  }
  template <class... Coors> HOST_DEVICE_INLINE const T& operator()(Coors... coors) const {
    static_assert(sizeof...(coors) == Rank, "error! wrong index dim");
    return data_[size_and_stride_(coors...)];
  }
  template <class... Coors> HOST_DEVICE_INLINE ssize_t offset(Coors... coors) const {
    static_assert(sizeof...(coors) == Rank, "error! wrong index dim");
    return size_and_stride_(coors...);
  }
  template <class Index> HOST_DEVICE_INLINE void deserialize(Index* const coor, ssize_t offset) const {
    size_and_stride_.template deserialize(coor, offset);
  }
  template <class... Coors> HOST_DEVICE_INLINE RefND<Rank - sizeof...(Coors), T> subview(Coors... coors) {
    constexpr ssize_t rm_dim = sizeof...(Coors);
    constexpr ssize_t new_dim = Rank - rm_dim;
    ptr_t new_data = data_ + size_and_stride_(coors...);
    Vec<new_dim, ssize_t> new_size;
#pragma unroll
    for (ssize_t i = 0; i < new_dim; i++) new_size[i] = size_and_stride_.size(i + rm_dim);
    return RefND<new_dim, T>(new_data, new_size);
  }
  template <class... Coors> HOST_DEVICE_INLINE RefND<Rank - sizeof...(Coors), const T> subview(Coors... coors) const {
    constexpr ssize_t rm_dim = sizeof...(Coors);
    constexpr ssize_t new_dim = Rank - rm_dim;
    const T* new_data = data_ + size_and_stride_(coors...);
    Vec<new_dim, ssize_t> new_size;
#pragma unroll
    for (ssize_t i = 0; i < new_dim; i++) new_size[i] = size_and_stride_.size(i + rm_dim);
    return RefND<new_dim, const T>(new_data, new_size);
  }

 private:
  ptr_t data_ = nullptr;
  Size<Rank> size_and_stride_;
};

template <class T> using Ref1D = RefND<1, T>;
template <class T> using Ref2D = RefND<2, T>;
template <class T> using Ref3D = RefND<3, T>;
template <class T> using Ref4D = RefND<4, T>;
template <class T> using Ref5D = RefND<5, T>;

#if NV_TENSORRT_MAJOR < 8
#  define AsciiChar char
#endif

template <ssize_t Rank> Size<Rank> fromTensorRT(nvinfer1::PluginTensorDesc inputDesc) {
  Vec<Rank, ssize_t> size;
#pragma unroll
  for (size_t i = 0; i < Rank; i++) { size[i] = inputDesc.dims.d[i]; }
  return Size<Rank>(size);
}

template <ssize_t Rank, class T, class T_void>
std::enable_if_t<std::is_void<T_void>::value, RefND<Rank, T>> fromTensorRT(T_void* data,
                                                                           nvinfer1::PluginTensorDesc inputDesc) {
  using namespace nvinfer1;
  assert(inputDesc.format == TensorFormat::kLINEAR);
  assert(inputDesc.dims.nbDims == Rank);
  constexpr bool T_is_float = std::is_same<std::remove_cv_t<T>, float>::value;
  constexpr bool T_is_half = std::is_same<std::remove_cv_t<T>, half>::value;
  constexpr bool T_is_cutlass_half = std::is_same<std::remove_cv_t<T>, cutlass::half_t>::value;
  constexpr bool T_is_int8 = std::is_same<std::remove_cv_t<T>, int8_t>::value;
  constexpr bool T_is_int32 = std::is_same<std::remove_cv_t<T>, int32_t>::value;
  constexpr bool T_is_bool = std::is_same<std::remove_cv_t<T>, bool>::value;
  constexpr bool unknown_type = false;
  switch (inputDesc.type) {
  case DataType::kFLOAT: assert(T_is_float); break;
  case DataType::kHALF: assert(T_is_half || T_is_cutlass_half); break;
  case DataType::kINT8: assert(T_is_int8); break;
  case DataType::kINT32: assert(T_is_int32); break;
  case DataType::kBOOL: assert(T_is_bool); break;
  default: assert(unknown_type); break;
  }
  Vec<Rank, ssize_t> size;
  T* data_ptr = reinterpret_cast<T*>(data);
#pragma unroll
  for (size_t i = 0; i < Rank; i++) {
    size[i] = inputDesc.dims.d[i];
    if (size[i] == 0) data_ptr = nullptr;
  }
  return RefND<Rank, T>(data_ptr, size);
}
}  // namespace nd

namespace io {

class SerializeStream {
  template <class T> friend SerializeStream& operator<<(SerializeStream& s, const T& x);
  template <class T> friend SerializeStream& operator>>(SerializeStream& s, T& x);

  const char* data_;
  size_t pos_;
  size_t size_;

 public:
  SerializeStream() : data_(nullptr), pos_(0), size_(std::numeric_limits<size_t>::max()){};
  SerializeStream(const void* data, size_t size = std::numeric_limits<size_t>::max())
      : data_(reinterpret_cast<const char*>(data)), pos_(0), size_(size){};
  inline size_t curPos() const { return pos_; };

  template <class Iter> SerializeStream& dumpRange(Iter first, Iter last) {
    while (pos_ + sizeof(*first) <= size_ && first != last) { *this << *(first++); }
    if (first != last) { printf("serialize out of buffer boundary!"); }
    return *this;
  }
  template <class Iter> SerializeStream& loadRange(Iter first, Iter last) {
    while (pos_ + sizeof(*first) <= size_ && first != last) { *this >> *(first++); }
    if (first != last) { printf("deserialize out of buffer boundary!"); }
    return *this;
  }
};

template <class T> SerializeStream& operator<<(SerializeStream& s, const T& x) {
  if (s.pos_ + sizeof(T) > s.size_) {
    printf("serialize out of buffer boundary!");
    return s;
  }
  if (s.data_) {
    *reinterpret_cast<T*>(const_cast<char*>(s.data_)) = x;
    s.data_ += sizeof(T);
  }
  s.pos_ += sizeof(T);
  return s;
};
template <class T> SerializeStream& operator>>(SerializeStream& s, T& x) {
  if (s.pos_ + sizeof(T) > s.size_) {
    printf("deserialize out of buffer boundary!");
    return s;
  }
  x = *reinterpret_cast<const T*>(s.data_);
  s.data_ += sizeof(T);
  s.pos_ += sizeof(T);
  return s;
};

}  // namespace io

namespace mem {
class DeviceVector {
  void* data_;

 public:
  DeviceVector() : data_(nullptr){};
  DeviceVector(const DeviceVector&) = delete;
  explicit DeviceVector(size_t bytes) : data_(nullptr) {
    auto err = cudaMalloc(&data_, bytes);
    assert(err == cudaSuccess);
  }
  ~DeviceVector() {
    if (data_) cudaFree(data_);
  }
  inline void* data() { return data_; }
};
}  // namespace mem
}  // namespace utils
