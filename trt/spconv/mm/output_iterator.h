#pragma once
#include "cutlass/arch/arch.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/threadblock/output_tile_thread_map.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator_params.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/permute.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/transform/pitch_linear_thread_map.h"

namespace spconv {
namespace out_iter {

template <typename AccessType> struct global_atomic_add;

template <> struct global_atomic_add<cutlass::AlignedArray<float, 1>> {
  CUTLASS_DEVICE
  global_atomic_add(cutlass::AlignedArray<float, 1> const& D, void* ptr, bool pred_guard) {
    float const& data = reinterpret_cast<float const&>(D);
    if (pred_guard) atomicAdd((float*)ptr, data);
  }
};

template <> struct global_atomic_add<cutlass::AlignedArray<cutlass::half_t, 1>> {
  CUTLASS_DEVICE
  global_atomic_add(cutlass::AlignedArray<cutlass::half_t, 1> const& D, void* ptr, bool pred_guard) {
    half const& data = reinterpret_cast<half const&>(D);
    if (pred_guard) atomicAdd((half*)ptr, data);
  }
};

template <> struct global_atomic_add<cutlass::AlignedArray<cutlass::half_t, 8>> {
  CUTLASS_DEVICE
  global_atomic_add(cutlass::AlignedArray<cutlass::half_t, 8> const& D, void* ptr, bool pred_guard) {
    half2 const* data = reinterpret_cast<half2 const*>(&D);
    half2* typed_ptr = reinterpret_cast<half2*>(ptr);
    if (pred_guard) {
      atomicAdd(&typed_ptr[0], data[0]);
      atomicAdd(&typed_ptr[1], data[1]);
      atomicAdd(&typed_ptr[2], data[2]);
      atomicAdd(&typed_ptr[3], data[3]);
    }
  }
};

template <typename ThreadMap_, typename Element_> class FeatureScatterTileIterator {
 public:
  using ThreadMap = ThreadMap_;
  using Shape = typename ThreadMap::Shape;

  using Element = Element_;

  using Layout = cutlass::layout::RowMajor;
  using TensorRef = cutlass::TensorRef<Element, Layout>;
  using ConstTensorRef = typename TensorRef::ConstTensorRef;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using TensorCoord = cutlass::MatrixCoord;

  static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
  static int const kThreads = ThreadMap::kThreads;
  static int const kIterations = ThreadMap::Count::kTile;

  static_assert(ThreadMap::Iterations::kRow > 0, "ThreadMap::Iterations::kRow must be > 0");
  static_assert(ThreadMap::Iterations::kGroup > 0, "ThreadMap::Iterations::kGroup must be > 0");
  static_assert(ThreadMap::Iterations::kCluster > 0, "ThreadMap::Iterations::kCluster must be > 0");
  static_assert(ThreadMap::Iterations::kColumn > 0, "ThreadMap::Iterations::kColumn must be > 0");

  using Fragment =
      cutlass::Array<Element,
                     ThreadMap::Iterations::kColumn * ThreadMap::Iterations::kRow * ThreadMap::Iterations::kGroup *
                         ThreadMap::Iterations::kCluster * ThreadMap::kElementsPerAccess>;

  using AccessType = cutlass::AlignedArray<Element, ThreadMap::kElementsPerAccess>;

  struct Params {
    LongIndex stride_;
    Params() = default;
    CUTLASS_HOST_DEVICE
    Params(Layout const& layout) : stride_(layout.stride(0)) {}
    CUTLASS_HOST_DEVICE
    Params(LongIndex stride) : stride_(stride) {}
  };

  /// Mask object
  struct Mask {
    static int const kCount = ThreadMap::Iterations::kColumn;

    /// Predicate state
    bool predicates[kCount];

    //
    // Mask
    //
    CUTLASS_HOST_DEVICE
    Mask() { enable(); }

    ///< Efficiently disables all accesses guarded by mask
    CUTLASS_HOST_DEVICE void clear() {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kCount; ++i) { predicates[i] = false; }
    }

    ///< CUTLASS_HOST_DEVICE enables all accesses guarded by mask
    CUTLASS_DEVICE void enable() {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kCount; ++i) { predicates[i] = true; }
    }
  };

 private:
  Params params_;
  uint8_t* byte_pointer_;
  Mask mask_;
  Index extent_row_;
  Index extent_column_;
  Index thread_start_row_;
  Index thread_start_column_;
  int state_[3];
  int const* indices_;

  static_assert(sizeof(extent_row_) == 4, "Expected 32b extents");
  static_assert(sizeof(thread_start_row_) == 4, "Expected 32b extents");
  static_assert(sizeof(Params::stride_) == 8, "Expected 64b strides");

 public:
  CUTLASS_DEVICE
  FeatureScatterTileIterator(Params const& params,
                             Element* pointer,
                             TensorCoord extent,
                             int thread_idx,
                             TensorCoord threadblock_offset = TensorCoord(),
                             int const* indices = nullptr)
      : params_(params.stride_), indices_(indices) {
    TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx) + threadblock_offset;

    extent_row_ = extent.row();
    extent_column_ = extent.column();

    thread_start_row_ = thread_offset.row();
    thread_start_column_ = thread_offset.column();

    // Initialize predicates
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < ThreadMap::Iterations::kColumn; ++c) {
      mask_.predicates[c] = ((thread_offset.column() + ThreadMap::Delta::kColumn * c) < extent.column());
    }
    byte_pointer_ = reinterpret_cast<uint8_t*>(pointer) +
        LongIndex(thread_offset.column()) * sizeof(AccessType) / kElementsPerAccess;

    // Initialize internal state counter
    state_[0] = state_[1] = state_[2] = 0;
  }

  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    byte_pointer_ += pointer_offset * cutlass::sizeof_bits<Element>::value / 8;
  }

  CUTLASS_DEVICE
  void load(Fragment& frag) const {}

  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const& frag, int64_t byte_offset) const {
    uint8_t* byte_pointer = byte_pointer_;
    AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {
      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {
        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
          int frag_row_idx = (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

          int row_offset =
              row * ThreadMap::Delta::kRow + group * ThreadMap::Delta::kGroup + cluster * ThreadMap::Delta::kCluster;

          bool row_guard = ((row_offset + thread_start_row_) < extent_row_);

          AccessType* memory_pointer;

          if (row_guard) {
            auto indices_row = LongIndex(indices_[row_offset + thread_start_row_]);

            row_guard = row_guard && (indices_row >= 0);

            memory_pointer = reinterpret_cast<AccessType*>(byte_pointer + byte_offset +
                                                           indices_row * LongIndex(params_.stride_) *
                                                               cutlass::sizeof_bits<Element>::value / 8);
          }

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
            bool guard = row_guard && mask_.predicates[column];

            global_atomic_add<AccessType>(frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column],
                                          (void*)&memory_pointer[0], guard);

            memory_pointer += (ThreadMap::Delta::kColumn / kElementsPerAccess);
          }
        }
      }
    }
  }

  /// Stores a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const& frag) const { store_with_byte_offset(frag, 0); }

  CUTLASS_DEVICE
  cutlass::MatrixCoord thread_start() const { return cutlass::MatrixCoord(thread_start_row_, thread_start_column_); }

  /// Need to get the thread start row from the tile iterator
  CUTLASS_DEVICE
  int32_t thread_start_row() const { return thread_start_row_; }

  /// Need to get the thread start row from the tile iterator
  CUTLASS_DEVICE
  int32_t thread_start_column() const { return thread_start_column_; }

  CUTLASS_DEVICE
  Index extent_row() const { return extent_row_; }

  CUTLASS_DEVICE
  Index extent_column() const { return extent_column_; }

  CUTLASS_HOST_DEVICE
  FeatureScatterTileIterator& operator++() {
    ++state_[0];

    thread_start_row_ += ThreadMap::Shape::kRow;

    if (state_[0] == ThreadMap::Count::kRow) {
      state_[0] = 0;
      ++state_[1];

      thread_start_row_ += (ThreadMap::Shape::kGroup - 1) * ThreadMap::Shape::kRow * ThreadMap::Count::kRow;

      if (state_[1] == ThreadMap::Count::kGroup) {
        state_[1] = 0;
        ++state_[2];

        thread_start_row_ +=
            ThreadMap::Count::kGroup * ThreadMap::Shape::kGroup * ThreadMap::Count::kRow * ThreadMap::Shape::kRow;

        if (state_[2] == ThreadMap::Count::kCluster) { state_[2] = 0; }
      }
    }

    return *this;
  }

  CUTLASS_DEVICE void clear_mask() { mask_.clear(); }
  CUTLASS_DEVICE void enable_mask() { mask_.enable(); }
  CUTLASS_DEVICE void get_mask(Mask& mask) const { mask = mask_; }
  CUTLASS_DEVICE void set_mask(Mask const& mask) { mask_ = mask; }
};

}  // namespace out_iter
}  // namespace spconv