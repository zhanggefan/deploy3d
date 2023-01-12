#pragma once
#include "cutlass/gemm/device/gemm.h"

namespace spconv {

namespace in_iter {

template <typename Shape_, typename Element_, typename ThreadMap_, typename AccessType_> class FeatureGatherPredicates {
 public:
  using Shape = Shape_;
  using Element = Element_;
  using Layout = cutlass::layout::PitchLinear;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;
  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using TensorCoord = typename Layout::TensorCoord;

  static int const kAccessesPerVector = ThreadMap::kElementsPerAccess / AccessType::kElements;

  static_assert(!(ThreadMap::kElementsPerAccess % AccessType::kElements),
                "Vectors implied by the thread map must be divisible by the "
                "access type.");

  static int const kPredicatesPerByte = 4;
  static int const kPredicatesPerWord = 4 * kPredicatesPerByte;
  static int const kPredicateCount = ThreadMap::Iterations::kCount * kAccessesPerVector;
  static int const kPredicateByteCount = (kPredicateCount + kPredicatesPerByte - 1) / kPredicatesPerByte;
  static int const kPredicateWordCount = (kPredicateByteCount + 3) / 4;
  static unsigned const kPredicateMask = (1u << kPredicatesPerByte) - 1u;
  static_assert(kPredicateWordCount <= 4, "Too many predicates.");

  using Mask = cutlass::Array<uint32_t, kPredicateWordCount>;

  uint32_t predicates_[kPredicateWordCount];
  TensorCoord extent_;
  TensorCoord thread_offset_;
  TensorCoord residue_offset_;
  int iteration_vector_;
  int iteration_contiguous_;
  int iteration_strided_;

 public:
  CUTLASS_DEVICE
  void compute_predicates_(TensorCoord extent) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPredicateWordCount; ++i) { predicates_[i] = 0u; }

    CUTLASS_PRAGMA_UNROLL
    for (int access_idx = 0; access_idx < ThreadMap::Iterations::kCount * kAccessesPerVector; ++access_idx) {
      int access_residual = access_idx % (ThreadMap::Iterations::kContiguous * kAccessesPerVector);
      int c = access_residual / kAccessesPerVector;
      int v = access_residual % kAccessesPerVector;
      auto access_contiguous =
          thread_offset_.contiguous() + c * ThreadMap::Delta::kContiguous + v * AccessType::kElements;
      bool guard = access_contiguous < extent.contiguous();
      int word_idx = access_idx / kPredicatesPerWord;
      int residual = access_idx % kPredicatesPerWord;
      int byte_idx = residual / kPredicatesPerByte;
      int bit_idx = residual % kPredicatesPerByte;
      predicates_[word_idx] |= (unsigned(guard) << (byte_idx * 8 + bit_idx));
    }
  }

  CUTLASS_HOST_DEVICE
  void set_predicates(int thread_id, TensorCoord const& threadblock_offset) {
    TensorCoord residue_extent;
    /// K dim is along contiguous(0) dimension
    typename TensorCoord::Index residue_size = (extent_[0] - threadblock_offset.contiguous()) % Shape::kContiguous;
    if (!residue_size) { residue_size = Shape::kContiguous; }
    residue_offset_ = cutlass::make_Coord(residue_size, 0);
    residue_extent = cutlass::make_Coord(min(extent_.contiguous(), threadblock_offset.contiguous() + residue_size),
                                         extent_.strided());

    thread_offset_ = threadblock_offset + ThreadMap::initial_offset(thread_id);
    compute_predicates_(residue_extent);
    set_iteration_index(0);
  }

  FeatureGatherPredicates() = default;

  CUTLASS_HOST_DEVICE
  FeatureGatherPredicates(TensorCoord extent) : extent_(extent) {}

  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {
    iteration_vector_ = index % kAccessesPerVector;
    int residual_access = index / kAccessesPerVector;
    iteration_contiguous_ = residual_access % ThreadMap::Iterations::kContiguous;
    iteration_strided_ = residual_access / ThreadMap::Iterations::kContiguous;
  }

  CUTLASS_HOST_DEVICE
  FeatureGatherPredicates& operator++() { return *this; }

  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPredicateWordCount; ++i) { predicates_[i] = enable ? 0u : predicates_[i]; }
  }

  CUTLASS_HOST_DEVICE
  void enable_mask() {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPredicateWordCount; ++i) { predicates_[i] = 0xffffffff; }
  }

  CUTLASS_HOST_DEVICE
  void set_mask(Mask const& mask) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPredicateWordCount; ++i) { predicates_[i] = mask[i]; }
  }

  CUTLASS_HOST_DEVICE
  void get_mask(Mask& mask) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPredicateWordCount; ++i) { mask[i] = predicates_[i]; }
  }

  CUTLASS_HOST_DEVICE
  bool valid() const {
    int pred_idx = iteration_vector_ +
        kAccessesPerVector * (iteration_contiguous_ + iteration_strided_ * ThreadMap::Iterations::kContiguous);

    int word_idx = pred_idx / kPredicatesPerWord;
    int residual = pred_idx % kPredicatesPerWord;
    int byte_idx = residual / kPredicatesPerByte;
    int bit_idx = residual % kPredicatesPerByte;

    bool pred = (predicates_[word_idx] & (1u << (byte_idx * 8 + bit_idx))) != 0;
    return pred;
  }
};

template <typename Shape_, typename Element_, typename Layout, typename ThreadMap_, typename AccessType_>
class FeatureGatherTileAccessIterator;

template <typename Shape_, typename Element_, typename ThreadMap_, typename AccessType_>
class FeatureGatherTileAccessIterator<Shape_, Element_, cutlass::layout::PitchLinear, ThreadMap_, AccessType_> {
 public:
  using Shape = Shape_;
  using Element = Element_;
  using Layout = cutlass::layout::PitchLinear;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = cutlass::TensorRef<Element, Layout>;
  using TensorView = cutlass::TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element*;
  using NonConstPointer = typename cutlass::platform::remove_const<Element>::type*;

  using UnderlyingPredicates = FeatureGatherPredicates<Shape, Element, ThreadMap, AccessType>;

  static int const kAccessesPerVector = ThreadMap::kElementsPerAccess / AccessType::kElements;

  static_assert(!(ThreadMap::kElementsPerAccess % AccessType::kElements),
                "Vectors implied by the thread map must be divisible by the "
                "access type.");

  using Mask = typename UnderlyingPredicates::Mask;

  struct Params {
    LongIndex stride_;
    Params() = default;
    CUTLASS_HOST_DEVICE
    Params(Layout const& layout) : stride_(layout.stride(0)) {}
    CUTLASS_HOST_DEVICE
    Params(LongIndex stride) : stride_(stride) {}
  };

 private:
  using BytePointer = char*;

 private:
  UnderlyingPredicates the_predicates;
  BytePointer pointer_;
  bool is_residue_tile_;
  Params params_;
  int const* indices_;
  Index gather_offset_strided;
  Index gather_index;

 public:
  FeatureGatherTileAccessIterator() = default;

  CUTLASS_HOST_DEVICE
  FeatureGatherTileAccessIterator(Params const& params,
                                  Pointer pointer,
                                  TensorCoord extent,
                                  int thread_id,
                                  TensorCoord const& threadblock_offset,
                                  int const* indices)
      : params_(params), pointer_(reinterpret_cast<BytePointer>(const_cast<NonConstPointer>(pointer))),
        the_predicates(extent), is_residue_tile_(true), indices_(indices) {
    the_predicates.set_predicates(thread_id, threadblock_offset);

    Layout layout(params_.stride_);

    gather_offset_strided = the_predicates.thread_offset_.strided();
    add_pointer_offset(layout(cutlass::make_Coord(the_predicates.thread_offset_.contiguous(), 0)));
    update_gather_index();
  }

  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {
    the_predicates.set_iteration_index(index);
    update_gather_index();
  }

  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    pointer_ += cutlass::sizeof_bits<Element>::value * pointer_offset / 8;
  }

  CUTLASS_HOST_DEVICE
  void update_gather_index() {
    gather_index = indices_[gather_offset_strided + the_predicates.iteration_strided_ * ThreadMap::Delta::kStrided];
  }

  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const& tile_offset) {
    /// advance along contiguous(0) dim, gather along strided(1) dim
    if (is_residue_tile_) {
      the_predicates.thread_offset_ += the_predicates.residue_offset_;
      the_predicates.compute_predicates_(the_predicates.extent_);
      Layout layout(params_.stride_);
      gather_offset_strided = the_predicates.thread_offset_.strided() + Shape::kStrided * tile_offset.strided();
      add_pointer_offset(the_predicates.residue_offset_.contiguous() +
                         Shape::kContiguous * (tile_offset.contiguous() - 1));
    } else {
      gather_offset_strided += Shape::kStrided * tile_offset.strided();
      add_pointer_offset(Shape::kContiguous * tile_offset.contiguous());
    }
    if (tile_offset.strided() != 0) update_gather_index();
    is_residue_tile_ = false;
  }

  CUTLASS_HOST_DEVICE
  AccessType* get() const {
    if (!valid()) { return nullptr; }

    LongIndex contiguous_offset =
        (the_predicates.iteration_contiguous_ * ThreadMap::Delta::kContiguous + the_predicates.iteration_vector_) *
        AccessType::kElements;

    return reinterpret_cast<AccessType*>(pointer_ +
                                         (contiguous_offset + LongIndex(params_.stride_) * gather_index) *
                                             cutlass::sizeof_bits<Element>::value / 8);
  }

  CUTLASS_HOST_DEVICE
  FeatureGatherTileAccessIterator& operator++() {
    the_predicates.operator++();
    ++the_predicates.iteration_vector_;
    if (the_predicates.iteration_vector_ < kAccessesPerVector) { return *this; }
    the_predicates.iteration_vector_ = 0;
    ++the_predicates.iteration_contiguous_;
    if (the_predicates.iteration_contiguous_ < ThreadMap::Iterations::kContiguous) { return *this; }
    the_predicates.iteration_contiguous_ = 0;
    ++the_predicates.iteration_strided_;
    if (the_predicates.iteration_strided_ < ThreadMap::Iterations::kStrided) {
      update_gather_index();
      return *this;
    }
    the_predicates.iteration_strided_ = 0;
    update_gather_index();
    return *this;
  }

  CUTLASS_HOST_DEVICE
  FeatureGatherTileAccessIterator operator++(int) {
    FeatureGatherTileAccessIterator self(*this);
    operator++();
    return self;
  }

  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) { the_predicates.clear_mask(enable); }

  CUTLASS_HOST_DEVICE
  void enable_mask() { the_predicates.enable_mask(); }

  CUTLASS_HOST_DEVICE
  void set_mask(Mask const& mask) { the_predicates.set_mask(mask); }

  CUTLASS_HOST_DEVICE
  void get_mask(Mask& mask) { the_predicates.get_mask(mask); }

  CUTLASS_HOST_DEVICE
  bool valid() const { return the_predicates.valid() && gather_index >= 0; }
};

template <typename Shape_, typename Element_, typename ThreadMap_, typename AccessType_>
class FeatureGatherTileAccessIterator<Shape_, Element_, cutlass::layout::RowMajor, ThreadMap_, AccessType_> {
 public:
  using Shape = Shape_;
  using Element = Element_;
  using Layout = cutlass::layout::RowMajor;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = cutlass::TensorRef<Element, Layout>;
  using TensorView = cutlass::TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element*;
  using NonConstPointer = typename cutlass::platform::remove_const<Element>::type*;

  using UnderlyingIterator =
      FeatureGatherTileAccessIterator<cutlass::layout::PitchLinearShape<Shape::kColumn, Shape::kRow>,
                                      Element,
                                      cutlass::layout::PitchLinear,
                                      ThreadMap,
                                      AccessType>;

  using Mask = typename UnderlyingIterator::Mask;

  static int const kAccessesPerVector = UnderlyingIterator::kAccessesPerVector;

  struct Params {
    LongIndex stride_;
    Params() = default;
    CUTLASS_HOST_DEVICE
    Params(Layout const& layout) : stride_(layout.stride(0)) {}
    CUTLASS_HOST_DEVICE
    Params(LongIndex stride) : stride_(stride) {}
  };

 private:
  UnderlyingIterator iterator_;

 public:
  FeatureGatherTileAccessIterator() = default;

  CUTLASS_HOST_DEVICE
  FeatureGatherTileAccessIterator(Params const& params,
                                  Pointer pointer,
                                  TensorCoord extent,
                                  int thread_id,
                                  TensorCoord const& threadblock_offset,
                                  int const* indices)
      : iterator_({params.stride_},
                  pointer,
                  cutlass::layout::PitchLinearCoord(extent.column(), extent.row()),
                  thread_id,
                  cutlass::layout::PitchLinearCoord(threadblock_offset.column(), threadblock_offset.row()),
                  indices) {}
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) { iterator_.set_iteration_index(index); }
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) { iterator_.add_pointer_offset(pointer_offset); }
  CUTLASS_HOST_DEVICE
  void add_tile_offset(TensorCoord const& tile_offset) {
    iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()});
  }
  CUTLASS_HOST_DEVICE
  AccessType* get() const { return reinterpret_cast<AccessType*>(iterator_.get()); }
  CUTLASS_HOST_DEVICE
  FeatureGatherTileAccessIterator& operator++() {
    ++iterator_;
    return *this;
  }
  CUTLASS_HOST_DEVICE
  FeatureGatherTileAccessIterator operator++(int) {
    FeatureGatherTileAccessIterator self(*this);
    operator++();
    return self;
  }
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) { iterator_.clear_mask(enable); }
  CUTLASS_HOST_DEVICE
  void enable_mask() { iterator_.enable_mask(); }
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const& mask) { iterator_.set_mask(mask); }
  CUTLASS_HOST_DEVICE
  void get_mask(Mask& mask) { iterator_.get_mask(mask); }
  CUTLASS_HOST_DEVICE
  bool valid() const { return iterator_.valid(); }
};

template <typename Shape,
          typename Element,
          typename Layout,
          int AdvanceRank,
          typename ThreadMap,
          int AccessSize = ThreadMap::kElementsPerAccess>
class FeatureGatherTileIterator;

template <typename Shape_, typename Element_, typename ThreadMap_, int AccessSize>
class FeatureGatherTileIterator<Shape_, Element_, cutlass::layout::PitchLinear, 0, ThreadMap_, AccessSize> {
 public:
  using Shape = Shape_;
  using Element = Element_;
  using Layout = cutlass::layout::PitchLinear;
  using ThreadMap = ThreadMap_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = cutlass::TensorRef<Element, Layout>;
  using TensorView = cutlass::TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element*;
  using NonConstPointer = typename cutlass::platform::remove_const<Element>::type*;

  using AccessType =
      cutlass::AlignedArray<Element, AccessSize, (AccessSize * cutlass::sizeof_bits<Element>::value / 8)>;

  using TileAccessIterator =
      FeatureGatherTileAccessIterator<Shape, Element, cutlass::layout::PitchLinear, ThreadMap, AccessType>;

  static int const kAccessesPerVector = TileAccessIterator::kAccessesPerVector;

  using Fragment = cutlass::Array<Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;

  using Mask = typename TileAccessIterator::Mask;

  class Params {
    friend FeatureGatherTileIterator;

   private:
    LongIndex stride_;

   public:
    CUTLASS_HOST_DEVICE
    Params(Layout const& layout) : stride_(layout.stride(0)) {}
    Params() = default;
    CUTLASS_HOST_DEVICE
    Params(LongIndex stride) : stride_(stride) {}
  };

 private:
  using BytePointer = char*;

 private:
  TileAccessIterator address_iterator_;

 public:
  FeatureGatherTileIterator() = default;

  CUTLASS_HOST_DEVICE
  FeatureGatherTileIterator(Params const& params,
                            Pointer pointer,
                            TensorCoord extent,
                            int thread_id,
                            TensorCoord const& threadblock_offset,
                            int const* indices)
      : address_iterator_({params.stride_}, pointer, extent, thread_id, threadblock_offset, indices) {}

  CUTLASS_HOST_DEVICE
  FeatureGatherTileIterator(Params const& params, Pointer pointer, TensorCoord extent, int thread_id)
      : FeatureGatherTileIterator(params, pointer, extent, thread_id, cutlass::make_Coord(0, 0), nullptr) {}

  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) { address_iterator_.add_pointer_offset(pointer_offset); }

  CUTLASS_HOST_DEVICE
  FeatureGatherTileIterator& operator++() {
    address_iterator_.add_tile_offset({1, 0});
    return *this;
  }

  CUTLASS_HOST_DEVICE
  FeatureGatherTileIterator operator++(int) {
    FeatureGatherTileIterator self(*this);
    operator++();
    return self;
  }

  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) { address_iterator_.clear_mask(enable); }

  CUTLASS_HOST_DEVICE
  void enable_mask() { address_iterator_.enable_mask(); }

  CUTLASS_HOST_DEVICE
  void set_mask(Mask const& mask) { address_iterator_.set_mask(mask); }

  CUTLASS_HOST_DEVICE
  void get_mask(Mask& mask) { address_iterator_.get_mask(mask); }

  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment& frag, Index pointer_offset) {
    load_with_byte_offset(frag, pointer_offset * cutlass::sizeof_bits<Element>::value / 8);
  }

  CUTLASS_DEVICE
  void load_with_byte_offset(Fragment& frag, LongIndex byte_offset) {
    AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < kAccessesPerVector; ++v) {
          int idx = v + kAccessesPerVector * (c + s * ThreadMap::Iterations::kContiguous);

          address_iterator_.set_iteration_index(idx);
          char const* byte_ptr = reinterpret_cast<char const*>(address_iterator_.get()) + byte_offset;

          AccessType const* access_ptr = reinterpret_cast<AccessType const*>(byte_ptr);

          cutlass::arch::global_load<AccessType, sizeof(AccessType)>(frag_ptr[idx], access_ptr,
                                                                     address_iterator_.valid());

          ++address_iterator_;
        }
      }
    }
  }

  CUTLASS_DEVICE
  void load(Fragment& frag) { load_with_byte_offset(frag, 0); }

  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) {
    store_with_byte_offset(frag, pointer_offset * cutlass::sizeof_bits<Element>::value / 8);
  }

  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const& frag, LongIndex byte_offset) {
    address_iterator_.set_iteration_index(0);
    AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < kAccessesPerVector; ++v) {
          int idx = v + kAccessesPerVector * (c + s * ThreadMap::Iterations::kContiguous);

          char* byte_ptr = reinterpret_cast<char*>(address_iterator_.get()) + byte_offset;
          AccessType* access_ptr = reinterpret_cast<AccessType*>(byte_ptr);

          if (address_iterator_.valid()) { *access_ptr = frag_ptr[idx]; }
          ++address_iterator_;
        }
      }
    }
  }

  CUTLASS_DEVICE
  void store(Fragment const& frag) { store_with_byte_offset(frag, 0); }
};

template <typename Shape_, typename Element_, typename ThreadMap_, int AccessSize>
class FeatureGatherTileIterator<Shape_, Element_, cutlass::layout::RowMajor, 1, ThreadMap_, AccessSize> {
 public:
  using Shape = Shape_;
  using Element = Element_;
  using Layout = cutlass::layout::RowMajor;
  using ThreadMap = ThreadMap_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = cutlass::TensorRef<Element, Layout>;
  using TensorView = cutlass::TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element*;
  using NonConstPointer = typename cutlass::platform::remove_const<Element>::type*;

  using UnderlyingIterator = FeatureGatherTileIterator<cutlass::layout::PitchLinearShape<Shape::kColumn, Shape::kRow>,
                                                       Element,
                                                       cutlass::layout::PitchLinear,
                                                       0,
                                                       ThreadMap,
                                                       AccessSize>;
  using AccessType = typename UnderlyingIterator::AccessType;
  using Fragment = cutlass::Array<Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;
  using Mask = typename UnderlyingIterator::Mask;

  class Params {
    friend FeatureGatherTileIterator;

   private:
    LongIndex stride_;

   public:
    CUTLASS_HOST_DEVICE
    Params(Layout const& layout) : stride_(layout.stride(0)) {}
    Params() = default;
    CUTLASS_HOST_DEVICE
    Params(LongIndex stride) : stride_(stride) {}
  };

 private:
  UnderlyingIterator iterator_;

 public:
  FeatureGatherTileIterator() = default;

  CUTLASS_HOST_DEVICE
  FeatureGatherTileIterator(Params const& params,
                            Pointer pointer,
                            TensorCoord extent,
                            int thread_id,
                            TensorCoord const& threadblock_offset,
                            int const* indices)
      : iterator_({params.stride_},
                  pointer,
                  cutlass::layout::PitchLinearCoord(extent.column(), extent.row()),
                  thread_id,
                  cutlass::layout::PitchLinearCoord(threadblock_offset.column(), threadblock_offset.row()),
                  indices) {}

  CUTLASS_HOST_DEVICE
  FeatureGatherTileIterator(Params const& params, Pointer pointer, TensorCoord extent, int thread_id)
      : FeatureGatherTileIterator(params, pointer, extent, thread_id, cutlass::make_Coord(0, 0), nullptr) {}

  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) { iterator_.add_pointer_offset(pointer_offset); }

  CUTLASS_HOST_DEVICE
  FeatureGatherTileIterator& operator++() {
    ++iterator_;
    return *this;
  }

  CUTLASS_HOST_DEVICE
  FeatureGatherTileIterator operator++(int) {
    FeatureGatherTileIterator self(*this);
    operator++();
    return self;
  }

  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) { iterator_.clear_mask(enable); }

  CUTLASS_HOST_DEVICE
  void enable_mask() { iterator_.enable_mask(); }

  CUTLASS_HOST_DEVICE
  void set_mask(Mask const& mask) { iterator_.set_mask(mask); }

  CUTLASS_HOST_DEVICE
  void get_mask(Mask& mask) { iterator_.get_mask(mask); }

  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment& frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  CUTLASS_DEVICE
  void load_with_byte_offset(Fragment& frag, LongIndex byte_offset) {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  CUTLASS_DEVICE
  void load(Fragment& frag) { load_with_pointer_offset(frag, 0); }

  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) {
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const& frag, LongIndex byte_offset) {
    iterator_.store_with_byte_offset(frag, byte_offset);
  }

  CUTLASS_DEVICE
  void store(Fragment const& frag) { store_with_pointer_offset(frag, 0); }
};

}  // namespace in_iter

};  // namespace spconv