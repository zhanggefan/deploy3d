#pragma once
#include "input_iterator.h"
#include "kernel.h"
#include "output_iterator.h"

namespace spconv {
namespace mm {

template <int kStages, typename MmaConfig> struct ThreadblockMmaConfig;

template <typename MmaConfig> struct ThreadblockMmaConfig<3, MmaConfig> {
  static int const kStages = 3;
  using MmaCoreConfig = typename MmaConfig::MmaCore;

  using InFeatureIterator = spconv::in_iter::FeatureGatherTileAccessIterator<typename MmaConfig::IteratorA::Shape,
                                                                             typename MmaConfig::IteratorA::Element,
                                                                             typename MmaConfig::IteratorA::Layout,
                                                                             typename MmaConfig::IteratorA::ThreadMap,
                                                                             typename MmaConfig::AccessTypeA>;

  using InWeightIterator = typename MmaConfig::IteratorB;

  using ThreadblockMma = cutlass::gemm::threadblock::MmaMultistage<typename MmaCoreConfig::Shape,
                                                                   InFeatureIterator,
                                                                   typename MmaCoreConfig::SmemIteratorA,
                                                                   MmaCoreConfig::kCacheOpA,
                                                                   InWeightIterator,
                                                                   typename MmaCoreConfig::SmemIteratorB,
                                                                   MmaCoreConfig::kCacheOpB,
                                                                   typename MmaCoreConfig::ElementC,
                                                                   typename MmaCoreConfig::LayoutC,
                                                                   typename MmaCoreConfig::MmaPolicy,
                                                                   kStages>;
};

template <typename MmaConfig> struct ThreadblockMmaConfig<2, MmaConfig> {
  static int const kStages = 2;
  using MmaCoreConfig = typename MmaConfig::MmaCore;

  using InFeatureIterator = spconv::in_iter::FeatureGatherTileIterator<typename MmaConfig::IteratorA::Shape,
                                                                       typename MmaConfig::IteratorA::Element,
                                                                       typename MmaConfig::IteratorA::Layout,
                                                                       MmaConfig::IteratorA::kAdvanceRank,
                                                                       typename MmaConfig::IteratorA::ThreadMap,
                                                                       MmaConfig::IteratorA::AccessType::kElements>;

  using InWeightIterator = typename MmaConfig::IteratorB;

  using ThreadblockMma = cutlass::gemm::threadblock::MmaPipelined<typename MmaCoreConfig::Shape,
                                                                  InFeatureIterator,
                                                                  typename MmaCoreConfig::SmemIteratorA,
                                                                  InWeightIterator,
                                                                  typename MmaCoreConfig::SmemIteratorB,
                                                                  typename MmaCoreConfig::ElementC,
                                                                  typename MmaCoreConfig::LayoutC,
                                                                  typename MmaCoreConfig::MmaPolicy>;
};

template <typename Element_,
          int BlockM,
          int BlockN,
          int BlockK,
          int WarpM,
          int WarpN,
          int WarpK,
          typename ArchTag_ = cutlass::arch::Sm80>
struct TensorOp {
 private:
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = ArchTag_;
  using ElementA = Element_;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = Element_;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = Element_;
  using LayoutC = cutlass::layout::RowMajor;
  using ElementAccumulator = Element_;
  static constexpr bool SplitKSerial = false;
  using BaseTaskConfig = cutlass::gemm::device::
      DefaultGemmConfiguration<OperatorClass, ArchTag, ElementA, ElementB, ElementC, ElementAccumulator>;

 public:
  using Element = Element_;
  using ThreadblockShape = cutlass::gemm::GemmShape<BlockM, BlockN, BlockK>;
  using WarpShape = cutlass::gemm::GemmShape<WarpM, WarpN, WarpK>;
  using InstructionShape = typename BaseTaskConfig::InstructionShape;
  using ThreadblockSwizzle = typename cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

 private:
  static int const kAlignmentA = BaseTaskConfig::kAlignmentA;
  static int const kAlignmentB = BaseTaskConfig::kAlignmentB;
  static int const kStages = BaseTaskConfig::kStages;

  using EpilogueOutputOp =
      cutlass::epilogue::thread::LinearCombination<typename BaseTaskConfig::EpilogueOutputOp::ElementOutput,
                                                   BaseTaskConfig::EpilogueOutputOp::kCount,
                                                   typename BaseTaskConfig::EpilogueOutputOp::ElementAccumulator,
                                                   typename BaseTaskConfig::EpilogueOutputOp::ElementAccumulator,
                                                   cutlass::epilogue::thread::ScaleType::Nothing>;

  using Operator = typename BaseTaskConfig::Operator;

  using MmaConfig = typename cutlass::gemm::threadblock::DefaultMma<ElementA,
                                                                    LayoutA,
                                                                    kAlignmentA,
                                                                    ElementB,
                                                                    LayoutB,
                                                                    kAlignmentB,
                                                                    ElementAccumulator,
                                                                    LayoutC,
                                                                    OperatorClass,
                                                                    ArchTag,
                                                                    ThreadblockShape,
                                                                    WarpShape,
                                                                    InstructionShape,
                                                                    kStages,
                                                                    Operator,
                                                                    false,
                                                                    cutlass::gemm::SharedMemoryClearOption::kNone,
                                                                    false,
                                                                    false>;

  using ThreadblockMma = typename ThreadblockMmaConfig<kStages, MmaConfig>::ThreadblockMma;

  using EpilogueConfig =
      typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<ThreadblockShape,
                                                                       typename ThreadblockMma::Operator,
                                                                       ThreadblockShape::kK / WarpShape::kK,
                                                                       EpilogueOutputOp,
                                                                       EpilogueOutputOp::kCount,
                                                                       true,
                                                                       cutlass::layout::NoPermute>;

  using OutFeatureIterator = spconv::out_iter::FeatureScatterTileIterator<typename EpilogueConfig::OutputTileThreadMap,
                                                                          typename EpilogueConfig::ElementOutput>;

  using Epilogue =
      typename cutlass::epilogue::threadblock::Epilogue<ThreadblockShape,
                                                        typename ThreadblockMma::Operator,
                                                        EpilogueConfig::kPartitionsK,
                                                        OutFeatureIterator,
                                                        typename EpilogueConfig::AccumulatorFragmentIterator,
                                                        typename EpilogueConfig::WarpTileIterator,
                                                        typename EpilogueConfig::SharedLoadIterator,
                                                        typename EpilogueConfig::OutputOp,
                                                        typename EpilogueConfig::Padding,
                                                        EpilogueConfig::kFragmentsPerIteration>;

 public:
  using Kernel = spconv::kernel::SpConv<ThreadblockMma, Epilogue, ThreadblockSwizzle, SplitKSerial>;

 private:
  Kernel op;

 public:
  static int const kThreadCount = Kernel::kThreadCount;
  using Params = typename Kernel::Params;
  using SharedStorage = typename Kernel::SharedStorage;
  CUTLASS_HOST_DEVICE
  TensorOp() : op(){};
  CUTLASS_DEVICE
  void operator()(Params const& params, SharedStorage& shared_storage) { op(params, shared_storage); };
};

}  // namespace mm
}  // namespace spconv