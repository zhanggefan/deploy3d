#pragma once
#include "common/refnd.h"

namespace spconv {
namespace func {

using utils::GPU;
using utils::nd::Ref1D;
using utils::nd::Ref2D;
using utils::nd::Ref3D;

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
               const Ref1D<Index>& bufferKernelNumHost);

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
               const Ref1D<Index>& bufferKernelNumHost);

}  // namespace func
}  // namespace spconv