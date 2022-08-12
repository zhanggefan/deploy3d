#pragma once
#include "macros.h"
#include <limits>

namespace spconv {
namespace detail {
namespace hash {

template <int N> struct ubytes
{
};
template <> struct ubytes<1>
{
  using type = uint8_t;
};
template <> struct ubytes<2>
{
  using type = uint16_t;
};
template <> struct ubytes<4>
{
  using type = uint32_t;
};
template <> struct ubytes<8>
{
  using type = uint64_t;
};
template <class T> using uint_t = typename ubytes<sizeof(T)>::type;
template <class T> constexpr HOST_DEVICE_INLINE uint_t<T> uint_view(T x)
{
  return *reinterpret_cast<uint_t<T>*>(&x);
}

template <class T> constexpr HOST_DEVICE_INLINE uint_t<T>* uint_view_ptr(T* x)
{
  return reinterpret_cast<uint_t<T>*>(x);
}

template <class K> constexpr K empty_v = std::numeric_limits<K>::max();

struct Murmur3Hash4B
{
  template <typename K> size_t HOST_DEVICE_INLINE operator()(K key) const
  {
    uint_t<K> k = uint_view(key);
    k = k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return static_cast<size_t>(k);
  }
};

template <class K, class V, class HashFtor=Murmur3Hash4B> struct LinearHashTable
{
 public:
  using key_type = K;
  using value_type = V;
  const K EMPTY = empty_v<K>;
  const uint_t<K> EMPTY_U = uint_view(empty_v<K>);

 private:
  key_type* key_table_ = nullptr;
  value_type* value_table_ = nullptr;
  size_t size_;
  HashFtor hash_ftor_;

 public:
  explicit LinearHashTable(key_type* key_table, value_type* value_table, size_t size)
      : key_table_(key_table), value_table_(value_table), size_(size), hash_ftor_()
  {
  }

  HOST_DEVICE_INLINE size_t size() const { return size_; }

  HOST_DEVICE_INLINE const key_type* keys() const { return key_table_; }
  HOST_DEVICE_INLINE key_type* keys() { return key_table_; }
  HOST_DEVICE_INLINE const value_type* data() const { return value_table_; }
  HOST_DEVICE_INLINE value_type* data() { return value_table_; }

  HOST_DEVICE_INLINE size_t insert(const K& key, const V& value)
  {
    size_t slot = hash_ftor_(key) % size_;
    uint_t<K> key_u = uint_view(key);
    for (int i = 0; i < size_; i++) {
      uint_t<K> prev = atomicCAS(uint_view_ptr(key_table_ + slot), EMPTY_U, key_u);
      if (prev == EMPTY_U || prev == key_u) {
        value_table_[slot] = value;
        return slot;
      }
      slot = (slot + 1) % size_;
    }
    return empty_v<size_t>;
  }

  HOST_DEVICE_INLINE bool lookup(const K& key, V& value) const
  {
    size_t slot = hash_ftor_(key) % size_;
    for (int i = 0; i < size_; i++) {
      K found = key_table_[slot];
      if (found == key) {
        value = value_table_[slot];
        return true;
      }
      if (found == EMPTY) { return false; }
      slot = (slot + 1) % size_;
    }
    return false;
  }
};

}  // namespace hash
}  // namespace detail
}  // namespace spconv