#pragma once
#include <map>
#include <string>

namespace deploy3d {
namespace interface {
template <typename K_, typename V_, typename Scope_ = void> class KeyValueStore {  // meyer's singleton
 public:
  using Key = K_;
  using Value = V_;
  using Scope = Scope_;

 public:
  static KeyValueStore& instance() {
    static KeyValueStore s;
    return s;
  }
  KeyValueStore(const KeyValueStore&) = delete;
  KeyValueStore& operator=(const KeyValueStore&) = delete;
  const Value& query(const Key& k, const Value& default_val = Value()) {
    auto result = store.find(k);
    if (result != store.end())
      return result->second;
    else
      return default_val;
  }

  void set(const Key& k, const Value& val) { store[k] = val; }

 private:
  KeyValueStore() : store(){};
  ~KeyValueStore() = default;
  std::map<Key, Value> store;
};

using ProfilingParams = deploy3d::interface::KeyValueStore<std::string, int>;
}  // namespace interface
}  // namespace deploy3d