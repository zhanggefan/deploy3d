#include "common/py_interface.h"
#include <pybind11/pybind11.h>

namespace deploy3d {
namespace interface {
int queryProfilingParams(const ProfilingParams::Key& k) {
  return deploy3d::interface::ProfilingParams::instance().query(k, -1);
}
void setProfilingParams(const ProfilingParams::Key& k, const ProfilingParams::Value& val) {
  deploy3d::interface::ProfilingParams::instance().set(k, val);
}

}  // namespace interface
}  // namespace deploy3d

PYBIND11_MODULE(deploy3d_py_interface, m) {
  m.doc() = "deploy3d python interface";
  m.def("query_profiling_params", &deploy3d::interface::queryProfilingParams, "query profiling params");
  m.def("set_profiling_params", &deploy3d::interface::setProfilingParams, "set profiling params");
}