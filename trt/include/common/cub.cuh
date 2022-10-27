#pragma once
#include <cub/version.cuh>
#if CUB_VERSION >= 101400
#  define CUB_WRAPPED_NAMESPACE deploy3d_backend
#else
#  define CUB_NS_PREFIX namespace deploy3d_backend {
#  define CUB_NS_POSTFIX }
#  define CUB_NS_QUALIFIER ::deploy3d_backend
#endif
#define DEPLOY3D_CUB_NS_QUALIFIER ::deploy3d_backend
#include <cub/cub.cuh>