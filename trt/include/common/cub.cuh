#pragma once
#define CUB_NS_PREFIX                                                                                                  \
  namespace deploy3d {                                                                                                 \
  namespace backend {
#define CUB_NS_POSTFIX                                                                                                 \
  }                                                                                                                    \
  }
#define CUB_NS_QUALIFIER ::deploy3d::backend
#include <cub/cub.cuh>