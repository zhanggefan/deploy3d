find_package(CUDA REQUIRED)

set(TRT_INCLUDE_DIRECTORIES /usr/local/TensorRT/include/)
set(TRT_LINK_DIRECTORIES /usr/local/TensorRT/lib/)
set(TRT_LIBRARIES nvinfer nvinfer_plugin nvparsers nvonnxparser)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -D_GLIBCXX_USE_CXX11_ABI=0")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14 -Xcompiler -fPIC -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86")

include_directories(include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${TRT_INCLUDE_DIRECTORIES})
link_directories(${TRT_LINK_DIRECTORIES})

list(APPEND DEPLOY3D_SRC spconv/index.cu spconv/index_trt.cu spconv/mm.cu spconv/mm_trt.cu)
list(APPEND DEPLOY3D_SRC voxel_encoder/cylinder_trt.cu voxel_encoder/simple_mean_trt.cu)
list(APPEND DEPLOY3D_SRC point_voxel_mapping/gather_back.cu point_voxel_mapping/scatter_to.cu)

cuda_add_library(deploy3d SHARED ${DEPLOY3D_SRC})
target_link_libraries(deploy3d ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} ${TRT_LIBRARIES})
