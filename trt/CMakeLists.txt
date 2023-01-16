find_package(CUDA REQUIRED)

set(TRT_INCLUDE_DIRECTORIES /usr/local/TensorRT/include/)
set(TRT_LINK_DIRECTORIES /usr/local/TensorRT/lib/)
set(TRT_LIBRARIES nvinfer nvinfer_plugin nvparsers nvonnxparser)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -D_GLIBCXX_USE_CXX11_ABI=0")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14 -Xcompiler -fPIC -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86")

include_directories(include include/3rd/cutlass/include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${TRT_INCLUDE_DIRECTORIES})
link_directories(${TRT_LINK_DIRECTORIES})

list(APPEND DEPLOY3D_SRC spconv/bev_densify_trt.cu spconv/index.cu spconv/index_trt.cu spconv/mm_trt.cu)
list(APPEND DEPLOY3D_SRC voxel_encoder/cylinder_trt.cu voxel_encoder/simple_mean_trt.cu)
list(APPEND DEPLOY3D_SRC point_voxel_mapping/gather_back.cu point_voxel_mapping/scatter_to.cu)
list(APPEND DEPLOY3D_SRC decoder/nms3d.cu decoder/yolox3d_decode.cu)

cuda_add_library(deploy3d SHARED ${DEPLOY3D_SRC})
target_compile_definitions(deploy3d PRIVATE CUTLASS_NAMESPACE=deploy3d_backend_cutlass)
target_link_libraries(deploy3d ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} ${TRT_LIBRARIES})

find_package(PythonInterp)
message("-- use python at " ${PYTHON_EXECUTABLE})
if (DEFINED PYTHON_EXECUTABLE)
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())" OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (DEFINED pybind11_DIR)
        message("-- use pybind11 at " ${pybind11_DIR})
        find_package(pybind11)
        pybind11_add_module(deploy3d_py_interface py_interface/py_interface.cpp)

    endif ()
endif ()
