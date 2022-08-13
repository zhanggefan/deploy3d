find_package(CUDA REQUIRED)

cuda_add_library(voxel_encoder SHARED simple_mean_trt.cu cylinder_trt.cu)
target_link_libraries(voxel_encoder ${CUDA_LIBRARIES} ${TRT_LIBRARIES})