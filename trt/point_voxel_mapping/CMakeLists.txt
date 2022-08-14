find_package(CUDA REQUIRED)

cuda_add_library(point_voxel_mapping SHARED scatter_to.cu gather_back.cu)
target_link_libraries(point_voxel_mapping ${CUDA_LIBRARIES} ${TRT_LIBRARIES})