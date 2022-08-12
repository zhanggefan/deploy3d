find_package(CUDA REQUIRED)

cuda_add_library(spconv SHARED mm_trt.cu index_trt.cu index.cu mm.cu)
target_link_libraries(spconv ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} ${TRT_LIBRARIES})
