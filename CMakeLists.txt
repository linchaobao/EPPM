cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)


include_directories(
    basic
    )


# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
    runeppm
    main.cpp
    bao_pmflow_census_kernel.cu
    bao_pmflow_refine_kernel.cu
    bao_flow_patchmatch_multiscale_cuda.cpp
    bao_flow_patchmatch_multiscale_kernel.cu
    bao_pmflow_kernel.cu
    3rdparty/middlebury/colorcode.cpp
    3rdparty/middlebury/flowIO.cpp
    3rdparty/middlebury/imageLib/Image.cpp
    3rdparty/middlebury/imageLib/RefCntMem.cpp
    basic/bao_basic.cpp
    basic/bao_basic_cuda.cpp
    basic/bao_flow_tools.cpp
    )

