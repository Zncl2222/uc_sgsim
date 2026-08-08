cmake_minimum_required(VERSION 3.15)

project(c_core)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/uc_sgsim/c_core/")

# set the library type to share or static
if(NOT DEFINED IS_EXECUTE OR NOT IS_EXECUTE)
    set(LIB_TYPE SHARED)
else()
    set(LIB_TYPE STATIC)
endif()

# Include header files
include_directories(include/)

# Set compiler optimization flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")

# Link each file
add_library(
    uc_sgsim ${LIB_TYPE}
    c_array_tools/src/c_array.c
    c_array_tools/src/c_array_mt.c
    src/cov_model.c
    src/kriging.c
    src/matrix_tools.c
    src/random_tools.c
    src/sgsim.c
    src/sort_tools.c
    src/variogram.c
)

# compile and install the shared library
if(NOT DEFINED IS_EXECUTE OR NOT IS_EXECUTE)
    install(TARGETS uc_sgsim DESTINATION "${CMAKE_SOURCE_DIR}/uc_sgsim/c_core/")
    set_target_properties(uc_sgsim PROPERTIES PREFIX "")
else()
    if(NOT WIN32)
        target_link_libraries(uc_sgsim PRIVATE m)
    endif()
endif()
