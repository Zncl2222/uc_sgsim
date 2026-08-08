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

option(
    UC_SGSIM_NATIVE_OPTIMIZATION
    "Enable native engine release optimization"
    ON
)

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

set_target_properties(
    uc_sgsim
    PROPERTIES
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        C_EXTENSIONS ON
)

if(UC_SGSIM_NATIVE_OPTIMIZATION)
    if(MSVC)
        target_compile_options(uc_sgsim PRIVATE /O2)
    else()
        target_compile_options(uc_sgsim PRIVATE -O3)
    endif()
endif()

if(NOT WIN32)
    target_link_libraries(uc_sgsim PRIVATE m)
endif()

# compile and install the shared library
if(NOT DEFINED IS_EXECUTE OR NOT IS_EXECUTE)
    install(TARGETS uc_sgsim DESTINATION "${CMAKE_SOURCE_DIR}/uc_sgsim/c_core/")
    set_target_properties(uc_sgsim PROPERTIES PREFIX "")
endif()
