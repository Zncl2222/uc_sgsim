cmake_minimum_required(VERSION 3.15)
enable_testing()

project(c_example)
set(CMAKE_VERBOSE_MAKEFILE ON)

# Set the option to compile the executable file
option(IS_EXECUTE "Compile the executable file" ON)

# set output directory
if(IS_EXECUTE)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
else()
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/uc_sgsim/c_core")
endif()

include_directories(${CMAKE_SOURCE_DIR})

# run CMakeLists in uc_sgsim folder
add_subdirectory(uc_sgsim)

# compile and install executable file
if(IS_EXECUTE)
    add_executable(c_example c_example.c)
    target_link_libraries(c_example uc_sgsim)
    if(WIN32)
        set_target_properties(c_example PROPERTIES SUFFIX ".exe")
    else()
        set_target_properties(c_example PROPERTIES SUFFIX ".out")
    endif()

    # make sure example can run
    add_test(
        c_example
        ${CMAKE_COMMAND}
        --build ${CMAKE_BINARY_DIR}
        --target c_example
    )
    # memory check (valgrind)
    find_program(VALGRIND "valgrind")

    if(VALGRIND)
        add_custom_target(
            memcheck
            COMMAND ${VALGRIND}
            --tool=memcheck
            --leak-check=full
            --track-origins=yes
            --show-reachable=yes
            --error-exitcode=1
            $<TARGET_FILE:c_example>
        )
    endif()
endif()

# unittest
if(IS_EXECUTE)
    project(unittest)
    include(Dart)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage")
    set(CTEST_COVERAGE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/coverage")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test)

    set(CTEST_COVERAGE_EXCLUDE
        '/usr/include/*' # Exclude system headers
        '${CMAKE_CURRENT_SOURCE_DIR}/test/utest.h' # Exclude a specific file
        '${CMAKE_CURRENT_SOURCE_DIR}/src/c_array.c'
        '${CMAKE_CURRENT_SOURCE_DIR}/src/c_array_mt.c'
        '${CMAKE_CURRENT_SOURCE_DIR}/include/c_array.h'
    )
    add_custom_target(
        coverage
        COMMAND ctest -T Coverage
    )

    add_executable(
        unittest
        uc_sgsim/c_core/tests/sgsim_test.c
    )
    target_link_libraries(unittest uc_sgsim)

    add_test(
        unittest
        ${CMAKE_COMMAND}
        --build ${CMAKE_BINARY_DIR}
        --target unittest
    )
endif()
