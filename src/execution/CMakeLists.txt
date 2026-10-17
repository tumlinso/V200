cmake_minimum_required(VERSION 3.18)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(v200_execution LANGUAGES CXX)
  include(CTest)
endif()

if(NOT TARGET v200_build_options)
  add_library(v200_build_options INTERFACE)
  target_compile_features(v200_build_options INTERFACE cxx_std_17)
  target_include_directories(v200_build_options INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(v200_build_options INTERFACE -Wall -Wextra -Wpedantic)
  endif()
endif()

find_package(CUDAToolkit 12 REQUIRED)

add_library(v200_execution STATIC owner_execution.cpp owner_execution_c.cpp)
add_library(v200::execution ALIAS v200_execution)
set_target_properties(v200_execution PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(v200_execution PUBLIC v200_build_options)
target_include_directories(v200_execution PUBLIC ${CUDAToolkit_INCLUDE_DIRS})

if(BUILD_TESTING)
  add_executable(v200_execution_tests
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/execution/owner_execution_test.cpp")
  target_link_libraries(v200_execution_tests PRIVATE v200::execution)
  add_test(NAME v200_execution_host COMMAND v200_execution_tests)
endif()

option(V200_EXECUTION_BUILD_GPU_SMOKE
       "Build the finite opt-in real Driver API completion smoke (never runs it)" OFF)
if(V200_EXECUTION_BUILD_GPU_SMOKE)
  add_executable(v200_execution_gpu_smoke
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/execution/owner_execution_gpu_smoke.cpp")
  target_link_libraries(v200_execution_gpu_smoke PRIVATE
    v200::execution CUDA::cuda_driver)
endif()
