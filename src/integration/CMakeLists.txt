cmake_minimum_required(VERSION 3.18)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  message(FATAL_ERROR "v200_pair_init is built from the V200 root so its singleton modules are linked exactly once")
endif()

add_library(v200_pair_init SHARED runtime_init.cpp)
add_library(v200::pair_init ALIAS v200_pair_init)
set_target_properties(v200_pair_init PROPERTIES
  OUTPUT_NAME v200_pair_init
  POSITION_INDEPENDENT_CODE ON)
target_include_directories(v200_pair_init PUBLIC
  "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>")
target_link_libraries(v200_pair_init PRIVATE
  v200::memory v200::execution v200::frontdoor ${CMAKE_DL_LIBS})
target_link_libraries(v200_pair_init PRIVATE v200::build_options)

if(BUILD_TESTING)
  add_executable(v200_runtime_init_tests
    "${PROJECT_SOURCE_DIR}/tests/integration/runtime_init_test.cpp")
  target_link_libraries(v200_runtime_init_tests PRIVATE v200::pair_init v200::build_options)
  add_test(NAME v200_runtime_init_host COMMAND v200_runtime_init_tests)
endif()

option(V200_INTEGRATION_BUILD_GPU_SMOKE
       "Build the opt-in real pair initializer and both-owner memory closure (never runs it)" OFF)
if(V200_INTEGRATION_BUILD_GPU_SMOKE)
  find_package(CUDAToolkit 12 REQUIRED)
  add_executable(v200_pair_memory_gpu_smoke
    "${PROJECT_SOURCE_DIR}/tests/integration/pair_memory_gpu_smoke.cpp")
  target_include_directories(v200_pair_memory_gpu_smoke PRIVATE
    "${PROJECT_SOURCE_DIR}/include")
  target_link_libraries(v200_pair_memory_gpu_smoke PRIVATE
    v200::pair_init v200::frontdoor CUDA::cuda_driver)
  target_compile_features(v200_pair_memory_gpu_smoke PRIVATE cxx_std_17)
  target_compile_options(v200_pair_memory_gpu_smoke PRIVATE -Wall -Wextra -Wpedantic)
endif()
