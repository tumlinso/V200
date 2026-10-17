cmake_minimum_required(VERSION 3.18)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(V200Memory LANGUAGES CXX)
  include(CTest)
  find_package(CUDAToolkit 12 REQUIRED)
  if(NOT TARGET v200_build_options)
    add_library(v200_build_options INTERFACE)
    add_library(v200::build_options ALIAS v200_build_options)
    target_compile_features(v200_build_options INTERFACE cxx_std_17)
    target_compile_options(v200_build_options INTERFACE -Wall -Wextra -Wpedantic)
  endif()
  set(V200_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")
else()
  find_package(CUDAToolkit 12 REQUIRED)
  set(V200_SOURCE_ROOT "${PROJECT_SOURCE_DIR}")
endif()

add_library(v200_memory STATIC
  manager.cpp
  planner.cpp)
add_library(v200::memory ALIAS v200_memory)
set_target_properties(v200_memory PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(v200_memory PUBLIC v200::build_options)
target_include_directories(v200_memory PUBLIC
  "$<BUILD_INTERFACE:${V200_SOURCE_ROOT}/include>"
  "$<BUILD_INTERFACE:${CUDAToolkit_INCLUDE_DIRS}>")

option(V200_MEMORY_BUILD_TESTS "Build CPU-only memory ledger tests" ${BUILD_TESTING})
if(V200_MEMORY_BUILD_TESTS)
  find_package(Threads REQUIRED)
  add_executable(v200_memory_tests "${V200_SOURCE_ROOT}/tests/memory/test_memory.cpp")
  target_link_libraries(v200_memory_tests PRIVATE v200::memory Threads::Threads)
  add_test(NAME v200_memory_tests COMMAND v200_memory_tests)
endif()
