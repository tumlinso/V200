cmake_minimum_required(VERSION 3.18)
project(V200 LANGUAGES CXX)

include(CTest)
if(BUILD_TESTING)
  enable_testing()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(V200_SHARED_INCLUDE_ROOT "${PROJECT_SOURCE_DIR}/include" CACHE PATH
    "Shared public include root for V200 components")
add_library(v200_build_options INTERFACE)
add_library(v200::build_options ALIAS v200_build_options)
target_compile_features(v200_build_options INTERFACE cxx_std_17)
if(EXISTS "${V200_SHARED_INCLUDE_ROOT}")
  target_include_directories(v200_build_options INTERFACE "${V200_SHARED_INCLUDE_ROOT}")
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(v200_build_options INTERFACE -Wall -Wextra -Wpedantic)
endif()

option(V200_BUILD_BOOTSTRAP_EXAMPLES
       "Build the package's CPU planning example and its host test" ON)
option(V200_ENABLE_GPU_TESTS
       "Permit GPU tests in worker subprojects after controller admission" OFF)
option(V200_BUILD_CUDA_PROBE
       "Compile the package's explicit CUDA Driver API probe (never runs it)" OFF)

if(V200_BUILD_BOOTSTRAP_EXAMPLES)
  add_subdirectory(planning/v200-bootstrap/examples)
endif()

if(BUILD_TESTING)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    add_test(
      NAME v200_source_profile_schema
      COMMAND "${Python3_EXECUTABLE}"
              "${PROJECT_SOURCE_DIR}/tests/host/validate_source_profile.py"
              "${PROJECT_SOURCE_DIR}/docs/v200/source-profile.json")
  endif()
endif()

# Work streams may add these bounded subprojects independently.  Missing
# directories remain harmless during the CPU foundation build.
foreach(v200_component IN ITEMS
    src/frontdoor
    src/memory
    src/execution
    tests/consumer)
  if(EXISTS "${PROJECT_SOURCE_DIR}/${v200_component}/CMakeLists.txt")
    add_subdirectory("${v200_component}")
  endif()
endforeach()
