cmake_minimum_required(VERSION 3.18)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(v200_frontdoor_standalone LANGUAGES CXX)
  include(CTest)
  set(V200_FRONTDOOR_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
else()
  set(V200_FRONTDOOR_ROOT "${PROJECT_SOURCE_DIR}")
endif()

if(NOT TARGET v200_frontdoor)
  find_path(V200_CUDA_INCLUDE_DIR cuda.h
    HINTS
      "${V200_CUDA_TOOLKIT_ROOT}/include"
      "$ENV{CUDA_HOME}/include"
      "/usr/local/cuda/include"
      "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/targets/x86_64-linux/include"
      "/usr/include")
  find_path(V200_CUDART_INCLUDE_DIR cuda_runtime_api.h
    HINTS
      "${V200_CUDA_TOOLKIT_ROOT}/include"
      "$ENV{CUDA_HOME}/include"
      "/usr/local/cuda/include"
      "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/targets/x86_64-linux/include"
      "/usr/include")
  if(NOT V200_CUDA_INCLUDE_DIR OR NOT V200_CUDART_INCLUDE_DIR)
    message(FATAL_ERROR "V200 frontdoor requires authentic CUDA 12.x headers; set V200_CUDA_TOOLKIT_ROOT")
  endif()

  file(STRINGS "${V200_CUDA_INCLUDE_DIR}/cuda.h" _v200_cuda_version_line
    REGEX "^#define CUDA_VERSION [0-9]+")
  if(NOT _v200_cuda_version_line MATCHES "CUDA_VERSION ([0-9]+)")
    message(FATAL_ERROR "Could not identify CUDA header version at ${V200_CUDA_INCLUDE_DIR}")
  endif()
  set(_v200_cuda_version "${CMAKE_MATCH_1}")
  if(_v200_cuda_version LESS 12000 OR _v200_cuda_version GREATER_EQUAL 13000)
    message(FATAL_ERROR "V200 frontdoor requires CUDA 12.x headers; found CUDA_VERSION=${_v200_cuda_version}")
  endif()

  add_library(v200_frontdoor SHARED
    abi_profile.cpp
    runtime_state.cpp
    cuda_wrappers.cpp)
  add_library(v200::frontdoor ALIAS v200_frontdoor)
  target_include_directories(v200_frontdoor
    PUBLIC "${V200_FRONTDOOR_ROOT}/include"
    PRIVATE "${V200_CUDA_INCLUDE_DIR}" "${V200_CUDART_INCLUDE_DIR}" .)
  target_link_libraries(v200_frontdoor PRIVATE ${CMAKE_DL_LIBS} pthread)
  target_compile_definitions(v200_frontdoor PRIVATE V200_FRONTDOOR_BUILD=1)
  set_target_properties(v200_frontdoor PROPERTIES OUTPUT_NAME v200_frontdoor)
  if(TARGET v200::build_options)
    target_link_libraries(v200_frontdoor PRIVATE v200::build_options)
  endif()

  add_library(v200_abi_audit SHARED audit.cpp)
  set_target_properties(v200_abi_audit PROPERTIES OUTPUT_NAME v200_abi_audit)

  if(BUILD_TESTING)
    add_executable(v200_abi_profile_test
      "${V200_FRONTDOOR_ROOT}/tests/abi/abi_profile_test.cpp"
      abi_profile.cpp)
    target_include_directories(v200_abi_profile_test PRIVATE "${V200_FRONTDOOR_ROOT}/include")
    if(TARGET v200::build_options)
      target_link_libraries(v200_abi_profile_test PRIVATE v200::build_options)
    else()
      target_compile_features(v200_abi_profile_test PRIVATE cxx_std_17)
      target_compile_options(v200_abi_profile_test PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME v200_abi_profile COMMAND v200_abi_profile_test)

    add_executable(v200_frontdoor_closure
      "${V200_FRONTDOOR_ROOT}/tests/abi/frontdoor_closure.cpp")
    target_include_directories(v200_frontdoor_closure PRIVATE
      "${V200_FRONTDOOR_ROOT}/include" "${V200_CUDA_INCLUDE_DIR}" "${V200_CUDART_INCLUDE_DIR}")
    target_link_libraries(v200_frontdoor_closure PRIVATE v200_frontdoor ${CMAKE_DL_LIBS})
    target_compile_features(v200_frontdoor_closure PRIVATE cxx_std_17)
    add_test(NAME v200_frontdoor_export_closure COMMAND v200_frontdoor_closure)

    add_library(v200_fake_cuda_driver SHARED
      "${V200_FRONTDOOR_ROOT}/tests/abi/fake_cuda_library.c")
    set_source_files_properties("${V200_FRONTDOOR_ROOT}/tests/abi/fake_cuda_library.c"
      PROPERTIES LANGUAGE CXX)
    target_compile_definitions(v200_fake_cuda_driver PRIVATE V200_FAKE_DRIVER=1)
    target_include_directories(v200_fake_cuda_driver PRIVATE "${V200_CUDA_INCLUDE_DIR}" "${V200_CUDART_INCLUDE_DIR}")
    set_target_properties(v200_fake_cuda_driver PROPERTIES
      OUTPUT_NAME cuda VERSION 1.0 SOVERSION 1)

    add_library(v200_fake_cudart SHARED
      "${V200_FRONTDOOR_ROOT}/tests/abi/fake_cuda_library.c")
    target_include_directories(v200_fake_cudart PRIVATE "${V200_CUDA_INCLUDE_DIR}" "${V200_CUDART_INCLUDE_DIR}")
    set_target_properties(v200_fake_cudart PROPERTIES
      OUTPUT_NAME cudart VERSION 12.0 SOVERSION 12)

    add_executable(v200_rtld_local_closure_test
      "${V200_FRONTDOOR_ROOT}/tests/abi/rtld_local_closure.cpp")
    target_include_directories(v200_rtld_local_closure_test PRIVATE
      "${V200_CUDA_INCLUDE_DIR}" "${V200_CUDART_INCLUDE_DIR}")
    target_link_libraries(v200_rtld_local_closure_test PRIVATE v200_frontdoor ${CMAKE_DL_LIBS})
    target_compile_features(v200_rtld_local_closure_test PRIVATE cxx_std_17)
    add_test(NAME v200_rtld_local_native_dispatch
      COMMAND v200_rtld_local_closure_test
        "$<TARGET_FILE:v200_fake_cuda_driver>" "$<TARGET_FILE:v200_fake_cudart>")

    add_library(v200_fake_lazy_cuda_driver SHARED
      "${V200_FRONTDOOR_ROOT}/tests/abi/fake_cuda_library.c")
    target_compile_definitions(v200_fake_lazy_cuda_driver PRIVATE
      V200_FAKE_DRIVER=1 V200_FAKE_LAZY_DRIVER=1)
    target_include_directories(v200_fake_lazy_cuda_driver PRIVATE
      "${V200_CUDA_INCLUDE_DIR}" "${V200_CUDART_INCLUDE_DIR}")
    set_target_properties(v200_fake_lazy_cuda_driver PROPERTIES
      OUTPUT_NAME cuda VERSION 1.0 SOVERSION 1
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lazy-driver")

    add_library(v200_fake_lazy_cudart SHARED
      "${V200_FRONTDOOR_ROOT}/tests/abi/fake_cuda_library.c")
    target_compile_definitions(v200_fake_lazy_cudart PRIVATE V200_FAKE_LAZY_RUNTIME=1)
    target_include_directories(v200_fake_lazy_cudart PRIVATE
      "${V200_CUDA_INCLUDE_DIR}" "${V200_CUDART_INCLUDE_DIR}")
    target_link_libraries(v200_fake_lazy_cudart PRIVATE ${CMAKE_DL_LIBS})
    set_target_properties(v200_fake_lazy_cudart PROPERTIES
      OUTPUT_NAME cudart VERSION 12.0 SOVERSION 12
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lazy-runtime")

    add_executable(v200_lazy_driver_discovery_test
      "${V200_FRONTDOOR_ROOT}/tests/abi/lazy_driver_discovery.cpp")
    target_include_directories(v200_lazy_driver_discovery_test PRIVATE
      "${V200_FRONTDOOR_ROOT}/include" "${V200_CUDA_INCLUDE_DIR}"
      "${V200_CUDART_INCLUDE_DIR}")
    target_link_libraries(v200_lazy_driver_discovery_test PRIVATE
      v200_frontdoor ${CMAKE_DL_LIBS})
    target_compile_features(v200_lazy_driver_discovery_test PRIVATE cxx_std_17)
    add_test(NAME v200_lazy_runtime_driver_discovery
      COMMAND v200_lazy_driver_discovery_test
        "$<TARGET_FILE:v200_fake_lazy_cudart>"
        "$<TARGET_FILE:v200_fake_lazy_cuda_driver>")

    add_executable(v200_vmm_location_test
      "${V200_FRONTDOOR_ROOT}/tests/abi/vmm_location_test.cpp")
    target_include_directories(v200_vmm_location_test PRIVATE .)
    target_compile_features(v200_vmm_location_test PRIVATE cxx_std_17)
    if(TARGET v200::build_options)
      target_link_libraries(v200_vmm_location_test PRIVATE v200::build_options)
    else()
      target_compile_options(v200_vmm_location_test PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME v200_vmm_device_location_mapping COMMAND v200_vmm_location_test)
  endif()
endif()
