cmake_minimum_required(VERSION 3.18)

if(NOT TARGET v200_build_options)
  add_library(v200_build_options INTERFACE)
  add_library(v200::build_options ALIAS v200_build_options)
  target_compile_features(v200_build_options INTERFACE cxx_std_17)
  target_include_directories(v200_build_options INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
endif()

find_package(CUDAToolkit 12 REQUIRED)

add_library(v200_operators SHARED
  graph_rewrite.cpp
  launch.cpp
  mmvq_plan.cpp
  pair_runtime.cpp
  ../multi_pair/copy.cpp)
add_library(v200::operators ALIAS v200_operators)
set_target_properties(v200_operators PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(v200_operators PRIVATE
  v200::build_options v200::pair_init v200::frontdoor ${CMAKE_DL_LIBS})
target_include_directories(v200_operators PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
  "${CUDAToolkit_INCLUDE_DIRS}")
set_target_properties(v200_operators PROPERTIES OUTPUT_NAME v200_operators)

if(BUILD_TESTING)
  add_executable(v200_operator_plan_tests
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/operators/test_mmvq_plan.cpp"
    mmvq_plan.cpp)
  target_link_libraries(v200_operator_plan_tests PRIVATE v200::build_options)
  target_include_directories(v200_operator_plan_tests PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  add_test(NAME v200_operator_mmvq_plan COMMAND v200_operator_plan_tests)

  add_executable(v200_operator_metadata_tests
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/operators/test_operator_metadata.cpp")
  target_link_libraries(v200_operator_metadata_tests PRIVATE v200::operators)
  add_test(NAME v200_operator_mmvq_metadata COMMAND v200_operator_metadata_tests)

endif()
