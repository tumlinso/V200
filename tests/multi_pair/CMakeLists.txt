cmake_minimum_required(VERSION 3.18)

find_package(CUDAToolkit 12 REQUIRED)

# This executable is launched only by the finite controller-backed B06
# consumer runner. Building it is CPU-only; the LD_PRELOAD frontdoor supplied
# at runtime turns its CUDA calls into the two-logical-device smoke probe.
add_executable(v200_multi_pair_copy_probe copy_probe.cpp)
target_link_libraries(v200_multi_pair_copy_probe PRIVATE CUDA::cudart ${CMAKE_DL_LIBS})
target_compile_features(v200_multi_pair_copy_probe PRIVATE cxx_std_17)
target_compile_options(v200_multi_pair_copy_probe PRIVATE -Wall -Wextra -Wpedantic)

if(BUILD_TESTING)
  # The copy contract uses fake Driver API functions and does not initialize a
  # CUDA device, so it remains safe in the ordinary host test gate.
  if(EXISTS "${PROJECT_SOURCE_DIR}/tests/multi_pair/copy_contract.cpp"
      AND EXISTS "${PROJECT_SOURCE_DIR}/src/multi_pair/copy.cpp")
    add_executable(v200_multi_pair_copy_contract
      "${PROJECT_SOURCE_DIR}/tests/multi_pair/copy_contract.cpp"
      "${PROJECT_SOURCE_DIR}/src/multi_pair/copy.cpp")
    target_include_directories(v200_multi_pair_copy_contract PRIVATE
      "${PROJECT_SOURCE_DIR}/include" "${CUDAToolkit_INCLUDE_DIRS}")
    target_compile_features(v200_multi_pair_copy_contract PRIVATE cxx_std_17)
    target_compile_options(v200_multi_pair_copy_contract PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(v200_multi_pair_copy_contract PRIVATE v200::build_options ${CMAKE_DL_LIBS})
    add_test(NAME v200_multi_pair_copy_contract COMMAND v200_multi_pair_copy_contract)
  endif()

  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    add_test(NAME v200_multi_pair_host
      COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/multi_pair/test_multi_pair.py")
  endif()
endif()
