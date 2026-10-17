find_package(Python3 COMPONENTS Interpreter QUIET)

if(BUILD_TESTING AND Python3_Interpreter_FOUND)
  add_test(
    NAME v200_consumer_fixture_cpu
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_consumer.py")
  set_tests_properties(v200_consumer_fixture_cpu PROPERTIES LABELS "CPU;consumer")
endif()
