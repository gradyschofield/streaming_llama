cmake_minimum_required(VERSION 3.26)
project(streaming_llama)

set(CMAKE_CXX_STANDARD 23)

if(APPLE)
    set(EXTRA_CXX_FLAGS "-DACCELERATE_NEW_LAPACK")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g ${EXTRA_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 ${EXTRA_CXX_FLAGS}")

include_directories(src/)

set(COMMON_SRC src/Common.h)

add_executable(deparallelizer ${COMMON_SRC} src/Deparallelizer.cpp)
add_executable(evaluator ${COMMON_SRC} src/Evaluator.cpp src/Timer.h src/Timer.cpp)
if(APPLE)
    target_link_libraries(evaluator "-framework Accelerate")
else()
    target_link_libraries(evaluator)
endif()

add_executable(testSgemm src/TestSgemm.cpp src/Timer.cpp src/Timer.h)
target_link_libraries(testSgemm "-framework Accelerate")