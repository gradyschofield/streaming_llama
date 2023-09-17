cmake_minimum_required(VERSION 3.26)
project(streaming_llama)

set(CMAKE_CXX_STANDARD 23)

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories(src/)

set(COMMON_SRC src/Common.h)

add_executable(deparallelizer ${COMMON_SRC} src/Deparallelizer.cpp)
add_executable(evaluator ${COMMON_SRC} src/Evaluator.cpp)
if(APPLE)
    target_link_libraries(evaluator "-framework Accelerate")
else()
    target_link_libraries(evaluator)
endif()
