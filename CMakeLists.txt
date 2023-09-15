cmake_minimum_required(VERSION 3.26)
project(streaming_llama)

set(CMAKE_CXX_STANDARD 23)

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_executable(deparallelizer src/Deparallelizer.cpp)
