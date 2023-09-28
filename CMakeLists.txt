cmake_minimum_required(VERSION 3.26)
project(streaming_llama)

set(CMAKE_CXX_STANDARD 23)

if(APPLE)
    set(EXTRA_CXX_FLAGS "-DACCELERATE_NEW_LAPACK")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Werror ${EXTRA_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 ${EXTRA_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_SANADDR "-fsanitize=address -fno-omit-frame-pointer -O1 -g -Wno-unqualified-std-cast-call ${EXTRA_CXX_FLAGS}")
if(NOT APPLE)
    set(CMAKE_CXX_FLAGS_SANMEM "-fsanitize=leak -fsanitize-recover=all -fPIE -fno-omit-frame-pointer -O1 -g -Wno-unqualified-std-cast-call ${EXTRA_CXX_FLAGS}")
endif()
set(CMAKE_LINKER_FLAGS_SANMEM "-pie")

include_directories(src/)

set(COMMON_SRC
        src/Common.h
        src/Common.cpp
)

set(SRC src/Socket.h
        src/Socket.cpp
        src/Timer.cpp
        src/Timer.h)

add_executable(deparallelizer ${COMMON_SRC} src/Deparallelizer.cpp)
add_executable(evaluator ${COMMON_SRC} ${SRC} src/Evaluator.cpp)
if(APPLE)
    target_link_libraries(evaluator "-framework Accelerate -framework Foundation -framework MetalPerformanceShaders")
else()
    target_link_libraries(evaluator)
endif()

add_executable(testSgemm src/TestSgemm.cpp src/Timer.cpp src/Timer.h)
target_link_libraries(testSgemm "-framework Accelerate")
