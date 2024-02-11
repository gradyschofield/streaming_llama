cmake_minimum_required(VERSION 3.24)
project(streaming_llama LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)

set(EXTRA_CXX_FLAGS "-DACCELERATE_NEW_LAPACK")
set(EXTRA_LFLAGS "-framework Accelerate -framework Foundation -framework QuartzCore -framework Metal")

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall ${EXTRA_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g ${EXTRA_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_SANADDR "-fsanitize=address -fno-omit-frame-pointer -O1 -g -Wno-unqualified-std-cast-call ${EXTRA_CXX_FLAGS}")
set(CMAKE_LINKER_FLAGS_SANMEM "-pie")

include_directories(src/ ${EXTRA_INCDIRS})

set(COMMON_SRC
        src/Common.h
        src/Common.cpp
)

set(SRC
        src/Bf16.cpp
        src/Bf16.h
        src/Checker.cpp
        src/Checker.h
        src/EvaluationTimings.h
        src/Exception.h
        src/LlamaModel.h
        src/LayerNormalization.h
        src/Masking.h
        src/Matmul.cpp
        src/Matmul.h
        src/Metal.hpp
        src/MetalHelpers.cpp
        src/MetalHelpers.h
        src/NonTransformerWeights.h
        src/Scratch.h
        src/Socket.h
        src/Socket.cpp
        src/Timer.cpp
        src/Timer.h
        src/TransformerBlockScratch.h
        src/TransformerBlock.h
        src/Weights.h
)

add_executable(deparallelizer ${COMMON_SRC} src/Deparallelizer.cpp)

add_executable(evaluator ${COMMON_SRC} ${SRC} src/Evaluator.cpp)
target_link_libraries(evaluator ${EXTRA_LFLAGS})

add_executable(testSgemm src/TestSgemm.cpp src/Timer.cpp src/Timer.h)
target_link_libraries(testSgemm ${EXTRA_LFLAGS})

add_executable(testMemoryBandwidth src/TestMemoryBandwidth.cpp src/Timer.cpp src/Timer.h)
target_link_libraries(testMemoryBandwidth ${EXTRA_LFLAGS})

add_executable(testSimpleMatvec src/TestSimpleMatvec.cpp src/Timer.cpp src/Timer.h)
target_link_libraries(testSimpleMatvec ${EXTRA_LFLAGS})
