cmake_minimum_required(VERSION 3.24)
if(APPLE)
    project(streaming_llama LANGUAGES CXX)
else()
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
    project(streaming_llama LANGUAGES CXX CUDA)
endif()

set(CMAKE_CXX_STANDARD 23)

if(APPLE)
    set(EXTRA_CXX_FLAGS "-DACCELERATE_NEW_LAPACK")
    set(EXTRA_LFLAGS "-framework Accelerate -framework Foundation -framework MetalPerformanceShaders")
else()
    set(EXTRA_INCDIRS "${MKLROOT}/include" "/usr/local/cuda/include")
    set(EXTRA_CXX_FLAGS "-DMKL_ILP64  -m64  -I${MKLROOT}/include")
    #set(EXTRA_LFLAGS "-L${MKLROOT}/lib/ -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl") # single thread
    #set(EXTRA_LFLAGS "-L${MKLROOT}/lib/ -Wl,--no-as-needed -lmkl_intel_ilp64 -ltbb -lmkl_tbb_thread -lmkl_core -lpthread -lm -ldl") # "thread building blocks" threading
    set(EXTRA_LFLAGS "stdc++_libbacktrace -L/usr/local/cuda/lib64 -L${MKLROOT}/lib/ -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl") # intel openmp  threading
endif()


set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Werror ${EXTRA_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g ${EXTRA_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_SANADDR "-fsanitize=address -fno-omit-frame-pointer -O1 -g -Wno-unqualified-std-cast-call ${EXTRA_CXX_FLAGS}")
if(NOT APPLE)
    set(CMAKE_CXX_FLAGS_SANMEM "-fsanitize=leak -fsanitize-recover=all -fPIE -fno-omit-frame-pointer -O1 -g -Wno-unqualified-std-cast-call ${EXTRA_CXX_FLAGS}")
endif()
set(CMAKE_LINKER_FLAGS_SANMEM "-pie")

include_directories(src/ ${EXTRA_INCDIRS})

if(NOT APPLE)
    find_package(CUDA REQUIRED)
    set(CUDA_PTX_COMPILATION ON)

    set(CUDA_SRC
        src/cuda/swilu.cu
        src/cuda/rope.cu
        src/cuda/layerNorm.cu
    )

    foreach(src ${CUDA_SRC})
        get_filename_component(filename ${src} NAME_WLE)
        set(ptx_filename "${CMAKE_BINARY_DIR}/${filename}.ptx")
        add_custom_command(
            OUTPUT ${ptx_filename}
            COMMAND ${CMAKE_CUDA_COMPILER} -ptx -o ${ptx_filename} ${CMAKE_SOURCE_DIR}/${src}
            DEPENDS ${src}
        )
        list(APPEND PTX_FILES ${ptx_filename})
    endforeach()

    add_custom_target(cuda_ptx ALL DEPENDS ${PTX_FILES})
    set_target_properties(cuda_ptx PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
endif()

set(COMMON_SRC
        src/Common.h
        src/Common.cpp
)

set(SRC
        src/Bf16.cpp
        src/Bf16.h
        src/Checker.cpp
        src/Checker.h
        src/Cuda.cpp
        src/Cuda.h
        src/Exception.h
        src/LlamaModel.h
        src/Matmul.cpp
        src/Matmul.h
        src/NonTransformerWeights.h
        src/Scratch.h
        src/Socket.h
        src/Socket.cpp
        src/Timer.cpp
        src/Timer.h
        src/TransformerBlockScratch.cpp
        src/TransformerBlockScratch.h
        src/TransformerBlock.h
        src/Weights.h
)

add_executable(deparallelizer ${COMMON_SRC} src/Deparallelizer.cpp)

add_executable(evaluator ${COMMON_SRC} ${SRC} src/Evaluator.cpp)
target_link_libraries(evaluator ${EXTRA_LFLAGS})

add_executable(testSgemm src/TestSgemm.cpp src/Timer.cpp src/Timer.h)
target_link_libraries(testSgemm ${EXTRA_LFLAGS})

add_executable(testCuda ${SRC} ${COMMON_SRC} src/TestCuda.cpp)
target_link_libraries(testCuda  ${EXTRA_LFLAGS} cuda cublas)
