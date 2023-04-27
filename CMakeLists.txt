cmake_minimum_required(VERSION 3.5)
project(ncnn_sam)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pie -fPIE -fPIC -Wall -g")

find_package(OpenCV REQUIRED)

if(OpenCV_FOUND)
    message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
    message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "opencv Not Found!")
endif(OpenCV_FOUND)

find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message(FATAL_ERROR "OpenMP Not Found!")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ncnn/include/ncnn)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/ncnn/lib)

# set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/)
add_executable(ncnn_sam src/demo.cpp src/segment_anything.cpp)
target_link_libraries(ncnn_sam ncnn ${OpenCV_LIBS})
