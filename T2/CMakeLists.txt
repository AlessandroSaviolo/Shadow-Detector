cmake_minimum_required(VERSION 2.8)
project(T2)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} include)

add_library(ShadowDetector src/ShadowDetector.cpp src/Utility.cpp)

target_link_libraries(ShadowDetector ${OpenCV_LIBS})

add_executable(T2 src/main.cpp)

target_link_libraries(T2 ShadowDetector)