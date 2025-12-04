cmake_minimum_required(VERSION 3.15)

project(demo)

add_executable(demo main.cpp)
set_property(TARGET demo PROPERTY CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
target_link_libraries(demo ${OpenCV_LIBS})
include_directories(${OpenCV_INCLUDE_DIRS})