cmake_minimum_required(VERSION 3.8)

project(main)

add_executable(main main.cpp)

set_property(TARGET main PROPERTY CXX_STANDARD 17)

add_library(writer STATIC writer/writer.h writer/writer.cpp)
target_include_directories(writer PUBLIC writer)

target_link_libraries(main PUBLIC writer)