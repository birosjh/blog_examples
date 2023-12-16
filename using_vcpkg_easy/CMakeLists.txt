cmake_minimum_required(VERSION 3.8)

set(CMAKE_TOOLCHAIN_FILE /usr/local/vcpkg/scripts/buildsystems/vcpkg.cmake)

project(main)

add_executable(main main.cpp)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries(main PRIVATE Eigen3::Eigen)