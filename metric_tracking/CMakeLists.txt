cmake_minimum_required(VERSION 3.8)

project(main)

add_executable(main main.cpp)

set_property(TARGET main PROPERTY CXX_STANDARD 17)

add_subdirectory(restclient-cpp)
target_link_libraries(main PUBLIC restclient-cpp)

# CMakeLists.txt
add_subdirectory(json)
target_link_libraries(main PRIVATE nlohmann_json::nlohmann_json)