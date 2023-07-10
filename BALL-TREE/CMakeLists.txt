project("Ball-tree")
cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
 
include_directories($(CMAKE_CURRENT_SOURCE_DIR)/src)

find_package(Eigen3)

add_executable(ball_tree src/main.cpp)
target_link_libraries(ball_tree Eigen3::Eigen)
