cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(cbow)

find_package(Torch REQUIRED)
INCLUDE_DIRECTORIES(${TORCH_LIBRARIES})
add_executable(cbow main.cpp)
target_link_libraries(cbow "${TORCH_LIBRARIES}")
set_property(TARGET cbow PROPERTY CXX_STANDARD 11)