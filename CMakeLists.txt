cmake_minimum_required(VERSION 3.10)
project(intnn C)

set(CMAKE_C_STANDARD 99)

include_directories(include)

file(GLOB SOURCES "src/*.c")

add_executable(main ${SOURCES})

file(COPY dataset DESTINATION ${CMAKE_BINARY_DIR})
