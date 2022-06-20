cmake_minimum_required(VERSION 3.21)
project(static-huffman)

set(CMAKE_CXX_STANDARD 20)

add_executable(static-huffman main.cpp)
set_target_properties(static-huffman PROPERTIES COMPILE_FLAGS "-O3 -flto -s -fvisibility=hidden -fvisibility-inlines-hidden -funroll-loops -fno-asynchronous-unwind-tables -fno-exceptions -fno-rtti -fno-plt -fPIC -Xlinker -unexported_symbol -Xlinker *")

find_package(fmt)
target_link_libraries(static-huffman fmt::fmt)