cmake_minimum_required(VERSION 3.3)
project(calculate_param_dwa)

# 添加c++ 11标准支持
set( CMAKE_CXX_FLAGS "-std=c++11" )

# 添加 OpenCV 和eign 库
find_package(OpenCV REQUIRED )
find_package(Eigen3 REQUIRED)

# 添加头文件
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(${EIGEN3_INCLUDE_DIRS})

add_executable(calculate_param_dwa src/main.cc src/calculate_param_dwa.cc)

# 链接OpenCV库
target_link_libraries(calculate_param_dwa ${OpenCV_LIBS} )