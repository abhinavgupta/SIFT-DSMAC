cmake_minimum_required(VERSION 2.3)

project(SIFT_Descriptor)
find_package (OpenCV REQUIRED)
add_executable (SIFT_Descriptor SIFT_Descriptor.cpp)
target_link_libraries ( SIFT_Descriptor ${OpenCV_LIBS} )
