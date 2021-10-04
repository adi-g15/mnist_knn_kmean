cmake_minimum_required(VERSION 3.16)

project(mnist_knn_kmean
	LANGUAGES CXX
	)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)
#add_compile_options(-fmodules-ts)

# Include CPP.cmake
include(cmake/CPM.cmake)
CPMAddPackage("gh:adi-g15/endian@1.0")

include_directories(include/)

add_executable(${CMAKE_PROJECT_NAME} src/main.cpp)
add_custom_command( TARGET ${CMAKE_PROJECT_NAME} POST_BUILD
		    COMMAND ${CMAKE_COMMAND} -E copy_directory
		    ${CMAKE_SOURCE_DIR}/dataset $<TARGET_FILE_DIR:${CMAKE_PROJECT_NAME}>/dataset
			)

add_library(knn src/knn.cpp)
add_library(kmean src/kmean.cpp)
find_package(Threads REQUIRED)
find_package(TBB REQUIRED)

target_link_libraries(${CMAKE_PROJECT_NAME} Threads::Threads tbb)
target_link_libraries(${CMAKE_PROJECT_NAME} kmean knn)
target_link_libraries(${CMAKE_PROJECT_NAME} endian)
