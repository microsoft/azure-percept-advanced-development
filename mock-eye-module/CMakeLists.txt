# CMake project set up
cmake_minimum_required(VERSION 3.1)
project(MockPerceptApp)
set(CMAKE_CXX_STANDARD 11)

# Set the application name. We'll use this as the executable name.
set(APP_NAME mock_eye_app)

# Locate the OpenCV directory. The user or a higher level CMake file must supply
# this, as OpenCV does not contain a FindOpenCV.cmake or OpenCVConfig.cmake file.
find_package(OpenCV REQUIRED gapi highgui)

# Find PkgConfig dependency, which must contain gstreamer
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER gstreamer-1.0>=1.14 REQUIRED)

# For debug purposes
find_package(InferenceEngine 2.0 REQUIRED)

# Slurp up all the .c* and .h* files for the application
file (GLOB MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.c*)
file (GLOB MAIN_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h*)
file (GLOB_RECURSE KERNELS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.c*)
file (GLOB_RECURSE KERNELS_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.h*)
file (GLOB_RECURSE MODULES_SRC ${CMAKE_CURRENT_SOURCE_DIR}/modules/*.c*)
file (GLOB_RECURSE MODULES_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/modules/*.h*)

# This is our target
add_executable(${APP_NAME} ${MAIN_SRC} ${MAIN_HEADERS} ${KERNELS_SRC} ${KERNELS_HEADERS} ${MODULES_SRC} ${MODULES_HEADERS})

# Add Warnings
target_compile_options(${APP_NAME} PRIVATE -Wall -Wextra -pedantic -Werror -Wno-unused-parameter)

# We need a few directories.
target_include_directories(${APP_NAME}
  PRIVATE
    ${GSTREAMER_INCLUDE_DIRS}
  )

# We need to link to some shared object files
target_link_libraries(${APP_NAME}
  PRIVATE
    ${OpenCV_LIBS}
    ${InferenceEngine_LIBRARIES}  # for debug
    ${GSTREAMER_LIBRARIES}
    pthread
    m
)
