cmake_minimum_required(VERSION 3.10)
project(ToolfilmAI_CPP_Modules LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add flexibility in finding Python
# Look for different Python versions
find_package(Python3 COMPONENTS Development REQUIRED)
message(STATUS "Found Python: ${Python3_EXECUTABLE}")
message(STATUS "Python include dirs: ${Python3_INCLUDE_DIRS}")

# Find pybind11
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    # Try to find pybind11 from Python
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_include())"
        OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(PYBIND11_INCLUDE_DIR)
        message(STATUS "Found pybind11 via Python: ${PYBIND11_INCLUDE_DIR}")
        include_directories(${PYBIND11_INCLUDE_DIR})
    else()
        message(FATAL_ERROR "Could not find pybind11. Please install it with: pip install pybind11")
    endif()
else()
    message(STATUS "Found pybind11 via CMake")
    include_directories(${pybind11_INCLUDE_DIRS})
endif()

# Find OpenCV with more flexibility
find_package(OpenCV QUIET)
if(NOT OpenCV_FOUND)
    # Try to find OpenCV in common locations
    set(OpenCV_PATHS
        $ENV{OPENCV_DIR}
        "C:/opencv/build"
        "C:/tools/opencv/build"
        "/usr/local"
        "/opt/homebrew"
    )
    
    foreach(path ${OpenCV_PATHS})
        if(EXISTS "${path}/include/opencv2/opencv.hpp")
            message(STATUS "Found OpenCV at: ${path}")
            set(OpenCV_INCLUDE_DIRS "${path}/include")
            
            # Find OpenCV libraries
            file(GLOB OpenCV_LIBS 
                "${path}/lib/opencv_*.lib"
                "${path}/lib/libopencv_*.so"
                "${path}/lib/libopencv_*.dylib"
            )
            
            if(OpenCV_LIBS)
                set(OpenCV_FOUND TRUE)
                break()
            endif()
        endif()
    endforeach()
    
    if(NOT OpenCV_FOUND)
        message(WARNING "OpenCV not found. The build may fail.")
    endif()
endif()

message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

# Add include directories for Python and pybind11
include_directories(${Python3_INCLUDE_DIRS})

# Check for CUDA with more flexibility
find_package(CUDA QUIET)
if(NOT CUDA_FOUND AND DEFINED ENV{CUDA_HOME})
    set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_HOME})
    find_package(CUDA QUIET)
endif()

if(CUDA_FOUND)
    message(STATUS "CUDA found at: ${CUDA_TOOLKIT_ROOT_DIR}")
    enable_language(CUDA)
    add_definitions(-DWITH_CUDA)
    set(CUDA_LIBS ${CUDA_LIBRARIES})
    include_directories(${CUDA_INCLUDE_DIRS})
else()
    message(STATUS "CUDA not found, building without CUDA support")
endif()

# Check for FFmpeg
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED
    libavformat
    libavcodec
    libavutil
    libswscale
)
include_directories(${FFMPEG_INCLUDE_DIRS})

# Optical Flow module
if(CUDA_FOUND)
    add_library(optical_flow_cpp MODULE 
        optical_flow.cpp 
        optical_flow_cuda.cu
    )
else()
    add_library(optical_flow_cpp MODULE optical_flow.cpp)
endif()

target_include_directories(optical_flow_cpp PRIVATE 
    ${Python3_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(optical_flow_cpp PRIVATE 
    ${Python3_LIBRARIES}
    ${OpenCV_LIBS}
    pybind11::module
)

if(CUDA_FOUND)
    target_link_libraries(optical_flow_cpp PRIVATE ${CUDA_LIBS})
endif()

# Video Extract module
add_library(video_extract_cpp MODULE video_extract.cpp)
target_include_directories(video_extract_cpp PRIVATE 
    ${Python3_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(video_extract_cpp PRIVATE 
    ${Python3_LIBRARIES}
    ${OpenCV_LIBS}
    pybind11::module
)

# Video Concat module
add_library(video_concat_cpp MODULE video_concat.cpp)
target_include_directories(video_concat_cpp PRIVATE 
    ${Python3_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
    ${FFMPEG_INCLUDE_DIRS}
)
target_link_libraries(video_concat_cpp PRIVATE 
    ${Python3_LIBRARIES}
    ${FFMPEG_LIBRARIES}
    pybind11::module
)

# Set the output directory for the modules
set_target_properties(
    optical_flow_cpp
    video_extract_cpp
    video_concat_cpp
    PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/.."
)
