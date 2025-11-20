cmake_minimum_required(VERSION 3.18)
project(mmapeak CUDA CXX)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA package
find_package(CUDA REQUIRED)

# Add CUDA executable
add_executable(mmapeak mmapeak.cu)
if (EXISTS "/etc/nv_tegra_release")
  set(CUDA_ARCH "72")
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.4)
    list(APPEND CUDA_ARCH "87")
  endif()
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
    list(APPEND CUDA_ARCH "110")
	list(REMOVE_ITEM CUDA_ARCH "72")
  endif()
else()
  set(CUDA_ARCH "70" "75")
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.1)
    list(APPEND CUDA_ARCH "80" "86")
  endif()
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
    list(APPEND CUDA_ARCH "89")
  endif()
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)
    list(APPEND CUDA_ARCH "90")
  endif()
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
    list(APPEND CUDA_ARCH "100" "120a")
  endif()
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
    list(APPEND CUDA_ARCH "121a")
    list(REMOVE_ITEM CUDA_ARCH "70")
  endif()
endif()
message(STATUS "CUDA_ARCHITECTURES: ${CUDA_ARCH}")
set_target_properties(mmapeak PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH}")

# Set compiler flags
set_target_properties(mmapeak PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF)

target_compile_options(mmapeak PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Wno-deprecated-gpu-targets>)

# Install the executable
install(TARGETS mmapeak
    RUNTIME DESTINATION bin)
