cmake_minimum_required(VERSION 3.20)
project(mccl LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0")
endif()

# Standard install layout — defines CMAKE_INSTALL_INCLUDEDIR etc.; needed before the install-interface
# include path below references it.
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Publish the central definitions header (generated from .in).
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/definitions.h.in
               ${CMAKE_CURRENT_SOURCE_DIR}/src/definitions.h COPYONLY)

# Every TU; probe main()s are #ifdef MCCL_*_MAIN so the library builds main-free.
file(GLOB_RECURSE MCCL_SOURCES CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
     ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/src/*.mm)

add_library(mccl STATIC ${MCCL_SOURCES})
target_compile_options(mccl PRIVATE -Wall -Wextra)
target_include_directories(mccl PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/mccl/include>)
target_link_libraries(mccl PUBLIC "-framework Metal" "-framework Foundation" "-framework IOKit" "-framework CoreFoundation")

# ARC for the Obj-C++ (Metal) sources.
file(GLOB_RECURSE MCCL_MM CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.mm)
set_source_files_properties(${MCCL_MM} PROPERTIES COMPILE_FLAGS "-fobjc-arc")

# reduce.metal is compiled at runtime (newLibraryWithSource); ship it beside the binaries so the
# executable-dir fallback in metal.mm finds it.
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/device/reduce.metal
               ${CMAKE_CURRENT_BINARY_DIR}/reduce.metal COPYONLY)

# Install: libmccl.a + the public headers (src/ layout preserved so the relative includes resolve) +
# the runtime Metal source (mccl compiles reduce.metal at load; point MCCL_METAL_DIR at the installed
# copy). A CMake package-config is generated so consumers can `find_package(MCCL)`.
install(TARGETS mccl EXPORT mcclTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mccl
        FILES_MATCHING PATTERN "*.h")
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/definitions.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mccl)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/device/reduce.metal DESTINATION ${CMAKE_INSTALL_DATADIR}/mccl)

set(MCCL_CMAKE_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/MCCL)
install(EXPORT mcclTargets NAMESPACE mccl:: DESTINATION ${MCCL_CMAKE_DIR})
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/MCCLConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/MCCLConfig.cmake
    INSTALL_DESTINATION ${MCCL_CMAKE_DIR}
    PATH_VARS CMAKE_INSTALL_DATADIR)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/MCCLConfigVersion.cmake
    VERSION 0.1.0 COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/MCCLConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/MCCLConfigVersion.cmake
        DESTINATION ${MCCL_CMAKE_DIR})
