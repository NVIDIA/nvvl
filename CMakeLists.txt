cmake_minimum_required(VERSION 3.8)

if(NOT CMAKE_CUDA_FLAGS)
    set(CMAKE_CUDA_FLAGS "-cudart shared")
endif()

project(NVVL VERSION 1.0 LANGUAGES C CXX CUDA)

include(CheckStructHasMember)
include(CheckTypeSize)

include(cmake/BuildType.cmake)
include(cmake/CudaArchFlags.cmake)

# Dependencies
##############
find_package(PkgConfig REQUIRED)
foreach(m avformat avcodec avfilter avutil)
    string(TOUPPER ${m} M)
    pkg_check_modules(${m} REQUIRED IMPORTED_TARGET lib${m})
    list(APPEND FFmpeg_LIBS PkgConfig::${m})
endforeach(m)

set(CMAKE_REQUIRED_INCLUDES ${avformat_INCLUDE_DIRS})
set(CMAKE_REQUIRED_LIBRARIES ${avformat_LIBRARIES})
check_struct_has_member("struct AVStream" codecpar libavformat/avformat.h HAVE_AVSTREAM_CODECPAR LANGUAGE C)
set(CMAKE_EXTRA_INCLUDE_FILES libavcodec/avcodec.h)
check_type_size("AVBSFContext" AVBSFCONTEXT LANGUAGE CXX)

# Targets
#########
add_library(nvvl SHARED
    src/VideoLoader.cpp
    src/PictureSequence.cpp
    src/detail/CUContext.cpp
    src/detail/CUVideoDecoder.cpp
    src/detail/CUVideoParser.cpp
    src/detail/Decoder.cpp
    src/detail/JoiningThread.cpp
    src/detail/NvDecoder.cpp
    src/detail/cuda/imgproc.cu
)

add_library(NVVL::NVVL ALIAS nvvl)

target_compile_features(nvvl PUBLIC cxx_std_14)

target_compile_options(nvvl PRIVATE
    $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wall;-Wno-unused-function>)

if (HAVE_AVSTREAM_CODECPAR)
    target_compile_definitions(nvvl PRIVATE HAVE_AVSTREAM_CODECPAR=1)
endif()

if (HAVE_AVBSFCONTEXT)
    target_compile_definitions(nvvl PRIVATE HAVE_AVBSFCONTEXT=1)
endif()

set_target_properties(nvvl PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    EXPORT_NAME NVVL
    )

target_include_directories(nvvl
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
target_link_libraries(nvvl PRIVATE ${FFmpeg_LIBS} cuda nvcuvid nvidia-ml)

# Installation
##############
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/NVVL)

install(TARGETS nvvl
    EXPORT nvvl-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

#Export the targets to a script
install(EXPORT nvvl-targets
  FILE
    NVVLTargets.cmake
  NAMESPACE
    NVVL::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

#Create a ConfigVersion.cmake file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/NVVLConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(${CMAKE_CURRENT_LIST_DIR}/cmake/NVVLConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/NVVLConfig.cmake
    INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

#Install the config, configversion and custom find modules
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/NVVLConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/NVVLConfigVersion.cmake
    DESTINATION ${INSTALL_CONFIGDIR}
)

# Exporting from the build tree
export(EXPORT nvvl-targets FILE ${CMAKE_CURRENT_BINARY_DIR}/NVVLTargets.cmake NAMESPACE NVVL::)

#Register package in user's package registry
export(PACKAGE NVVL)

add_subdirectory(doc EXCLUDE_FROM_ALL)
add_subdirectory(doc/examples EXCLUDE_FROM_ALL)
