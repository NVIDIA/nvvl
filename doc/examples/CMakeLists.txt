add_executable(extract_frames
    extract_frames.cpp
    cuda/utils.cu
    )
target_link_libraries(extract_frames PUBLIC NVVL::NVVL)

find_package(OpenCV CONFIG OPTIONAL_COMPONENTS core cudaarithm cudaimgproc imgcodecs)

if(OPENCV_FOUND)
    target_compile_definitions(extract_frames PRIVATE HAVE_OPENCV=1)
    target_link_libraries(extract_frames PUBLIC opencv_core PRIVATE ${OpenCV_LIBS})
endif()

add_executable(extract_frames_c
    extract_frames_c.c
    cuda/utils.cu
    )
target_link_libraries(extract_frames_c PUBLIC NVVL::NVVL)

add_custom_target(examples DEPENDS extract_frames extract_frames_c)
