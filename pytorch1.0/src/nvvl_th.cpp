#include <torch/extension.h>
#include <THC/THC.h>
#include "PictureSequence.h"
#include "VideoLoader.h"

extern THCState *state;

namespace py = pybind11;


PYBIND11_MODULE(_nvvl, m) {
    py::enum_<NVVL_ScaleMethod>(m, "ScaleMethod")
            .value("ScaleMethod_Nearest", ScaleMethod_Nearest,
                   "The value for the nearest neighbor is used, no interpolation")
            .value("ScaleMethod_Linear", ScaleMethod_Linear,
                   "Simple bilinear interpolation of four nearest neighbors")
            .export_values();

    py::enum_<NVVL_ChromaUpMethod>(m, "ChromaUpMethod")
            .value("ChromaUpMethod_Linear", ChromaUpMethod_Linear,
                   "Simple bilinear interpolation of four nearest neighbors")
            .export_values();

    py::enum_<NVVL_ColorSpace>(m, "ColorSpace")
            .value("ColorSpace_RGB", ColorSpace_RGB,
                   "channel 0 is red, 1 is green, 2 is blue")
            .value("ColorSpace_YCbCr", ColorSpace_YCbCr,
                   "channel 0 is luma, 1 is blue-difference chroma, 2 is red-difference chroma")
            .export_values();

    py::class_<NVVL_Strides>(m, "NVVL_Strides")
            .def_readwrite("x", &NVVL_Strides::x)
            .def_readwrite("y", &NVVL_Strides::y)
            .def_readwrite("c", &NVVL_Strides::c)
            .def_readwrite("n", &NVVL_Strides::n);

    py::class_<NVVL_LayerDesc>(m, "LayerDesc")
            .def(py::init<>())
            .def_readwrite("count", &NVVL_LayerDesc::count)
            .def_readwrite("channels", &NVVL_LayerDesc::channels)
            .def_readwrite("width", &NVVL_LayerDesc::width)
            .def_readwrite("height", &NVVL_LayerDesc::height)
            .def_readwrite("crop_x", &NVVL_LayerDesc::crop_x)
            .def_readwrite("crop_y", &NVVL_LayerDesc::crop_y)
            .def_readwrite("scale_width", &NVVL_LayerDesc::scale_width)
            .def_readwrite("scale_height", &NVVL_LayerDesc::scale_height)
            .def_readwrite("horiz_flip", &NVVL_LayerDesc::horiz_flip)
            .def_readwrite("normalized", &NVVL_LayerDesc::normalized)
            .def_readwrite("color_space", &NVVL_LayerDesc::color_space)
            .def_readwrite("chroma_up_method", &NVVL_LayerDesc::chroma_up_method)
            .def_readwrite("scale_method", &NVVL_LayerDesc::scale_method)
            .def_readwrite("stride", &NVVL_LayerDesc::stride);

    py::class_<NVVL::PictureSequence> PictureSequence(m, "PictureSequence");

    py::class_<NVVL::PictureSequence::Layer<float>>(PictureSequence, "FloatLayer")
               .def(py::init<>())
               .def_readwrite("desc", &NVVL::PictureSequence::Layer<float>::desc)
               .def_readwrite("index_map", &NVVL::PictureSequence::Layer<float>::index_map)
               .def("set_data", [](NVVL::PictureSequence::Layer<float>& l, int64_t ptr) {
                       l.data = reinterpret_cast<float*>(ptr);
                   });

    py::class_<NVVL::PictureSequence::Layer<half>>(PictureSequence, "HalfLayer")
               .def(py::init<>())
               .def_readwrite("desc", &NVVL::PictureSequence::Layer<half>::desc)
               .def_readwrite("index_map", &NVVL::PictureSequence::Layer<half>::index_map)
               .def("set_data", [](NVVL::PictureSequence::Layer<half>& l, int64_t ptr) {
                       l.data = reinterpret_cast<half*>(ptr);
                   });

    py::class_<NVVL::PictureSequence::Layer<uint8_t>>(PictureSequence, "ByteLayer")
               .def(py::init<>())
               .def_readwrite("desc", &NVVL::PictureSequence::Layer<uint8_t>::desc)
               .def_readwrite("index_map", &NVVL::PictureSequence::Layer<uint8_t>::index_map)
               .def("set_data", [](NVVL::PictureSequence::Layer<uint8_t>& l, int64_t ptr) {
                       l.data = reinterpret_cast<uint8_t*>(ptr);
                   });

    PictureSequence
               .def(py::init<uint16_t, int>())
               .def("set_layer",
                    (void (NVVL::PictureSequence::*)
                     (std::string, const NVVL::PictureSequence::Layer<float>&))
                    &NVVL::PictureSequence::set_layer<float>)
               .def("set_layer",
                    (void (NVVL::PictureSequence::*)
                     (std::string, const NVVL::PictureSequence::Layer<half>&))
                    &NVVL::PictureSequence::set_layer<half>)
               .def("set_layer",
                    (void (NVVL::PictureSequence::*)
                     (std::string, const NVVL::PictureSequence::Layer<uint8_t>&))
                    &NVVL::PictureSequence::set_layer<uint8_t>)
               .def("count", &NVVL::PictureSequence::count)
               .def("wait", (void (NVVL::PictureSequence::*)() const)
                    &NVVL::PictureSequence::wait)
               .def("wait_stream", [](NVVL::PictureSequence& seq) {
                       seq.wait(THCState_getCurrentStream(state));
                   });


    py::enum_<LogLevel>(m, "LogLevel")
               .value("LogLevel_Debug", LogLevel_Debug, "")
               .value("LogLevel_Info", LogLevel_Info, "")
               .value("LogLevel_Warn", LogLevel_Warn, "")
               .value("LogLevel_Error", LogLevel_Error, "")
               .value("LogLevel_None", LogLevel_None, "")
               .export_values();

    py::class_<Size>(m, "Size")
               .def(py::init<>())
               .def_readwrite("width", &Size::width)
               .def_readwrite("height", &Size::height);

    py::class_<VideoLoaderStats>(m, "VideoLoaderStats")
               .def(py::init<>())
               .def_readwrite("bytes_read", &VideoLoaderStats::bytes_read)
               .def_readwrite("packets_read", &VideoLoaderStats::packets_read)
               .def_readwrite("bytes_decoded", &VideoLoaderStats::bytes_decoded)
               .def_readwrite("packets_decoded", &VideoLoaderStats::packets_decoded)
               .def_readwrite("frames_used", &VideoLoaderStats::frames_used);

    py::class_<NVVL::VideoLoader>(m, "VideoLoader")
               .def(py::init<int, LogLevel>())
               .def("frame_count", &NVVL::VideoLoader::frame_count)
               .def("size", &NVVL::VideoLoader::size)
               .def("read_sequence", &NVVL::VideoLoader::read_sequence)
               .def("receive_frames", &NVVL::VideoLoader::receive_frames)
               .def("receive_frames_sync", &NVVL::VideoLoader::receive_frames_sync)
               .def("get_stats", &NVVL::VideoLoader::get_stats)
               .def("reset_stats", &NVVL::VideoLoader::reset_stats)
               .def("set_log_level", &NVVL::VideoLoader::set_log_level);

    m.def("video_size_from_file", &nvvl_video_size_from_file);
}
