#include <iostream>

#ifdef HAVE_OPENCV
# include <opencv2/imgcodecs.hpp>
# include <opencv2/cudaimgproc.hpp>
# include <opencv2/cudaarithm.hpp>
#endif

#include <cuda.h>

#include "VideoLoader.h"
#include "cuda/utils.h"

constexpr auto sequence_width = uint16_t{1280/2};
constexpr auto sequence_height = uint16_t{720/2};
constexpr auto sequence_count = uint16_t{4};
constexpr auto scale_width = int16_t{1280/2};
constexpr auto scale_height = int16_t{720/2};

using PictureSequence = NVVL::PictureSequence;

template<typename T>
T* new_data(size_t* pitch, size_t width, size_t height) {
    T* data;
    if(cudaMallocPitch(&data, pitch, width * sizeof(T), height) != cudaSuccess) {
        throw std::runtime_error("Unable to allocate buffer in device memory");
    }
    return data;
}

// just use one buffer for each different type
template<typename T>
auto get_data(size_t* ret_pitch) {
    static size_t pitch;
    static auto data = std::unique_ptr<T, decltype(&cudaFree)>{
        new_data<T>(&pitch, sequence_width, sequence_height * sequence_count * 3),
        cudaFree};
    *ret_pitch = pitch / sizeof(T);
    return data.get();
}

#ifdef HAVE_OPENCV
template<typename T>
cv::cuda::GpuMat get_pixels(const PictureSequence& sequence, int index,
                            std::initializer_list<int> channel_order) {
    auto pixels = sequence.get_layer<T>("data", index);
    auto type = cv::DataType<T>::type;
    auto channels = std::vector<cv::cuda::GpuMat>();
    for (auto i : channel_order) {
        channels.emplace_back(pixels.desc.height, pixels.desc.width, type,
                              pixels.data + pixels.desc.stride.c*i,
                              pixels.desc.stride.y * sizeof(T));
    }
    auto tmp = cv::cuda::GpuMat();
    cv::cuda::merge(channels, tmp);
    auto out = cv::cuda::GpuMat();
    tmp.convertTo(out, CV_8U, pixels.desc.normalized ? 255.0 : 1.0);
    return out;
}

template<>
cv::cuda::GpuMat get_pixels<half>(const PictureSequence& sequence, int index,
                                  std::initializer_list<int> channel_order) {
    auto pixels = sequence.get_layer<half>("data", index);
    auto channels = std::vector<cv::cuda::GpuMat>();
    for (auto i : channel_order) {
        auto channel = cv::cuda::GpuMat(pixels.desc.height, pixels.desc.width, CV_32FC1);

        half2float(pixels.data + pixels.desc.stride.c*i, pixels.desc.stride.y,
                   pixels.desc.width, pixels.desc.height,
                   channel.ptr<float>(), channel.step1());
        channels.push_back(channel);
    }
    auto tmp = cv::cuda::GpuMat();
    cv::cuda::merge(channels, tmp);
    auto out = cv::cuda::GpuMat();
    tmp.convertTo(out, CV_8U, pixels.desc.normalized ? 255.0 : 1.0);
    return out;
}

template<typename T>
void write_frame(const PictureSequence& sequence) {
    auto frame_nums = sequence.get_meta<int>("frame_num");
    for (int i = 0; i < sequence.count(); ++i) {
        auto pixels = sequence.get_layer<T>("data", i);

        auto gpu_bgr = cv::cuda::GpuMat();
        if (pixels.desc.color_space == ColorSpace_RGB) {
            gpu_bgr = get_pixels<T>(sequence, i, {2, 1, 0});
        } else {
            auto gpu_yuv = get_pixels<T>(sequence, i, {0, 2, 1});
            cv::cuda::cvtColor(gpu_yuv, gpu_bgr, CV_YCrCb2BGR);
        }

        cv::Mat host_bgr;
        gpu_bgr.download(host_bgr);

        char output_file[256];
        auto frame_num = frame_nums[i];
        sprintf(output_file,"./output/frames/%05d.png",frame_num);
        cv::imwrite(output_file,host_bgr);
        std::cout << "Wrote frame " << frame_num << std::endl;
    }
}
#else // no OpenCV
template<typename T>
struct host_type{ using type = T; };

template<>
struct host_type<half>{ using type = float; };

template<typename T>
typename host_type<T>::type* dev_data(const PictureSequence::Layer<T>& layer, size_t* pitch) {
    *pitch = layer.desc.stride.y;
    return layer.data;
}

template<>
host_type<half>::type* dev_data<half>(const PictureSequence::Layer<half>& layer, size_t* pitch) {
    auto dev_floats = get_data<float>(pitch);

    half2float(layer.data, layer.desc.stride.y,
               layer.desc.width, layer.desc.height,
               dev_floats, *pitch);
    return dev_floats;
}

template<typename T>
void write_frame(const PictureSequence& sequence) {
    constexpr auto sample_count = 100;
    auto frame_nums = sequence.get_meta<int>("frame_num");
    std::cout << "Got a sequence of size: " << sequence.count() << std::endl;
    for (int i = 0; i < sequence.count(); ++i) {
        auto pixels = sequence.get_layer<T>("data", i);
        size_t data_stride = 0;
        auto data = dev_data(pixels, &data_stride);

        typename host_type<T>::type tmp[sample_count];
        uint32_t sum = 0;

        for (int c = 0; c < 3; ++c) {
            if (cudaMemcpy(tmp, data + data_stride*pixels.desc.height*c,
                           sample_count * sizeof(*data), cudaMemcpyDeviceToHost)
                != cudaSuccess) {
                throw std::runtime_error("Couldn't copy frame data to cpu");
            }

            for (int i = 0; i < sample_count; i++) {
                sum += static_cast<uint32_t>(tmp[i]);
            }
        }
        std::cout << " Frame " << frame_nums[i]
                  << " sum (first " << sample_count << " of each channel): "
                  << sum << std::endl;
    }
}
#endif

NVVL::VideoLoader* loader;

template<typename T>
void process_frames(NVVL::VideoLoader& loader, NVVL::ColorSpace color_space,
                    bool scale, bool normalized, bool flip,
                    NVVL::ScaleMethod scale_method = ScaleMethod_Linear)
{
    auto s = PictureSequence{sequence_count};

    auto pixels = PictureSequence::Layer<T>{};
    pixels.data = get_data<T>(&pixels.desc.stride.y);
    pixels.desc.count = sequence_count;
    pixels.desc.channels = 3;
    pixels.desc.width = sequence_width;
    pixels.desc.height = sequence_height;
    if (scale) {
        pixels.desc.scale_width = scale_width;
        pixels.desc.scale_height = scale_height;
    }
    pixels.desc.horiz_flip = flip;
    pixels.desc.normalized = normalized;
    pixels.desc.color_space = color_space;
    pixels.desc.scale_method = scale_method;
    pixels.desc.stride.x = 1;
    pixels.desc.stride.c = pixels.desc.stride.y * pixels.desc.height;
    pixels.desc.stride.n = pixels.desc.stride.c * 3;
    s.set_layer("data", pixels);

    loader.receive_frames_sync(s);
    write_frame<T>(s);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "usage: " << argv[0] << " <video file>\n";
        return -1;
    }

    auto loader = NVVL::VideoLoader{0};

    auto filename = argv[1];
    auto frame_count = loader.frame_count(filename);
    std::cout << "Looks like there are " << frame_count << " frames" << std::endl;

    // just enqueue all the frames, probably won't use them all
    loader.read_sequence(filename, 0, frame_count);

    //             type               color space     scale  norm   flip
    process_frames<uint8_t>(loader, ColorSpace_RGB,   false, false, false); // 0-3
    process_frames<uint8_t>(loader, ColorSpace_RGB,   false, false, true);  // 4-7
    process_frames<uint8_t>(loader, ColorSpace_RGB,   true,  false, false); // 8-11
    process_frames<uint8_t>(loader, ColorSpace_RGB,   true,  false, true);  // 12-15
    process_frames<uint8_t>(loader, ColorSpace_YCbCr, false, false, false); // 16-19
    process_frames<uint8_t>(loader, ColorSpace_YCbCr, true,  false, false); // 20-23
    process_frames<float>  (loader, ColorSpace_RGB,   false, false, false); // 24-27
    process_frames<float>  (loader, ColorSpace_RGB,   true,  false, false); // 28-31
    process_frames<float>  (loader, ColorSpace_RGB,   true,  true,  false); // 32-35
    process_frames<float>  (loader, ColorSpace_YCbCr, true,  false, false); // 36-39
    process_frames<float>  (loader, ColorSpace_YCbCr, true,  true,  false); // 40-43
    process_frames<half>   (loader, ColorSpace_RGB,   false, false, false); // 44-47
    process_frames<half>   (loader, ColorSpace_RGB,   true,  false, false); // 48-51
    process_frames<half>   (loader, ColorSpace_RGB,   true,  true,  false); // 52-55
    process_frames<half>   (loader, ColorSpace_YCbCr, true,  false, false); // 56-59
    process_frames<half>   (loader, ColorSpace_YCbCr, true,  true,  false); // 60-63

    auto stats = loader.get_stats();
    std::cout << "Total video packets read: " << stats.packets_read
              << " (" << stats.bytes_read << " bytes)\n"
              << "Total frames used: " << stats.frames_used
              << std::endl;

    return 0;
}
