#include <algorithm>
#include <chrono>
#include <iostream>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>

#include <cuda.h>
#include <nvml.h>

extern "C" {
#include <libavformat/avformat.h>
}

#include "VideoLoader.h"
#include "detail/CUContext.h"
#include "detail/Logger.h"
#include "detail/NvDecoder.h"
#include "detail/utils.h"
#include "detail/cuda/imgproc.h"

#include "nvcuvid/nvcuvid.h"

namespace NVVL {
namespace detail {

NvDecoder::NvDecoder() {
}

NvDecoder::NvDecoder(int device_id,
                     Logger& logger,
                     const CodecParameters* codecpar,
                     AVRational time_base)
    : Decoder{device_id, logger, codecpar},
      device_{}, context_{}, parser_{}, decoder_{},
      time_base_{time_base.num, time_base.den},
      frame_in_use_(32), // 32 is cuvid's max number of decode surfaces
      recv_queue_{}, frame_queue_{}, output_queue_{},
      current_recv_{}, textures_{}, done_{false}
{
    if (!codecpar) {
        return;
    }

    if (!cucall(cuInit(0))) {
        throw std::runtime_error("Unable to initial cuda driver. Is the kernel module installed?");
    }

    if (!cucall(cuDeviceGet(&device_, device_id_))) {
        std::cerr << "Problem getting device info for device "
                  << device_id_ << ", not initializing VideoDecoder\n";
        return;
    }

    char device_name[100];
    if (!cucall(cuDeviceGetName(device_name, 100, device_))) {
        std::cerr << "Problem getting device name for device "
                  << device_id_ << ", not initializing VideoDecoder\n";
        return;
    }
    log_.info() << "Using device: " << device_name << std::endl;

    try {
        auto nvml_ret = nvmlInit();
        if (nvml_ret != NVML_SUCCESS) {
            std::stringstream ss;
            ss << "nvmlInit returned error " << nvml_ret;
            throw std::runtime_error(ss.str());
        }
        char nvmod_version_string[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];
        nvml_ret = nvmlSystemGetDriverVersion(nvmod_version_string,
                                              sizeof(nvmod_version_string));
        if (nvml_ret != NVML_SUCCESS) {
            std::stringstream ss;
            ss << "nvmlSystemGetDriverVersion returned error " << nvml_ret;
            throw std::runtime_error(ss.str());
        }
        auto nvmod_version = std::stof(nvmod_version_string);
        if (nvmod_version < 384.0f) {
            log_.info() << "Older kernel module version " << nvmod_version
                        << " so using the default stream."
                        << std::endl;
            use_default_stream();
        } else {
            log_.info() << "Kernel module version " << nvmod_version
                        << ", so using our own stream."
                        << std::endl;
        }
    } catch(const std::exception& e) {
        log_.warn() << "Unable to get nvidia kernel module version from NVML, "
                    << "conservatively assuming it is an older version.\n"
                    << "The error was: " << e.what()
                    << std::endl;
        use_default_stream();
    }

    context_ = CUContext(device_);
    if (!context_.initialized()) {
        std::cerr << "Problem initializing context, not initializing VideoDecoder\n";
        return;
    }

    auto codec = Codec::H264;
    switch (codecpar->codec_id) {
        case AV_CODEC_ID_H264:
            codec = Codec::H264;
            break;

        case AV_CODEC_ID_HEVC:
            codec = Codec::HEVC;
            break;

        default:
            std::cerr << "Invalid codec for NvDecoder\n";
            return;
    }

    parser_ = CUVideoParser(codec, this, 20, codecpar->extradata,
                            codecpar->extradata_size);
    if (!parser_.initialized()) {
        std::cerr << "Problem creating video parser\n";
        return;
    }

    convert_thread_ = std::thread{&NvDecoder::convert_frames, this};
}

bool NvDecoder::initialized() const {
    return parser_.initialized();
}

int NvDecoder::decode_av_packet(AVPacket* avpkt) {
    if (done_) return 0;

    CUVIDSOURCEDATAPACKET cupkt = {0};

    context_.push();

    if (avpkt && avpkt->size) {
        cupkt.payload_size = avpkt->size;
        cupkt.payload = avpkt->data;
        if (avpkt->pts != AV_NOPTS_VALUE) {
            cupkt.flags = CUVID_PKT_TIMESTAMP;
            if (time_base_.num && time_base_.den) {
                cupkt.timestamp = av_rescale_q(avpkt->pts, time_base_, nv_time_base_);
            } else {
                cupkt.timestamp = avpkt->pts;
            }
        }
    } else {
        cupkt.flags = CUVID_PKT_ENDOFSTREAM;
        // mark as flushing?
    }

    if (!cucall(cuvidParseVideoData(parser_, &cupkt))) {
        std::cerr << "Problem decoding packet" << std::endl;
    }
    return 0;
}

int CUDAAPI NvDecoder::handle_sequence(void* user_data, CUVIDEOFORMAT* format) {
    auto decoder = reinterpret_cast<NvDecoder*>(user_data);
    return decoder->handle_sequence_(format);
}

int CUDAAPI NvDecoder::handle_decode(void* user_data,
                                            CUVIDPICPARAMS* pic_params) {
    auto decoder = reinterpret_cast<NvDecoder*>(user_data);
    return decoder->handle_decode_(pic_params);
}

int CUDAAPI NvDecoder::handle_display(void* user_data,
                                             CUVIDPARSERDISPINFO* disp_info) {
    auto decoder = reinterpret_cast<NvDecoder*>(user_data);
    return decoder->handle_display_(disp_info);
}

int NvDecoder::handle_sequence_(CUVIDEOFORMAT* format) {
    // std::cout << "Frame base is " << format->frame_rate.denominator
    //           << " / " << format->frame_rate.numerator << std::endl;
    // std::cout << "handle_sequence" << std::endl;
    frame_base_ = {static_cast<int>(format->frame_rate.denominator),
                   static_cast<int>(format->frame_rate.numerator)};
    return decoder_.initialize(format);
}

int NvDecoder::handle_decode_(CUVIDPICPARAMS* pic_params) {
    int total_wait = 0;
    constexpr auto sleep_period = 500;
    constexpr auto timeout_sec = 20;
    constexpr auto enable_timeout = false;
    while(frame_in_use_[pic_params->CurrPicIdx]) {
        if (enable_timeout &&
            total_wait++ > timeout_sec * 1000000 / sleep_period) {
            std::cout << device_id_ << ": Waiting for picture "
                      << pic_params->CurrPicIdx
                      << " to become available..." << std::endl;
            std::stringstream ss;
            ss << "Waited too long (" << timeout_sec << " seconds) "
               << "for decode output buffer to become available";
            throw std::runtime_error(ss.str());
        }
        usleep(sleep_period);
        if (done_) return 0;
    }

    log_.info() << "Sending a picture for decode"
                << " size: " << pic_params->nBitstreamDataLen
                << " pic index: " << pic_params->CurrPicIdx
                << std::endl;

    cucall(cuvidDecodePicture(decoder_, pic_params));
    return 1;
}

NvDecoder::MappedFrame::MappedFrame()
    : disp_info{nullptr}, valid_{false} {
}

NvDecoder::MappedFrame::MappedFrame(CUVIDPARSERDISPINFO* disp_info,
                                    CUvideodecoder decoder,
                                    CUstream stream)
    : disp_info{disp_info}, valid_{false}, decoder_(decoder), params_{0} {

    if (!disp_info->progressive_frame) {
        throw std::runtime_error("Got an interlaced frame. We don't do interlaced frames.");
    }

    params_.progressive_frame = disp_info->progressive_frame;
    params_.top_field_first = disp_info->top_field_first;
    params_.second_field = 0;
    params_.output_stream = stream;

    if (!cucall(cuvidMapVideoFrame(decoder_, disp_info->picture_index,
                                   &ptr_, &pitch_, &params_))) {
        throw std::runtime_error("Unable to map video frame");
    }
    valid_ = true;
}

NvDecoder::MappedFrame::MappedFrame(MappedFrame&& other)
    : disp_info{other.disp_info}, valid_{other.valid_}, decoder_{other.decoder_},
      ptr_{other.ptr_}, pitch_{other.pitch_}, params_{other.params_} {
    other.disp_info = nullptr;
    other.valid_ = false;
}

NvDecoder::MappedFrame::~MappedFrame() {
    if (valid_) {
        if (!cucall(cuvidUnmapVideoFrame(decoder_, ptr_))) {
            std::cerr << "Error unmapping video frame\n";
        }
    }
}

uint8_t* NvDecoder::MappedFrame::get_ptr() const {
    return reinterpret_cast<uint8_t*>(ptr_);
}

unsigned int NvDecoder::MappedFrame::get_pitch() const {
    return pitch_;
}

NvDecoder::TextureObject::TextureObject() : valid_{false} {
}

NvDecoder::TextureObject::TextureObject(const cudaResourceDesc* pResDesc,
                                        const cudaTextureDesc* pTexDesc,
                                        const cudaResourceViewDesc* pResViewDesc)
    : valid_{false}
{
    if (!cucall(cudaCreateTextureObject(&object_, pResDesc, pTexDesc, pResViewDesc))) {
        throw std::runtime_error("Unable to create a texture object");
    }
    valid_ = true;
}

NvDecoder::TextureObject::~TextureObject() {
    if (valid_) {
        cudaDestroyTextureObject(object_);
    }
}

NvDecoder::TextureObject::TextureObject(NvDecoder::TextureObject&& other)
    : valid_{other.valid_}, object_{other.object_}
{
    other.valid_ = false;
}

NvDecoder::TextureObject& NvDecoder::TextureObject::operator=(NvDecoder::TextureObject&& other) {
    valid_ = other.valid_;
    object_ = other.object_;
    other.valid_ = false;
    return *this;
}

NvDecoder::TextureObject::operator cudaTextureObject_t() const {
    if (valid_) {
        return object_;
    } else {
        return cudaTextureObject_t{};
    }
}

int NvDecoder::handle_display_(CUVIDPARSERDISPINFO* disp_info) {
    auto frame = av_rescale_q(disp_info->timestamp,
                              nv_time_base_, frame_base_);

    if (current_recv_.count <= 0) {
        if (recv_queue_.empty()) {
            // we aren't expecting anything so just ditch this,
            // guessing it is extra frames.  There is a small chance
            // we are throwing out frames that will later be requested
            // but if we wait here for a request to come in to check,
            // we're stalling the loop that sends requests. We could
            // send requests to the decoder outside of the read_file
            // loop, but that has its own synchronization problems
            // since the decoder is created in that loop, not worth
            // the hassle on the tiny chance we are throwing way good
            // frames here.
            log_.info() << "Ditching frame " << frame << " since "
                        << "the receive queue is empty." << std::endl;
            return 1;
        }
        // std::cout << "Moving on to next request, " << recv_queue_.size()
        //           << " reqs left" << std::endl;
        current_recv_ = recv_queue_.pop();
    }

    if (done_) return 0;

    if (current_recv_.count <= 0) {
        // a new req with count <= 0 probably means we are finishing
        // up and should just ditch this frame
        log_.info() << "Ditching frame " << frame << "since current_recv_.count <= 0" << std::endl;
        return 1;
    }

    if (frame != current_recv_.frame) {
        // TODO This definitely needs better error handling... what if
        // we never get the frame we are waiting for?!
        log_.info() << "Ditching frame " << frame << " since we are waiting for "
                    << "frame " << current_recv_.frame << std::endl;
        return 1;
    }

    log_.info() << "\e[1mGoing ahead with frame " << frame
                << " wanted count: " << current_recv_.count
                << " disp_info->picture_index: " << disp_info->picture_index
                << "\e[0m" << std::endl;

    current_recv_.frame++;
    current_recv_.count--;

    frame_in_use_[disp_info->picture_index] = true;
    frame_queue_.push(disp_info);
    return 1;
}


void NvDecoder::push_req(FrameReq req) {
    recv_queue_.push(std::move(req));
}

void NvDecoder::receive_frames(PictureSequence& sequence) {
    output_queue_.push(&sequence);
}

// we assume here that a pointer, scale_method, and chroma_up_method
// uniquely identifies a texture
const NvDecoder::TextureObjects&
NvDecoder::get_textures(uint8_t* input, unsigned int input_pitch,
                        uint16_t input_width, uint16_t input_height,
                        ScaleMethod scale_method, ChromaUpMethod chroma_method) {
    auto tex_id = std::make_tuple(input, scale_method, chroma_method);
    auto tex = textures_.find(tex_id);
    if (tex != textures_.end()) {
        return tex->second;
    }
    TextureObjects objects;
    cudaTextureDesc tex_desc = {};
    tex_desc.addressMode[0]   = cudaAddressModeClamp;
    tex_desc.addressMode[1]   = cudaAddressModeClamp;
    if (scale_method == ScaleMethod_Nearest) {
        tex_desc.filterMode   = cudaFilterModePoint;
    } else {
        tex_desc.filterMode   = cudaFilterModeLinear;
    }
    tex_desc.readMode         = cudaReadModeNormalizedFloat;
    tex_desc.normalizedCoords = 0;

    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypePitch2D;
    res_desc.res.pitch2D.devPtr = input;
    res_desc.res.pitch2D.desc = cudaCreateChannelDesc<uchar1>();
    res_desc.res.pitch2D.width = input_width;
    res_desc.res.pitch2D.height = input_height;
    res_desc.res.pitch2D.pitchInBytes = input_pitch;

    objects.luma = TextureObject{&res_desc, &tex_desc, nullptr};

    tex_desc.addressMode[0]   = cudaAddressModeClamp;
    tex_desc.addressMode[1]   = cudaAddressModeClamp;
    // only one ChromaUpMethod for now...
    tex_desc.filterMode       = cudaFilterModeLinear;
    tex_desc.readMode         = cudaReadModeNormalizedFloat;
    tex_desc.normalizedCoords = 0;

    res_desc.resType = cudaResourceTypePitch2D;
    res_desc.res.pitch2D.devPtr = input + (input_height * input_pitch);
    res_desc.res.pitch2D.desc = cudaCreateChannelDesc<uchar2>();
    res_desc.res.pitch2D.width = input_width;
    res_desc.res.pitch2D.height = input_height / 2;
    res_desc.res.pitch2D.pitchInBytes = input_pitch;

    objects.chroma = TextureObject{&res_desc, &tex_desc, nullptr};

    auto p = textures_.emplace(tex_id, std::move(objects));
    if (!p.second) {
        throw std::runtime_error("Unable to cache a new texture object.");
    }
    return p.first->second;
}

void NvDecoder::convert_frames() {
    context_.push();
    while (!done_) {
        auto& sequence = *output_queue_.pop();
        if (done_) break;
        for (int i = 0; i < sequence.count(); ++i) {
            log_.debug() << "popping frame (" << i << "/" << sequence.count() << ") "
                         << frame_queue_.size() << " reqs left"
                         << std::endl;
            auto frame = MappedFrame{frame_queue_.pop(), decoder_, stream_};
            if (done_) break;
            convert_frame(frame, sequence, i);
        }
        if (done_) break;
        record_sequence_event_(sequence);
    }
    log_.info() << "Leaving convert frames" << std::endl;
}

void NvDecoder::convert_frame(const MappedFrame& frame, PictureSequence& sequence,
                              int index) {
    auto input_width = decoder_.width();
    auto input_height = decoder_.height();

    foreach_layer(sequence, [&](auto& l) -> void {
            auto output_idx = index;
            if (!l.index_map.empty()) {
                if (l.index_map.size() > static_cast<size_t>(index)) {
                    output_idx = l.index_map[index];
                } else {
                    output_idx = -1;
                }
            }
            if (output_idx < 0) {
                return;
            }
            auto& textures = this->get_textures(frame.get_ptr(),
                                                frame.get_pitch(),
                                                input_width,
                                                input_height,
                                                l.desc.scale_method,
                                                l.desc.chroma_up_method);
            process_frame(textures.chroma, textures.luma,
                          l, output_idx, stream_,
                          input_width, input_height);
        });

    frame_in_use_[frame.disp_info->picture_index] = false;
    auto frame_num = av_rescale_q(frame.disp_info->timestamp,
                                  nv_time_base_, frame_base_);

    sequence.get_or_add_meta<int>("frame_num")[index] = frame_num;
}

void NvDecoder::finish() {
    done_ = true;
    recv_queue_.cancel_pops();
    frame_queue_.cancel_pops();
    output_queue_.cancel_pops();
}

}
}
