#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>

#include <cuda.h>

#include <libavcodec/avcodec.h>

#include "VideoLoader.h"
#include "detail/Decoder.h"
#include "detail/CUContext.h"
#include "detail/CUVideoParser.h"
#include "detail/CUVideoDecoder.h"
#include "detail/JoiningThread.h"
#include "detail/Queue.h"

#include "nvcuvid/nvcuvid.h"

namespace NVVL {
namespace detail {

class Logger;

class NvDecoder : public Decoder
{
  public:
    NvDecoder();

    NvDecoder(int device_id, Logger& logger,
              const AVCodecParameters* codecpar,
              AVRational time_base);

    bool initialized() const;

    static int CUDAAPI handle_sequence(void* user_data, CUVIDEOFORMAT* format);
    static int CUDAAPI handle_decode(void* user_data, CUVIDPICPARAMS* pic_params);
    static int CUDAAPI handle_display(void* user_data, CUVIDPARSERDISPINFO* disp_info);

    void push_req(FrameReq req) final;

    void receive_frames(PictureSequence& batch) final;

    void finish() final;

  protected:
    int decode_av_packet(AVPacket* pkt) final;

  private:

    class MappedFrame {
      public:
        MappedFrame();
        MappedFrame(CUVIDPARSERDISPINFO* disp_info, CUvideodecoder decoder,
                    CUstream stream);
        ~MappedFrame();
        MappedFrame(const MappedFrame&) = delete;
        MappedFrame& operator=(const MappedFrame&) = delete;
        MappedFrame(MappedFrame&& other);
        MappedFrame& operator=(MappedFrame&&) = delete;

        uint8_t* get_ptr() const;
        unsigned int get_pitch() const;

        CUVIDPARSERDISPINFO* disp_info;
      private:
        bool valid_;
        CUvideodecoder decoder_;
        CUdeviceptr ptr_;
        unsigned int pitch_;
        CUVIDPROCPARAMS params_;
    };

    class TextureObject {
      public:
        TextureObject();
        TextureObject(const cudaResourceDesc* pResDesc,
                      const cudaTextureDesc* pTexDesc,
                      const cudaResourceViewDesc* pResViewDesc);
        ~TextureObject();
        TextureObject(TextureObject&& other);
        TextureObject& operator=(TextureObject&& other);
        TextureObject(const TextureObject&) = delete;
        TextureObject& operator=(const TextureObject&) = delete;
        operator cudaTextureObject_t() const;
      private:
        bool valid_;
        cudaTextureObject_t object_;
    };

    struct TextureObjects {
        TextureObject luma;
        TextureObject chroma;
    };

    CUdevice device_;
    CUContext context_;
    CUVideoParser parser_;
    CUVideoDecoder decoder_;

    AVRational time_base_;
    AVRational nv_time_base_ = {1, 10000000};
    AVRational frame_base_;

    std::vector<uint8_t> frame_in_use_;
    Queue<FrameReq> recv_queue_;
    Queue<CUVIDPARSERDISPINFO*> frame_queue_;
    Queue<PictureSequence*> output_queue_;
    FrameReq current_recv_;

    using TexID = std::tuple<uint8_t*, ScaleMethod, ChromaUpMethod>;
    struct tex_hash {
        std::hash<uint8_t*> ptr_hash;
        std::hash<int> scale_hash;
        std::hash<int> up_hash;
        std::size_t operator () (const TexID& tex) const {
            return ptr_hash(std::get<0>(tex))
                    ^ scale_hash(std::get<1>(tex))
                    ^ up_hash(std::get<2>(tex));
        }
    };

    std::unordered_map<TexID, TextureObjects, tex_hash> textures_;

    bool done_;

    JoiningThread convert_thread_;

    int handle_sequence_(CUVIDEOFORMAT* format);
    int handle_decode_(CUVIDPICPARAMS* pic_params);
    int handle_display_(CUVIDPARSERDISPINFO* disp_info);

    const TextureObjects& get_textures(uint8_t* input, unsigned int input_pitch,
                                       uint16_t input_width, uint16_t input_height,
                                       ScaleMethod scale_method, ChromaUpMethod chroma_method);
    void convert_frames();
    void convert_frame(const MappedFrame& frame, PictureSequence& sequence,
                       int index);
};

}
}
