#include <assert.h>

#include <chrono>
#include <thread>

#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>

extern "C" {
#include <libavformat/avformat.h>
}

#include "VideoLoader.h"
#include "PictureSequence.h"
#include "detail/Decoder.h"
#include "detail/JoiningThread.h"
#include "detail/Logger.h"
#include "detail/NvDecoder.h"
#include "detail/utils.h"

namespace {

#undef av_err2str
std::string av_err2str(int errnum) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(errnum, errbuf, AV_ERROR_MAX_STRING_SIZE);
    return std::string{errbuf};
}

// libav resource free function take the address of a pointer...
template<typename T>
class AVDeleter {
  public:
    AVDeleter() : deleter_(nullptr) {}
    AVDeleter(std::function<void(T**)> deleter) : deleter_{deleter} {}

    void operator()(T *p) {
        deleter_(&p);
    }
  private:
    std::function<void(T**)> deleter_;
};

// except for the old AVBitSTreamFilterContext
#ifndef HAVE_AVBSFCONTEXT
class BSFDeleter {
  public:
    void operator()(AVBitStreamFilterContext* bsf) {
        av_bitstream_filter_close(bsf);
    }
};
#endif

template<typename T>
using av_unique_ptr = std::unique_ptr<T, AVDeleter<T>>;

template<typename T>
av_unique_ptr<T> make_unique_av(T* raw_ptr, void (*deleter)(T**)) {
    return av_unique_ptr<T>(raw_ptr, AVDeleter<T>(deleter));
}

#ifdef HAVE_AVSTREAM_CODECPAR
auto codecpar(AVStream* stream) -> decltype(stream->codecpar) {
    return stream->codecpar;
}
#else
auto codecpar(AVStream* stream) -> decltype(stream->codec) {
    return stream->codec;
}
#endif

} // anonymous namespace

namespace NVVL {

class VideoLoader::impl {
  public:
    impl(int device_id, LogLevel log_level);
    int frame_count(std::string filename);
    Size size() const;
    void read_sequence(std::string filename, int frame, int count=1);
    void receive_frames(PictureSequence& sequence);
    VideoLoaderStats get_stats() const;
    void reset_stats();
    void set_log_level(LogLevel level);
    void finish();
  private:
    struct OpenFile {
        bool open = false;
        AVRational frame_base_;
        AVRational stream_base_;
        int frame_count_;

        int vid_stream_idx_;
        int last_frame_;

#ifdef HAVE_AVBSFCONTEXT
        av_unique_ptr<AVBSFContext> bsf_ctx_;
#else
        using bsf_ptr = std::unique_ptr<AVBitStreamFilterContext, BSFDeleter>;
        bsf_ptr bsf_ctx_;
        AVCodecContext* codec;
#endif
        av_unique_ptr<AVFormatContext> fmt_ctx_;
    };

    OpenFile& get_or_open_file(std::string filename);

    std::unordered_map<std::string, OpenFile> open_files_;

    int device_id_;
    VideoLoaderStats stats_;

    uint16_t width_;
    uint16_t height_;
    int codec_id_;

    std::atomic<bool> done_;
    detail::Logger log_;
    std::unique_ptr<detail::Decoder> vid_decoder_;
    detail::Queue<detail::FrameReq> send_queue_;

    // this needs to be last so that it is destroyed first so that the
    // above remain valid until the file_reader_ thread is done
    detail::JoiningThread file_reader_;

    // Are these good numbers? Allow them to be set?
    static constexpr auto frames_used_warning_ratio = 3.0f;
    static constexpr auto frames_used_warning_minimum = 1000;
    static constexpr auto frames_used_warning_interval = 10000;

    void read_file();
    void seek(OpenFile& file, int frame);
};

VideoLoader::VideoLoader(int device_id)
    : VideoLoader(device_id, LogLevel_Warn) {
}

VideoLoader::VideoLoader(int device_id, LogLevel log_level)
    : pImpl{std::make_unique<impl>(device_id, log_level)}
{}

VideoLoader::impl::impl(int device_id, LogLevel log_level)
    : device_id_{device_id}, stats_{},
      width_{0}, height_{0}, codec_id_{0},
      done_{false}, log_{log_level} {
    av_register_all();

    file_reader_ = std::thread{&VideoLoader::impl::read_file, this};
}

// Willing to break rule of zero here to clean up threads
VideoLoader::~VideoLoader() {
    pImpl->finish();
}

void VideoLoader::impl::finish() {
    done_ = true;
    send_queue_.cancel_pops();
    log_.info() << "Finishing VideoLoader" << std::endl;
    if (vid_decoder_) {
        vid_decoder_->finish();
    }
}

VideoLoader::VideoLoader(VideoLoader&&) = default;
VideoLoader& VideoLoader::operator=(VideoLoader&&) = default;

int VideoLoader::frame_count(std::string filename) {
    return pImpl->frame_count(filename);
}

int VideoLoader::impl::frame_count(std::string filename) {
    return get_or_open_file(filename).frame_count_;
}


void VideoLoader::read_sequence(std::string filename, int frame, int count) {
    pImpl->read_sequence(filename, frame, count);
}

void VideoLoader::impl::read_sequence(std::string filename, int frame, int count) {
    auto req = detail::FrameReq{filename, frame, count};
    // give both reader thread and decoder a copy of what is coming
    send_queue_.push(req);
}

Size VideoLoader::size() const {
    return pImpl->size();
}

Size VideoLoader::impl::size() const {
    return Size{width_, height_};
}

VideoLoader::impl::OpenFile& VideoLoader::impl::get_or_open_file(std::string filename) {
    auto& file = open_files_[filename];

    if (!file.open) {
        log_.debug() << "Opening file " << filename << std::endl;

        AVFormatContext* raw_fmt_ctx = nullptr;
        if (avformat_open_input(&raw_fmt_ctx, filename.c_str(), NULL, NULL) < 0) {
            throw std::runtime_error(std::string("Could not open file ") + filename);
        }
        file.fmt_ctx_ = make_unique_av<AVFormatContext>(raw_fmt_ctx, avformat_close_input);

        // is this needed?
        if (avformat_find_stream_info(file.fmt_ctx_.get(), nullptr) < 0) {
            throw std::runtime_error(std::string("Could not find stream information in ")
                                     + filename);
        }

        if (file.fmt_ctx_->nb_streams > 1) {
            log_.warn() << "There are " << file.fmt_ctx_->nb_streams << " streams in "
                        << filename << " which will degrade performance. "
                        << "Consider removing all but the main video stream."
                        << std::endl;
        }

        file.vid_stream_idx_ = av_find_best_stream(file.fmt_ctx_.get(), AVMEDIA_TYPE_VIDEO,
                                      -1, -1, nullptr, 0);
        if (file.vid_stream_idx_ < 0) {
            throw std::runtime_error(std::string("Could not find video stream in ") + filename);
        }


        auto stream = file.fmt_ctx_->streams[file.vid_stream_idx_];
        auto codec_id = codecpar(stream)->codec_id;
        if (width_ == 0) { // first file to open
            width_ = codecpar(stream)->width;
            height_ = codecpar(stream)->height;
            codec_id_ = codec_id;

            if (vid_decoder_) {
                throw std::logic_error("width and height not set, but we have a decoder?");
            }
            log_.info() << "Opened the first file, creating a video decoder" << std::endl;

            vid_decoder_ = std::unique_ptr<detail::Decoder>{
                new detail::NvDecoder(device_id_, log_,
                                      codecpar(stream),
                                      stream->time_base)};
        } else { // already opened a file
            if (!vid_decoder_) {
                throw std::logic_error("width is already set but we don't have a vid_decoder_");
            }

            if (width_ != codecpar(stream)->width ||
                height_ != codecpar(stream)->height ||
                codec_id_ != codec_id) {
                std::stringstream err;
                err << "File " << filename << " is not the same size and codec as previous files."
                    << " This is not yet supported. ("
                    << codecpar(stream)->width << "x" << codecpar(stream)->height
                    << " instead of "
                    << width_ << "x" << height_ << " or codec "
                    << codec_id << " != " << codec_id_ << ")";
                throw std::runtime_error(err.str());
            }
        }
        file.stream_base_ = stream->time_base;
        // 1/frame_rate is duration of each frame (or time base of frame_num)
        file.frame_base_ = AVRational{stream->avg_frame_rate.den,
                                      stream->avg_frame_rate.num};
        file.frame_count_ = av_rescale_q(stream->duration,
                                         stream->time_base,
                                         file.frame_base_);

        // todo check chroma format

        if (codec_id == AV_CODEC_ID_H264 || codec_id == AV_CODEC_ID_HEVC) {
            const char* filtername = nullptr;
            if (codec_id == AV_CODEC_ID_H264) {
                filtername = "h264_mp4toannexb";
            } else {
                filtername = "hevc_mp4toannexb";
            }

#ifdef HAVE_AVBSFCONTEXT
            auto bsf = av_bsf_get_by_name(filtername);
            if (!bsf) {
                throw std::runtime_error("Error finding bit stream filter.");
            }
            AVBSFContext* raw_bsf_ctx_ = nullptr;
            if (av_bsf_alloc(bsf, &raw_bsf_ctx_) < 0) {
                throw std::runtime_error("Error allocating bit stream filter context.");
            }
            file.bsf_ctx_ = make_unique_av<AVBSFContext>(raw_bsf_ctx_, av_bsf_free);

            if (avcodec_parameters_copy(file.bsf_ctx_->par_in, codecpar(stream)) < 0) {
                throw std::runtime_error("Error setting BSF parameters.");
            }

            if (av_bsf_init(file.bsf_ctx_.get()) < 0) {
                throw std::runtime_error("Error initializing BSF.");
            }

            avcodec_parameters_copy(codecpar(stream), file.bsf_ctx_->par_out);
#else
            auto raw_bsf_ctx_ = av_bitstream_filter_init(filtername);
            if (!raw_bsf_ctx_) {
                throw std::runtime_error("Error finding h264_mp4toannexb bit stream filter.");
            }
            file.bsf_ctx_ = OpenFile::bsf_ptr{raw_bsf_ctx_};
            file.codec = stream->codec;
#endif
        } else {
            // todo setup an ffmpeg decoder?
            std::stringstream err;
            err << "Unhandled codec " << codec_id << " in " << filename;
            throw std::runtime_error(err.str());
        }
        file.open = true;
    }
    return file;
}


void VideoLoader::impl::seek(OpenFile& file, int frame) {
    auto seek_time = av_rescale_q(frame,
                                  file.frame_base_,
                                  file.stream_base_);
    log_.debug() << "Seeking to frame " << frame << " timestamp " << seek_time << std::endl;
    // auto ret = avformat_seek_file(file.fmt_ctx_.get(), file.vid_stream_idx_,
    //                               INT64_MIN, seek_time, seek_time, 0);

    // auto ret = av_seek_frame(file.fmt_ctx_.get(), file.vid_stream_idx_,
    //                          frame, AVSEEK_FLAG_FRAME | AVSEEK_FLAG_BACKWARD);
    auto ret = av_seek_frame(file.fmt_ctx_.get(), file.vid_stream_idx_,
                             seek_time, AVSEEK_FLAG_BACKWARD);

    if (ret < 0) {
        std::cerr << "Unable to skip to ts " << seek_time
                  << ": " << av_err2str(ret) << std::endl;
    }

    // todo this seek may be unreliable and will sometimes start after
    // the promised time step.  So we need to calculate the end_time
    // after we actually get a frame to see where we are really
    // starting.
}

void VideoLoader::impl::read_file() {
    // av_packet_unref is unlike the other libav free functions
    using pkt_ptr = std::unique_ptr<AVPacket, decltype(&av_packet_unref)>;
    auto raw_pkt = AVPacket{};
    auto seek_hack = 1;
    while (!done_) {
        if (done_) {
            break;
        }

        auto req = send_queue_.pop();

        log_.info() << "Got a request for " << req.filename << " frame " << req.frame
                    << " send_queue_ has " << send_queue_.size() << " frames left"
                    << std::endl;

        if (done_) {
            break;
        }

        auto& file = get_or_open_file(req.filename);

        // give the vid_decoder a copy of the req before we trash it
        if (vid_decoder_) {
            vid_decoder_->push_req(std::move(req));
        } else {
            throw std::logic_error("No video decoder even after opening a file");
        }

        // we want to seek each time because even if we ended on the
        // correct key frame, we've flushed the decoder, so it needs
        // another key frame to start decoding again
        seek(file, req.frame);

        auto nonkey_frame_count = 0;
        while (req.count > 0 && av_read_frame(file.fmt_ctx_.get(), &raw_pkt) >= 0) {
            auto pkt = pkt_ptr(&raw_pkt, av_packet_unref);

            stats_.bytes_read += pkt->size;
            stats_.packets_read++;

            if (pkt->stream_index != file.vid_stream_idx_) {
                continue;
            }

            auto frame = av_rescale_q(pkt->pts,
                                      file.stream_base_,
                                      file.frame_base_);

            file.last_frame_ = frame;
            auto key = pkt->flags & AV_PKT_FLAG_KEY;

            // The following assumes that all frames between key frames
            // have pts between the key frames.  Is that true?
            if (frame >= req.frame) {
                if (key) {
                    static auto final_try = false;
                    if (frame > req.frame + nonkey_frame_count) {
                        log_.debug() << device_id_ << ": We got ahead of ourselves! "
                                     << frame << " > " << req.frame << " + "
                                     << nonkey_frame_count
                                     << " seek_hack = " << seek_hack << std::endl;
                        seek_hack *= 2;
                        if (final_try) {
                            std::stringstream ss;
                            ss << device_id_ << ": I give up, I can't get it to seek to frame "
                               << req.frame;
                            throw std::runtime_error(ss.str());
                        }
                        if (req.frame > seek_hack) {
                            seek(file, req.frame - seek_hack);
                        } else {
                            final_try = true;
                            seek(file, 0);
                        }
                        continue;
                    } else {
                        req.frame += nonkey_frame_count + 1;
                        req.count -= nonkey_frame_count + 1;
                        nonkey_frame_count = 0;
                    }
                    final_try = false;
                } else {
                    nonkey_frame_count++;
                    // A hueristic so we don't go way over... what should "20" be?
                    if (frame > req.frame + req.count + 20) {
                        // This should end the loop
                        req.frame += nonkey_frame_count;
                        req.count -= nonkey_frame_count;
                        nonkey_frame_count = 0;
                    }
                }
            }
            seek_hack = 1;

            log_.info() << device_id_ << ": Sending " << (key ? "  key " : "nonkey")
                        << " frame " << frame << " to the decoder."
                        << " size = " << pkt->size
                        << " req.frame = " << req.frame
                        << " req.count = " << req.count
                        << " nonkey_frame_count = " << nonkey_frame_count
                        << std::endl;

            stats_.bytes_decoded += pkt->size;
            stats_.packets_decoded++;

            if (file.bsf_ctx_ && pkt->size > 0) {
                int ret;
#ifdef HAVE_AVBSFCONTEXT
                auto raw_filtered_pkt = AVPacket{};

                if ((ret = av_bsf_send_packet(file.bsf_ctx_.get(), pkt.release())) < 0) {
                    throw std::runtime_error(std::string("BSF send packet failed:") +
                                             av_err2str(ret));
                }
                while ((ret = av_bsf_receive_packet(file.bsf_ctx_.get(), &raw_filtered_pkt)) == 0) {
                    auto fpkt = pkt_ptr(&raw_filtered_pkt, av_packet_unref);
                    vid_decoder_->decode_packet(fpkt.get());
                }
                if (ret != AVERROR(EAGAIN)) {
                    throw std::runtime_error(std::string("BSF receive packet failed:") +
                                             av_err2str(ret));
                }
#else
                AVPacket fpkt;
                for (auto bsf = file.bsf_ctx_.get(); bsf; bsf = bsf->next) {
                    fpkt = *pkt.get();
                    ret = av_bitstream_filter_filter(bsf, file.codec, nullptr,
                                                     &fpkt.data, &fpkt.size,
                                                     pkt->data, pkt->size,
                                                     !!(pkt->flags & AV_PKT_FLAG_KEY));
                    if (ret < 0) {
                        throw std::runtime_error(std::string("BSF error:") +
                                                 av_err2str(ret));
                    }
                    if (ret == 0 && fpkt.data != pkt->data) {
                        // fpkt is an offset into pkt, copy the smaller portion to the start
                        if ((ret = av_copy_packet(&fpkt, pkt.get())) < 0) {
                            av_free(fpkt.data);
                            throw std::runtime_error(std::string("av_copy_packet error:") +
                                                     av_err2str(ret));
                        }
                        ret = 1;
                    }
                    if (ret > 0) {
                        // free the buffer in pkt and replace it with the newly created buffer in fpkt
                        av_free_packet(pkt.get());
                        fpkt.buf = av_buffer_create(fpkt.data, fpkt.size, av_buffer_default_free,
                                                    nullptr, 0);
                        if (!fpkt.buf) {
                            av_free(fpkt.data);
                            throw std::runtime_error(std::string("Unable to create buffer during bsf"));
                        }
                    }
                    *pkt.get() = fpkt;
                }
                vid_decoder_->decode_packet(pkt.get());
#endif
            } else {
                vid_decoder_->decode_packet(pkt.get());
            }
        }
        // flush the decoder
        vid_decoder_->decode_packet(nullptr);
    }

    if (vid_decoder_) {
        vid_decoder_->decode_packet(nullptr); // stop decoding
    }
    log_.info() << "Leaving read_file" << std::endl;
}

void VideoLoader::receive_frames(PictureSequence& sequence) {
    pImpl->receive_frames(sequence);
}

void VideoLoader::impl::receive_frames(PictureSequence& sequence) {
    auto startup_timeout = 1000;
    while (!vid_decoder_) {
        usleep(500);
        if (startup_timeout-- == 0) {
            throw std::runtime_error("Timeout waiting for a valid decoder");
        }
    }
    vid_decoder_->receive_frames(sequence);

    stats_.frames_used += sequence.count();

    static auto frames_since_warn = 0;
    static auto frames_used_warned = false;
    frames_since_warn += sequence.count();
    auto ratio_used = static_cast<float>(stats_.packets_decoded) / stats_.frames_used;
    if (ratio_used > frames_used_warning_ratio &&
        frames_since_warn > (frames_used_warned ? frames_used_warning_interval :
                             frames_used_warning_minimum)) {
        frames_since_warn = 0;
        frames_used_warned = true;
        log_.warn() << "\e[1mThe video loader is performing suboptimally due to reading "
                    << std::setprecision(2) << ratio_used << "x as many packets as "
                    << "frames being used.\e[0m  Consider reencoding the video with a "
                    << "smaller key frame interval (GOP length).";
    }
}

void VideoLoader::receive_frames_sync(PictureSequence& sequence) {
    receive_frames(sequence);
    sequence.wait();
}

VideoLoaderStats VideoLoader::get_stats() const {
    return pImpl->get_stats();
}
VideoLoaderStats VideoLoader::impl::get_stats() const {
    // Possible race here with reading thread, but do we really care for stats? Worth putting a lock in the read loop?
    return stats_;
}

void VideoLoader::reset_stats() {
    pImpl->reset_stats();
}
void VideoLoader::impl::reset_stats() {
    stats_ = {};
}

void VideoLoader::set_log_level(LogLevel level) {
    pImpl->set_log_level(level);
}
void VideoLoader::impl::set_log_level(LogLevel level) {
    log_.set_level(level);
}

} // end namespace NVVL

// now the c interface

VideoLoaderHandle nvvl_create_video_loader(int device_id) {
    auto vl = new NVVL::VideoLoader(device_id);
    return reinterpret_cast<VideoLoaderHandle>(vl);
}

VideoLoaderHandle nvvl_create_video_loader_with_log(int device_id, LogLevel log_level) {
    auto vl = new NVVL::VideoLoader(device_id, log_level);
    return reinterpret_cast<VideoLoaderHandle>(vl);
}

void nvvl_destroy_video_loader(VideoLoaderHandle loader) {
    auto vl = reinterpret_cast<NVVL::VideoLoader*>(loader);
    delete vl;
}

struct Size nvvl_video_size_from_file(const char* filename) {
    av_register_all();

    AVFormatContext* raw_fmt_ctx = nullptr;
    auto ret = avformat_open_input(&raw_fmt_ctx, filename, NULL, NULL);
    if (ret < 0) {
        std::stringstream err;
        err << "Could not open file " << filename
            << ": " << av_err2str(ret);
        throw std::runtime_error(err.str());
    }

    auto fmt_ctx = make_unique_av<AVFormatContext>(raw_fmt_ctx, avformat_close_input);

    // is this needed?
    if (avformat_find_stream_info(fmt_ctx.get(), nullptr) < 0) {
        throw std::runtime_error(std::string("Could not find stream information in ")
                                 + filename);
    }

    auto vid_stream_idx_ = av_find_best_stream(fmt_ctx.get(), AVMEDIA_TYPE_VIDEO,
                                               -1, -1, nullptr, 0);
    if (vid_stream_idx_ < 0) {
        throw std::runtime_error(std::string("Could not find video stream in ") + filename);
    }

    auto stream = fmt_ctx->streams[vid_stream_idx_];
    return Size{static_cast<uint16_t>(codecpar(stream)->width),
                static_cast<uint16_t>(codecpar(stream)->height)};
}

struct Size nvvl_video_size(VideoLoaderHandle loader) {
    auto vl = reinterpret_cast<NVVL::VideoLoader*>(loader);
    return vl->size();
}

int nvvl_frame_count(VideoLoaderHandle loader, const char* filename) {
    auto vl = reinterpret_cast<NVVL::VideoLoader*>(loader);
    return vl->frame_count(filename);
}

void nvvl_read_sequence(VideoLoaderHandle loader, const char* filename,
                   int frame, int count) {
    auto vl = reinterpret_cast<NVVL::VideoLoader*>(loader);
    vl->read_sequence(filename, frame, count);
}

PictureSequenceHandle nvvl_receive_frames(VideoLoaderHandle loader, PictureSequenceHandle sequence) {
    auto vl = reinterpret_cast<NVVL::VideoLoader*>(loader);
    auto pb = reinterpret_cast<NVVL::PictureSequence*>(sequence);
    vl->receive_frames(*pb);
    return sequence;
}

PictureSequenceHandle nvvl_receive_frames_sync(VideoLoaderHandle loader, PictureSequenceHandle sequence) {
    auto vl = reinterpret_cast<NVVL::VideoLoader*>(loader);
    auto pb = reinterpret_cast<NVVL::PictureSequence*>(sequence);
    vl->receive_frames_sync(*pb);
    return sequence;
}

struct VideoLoaderStats nvvl_get_stats(VideoLoaderHandle loader) {
    auto vl = reinterpret_cast<NVVL::VideoLoader*>(loader);
    return vl->get_stats();
}

void nvvl_reset_stats(VideoLoaderHandle loader) {
    auto vl = reinterpret_cast<NVVL::VideoLoader*>(loader);
    vl->reset_stats();
}

void nvvl_set_log_level(VideoLoaderHandle loader, LogLevel level) {
    auto vl = reinterpret_cast<NVVL::VideoLoader*>(loader);
    vl->set_log_level(level);
}
