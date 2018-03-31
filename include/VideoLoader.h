#pragma once

#include "PictureSequence.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

struct Size {
    uint16_t width;
    uint16_t height;
};

/// Provides statistics, see VideoLoader::get_stats() and VideoLoader::reset_stats()
struct VideoLoaderStats {
    /** Total number of bytes read from disk
     */
    uint64_t bytes_read;

    /** Number of compressed packets read from disk
     */
    uint64_t packets_read;

    /** Total number of bytes sent to NVDEC for decoding, can be
     *  different from bytes_read when seeking is a bit off or if
     *  there are extra streams in the video file.
     */
    uint64_t bytes_decoded;

    /** Total number of packets sent to NVDEC for decoding, see bytes_decoded
     */
    uint64_t packets_decoded;

    /** Total number of frames actually used. This is usually less
     *  than packets_decoded because decoding must happen key frame to
     *  key frame and output sequences often span key frame sequences,
     *  requiring more frames to be decoded than are actually used in
     *  the output.
     */
    uint64_t frames_used;
};

/** Indicate which log level to use, sent to VideoLoader::VideoLoader
 * and VideoLoader::set_log_level()
 */
enum LogLevel {
    LogLevel_Debug,
    LogLevel_Info,
    LogLevel_Warn,
    LogLevel_Error,
    LogLevel_None,
};

/**
 * Opaque handle to a VideoLoader
 */
typedef void* VideoLoaderHandle;

/**
 * Wrapper for VideoLoader::VideoLoader()
 *
 * Uses default log_level, use nvvl_set_log_level to set a level
 */
VideoLoaderHandle nvvl_create_video_loader(int device_id);

/**
 * Wrapper for VideoLoader::VideoLoader(int device_id, LogLevel log_level)
 *
 * Use when you want to set the log level right from creation
 */
VideoLoaderHandle nvvl_create_video_loader_with_log(int device_id, enum LogLevel log_level);

/**
 * Frees the VideoLoader
 */
void nvvl_destroy_video_loader(VideoLoaderHandle loader);

/**
 * Parses headers of filename to return width and height of the video
 */
struct Size nvvl_video_size_from_file(const char* filename);

/**
 * Wrapper for VideoLoader::video_size()
 */
struct Size nvvl_video_size(VideoLoaderHandle loader);

/**
 * Wrapper for VideoLoader::frame_count()
 */
int nvvl_frame_count(VideoLoaderHandle loader, const char* filename);

/**
 * Wrapper for VideoLoader::read_sequence()
 */
void nvvl_read_sequence(VideoLoaderHandle loader, const char* filename,
                        int frame, int count);

/**
 * Wrapper for VideoLoader::receive_frames()
 */
PictureSequenceHandle nvvl_receive_frames(VideoLoaderHandle loader, PictureSequenceHandle sequence);

/**
 * Wrapper for VideoLoader::receive_frames_sync()
 */
PictureSequenceHandle nvvl_receive_frames_sync(VideoLoaderHandle loader, PictureSequenceHandle sequence);

/**
 * Wrapper for VideoLoader::get_stats()
 */
struct VideoLoaderStats nvvl_get_stats(VideoLoaderHandle loader);

/**
 * Wrapper for VideoLoader::reset_stats()
 */
void nvvl_reset_stats(VideoLoaderHandle loader);

/**
 * Wrapper for VideoLoader::set_log_level()
 */
void nvvl_set_log_level(VideoLoaderHandle loader, enum LogLevel level);

#ifdef __cplusplus
} // close extern "C" {

namespace NVVL {

class VideoLoader {
  public:
    /**
     * Overload of full constructor, see VideoLoader::VideoLoader
     */
    VideoLoader(int device_id);

    /**
     * Allocates a nvcuvid decoder and launches some internal threads to
     * assist in the decoding.
     *
     * \param device_id GPU device number to perform the decoding on
     *
     * \param log_level \a optional Set the amount of logging to
     * output. See \enum LogLevel.
     */
    VideoLoader(int device_id, LogLevel log_level);

    /**
     * Retrieve the number of frames in a video file and keeps \c
     * filename open ready to be read from.
     *
     * \param filename path to the video file
     *
     * \return Total number frames in the file
     */
    int frame_count(std::string filename);

    /**
     * Get the size of the video files this loader is reading.
     *
     * \return A \c Size struct with \c width and \c height.
     */
    Size size() const;

    /**
     * Enqueue the reading and decoding of video frames from a file.
     *
     * The specified frames will be sent to the decoder after all
     * frames from previous calls to \c read_sequence(). Note that due
     * to the necessity to begin decoding at a key frame and the
     * possibility that frames appear out of order in the encoded
     * file, more frames than requested will probably be read and sent
     * to the decoder. However, only the frames specified will be used
     * to fill sequences sent to \c receive_frames().
     *
     * \param filename path to the video file
     * \param frame the frame number to start reading
     * \param count the number of frames to read
     */
    void read_sequence(std::string filename, int frame, int count=1);

    /**
     * Enqueue transfer of the next set of frames into a
     * PictureSeqeunce.
     *
     * Enqueues an asynchronous transfer of next \c sequence.count()
     * frames into \c sequence. One of the two \c
     * PictureSequence::wait() functions should be used before the
     * sequence is consumed.
     *
     * There does not need to be any correlation between calls to \c
     * read_sequence() and calls to \c receive_frames.  \c
     * receive_frames() will extract the next frames out of the
     * decoder regardless of if that consumes some, all, or multiple
     * sequences read using \c read_sequence().
     *
     * \param sequence The PictureSequence to transfer the frames
     * into, \c sequence must be configured with the desired
     * parameters before calling \c receive_frames.
     *
     * \see PictureSequence::wait()
     */
    void receive_frames(PictureSequence& sequence);

    /**
     * Synchronously transfer the next set of frames into a
     * PictureSequence.
     *
     * Synchronous version of \c receive_frames(), see that for details.
     *
     * \see VideoLoader::receive_frames()
     */
    void receive_frames_sync(PictureSequence& sequence);

    /**
     * Get current statistics
     *
     * \see VideoLoaderStats
     */
    VideoLoaderStats get_stats() const;

    /**
     * Reset statistic counters to zero
     */
    void reset_stats();

    /**
     * Set the log level
     *
     * \see LogLevel
     */
    void set_log_level(LogLevel level);

    // need these for pImpl pointer to be happy
    ~VideoLoader();
    VideoLoader(VideoLoader&&);
    VideoLoader& operator=(VideoLoader&&);
    VideoLoader(const VideoLoader&) = delete;
    VideoLoader& operator=(VideoLoader&) = delete;

  private:
    // we use pImpl here to prevent copying a slew of headers for installation
    class impl;
    std::unique_ptr<impl> pImpl;
};

}
#endif // ifdef __cplusplus
