#pragma once

#ifndef CFFI
# include <stddef.h>
# include <stdint.h>
# include <cuda_runtime.h>
#else
struct CUstream_st;
typedef struct CUstream_st* cudaStream_t;
#endif

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

/**
 * How the image is scaled up/down from the original
 */
enum NVVL_ScaleMethod {
    /**
     * The value for the nearest neighbor is used, no interpolation
     */
    ScaleMethod_Nearest,

    /**
     * Simple bilinear interpolation of four nearest neighbors
     */
    ScaleMethod_Linear

    // These are possibilities but currently unimplemented (PRs welcome)
    // ScaleMethod_Area
    // ScaleMethod_Cubic
    // ScaleMethod_Lanczos
};

/**
 * How the chroma channels are upscaled from yuv 4:2:0 to 4:4:4
 */
enum NVVL_ChromaUpMethod {
    /**
     * Simple bilinear interpolation of four nearest neighbors
     */
    ChromaUpMethod_Linear

    // These are possibilities but currently unimplemented (PRs welcome)
    // ChromaUpMethod_CatmullRom
};

/**
 * Color space to return picture in
 */
enum NVVL_ColorSpace {
    /**
     * RGB. channel 0 is red, 1 is green, 2 is blue
     */
    ColorSpace_RGB,
    /**
     * YCbCr. channel 0 is luma, 1 is blue-difference chroma, 2 is red-difference chroma
     */
    ColorSpace_YCbCr
    //ColorSpace_YCgCo
};

/**
 * Different types of data that a layer in a PictureSequence can hold, used for C interface
 */
enum NVVL_PicDataType {
    PDT_NONE,
    PDT_BYTE,
    PDT_HALF,
    PDT_FLOAT
};

/**
 * Description of a layer in a frame sequence.
 */
struct NVVL_LayerDesc {
    /**
     * Number of frames in this Layer.
     *
     * This does not need to be the same as the number of frames in the sequence.
     */
    uint16_t count;

    /**
     * Number of color channels in this layer.
     *
     * Must match the number of channels in the color space requested.
     */
    uint8_t channels;

    /**
     * Width of this layer.
     *
     * Each frame is cropped to this size after scaling.
     */
    uint16_t width;

    /**
     * Height of this layer.
     *
     * Each frame is cropped to this size after scaling.
     */
    uint16_t height;

    /**
     * Location of the crop within the scaled frame.
     *
     * Must be set such that crop_x + width <= sequence_width
     */
    uint16_t crop_x;

    /**
     * Location of the crop within the scaled frame.
     *
     * Must be set such that crop_y + height <= sequence_height
     */
    uint16_t crop_y;

    /**
     * Size the original frame is scaled to before cropping.
     *
     * If left 0, no scaling is performed.
     */
    uint16_t scale_width;

    /**
     * Size the original frame is scaled to before cropping.
     *
     * If left 0, no scaling is performed.
     */
    uint16_t scale_height;

    /**
     * Indicates if the frame should be flipped horizontally before scaling.
     */
    bool horiz_flip;

    /**
     * Indicates if the pixel values should be normalized to [0,1]
     *
     * If False, pixels will be in the standard [0,255] range. If
     * True, T must be a floating point type.
     */
    bool normalized;

    /**
     * Color space of output
     *
     * \see NVVL_ColorSpace
     */
    enum NVVL_ColorSpace color_space;

    /**
     * Method to upscale the chroma channel in 4:2:0 to 4:4:4 conversion.
     *
     * \see NVVL_ChromaUpMethod
     */
    enum NVVL_ChromaUpMethod chroma_up_method;

    /**
     * Method used to scale frame from original size to scale_width x
     * scale_height.
     *
     * \see NVVL_ScaleMethod
     */
    enum NVVL_ScaleMethod scale_method;

    /**
     * Strides for the various dimensions.
     *
     * These are element strides, not byte strides. So, for
     * example, if T is float, a stride of "10" is a stride of
     * 10*sizeof(float) bytes.
     */
    struct {
        size_t x;
        size_t y;
        size_t c;
        size_t n;
    } stride;

};

/**
 * Dynamically typed C version of NVVL::PictureSequence::Layer.
 *
 */
struct NVVL_PicLayer {
    /**
     * Type of data
     *
     * \see NVVL_PicDataType
     */
    enum NVVL_PicDataType type;

    /**
     * Elements of the description shared with the C++ NVVL::PictureSequence::Layer
     *
     * \see NVVL_LayerDesc
     */
    struct NVVL_LayerDesc desc;

    /**
     * Equivalent to NVVL::PictureSequence::Layer::index_map
     *
     * If NULL, use a 1-to-1 mapping.  Otherwise must point to an
     * array of length index_map_length.
     *
     * \see NVVL::PictureSequence::Layer::index_map
     */
    const int* index_map;

    /**
     * Number of ints in \c index_map
     */
    int index_map_length;

    /**
     * The actual data of the layer
     *
     * The type of data pointed to by this is determined by \c type
     */
    void* data;
};

/**
 * Opaque handle to a PictureSequence
 */
typedef void* PictureSequenceHandle;

/**
 * Different types of metadata that can be retrieved
 */
enum NVVL_PicMetaType {
    PMT_INT,
    PMT_STRING
};

/**
 * Wrapper for PictureSequence::PictureSequence
 */
PictureSequenceHandle nvvl_create_sequence(uint16_t count);

/**
 * Wrapper for PictureSequence::SetLayer()
 *
 * All options and index_map are copied into the sequence.
 */
void nvvl_set_layer(PictureSequenceHandle sequence,
                    const struct NVVL_PicLayer* layer,
                    const char* name);

/**
 * Wrapper for PictureSequence::get_or_add_meta()
 *
 * \return a pointer to an array of type `type` containing the
 * metadata `name`, return NULL if the named metadata does not
 * exist. Does not (currently) support strings.
 */
void* nvvl_get_or_add_meta_array(PictureSequenceHandle sequence, enum NVVL_PicMetaType type, const char* name);

/**
 * Wrapper for PictureSequence::get_meta()
 */
const void* nvvl_get_meta_array(PictureSequenceHandle sequence, enum NVVL_PicMetaType type, const char* name);

/**
 * Get the string of metadata `name` corresponding to `index`
 *
 * \param sequence handle to a valid PictureSequence
 * \param name name of the metadata to retrieve
 * \param index Which frame index to get
 */
const char* nvvl_get_meta_str(PictureSequenceHandle sequence, const char* name, int index);

/**
 * Wrapper for PictureSequence::count()
 */
int nvvl_get_sequence_count(PictureSequenceHandle sequence);

/**
 * Wrapper for PictureSequence::get_layer()
 *
 * Note that the PictureSequence retains ownership of the index_map,
 * so the returned index_map is only valid while the PictureSequence
 * exists.
 */
struct NVVL_PicLayer nvvl_get_layer(PictureSequenceHandle sequence,
                               enum NVVL_PicDataType type,
                               const char* name);

/**
 * Wrapper for PictureSequence::get_layer()
 *
 * Since this is a copy of the layer description, index_map is left
 * NULL to avoid returning a pointer to dynamically allocated memory.
 */
struct NVVL_PicLayer nvvl_get_layer_indexed(PictureSequenceHandle sequence,
                                       enum NVVL_PicDataType type,
                                       const char* name,
                                       int index);

/**
 * Wrapper for PictureSequence::wait()
 */
void nvvl_sequence_wait(PictureSequenceHandle sequence);

/**
 * Wrapper for PictureSequence::wait(cudaStream_t)
 */
void nvvl_sequence_stream_wait(PictureSequenceHandle sequence, cudaStream_t stream);

/**
 * Free a PictureSequence
 */
void nvvl_free_sequence(PictureSequenceHandle sequence);

#ifdef __cplusplus
} // end extern "C"

#include <cuda.h>
#include <memory>
#include <string>
#include <vector>

namespace NVVL {

namespace detail {
class Decoder;
}

// C++ scoping for these C structs
using ScaleMethod = NVVL_ScaleMethod;
using ChromaUpMethod = NVVL_ChromaUpMethod;
using ColorSpace = NVVL_ColorSpace;
using PicLayer = NVVL_PicLayer;
using LayerDesc = NVVL_LayerDesc;

class PictureSequence {
  public:
    /** Create an empty PictureSequence.
     *
     * \param count The number of frames to receive from the
     * decoder. Not all of the frames received from the decoder need
     * to be used in any layer, so some layers may have a different
     * count.
     */
    PictureSequence(uint16_t count);

    /**
     * A full description of a layer
     */
    template<typename T>
    struct Layer {
        /**
         * Elements of description shared with the C version PicLayer.
         *
         * \see NVVL_LayerDesc
         */
        LayerDesc desc;

        /**
         * Map from indices into the decoded sequence to indices in this Layer.
         *
         * An empty vector indicates a 1-to-1 mapping from sequence to layer.
         *
         * For examples, To reverse the frames, set index_map
         * to {4, 3, 2, 1, 0}.
         *
         * An index of -1 indicates that the decoded frame should not
         * be used in this layer. For example, to extract just the
         * middle frame from a sequence of 5 frames, set index_map to
         * {-1, -1, 0, -1, -1}.
         *
         * If the size of index_map is less than the number of frames
         * in the sequence, then those extra frames will not be
         * used. For example, if index_map is {-1, 0} and the sequence
         * is 5 frames, only the second frame in the sequence will be
         * placed into the output array (at index 0).
         *
         * It is up to the user to ensure that all indices are smaller
         * than the size of this layer.
         */
        std::vector<int> index_map;

        /**
         * Pointer to the multi-dimensional tensor to place the frames into.
         *
         * The smallest dimension should be padded for optimal
         * performance, see the CUDA documentation for cudaMallocPitch
         * for details.
         */
        T* data;
    };

    /**
     * Add a layer to this sequence
     *
     * All the options and the index_map are copied in, but the caller
     * maintains ownership of the data, which it sould keep valid until
     * the data has been retrieved from the picture sequence.
     *
     * \param name name the layer should be given
     * \param layer description of the layer
     */
    template<typename T>
    void set_layer(std::string name, const Layer<T>& layer);

    /**
     * Overload for set_layer that takes a C-style PicLayer
     */
    template<typename T>
    void set_layer(std::string name, const PicLayer* layer) {
        auto l = PictureSequence::Layer<T>{};
        l.data = reinterpret_cast<decltype(l.data)>(layer->data);
        l.desc = layer->desc;
        if (layer->index_map) {
            l.index_map.insert(l.index_map.end(), layer->index_map, layer->index_map + layer->index_map_length);
        }
        set_layer(name, l);
    }

    /**
     * Retrieve a layer from the sequence with data pointing to specific index
     *
     * \param name name of the layer
     * \param index index to adjust the data pointer to
     *
     * \return Copy of the layer description
     */
    template<typename T>
    Layer<T> get_layer(std::string name, int index) const;

    /**
     * Retrieve a reference to a layer from the sequence
     *
     * \param name name of the layer
     *
     * \return const reference to the layer
     */
    template<typename T>
    const Layer<T>& get_layer(std::string name) const;

    /**
     * Check if sequence has the named layer
     *
     * \param name name of data layer to look for
     *
     * \return True if layer \c name exists in this sequence
     */
    bool has_layer(std::string name) const;

    /**
     * Get the vector for the named meta, adding it if it exists
     *
     * \param name name of meta array to add
     *
     * \return non-const reference to meta vector for \c name
     */
    template<typename T>
    std::vector<T>& get_or_add_meta(std::string name);

    /**
     * Get a const reference to meta aray for \c name
     *
     * \param name name of meta array to get
     *
     * \return const refernece to meta array
     */
    template<typename T>
    const std::vector<T>& get_meta(std::string name) const;

    /**
     * Check if sequence has the named meta array
     *
     * \param name name of meta array to look for
     *
     * \return True if meta array \c name exists in the sequence
     */
    bool has_meta(std::string name) const;

    /**
     * The number of frames retrieved from the decoder for this sequence
     *
     * Returned count is not necessarily the count for any of the layers
     *
     * \return the number of frames retrieved
     */
    int count() const;

    /**
     * Set the number of frames to retrieve from the decoder
     *
     * \param count the number of frames to retrieve
     */
    void set_count(int count);

    /**
     * Synchronously wait for the sequence to be ready to use
     */
    void wait() const;

    /**
     * Synchronously wait until ready, then insert a wait event into
     * stream.
     *
     * Waits until the transfer from the decoder in the data layer has
     * begun, then inserts a wait event into \c stream signalling the
     * completion of the transfer.
     *
     * Until the transfer from the decoder has begun, the event to
     * wait on has not captured any work so we don't have anything to
     * wait on.
     *
     * \param stream The CUDA stream to insert the wait event into.
     */
    void wait(cudaStream_t stream) const;

    // need these for pImpl pointer to be happy
    ~PictureSequence();
    PictureSequence(PictureSequence&&);
    PictureSequence& operator=(PictureSequence&&);
    PictureSequence(const PictureSequence&) = delete;
    PictureSequence& operator=(const PictureSequence&) = delete;

  private:
    // we use pImpl here to prevent copying a slew of headers for installation
    // (i.e. a chunk of boost for boost::variant)
    class impl;
    std::unique_ptr<impl> pImpl;

    // Decoder's needs to record the event and indicate transfer has started
    friend class detail::Decoder;
};

}
#endif // ifdef __cplusplus
