NVVL (**NV**IDIA **V**ideo **L**oader) is a library to load random
sequences of video frames from compressed video files to facilitate
machine learning training. It uses FFmpeg's libraries to parse and
read the compressed packets from video files and the video decoding
hardware available on NVIDIA GPUs to off-load and accelerate the
decoding of those packets, providing a ready-for-training tensor in
GPU device memory. NVVL can additionally perform data augmentation
while loading the frames. Frames can be scaled, cropped, and flipped
horizontally using the GPUs dedicated texture mapping units. Output
can be in RGB or YCbCr color space, normalized to [0, 1] or [0, 255],
and in `float`, `half`, or `uint8` tensors.

Using compressed video files instead of individual frame image files
significantly reduces the demands on the storage and I/O systems
during training. Storing video datasets as video files consumes an
order of magnitude less disk space, allowing for larger datasets to
both fit in system RAM as well as local SSDs for fast access. During
loading fewer bytes must be read from disk. Fitting on smaller, faster
storage and reading fewer bytes at load time allievates the bottleneck
of retrieving data from disks, which will only get worse as GPUs get
faster. For the dataset used in our example project, H.264 compressed
`.mp4` files were nearly 40x smaller than storing frames as `.png`
files.

Using the hardware decoder on NVIDIA GPUs to decode images
significantly reduces the demands on the host CPU. This means fewer
CPU cores need to be dedicated to data loading during training. This
is especially important in servers with a large number of GPUs per
CPU, such as the in the NVIDIA DGX-2 server, but also provides
benefits for other platforms. When training our example project on a
NVIDIA DGX-1, the CPU load when using NVVL was 50-60% of the load seen
when using a normal dataloader for `.png` files.

Measurements that quantify the performance advantages of using NVVL
are detailed in our [super resolution example
project](/examples/pytorch_superres).

Most users will want to use the deep learning framework wrappers
provided rather than using the library directly. Currently a wrapper
for PyTorch is provided (PR's for other frameworks are welcome). See
the [PyTorch wrapper README](/pytorch/README.md) for documentation on
using the PyTorch wrapper. Note that it is not required to build or
install the C++ library before building the PyTorch wrapper (its
setup scripts will do so for you).

# Building and Installing

NVVL depends on the following:
- CUDA Toolkit. We have tested versions 8.0 and later but earlier
  versions may work. NVVL will perform better with CUDA 9.0 or
  later<sup id="a1">[1](#f1)</sup>.
- FFmpeg's libavformat, libavcodec, libavfilter, and libavutil. These
  can be installed from source as in the [example
  Dockerfiles](/docker) or from the Ubuntu 16.04 packages
  `libavcodec-dev libavfilter-dev libavformat-dev
  libavutil-dev`. Other distributions should have similar packages.

Additionally, building from source requires CMake version 3.8 or above
and some examples optionally make use of some libraries from OpenCV if
they are installed.

The [docker](docker) directory contains Dockerfiles that can be used
as a starting point for creating an image to build or use the NVVL
library. The [example's docker directory](examples/pytorch/docker) has
an example Dockerfile that actually builds and installs the NVVL
library.

CMake 3.8 and above provides builtin CUDA language support that NVVL's
build system uses. Since CMake 3.8 is relatively new and not yet in
widely used Linux distribution, it may be required to install a new
version of CMake.  The easiest way to do so is to make use of their
package on PyPI:

```
pip install cmake
```

Alternatively, or if `pip` isn't available, you can install to
`/usr/local` from a binary distribution:

```sh
wget https://cmake.org/files/v3.10/cmake-3.10.2-Linux-x86_64.sh
/bin/sh cmake-3.10.2-Linux-x86_64.sh --prefix=/usr/local
```

See https://cmake.org/download/ for more options.

Building and installing NVVL follows the typical CMake pattern:

```sh
mkdir build && cd build
cmake ..
make -j
sudo make install
```

This will install `libnvvl.so` and development headers into
appropriate subdirectores under `/usr/local`. CMake can be passed the
following options using `cmake .. -DOPTION=Value`:

- `CUDA_ARCH` - Name of a CUDA architecture to generate device code
  for, seperated via a semicolon. Valid options are `Kepler`,
  `Maxwell`, `Pascal`, and `Volta`. You can also use specific
  architecture names such as `sm_61`. Default is
  `Maxwell;Pascal;Volta`.

- `CMAKE_CUDA_FLAGS` - A string of arguments to pass to `nvcc`. In
  particular, you can decide to link against the static or shared
  runtime library using `-cudart shared` or `-cudart static`. You can
  also use this for finer control of code generation than `CUDA_ARCH`,
  see the `nvcc` documentation. Default is `-cudart shared`.

- `WITH_OPENCV` - Set this to 1 to build the examples with the
  optional OpenCV functionality.

- `CMAKE_INSTALL_PREFIX` - Install directory. Default is
  `/usr/local`.

- `CMAKE_BUILD_TYPE` - `Debug` or `Release` build.

See the [CMake documentation](https://cmake.org/cmake/help/v3.8/) for
more options.

The examples in `doc/examples` can be built using the `examples` target:
```
make examples
```

Finally, if Doxygen is installed, API documentation can be built using
the `doc` target:
```
make doc
```
This will build html files in `doc/html`.

# Preparing Data

NVVL supports the H.264 and HEVC (H.265) video codecs in any container
format that FFmpeg is able to parse.  Video codecs only store certain
frames, called keyframes or intra-frames, as a complete image in the
data stream. All other frames require data from other frames, either
before or after it in time, to be decoded. In order to decode a
sequence of frames, it is necessary to start decoding at the keyframe
before the sequence, and continue past the sequence to the next
keyframe after it. This isn't a problem when streaming sequentially
through a video; however, when decoding small sequences of frames
randomly throughout the video, a large gap between keyframes results in
reading and decoding a large amount of frames that are never used.

Thus, to get good performance when randomly reading short sequences
from a video file, it is necessary to encode the file with frequent
key frames. We've found setting the keyframe interval to the length of
the sequences you will be reading provides a good compromise between
filesize and loading performance. Also, NVVL's seeking logic doesn't
support open GOPs in HEVC streams. To set the keyframe interval to `X`
when using `ffmpeg`:

- For `libx264` use `-g X`
- For `libx265` use `-x265-params "keyint=X:no-open-gop=1"`

The pixel format of the video must also be yuv420p to be supported by
the hardware decoder. This is done by passing `-pix_fmt yuv420p` to
`ffmpeg`. You should also remove any extra audio or video streams from
the video file by passing `-map v:0` to ffmpeg after the input but
before the output.

For example to transcode to H.264:
```
ffmpeg -i original.mp4 -map v:0 -c:v libx264 -crf 18 -pix_fmt yuv420p -g 5 -profile:v high prepared.mp4
```

# Basic Usage
This section describes the usage of the base C/C++ library, for usage
of the PyTorch wrapper, see the [README](/pytorch/README.md) in the
pytorch directory.

The library provides both a C++ and C interface. See the examples in
[doc/examples](doc/examples) for brief example code on how to use the
library. [extract_frames.cpp](doc/examples/extract_frames.cpp)
demonstrates the C++ interface and
[extract_frames_c.c](doc/examples/extract_frames_c.c) the C
interface. The API documentation built with `make doc` is the
canonical reference for the API.

The basic flow is to create a `VideoLoader` object, tell it which
frame sequences to read, and then give it buffers in device memory to
put the decoded sequences into. In C++, creating a video loader is
straight forward:

```C++
auto loader = NVVL::VideoLoader{device_id};
```

You can then tell it which sequences to read via `read_sequence`:

```C++
loader.read_sequence(filename, frame_num, sequence_length);

```

To receive the frames from the decoder, it is necessary to create a
`PictureSequence` to tell it how and where you want the decoded frames
provided. First, create a `PictureSequence`, providing a count of the
number of frames to receive from the decoder. Note that the count here
does not need to be the same as the sequence_length provided to
`read_sequence`; you can read a large sequence of frames and receive
them as multiple tensors, or read multiple smaller sequences and
receive them concatenated as a single tensor.

```C++
auto seq = PictureSequence{sequence_count};
```

You now create "Layers" in the sequence to provide the destination for
the frames. Each layer can be a different type, have different
processing, and contain different frames from the received
sequence. First, create a `PictureSequence::Layer` of the desired
type:

```C++
auto pixels = PictureSequence::Layer<float>{};
```

Next, fill in the pointer to the data and other details. See the
documentation in [PictureSequence.h](include/PictureSequence.h) for a
description of all the available options.

```C++
float* data = nullptr;
size_t pitch = 0;
cudaMallocPitch(&data, &pitch,
                crop_width * sizeof(float),
                crop_height * sequence_count * 3);
pixels.data = data;
pixels.desc.count = sequence_count;
pixels.desc.channels = 3;
pixels.desc.width = crop_width;
pixels.desc.height = crop_height;
pixels.desc.scale_width = scale_width;
pixels.desc.scale_height = scale_height;
pixels.desc.horiz_flip = false;
pixels.desc.normalized = true;
pixels.desc.color_space = ColorSpace_RGB;
pixels.desc.stride.x = 1;
pixels.desc.stride.y = pitch / sizeof(float);
pixels.desc.stride.c = pixels.desc.stride.y * crop_height;
pixels.desc.stride.n = pixels.desc.stride.c * 3;
```

Note that here we have set the strides such that the dimensions are
"nchw", we could have done "nhwc" or any other dimension order by
setting the strides appropriately. Also note that the strides in the
layer description are number of elements, not number of bytes.

We now add this layer to our `PictureSequence`, and send it to the loader:

```C++
seq.set_layer("pixels", pixels);
loader.receive_frames(seq);
```

This call to `receive_frames` will be
asynchronous. `receive_frames_sync` can be used if synchronous reading
is desired. When we are ready to use the frames we can insert a wait
event into the CUDA stream we are using for our computation:

```C++
seq.wait(stream);
```

This will insert a wait event into the stream `stream`, causing any
further kernels launched on `stream` to wait until the data is
ready.

The C interface follows a very similar pattern, see
[doc/examples/extract_frames_c.c](doc/examples/extract_frames_c.c)
for an example.

# Reference
If you find this library useful in your work, please cite it in your
publications using the following BibTeX entry:

```
@misc{nvvl,
  author = {Jared Casper and Jon Barker and Bryan Catanzaro},
  title = {NVVL: NVIDIA Video Loader},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/NVIDIA/nvvl}}
}
```

# Footnotes

<b id="f1">[1]</b> Specifically, with nvidia kernel modules version
384 and later, which come with CUDA 9.0+, CUDA kernels launched by
NVVL will run asynchronously on a separate stream. With earlier kernel
modules, all CUDA kernels are launched on the default stream. [↩](#a1)
