This directory contains a PyTorch wrapper for the NVVL library. See
the [README](/README.md) in the root directory for more information about the
library.

There is an example project that uses this wrapper in
[/examples/pytorch_superres](/examples/pytorch_superres) that trains a
video super-resolution network.

# Building and installing
The NVVL PyTorch wrapper requires:

- a python `torch` package with CUDA support, see [their
homepage](http://pytorch.org) for instructions.

- FFmpeg's libavformat, libavcodec, libavfilter, and libavutil. These
  should be available from your Linux distribution's package manager
  (be sure to get the `-dev` packages).

The setup scripts will install the `cmake` and `cffi` python packages
if necessary.

The [docker](docker) directory provides example `Dockerfile`s for
building a docker image to build and run NVVL using PyTorch. The
example project's
[Dockerfile](/examples/pytorch_superres/docker/Dockerfile) provides an
example of building an image with the NVVL library and wrapper built
in.

It is not necessary to build the NVVL C++ library before building and
installing the wrapper, it will be built for you.

From this directory (`pytorch`) you can simply run:

```
python setup.py install
```

to build and install the NVVL library and the PyTorch wrapper into the
current python environment (you may need to be root or use `sudo` if
you are not in a virtual environment).

If you want more control over building the NVVL library (to specify
GPU architectures for example), you can build the NVVL library as
described in the [root README](/README.md). You can then supply the
location of your build using the `--with-nvvl` option to `setup.py`:

```
python setup.py install --with-nvvl=../build
```

and the library will be included in the python package
installed. Finally, you can also do a full system install of the NVVL
C++ library (using `make install`) then tell `setup.py` to use the
system installed library with the `--system-nvvl` option:

```
python setup.py install --system-nvvl
```


# Usage
Usage of NVVL's video loader is similar to using a standard PyTorch
dataset and dataloader with a few tweaks. Training data should be
prepared as described in the root [README](/README.md)'s Preparing
Data section.

First, create a `nvvl.VideoDataset` by providing at least a list of
filenames and a sequence length. You can also provide a `device_id` to
indicate which GPU to use, and a `processing` dictionary, described
below. Each element of the dataset correspondes to a sequence of
frames beginning at the specified frame within the video files,
excluding starting frames whose sequence would go beyond the end of a
particular video. For example, consider a dataset that consists of two
video files, each with 100 frames, and a sequence of length 5. Each
file has 96 valid sequences, for a total of 192 samples; thus, indices
0 - 95 are sequences from the first file, and indices 96 - 191 are
sequences from the second file.

The `processing` dictionary passed to `nvvl.VideoDataset` describes
processing to be done to create multiple output tensors from a single
frame sequence. This can be used, for example, to create input and
target tensors to use for training. The `nvvl.video_size_from_file()`
function can be useful when setting up `processing`.

For example consider the case that from a sequence of 5 frames, we
want to downsample all 5 to create an input tensor, and use just the
middle frame full size as the target. Further, we want to train in
fp16. In this case we create a dictionary like this:

```Python
size = nvvl.video_size_from_file(filenames[0])
processing = {
    "input" : nvvl.ProcessDesc(type = "half",
                               scale_width = size.width/4,
                               scale_height = size.height/4),
    "target" : nvvl.ProcessDesc(type = "half",
                                index_map = [-1, -1, 0])
}
```

See the documentation for `nvvl.ProcessDesc` in
[nvvl/dataset.py](nvvl/dataset.py) for a list of available processing
options.

This dictionary can then be used to create a dataset:
```Python
dataset = nvvl.VideoDataset(filenames, 5, processing=processing)
```

If no processing dictionary is passed to the dataset, it will default
to providing all the full unscaled and uncropped frames in each
sequence in unnormalized RGB.

We then do the normal process of optionally creating a custom sampler
based on the size of the dataset, and pass that and the dataset to a
VideoLoader. Note that only an `nvvl.VideoDataset` can be used as a
dataset for an `nvvl.VideoLoader`.

For example, using the default PyTorch samplers to shuffle the data
and create batches of size 8 would look like this:

```Python
loader = nvvl.VideoLoader(dataset, batch_size=8, shuffle=True)
```

The `nvvl.VideoLoader` takes some of the options that a normal PyTorch
data loader takes, but not all. For example, `pin_memory` doesn't make
sense in this context so isn't an option. See the docstring for
`nvvl.VideoLoader` in [nvvl/loader.py](nvvl/loader.py) for the
available options.

Once the dataset and loader have been created, they can be used much
like normal datasets and loaders with one important exception: once an
iterator over the loader is created, it must be run to completion
before another iterator is created and before any elemental access to
the dataset.  This is because creation of the iterator queues up
frames to be decoded by the hardware decoder, so there is no way
(currently) to interrupt the stream of frames to decode other
frames.

Also note that element access to an `nvvl.VideoDataset` is not very
performant, as it transfers the compressed frame from the CPU to the
GPU, performs the decoding, then transfers the resulting raw frame
back to the CPU, all synchronously. So while the `nvvl.VideoDataset`
can technically be used as a dataset for other data loaders, this will
result in suboptimal performance. This functionality should be used
sparingly for debugging or similar use cases.
