The `Dockerfile`s in this directory build images containing the
minimum necessary to run, build, or interact with the NVVL library
through pytorch.  They probably won't be very useful as-is but serve
as a base for expanding or examples to take from. While these are
similar to the Dockerfile's in the base `Docker` directory, the base
pytorch images have more packages already installed than the base cuda
images, so we need to install less. Also, we don't include `_opencv`
options since it is assumed pytorch will be used for anything opencv
would have been.

- `Dockerfile.minimal` - The minimum necessary to use the nvvl pytorch
  library.

- `Dockerfile.build` - Includes tools necessary to build the library.

- `Dockerfile.interactive` - Includes tools we found useful for
  interactive use (mostly the `ffmpeg` commandline tool)

The final two are useful for the decidedly non-docker-ish practice of
developing while inside a container.
