The `Dockerfile`s in this directory build images containing the
minimum necessary to run, build, or interact with the NVVL
library.  They probably won't be very useful as-is but serve as a
base for expanding or examples to take from.

- `Dockerfile.minimal` - The minimum necessary to run a program linked
  against `libnvvl`

- `Dockerfile.build` - Includes tools necessary to build the library.

- `Dockerfile.interactive` - Includes tools we found useful for
  interactive use (mostly the `ffmpeg` commandline tool)

The final two are useful for the decidedly non-docker-ish practice of
developing while inside a container.

In addition, there is a version of each with an `_opencv` suffix that
also includes a build of opencv with enough modules to and run the
example code that uses opencv.
