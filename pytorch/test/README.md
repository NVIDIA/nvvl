This folder contains example scripts that can be used to compare the performance
of an NVVL PyTorch data loader to standard PyTorch PNG, JPEG and MP4 ffmpeg
based data loaders.

These scripts were written to be run in the Docker container built using
`NVVL/examples/pytorch_superres/docker/Dockerfile`.

Modify the `NVVL_DIR` and `ROOT` paths in `run_docker_test.sh` and then execute
`bash run_docker_test.sh`
