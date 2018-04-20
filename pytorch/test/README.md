This folder contains example scripts that can be used to compare the performance
of an NVVL PyTorch data loader to standard PyTorch PNG, JPEG and MP4 ffmpeg
based data loaders.

These scripts were written to be run in the Docker container built using
`./docker/Dockerfile`.

Modify the `NVVL_DIR` and `ROOT` paths in `run_docker_test.sh` and then execute
`bash run_docker_test.sh`

For reference, the table below contains performance metrics captured using these
scripts on one V100 GPU from a DGX-1 system running CUDA 9.1 and PyTorch 0.4.0a0+02b758f.
