This folder contains example scripts that can be used to compare the performance
of an NVVL PyTorch data loader to standard PyTorch PNG, JPEG and MP4 ffmpeg
based data loaders.

These scripts were written to be run in the Docker container built using
`./docker/Dockerfile`.

Modify the `NVVL_DIR` and `ROOT` paths in `run_docker_test.sh` and then execute
`bash run_docker_test.sh`

For reference, the table below contains performance metrics captured using these
scripts on one V100 GPU from a DGX-1 system running CUDA 9.1 and PyTorch 0.4.0a0+02b758f.

| Data loader | Input resolution | Numerical precision | CPU load (max of 80) | Host memory usage (GB) | Per iteration data time (ms) | Data loader overhead (ms) |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| PNG   | 540p  | fp32  | 6.68  | 16.55 | 64.50 | 23.5  |
| PNG   | 720p  | fp32  | 11.22 | 16.15 | 55.38 | 20.2  |
| PNG   | 1080p | fp32  | 11.05 | 16.15 | 89.70 | 21.67 |
| PNG   | 4K    | fp32  | 11.00 | 16.75 | 315.60| 32.11 |
| PNG   | 540p  | fp16  | 6.74  | 17.05 | 62.11 | 21.20 |
| PNG   | 720p  | fp16  | 11.50 | 16.30 | 53.80 | 21.49 |
| PNG   | 1080p | fp16  | 11.19 | 16.65 | 91.22 | 22.59 |
| PNG   | 4K    | fp16  | 10.92 | 16.90 | 317.18| 32.25 |
| JPEG  | 540p  | fp32  | 6.07  | 16.50 | 60.92 | 23.05 |
| JPEG  | 720p  | fp32  | 8.24  | 16.05 | 63.16 | 22.65 |
| JPEG  | 1080p | fp32  | 11.14 | 16.25 | 82.36 | 21.66 |
| JPEG  | 4K    | fp32  | 11.03 | 16.70 | 301.87| 32.62 |
| JPEG  | 540p  | fp16  | 5.86  | 16.95 | 66.08 | 18.04 |
| JPEG  | 720p  | fp16  | 8.112 | 17.05 | 65.64 | 19.90 |
| JPEG  | 1080p | fp16  | 11.25 | 16.70 | 81.06 | 22.82 |
| JPEG  | 4K    | fp16  | 10.93 | 17.05 | 297.42| 32.87 |
| FFMPEG| 540p  | fp32  | 6.83  | 16.50 | 66.93 | 22.65 |
| FFMPEG| 720p  | fp32  | 8.73  | 16.45 | 69.90 | 24.05 |
| FFMPEG| 1080p | fp32  | 10.42 | 16.15 | 102.24| 24.35 |
| FFMPEG| 4K    | fp32  | 10.56 | 16.60 | 354.63| 50.37 |
| FFMPEG| 540p  | fp16  | 6.99  | 17.05 | 67.88 | 20.37 |
| FFMPEG| 720p  | fp16  | 8.79  | 17.00 | 73.08 | 20.72 |
| FFMPEG| 1080p | fp16  | 9.94  | 16.85 | 108.44| 25.94 |
| FFMPEG| 4K    | fp16  | 10.85 | 17.10 | 361.48| 45.91 |
| NVVL  | 540p  | fp32  | 0.55  | 13.08 | 27.68 | 1.24  |
| NVVL  | 720p  | fp32  | 0.49  | 13.08 | 42.55 | 1.10  |
| NVVL  | 1080p | fp32  | 0.42  | 13.08 | 85.31 | 1.81  |
| NVVL  | 4K    | fp32  | 0.34  | 13.08 | 303.70| 2.92  |
| NVVL  | 540p  | fp16  | 0.54  | 13.08 | 27.64 | 1.31  |
| NVVL  | 720p  | fp16  | 0.50  | 13.08 | 42.56 | 1.52  |
| NVVL  | 1080p | fp16  | 0.48  | 13.08 | 85.18 | 1.78  |
| NVVL  | 4K    | fp16  | 0.34  | 13.08 | 303.34| 3.37  |

All experiments cropped frames to 540p, used a batch size of 8 and the CPU based data loaders each used 10 workers per GPU.
