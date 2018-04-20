export NVVL_DIR=/home/jbarker/Projects/nvvl
export NVC="NVIDIA_DRIVER_CAPABILITIES=video,compute,utility"

### FP32

## NVVL

ROOT=/raid/540p/scenes/benchmark

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader NVVL

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.05 --loader NVVL

ROOT=/raid/720p/scenes/benchmark

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader NVVL --is_cropped --crop_size 540 960

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.075 --loader NVVL --is_cropped --crop_size 540 960

ROOT=/raid/1080p/scenes/benchmark

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader NVVL --is_cropped --crop_size 540 960

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.13 --loader NVVL --is_cropped --crop_size 540 960

ROOT=/raid/4K/scenes/benchmark

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader NVVL --is_cropped --crop_size 540 960

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.45 --loader NVVL --is_cropped --crop_size 540 960

## lintel

ROOT=/raid/540p/scenes/benchmark

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader lintel

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.05 --loader lintel

ROOT=/raid/720p/scenes/benchmark

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader lintel --is_cropped --crop_size 540 960

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.075 --loader lintel --is_cropped --crop_size 540 960

ROOT=/raid/1080p/scenes/benchmark

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader lintel --is_cropped --crop_size 540 960

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.13 --loader lintel --is_cropped --crop_size 540 960

ROOT=/raid/4K/scenes/benchmark

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader lintel --is_cropped --crop_size 540 960

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.45 --loader lintel --is_cropped --crop_size 540 960

## pytorch - png

ROOT=/raid/540p/png/benchmark

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader pytorch

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.05 --loader pytorch

ROOT=/raid/720p/png/benchmark

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader pytorch --is_cropped --crop_size 540 960

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.075 --loader pytorch --is_cropped --crop_size 540 960

ROOT=/raid/1080p/png/benchmark

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader pytorch --is_cropped --crop_size 540 960

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.13 --loader pytorch --is_cropped --crop_size 540 960

ROOT=/raid/4K/png/benchmark

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader pytorch --is_cropped --crop_size 540 960

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.45 --loader pytorch --is_cropped --crop_size 540 960

## pytorch - jpg

ROOT=/raid/540p/jpg/benchmark

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader pytorch

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.05 --loader pytorch

ROOT=/raid/720p/jpg/benchmark

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader pytorch --is_cropped --crop_size 540 960

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.075 --loader pytorch --is_cropped --crop_size 540 960

ROOT=/raid/1080p/jpg/benchmark

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader pytorch --is_cropped --crop_size 540 960

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.13 --loader pytorch --is_cropped --crop_size 540 960

ROOT=/raid/4K/jpg/benchmark

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader pytorch --is_cropped --crop_size 540 960

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.45 --loader pytorch --is_cropped --crop_size 540 960

## FP16

## NVVL

ROOT=/raid/540p/scenes/benchmark

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader NVVL --fp16

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.05 --loader NVVL --fp16

ROOT=/raid/720p/scenes/benchmark

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader NVVL --is_cropped --crop_size 540 960 --fp16

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.075 --loader NVVL --is_cropped --crop_size 540 960 --fp16

ROOT=/raid/1080p/scenes/benchmark

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader NVVL --is_cropped --crop_size 540 960 --fp16

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.13 --loader NVVL --is_cropped --crop_size 540 960 --fp16

ROOT=/raid/4K/scenes/benchmark

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader NVVL --is_cropped --crop_size 540 960 --fp16

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.45 --loader NVVL --is_cropped --crop_size 540 960 --fp16

## lintel

ROOT=/raid/540p/scenes/benchmark

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader lintel --fp16

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.27 --loader lintel --fp16

ROOT=/raid/720p/scenes/benchmark

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader lintel --is_cropped --crop_size 540 960 --fp16

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.27 --loader lintel --is_cropped --crop_size 540 960 --fp16

ROOT=/raid/1080p/scenes/benchmark

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader lintel --is_cropped --crop_size 540 960 --fp16

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.28 --loader lintel --is_cropped --crop_size 540 960 --fp16

ROOT=/raid/4K/scenes/benchmark

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader lintel --is_cropped --crop_size 540 960 --fp16

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.45 --loader lintel --is_cropped --crop_size 540 960 --fp16

## pytorch - png

ROOT=/raid/540p/png/benchmark

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader pytorch --fp16

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.27 --loader pytorch --fp16

ROOT=/raid/720p/png/benchmark

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader pytorch --is_cropped --crop_size 540 960 --fp16

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.27 --loader pytorch --is_cropped --crop_size 540 960 --fp16

ROOT=/raid/1080p/png/benchmark

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader pytorch --is_cropped --crop_size 540 960 --fp16

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.28 --loader pytorch --is_cropped --crop_size 540 960 --fp16

ROOT=/raid/4K/png/benchmark

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader pytorch --is_cropped --crop_size 540 960 --fp16

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.45 --loader pytorch --is_cropped --crop_size 540 960 --fp16

## pytorch - jpg

ROOT=/raid/540p/jpg/benchmark

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader pytorch --fp16

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.27 --loader pytorch --fp16

ROOT=/raid/720p/jpg/benchmark

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader pytorch --is_cropped --crop_size 540 960 --fp16

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.27 --loader pytorch --is_cropped --crop_size 540 960 --fp16

ROOT=/raid/1080p/jpg/benchmark

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader pytorch --is_cropped --crop_size 540 960 --fp16

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.28 --loader pytorch --is_cropped --crop_size 540 960 --fp16

ROOT=/raid/4K/jpg/benchmark

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader pytorch --is_cropped --crop_size 540 960 --fp16

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0.45 --loader pytorch --is_cropped --crop_size 540 960 --fp16
