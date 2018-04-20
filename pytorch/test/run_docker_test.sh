export NVVL_DIR=/home/jbarker/Projects/nvvl
export NVC="NVIDIA_DRIVER_CAPABILITIES=video,compute,utility"

ROOT=/raid/540p/scenes/train
#ROOT=/raid/540p/frames/train

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 1 --loader NVVL

#nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader lintel

#nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $ROOT:/data -v $NVVL_DIR:/workspace -u $(id -u):$(id -g) vsrnet python /workspace/pytorch/test/benchmark.py --root /data --sleep 0 --loader pytorch
