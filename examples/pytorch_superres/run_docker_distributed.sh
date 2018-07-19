export NVVL_DIR=/home/jbarker/Projects/nvvl/
export DATA_DIR=/raid/

export NVC="NVIDIA_DRIVER_CAPABILITIES=video,compute,utility"

nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $NVVL_DIR:/workspace -v $DATA_DIR:$DATA_DIR  -u $(id -u):$(id -g) -p 3567:3567 -p 6006:6006 nvcr.io/nvidian_general/adlr_vsrnet:latest /workspace/examples/pytorch_superres/run_distributed.sh
