#!/bin/bash

###################
# TRAINING CONFIG #
###################

export DATA_DIR=/raid
export RESOLUTION=540p  # options: 540p, 720p, 1080p, 4K
export LOADER="NVVL"  # options: "NVVL" or "pytorch"
export DATA_TYPE=scenes # options: "scenes" or "frames"
#export CODEC="h264"  #
#export CRF="18"      # set these three only if used during preprocessing
#export KEYINT="4"    #
export ROOT=$DATA_DIR/$RESOLUTION/$DATA_TYPE/$CODEC/${CRF+crf$CRF}/${KEYINT+keyint$KEYINT}/
#export IS_CROPPED="--is_cropped"  # Uncomment to crop input images
export CROP_SIZE="-1 -1"
#export CROP_SIZE="540 960"  # Only applicable if --is_cropped uncommented
#export TIMING="--timing"  # Uncomment to time data loading and computation - slower
export MINLR=0.0001
export MAXLR=0.001
export BATCHSIZE=2
export FRAMES=3
export MAX_ITER=1000000
export AMP="--amp"    # Uncomment to load data and train model in fp16

tensorboard --logdir runs 2> /dev/null &
echo "Tensorboard launched"

# Launch one PyTorch distributed process per GPU
python -m apex.parallel.multiproc main.py --loader $LOADER --batchsize $BATCHSIZE --frames $FRAMES --root $ROOT $IS_CROPPED --max_iter $MAX_ITER --min_lr $MINLR --max_lr $MAXLR $TIMING --crop_size $CROP_SIZE $AMP
