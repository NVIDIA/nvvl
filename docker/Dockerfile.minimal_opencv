FROM nvcr.io/nvidia/pytorch:18.02-py3

ARG FFMPEG_VERSION=3.4.2
ARG OPENCV_VERSION=3.4.0
ARG OPENCV_CUDA_GENERATION=Volta

# nvcuvid deps
RUN apt-get update --fix-missing && \
    apt-get install -y libx11-6 libxext6
ENV NVIDIA_DRIVER_CAPABILITIES=video,compute,utility

# minimal ffmpeg from source
RUN apt-get install -y yasm && \
    cd /tmp && wget -q http://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.bz2 && \
    tar xf ffmpeg-$FFMPEG_VERSION.tar.bz2 && \
    rm ffmpeg-$FFMPEG_VERSION.tar.bz2 && \
    cd ffmpeg-$FFMPEG_VERSION && \
    ./configure \
      --prefix=/usr/local \
      --disable-static \
      --disable-all \
      --disable-autodetect \
      --disable-iconv \
      --enable-shared \
      --enable-avformat \
      --enable-avcodec \
      --enable-avfilter \
      --enable-protocol=file \
      --enable-demuxer=mov,matroska \
      --enable-bsf=h264_mp4toannexb,hevc_mp4toannexb && \
    make -j8 && make install && \
    cd /tmp && rm -rf ffmpeg-$FFMPEG_VERSION && \
    apt-get remove -y yasm

# paired down OpenCV build, just enough for video_reader and examples
RUN cd /tmp && \
    wget https://github.com/opencv/opencv/archive/$OPENCV_VERSION.tar.gz && \
    tar xzvf $OPENCV_VERSION.tar.gz && \
    rm $OPENCV_VERSION.tar.gz && \
    cd opencv-$OPENCV_VERSION && \
    mkdir build && cd build && \
    cmake -DCUDA_GENERATION=$OPENCV_CUDA_GENERATION \
      $(for m in cudabgsegm cudafeatures2d cudalegacy \
      cudaobjdetect dnn features2d flann highgui ml \
      objdetect photo python_bindings_generator shape \
      superres ts; do echo -DBUILD_opencv_$m=OFF; done) \
      $(for f in WEBP TIFF OPENEXR JASPER; do echo -DWITH_$f=OFF; done) \
      .. && \
    make -j8 && make install && \
    ldconfig && \
    cd /tmp && rm -rf opencv-$OPENCV_VERSION
