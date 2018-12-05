FROM nvcr.io/nvidia/pytorch:18.11-py3

ARG FFMPEG_VERSION=4.1

ENV NVIDIA_DRIVER_CAPABILITIES=video,compute,utility

# nvcuvid deps
RUN apt-get update --fix-missing && \
    apt-get install -y libx11-6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# minimal ffmpeg from source
RUN apt-get update --fix-missing && \
    apt-get install -y \
      yasm \
      libx264-148 libx264-dev \
      libx265-79 libx265-dev \
      pkg-config && \
    cd /tmp && wget -q http://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.bz2 && \
    tar xf ffmpeg-$FFMPEG_VERSION.tar.bz2 && \
    rm ffmpeg-$FFMPEG_VERSION.tar.bz2 && \
    cd /tmp/ffmpeg-$FFMPEG_VERSION && \
    ./configure \
    --prefix=/usr/local \
    --disable-static --enable-shared \
    --disable-all --disable-autodetect --disable-iconv \
    --enable-avformat --enable-avcodec --enable-avfilter --enable-avdevice \
    --enable-protocol=file \
    --enable-demuxer=mov,matroska,image2 \
    --enable-bsf=h264_mp4toannexb,hevc_mp4toannexb \
    --enable-gpl --enable-libx264 --enable-libx265 --enable-zlib \
    --enable-indev=lavfi \
    --enable-swresample --enable-ffmpeg \
    --enable-swscale --enable-filter=scale,testsrc \
    --enable-muxer=mp4,matroska,image2 \
    --enable-decoder=h264,hevc,png,mjpeg,rawvideo \
    --enable-encoder=libx264,libx265,png,mjpeg \
    --enable-parser=h264,hevc,png && \
    make -j8 && make install && \
    ldconfig && \
    cd /tmp && rm -rf ffmpeg-$FFMPEG_VERSION && \
    apt-get remove -y yasm libx264-dev libx265-dev && \
    apt-get auto-remove -y && \
    rm -rf /var/lib/apt/lists/*

# nvvl build deps
RUN pip install --upgrade cmake

# nvidia-docker only provides libraries for runtime use, not for
# development, to hack it so we can develop inside a container (not a
# normal or supported practice), we need to make an unversioned
# symlink so gcc can find the library.  Additional, different
# nvidia-docker versions put the lib in different places, so we make
# symlinks for both places.
RUN ln -s /usr/local/nvidia/lib64/libnvcuvid.so.1 /usr/local/lib/libnvcuvid.so && \
    ln -s libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so
