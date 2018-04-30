FROM nvcr.io/nvidia/pytorch:18.03-py3

ENV CPATH="$CPATH:/usr/local:/usr/local/lib:/usr/local/include:/usr/local/bin"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local:/usr/local/lib:/usr/local/include:/usr/local/bin"
ENV PATH="$PATH:/usr/local:/usr/local/lib:/usr/local/include:/usr/local/bin"
ENV LIBRARY_PATH="$LIBRARY_PATH:/usr/local:/usr/local/lib:/usr/local/include:/usr/local/bin"

# nvcuvid deps
RUN apt-get update --fix-missing && \
    apt-get install -y libx11-6 libxext6
ENV NVIDIA_DRIVER_CAPABILITIES=video,compute,utility

RUN cd /tmp && \
    wget http://www.nasm.us/pub/nasm/releasebuilds/2.13.01/nasm-2.13.01.tar.bz2 && \
    tar xvjf nasm-2.13.01.tar.bz2 && \
    cd nasm-2.13.01 && \
    ./configure --prefix=/usr/local && \
    make -j8 && \
    make install

RUN cd /tmp && git clone git://git.videolan.org/x264.git && cd x264 && \
    ./configure --enable-static --enable-shared --prefix=/usr/local && \
    make -j8 && make install


# minimal ffmpeg from source
ARG FFMPEG_VERSION=3.4.2
RUN apt-get install -y yasm wget && \
    cd /tmp && wget -q http://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.bz2 && \
    tar xf ffmpeg-$FFMPEG_VERSION.tar.bz2 && \
    rm ffmpeg-$FFMPEG_VERSION.tar.bz2 && \
    cd ffmpeg-$FFMPEG_VERSION && \
    ./configure \
      --prefix=/usr/local \
      --enable-shared \
      --enable-avformat \
      --enable-avcodec \
      --enable-avfilter \
      --enable-protocol=file \
      --enable-demuxer=mov,matroska \
      --enable-bsf=h264_mp4toannexb,hevc_mp4toannexb \
      --enable-gpl \
      --enable-libx264 \
      --enable-pic \
      --enable-runtime-cpudetect \
      --enable-ffprobe \
      --cc="gcc -fPIC" && \
    make -j8 && make install && \
    ldconfig && \
    cd /tmp && rm -rf ffmpeg-$FFMPEG_VERSION && \
    apt-get remove -y yasm

# install stub library since driver libs aren't available at image build time
# this is a temporary requirement that will go away in future cuda versions
# libnvcuvid.so was created using the make-stub.sh script
COPY libnvcuvid.so /usr/local/cuda/lib64/stubs

# install nvvl
RUN pip install --upgrade cmake && \
    apt-get install -y pkg-config && \
    cd /tmp && \
    git clone https://github.com/nvidia/nvvl.git && \
    mkdir -p nvvl/build && \
    cd nvvl/build && \
    cmake .. && make && make install && \
    cd ../pytorch && \
    python3 setup.py install && \
    pip uninstall -y cmake && \
    apt-get remove -y pkg-config && \
    apt-get autoremove -y

RUN pip install scikit-image tensorflow tensorboard tensorboardX psutil

RUN git clone https://github.com/dukebw/lintel.git /workspace/lintel

RUN cd /workspace/lintel && \
    pip install .
