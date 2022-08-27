FROM nvcr.io/nvidia/l4t-ml:r34.1.1-py3

RUN mkdir -p /src
COPY requirements.txt /src
WORKDIR /src

RUN pip3 install -U pip
RUN pip3 install -r requirements.txt
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt.git
RUN ls
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/aarch64-linux-gnu/tegra/
RUN ls /usr/lib/aarch64-linux-gnu/tegra/
#RUN rm /usr/lib/aarch64-linux-gnu/tegra
COPY tegra/* /usr/lib/aarch64-linux-gnu/tegra/
RUN ls /usr/lib/aarch64-linux-gnu/tegra/
# COPY libnvdla_compiler.so /usr/lib/aarch64-linux-gnu/tegra/libnvdla_compiler.so
# COPY libnvdla_runtime.so /usr/lib/aarch64-linux-gnu/tegra/libnvdla_runtime.so
# COPY libnvmedia.so /usr/lib/aarch64-linux-gnu/tegra/libnvmedia.so
# COPY libnvmedia_tensor.so /usr/lib/aarch64-linux-gnu/tegra/libnvmedia_tensor.so
# COPY libnvmedia_dla.so /usr/lib/aarch64-linux-gnu/tegra/libnvmedia_dla.so
# COPY libnvos.so /usr/lib/aarch64-linux-gnu/tegra/libnvos.so
# COPY libnvvideo.so /usr/lib/aarch64-linux-gnu/tegra/libnvvideo.so
# COPY libnvsocsys.so /usr/lib/aarch64-linux-gnu/tegra/libnvsocsys.so
# COPY libnvrm_mem.so /usr/lib/aarch64-linux-gnu/tegra/libnvrm_mem.so
# COPY libnvrm_host1x.so /usr/lib/aarch64-linux-gnu/tegra/libnvrm_host1x.so
# COPY libnvrm_surface.so /usr/lib/aarch64-linux-gnu/tegra/libnvrm_surface.so
# COPY libnvtvmr.so /usr/lib/aarch64-linux-gnu/tegra/libnvtvmr.so
# COPY libnvrm_chip.so /usr/lib/aarch64-linux-gnu/tegra/libnvrm_chip.so
# COPY libnvrm_sync.so /usr/lib/aarch64-linux-gnu/tegra/libnvrm_sync.so
# COPY libnvrm_stream.so /usr/lib/aarch64-linux-gnu/tegra/libnvrm_stream.so
# COPY libnvdc.so /usr/lib/aarch64-linux-gnu/tegra/libnvdc.so
# COPY libnvparser.so /usr/lib/aarch64-linux-gnu/tegra/libnvparser.so
RUN du -a /usr/lib/aarch64-linux-gnu/tegra/
#RUN ls -l /usr/lib/aarch64-linux-gnu/tegra/libnvdla_compiler.so
RUN cd torch2trt \
    && python3 setup.py install