# for cuda 12
# FROM nvcr.io/nvidia/tensorrt:23.06-py3
# for cuda 11
FROM nvcr.io/nvidia/tensorrt:22.12-py3 
ENV DEBIAN_FRONTEND noninteractive

# Build tools
RUN apt update && apt install -y libgl1-mesa-glx
RUN python3 -m pip install opencv-python \
                            line_profiler \
                            cupy-cuda12x \
                            pandas
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install typing-extensions --upgrade
WORKDIR /w
COPY . .

# docker build -t mri .
# docker run --runtime nvidia --rm -dit -v $PWD:/w -w /w mri bash
