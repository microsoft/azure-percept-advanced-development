#############################################################################################
# Copyright: Microsoft 2021
# License: MIT
# Purpose:  This Dockerfile is for use with Azure ML TF Object Detection experiments.
# It's the base of an Environment image for training object detection models.
# The base is from a build of the TensorFlow Object Detection API.
# Additionally, the following is built into the image:
# - Miniconda (custom Dockerfile method with Azure ML expects conda)
# - Azure ML SDK
#############################################################################################
FROM nvidia/cuda:10.0-cudnn7-runtime

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update \
      && apt-get install --no-install-recommends --no-install-suggests -y gnupg2 ca-certificates \
            git build-essential libopencv-dev python3-opencv \
      && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
      && apt-get install --no-install-recommends --no-install-suggests -y libopencv-highgui3.2 \
      wget \
      bzip2 \
      git \
      ffmpeg \
      libsm6 \
      libxext6 \
      unzip \
      curl \
      && rm -rf /var/lib/apt/lists/*
      
RUN apt-get update \
    && apt-get install -y -qq protobuf-compiler python-tk

# Get Python in the form of Miniconda
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
	bash ~/miniconda.sh -b -p /opt/conda && \
	rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

# Update package installers
RUN conda install python=3.6 && conda update conda -y

# Install Python packages
RUN python -m pip install --upgrade pip==19.0 setuptools
RUN python -m pip install azureml-core==1.25.0
RUN python -m pip install tensorflow-gpu==1.15
RUN python -m pip install tf-slim
RUN python -m pip install Cython \
                contextlib2 \
                pillow \
                lxml \
                matplotlib \
                PyDrive \
                pycocotools \
                build \
                utils \
                dataclasses \
                install \
                azure-iot-device \
                azure-iot-hub \
                numpy==1.17

# Update protocol buffers for TF object detection API
RUN curl -OL 'https://github.com/google/protobuf/releases/download/v3.2.0/protoc-3.2.0-linux-x86_64.zip' \
    && unzip -o protoc-3.2.0-linux-x86_64.zip -d protoc3 \
    && mv protoc3/bin/* /usr/local/bin/ \
    && mv protoc3/include/* /usr/local/include/

WORKDIR /home/

# Install TensorFlow models and scripts; compile contents of protos folder
RUN git clone --depth 1 --branch v1.13.0 https://github.com/tensorflow/models.git \
    && export PYTHONPATH=`pwd`/models/research:`pwd`/models/research/slim:`pwd`:$PYTHONPATH \
    && cd models/research \
    && /usr/local/bin/protoc object_detection/protos/*.proto --python_out=. \
    && python setup.py build \
    && python setup.py install \
    && python object_detection/builders/model_builder_test.py

RUN mkdir data \
    && cd data \
    && curl -OL 'http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz'


