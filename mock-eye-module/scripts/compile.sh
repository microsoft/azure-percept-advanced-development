# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Compiles the mock inference app inside the docker image

# usage: ./compile.sh

# Make the tmp directory, then copy everything into it
rm -rf tmp
mkdir -p tmp
cp -r kernels tmp/
cp -r modules tmp/
cp main.cpp tmp/
cp CMakeLists.txt tmp/

docker run --rm -t -v `realpath tmp`:/home/openvino/blah -w /home/openvino/blah openvino/ubuntu18_runtime:2021.1 bash -c \
    "source /opt/intel/openvino/bin/setupvars.sh && \
    mkdir -p build && \
    cd build && \
    cmake .. && \
    make"
