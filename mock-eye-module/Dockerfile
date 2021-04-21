# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Note that this Docker container bases itself off of https://hub.docker.com/r/openvino/ubuntu18_dev
# which is itself based off of Ubuntu 18.04. The Intel Distribution of OpenVINO (and therefore this
# Docker container) requires that you agree to an
# EULA (here: https://software.intel.com/content/www/us/en/develop/articles/end-user-license-agreement.html#inpage-nav-2)
# which you implicitly agree to when you use their software (and again, therefore this container).
#
# This Dockerfile is only necessary if you want to use GDB on the mock application.
# To use it, make sure you are in this directory, then run `docker build . -t mock-eye-module-debug`
FROM openvino/ubuntu18_runtime:2021.1
WORKDIR /
USER root

# Overwrite the default shell command form to include -xo pipefail
# See here for an explanation: https://gist.github.com/mohanpedala/1e2ff5661761d3abd0385e8223e16425
SHELL ["/bin/bash", "-xo", "pipefail", "-c"]

# Install stuff we need
RUN apt-get update && apt-get install -y gdb libc6-dbg && rm -rf /var/lib/apt/lists/*

# Use the OpenVINO user from the base image
USER openvino
WORKDIR /opt/intel/openvino

CMD ["/bin/bash"]