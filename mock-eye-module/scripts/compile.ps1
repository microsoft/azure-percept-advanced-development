# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Exit on errors
$ErrorActionPreference = "Stop"

# Create a tmp directory that will house the build artifacts, as well as all the source code
If (Test-Path "tmp") {
    Remove-Item -Recurse -Force tmp
}
New-Item -ItemType Directory -Name tmp
Copy-Item -Recurse kernels tmp/
Copy-Item -Recurse modules tmp/
Copy-Item main.cpp tmp/
Copy-Item CMakeLists.txt tmp/

# Full path to the tmp directory
$tmppath = $(Resolve-Path tmp).Path

# Create the bash command that I want to run as a variable to avoid having a bunch of carriage returns corrupt it.
# I wonder if whoever made the line-endings decisions on the different operating systems realized
# these decisions would still be relavent decades into the future...
$bashcmd =  "source /opt/intel/openvino/bin/setupvars.sh && "
$bashcmd += "mkdir -p build && "
$bashcmd += "cd build && "
$bashcmd += "cmake .. && "
$bashcmd += "make"

# Use Docker to do the build, mapping the tmp directory into the Docker container's filesystem
docker run --rm -t -v ${tmppath}:/home/openvino/blah -w /home/openvino/blah openvino/ubuntu18_runtime:2021.1 /bin/bash -c ${bashcmd}