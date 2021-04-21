# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

<#
  .SYNOPSIS
  Compiles and runs the mock Azure Eye Percept DK application.

  .DESCRIPTION
  Compiles and runs the mock application, accepting a bunch of command line
  arguments. This script will create a temporary folder, copy all the source
  code into it, then run a Docker container that maps that folder into the
  Docker container's filesystem. Inside the container, we compile the source
  code and run the resulting application.

  This script has several dependencies. See the README in the mock-eye-module directory.

  .PARAMETER Ipaddr
  Your host PC's IP address. Needed for the XServer forwarding.

  .PARAMETER DebugMode
  If given, we compile the application in Debug mode and drop you into a GDB session.
  If you want to use Debug mode, you must run `docker build . -t mock-eye-module-debug` from the mock-eye-module folder
  because the OpenVINO Docker image does not include GDB, so we need a custom Docker container to do it.

  .PARAMETER Labels
  The label file for the neural network. Not all models will require this.

  .PARAMETER Parser
  The string representation of the parser enum variant you want to use. See the application's
  help for accepted variants. Defaults to SSD.

  .PARAMETER Video
  If given, should be a path to a .mp4 file, which will serve as the input video source.
  If not given, we attempt to use your PC's webcam.

  .PARAMETER Xml
  The path to the .xml file of the neural network, which must be in OpenVINO IR format,
  and therefore must be composed of an .xml file and a .bin file. The .xml file and .bin
  file should have the same name other than the extensions, and should be in the same folder.

  .INPUTS
  None. You cannot pipe objects into this script.

  .OUTPUTS
  None.
#>

param (
    [Alias('i')][Parameter(Mandatory=$true)][string]$ipaddr,
    [Alias('g')][switch]$debugmode = $False,
    [Alias('l')][ValidateScript({Test-Path $_ -PathType "Leaf"})][string]$labels,
    [Alias('p')][string]$parser = "ssd",
    [Alias('v')][ValidateScript({Test-Path $_ -PathType "Leaf"})][string]$video,
    [Alias('x')][Parameter(Mandatory=$true)][ValidateScript({Test-Path $_ -PathType "Leaf"})][string]$xml
)

# Exit on errors
$ErrorActionPreference = "Stop"

# Make sure we are in the right directory
If (-NOT (Test-Path "main.cpp" -PathType "Leaf")) {
    Write-Host "You must run this script from the mock-eye-module directory."
    Exit 1
}

# Create a $bin variable analagous to $xml
$bin = [io.path]::GetDirectoryName($xml) + "/" + [io.path]::GetFileNameWithoutExtension($xml) + ".bin"

# Map the variables from where they are on the host system to the Docker container
$labelfile = "/home/openvino/tmp/labels.txt"
$videofile = ""
$weightfile = "/home/openvino/tmp/model.bin"
$xmlfile = "/home/openvino/tmp/model.xml"
$videofile = "/home/openvino/tmp/movie.mp4"

# Now create the tmp directory and copy everything into it
If (Test-Path "tmp") {
    Remove-Item -Recurse -Force tmp
}
New-Item -Type Directory -Name tmp
Copy-Item -Recurse kernels tmp/
Copy-Item -Recurse modules tmp/
Copy-Item main.cpp tmp/
Copy-Item CMakeLists.txt tmp/

# For each of these files, if they exist, copy them into the temp directory.
if ($labels) {
    $path = "tmp/labels.txt"
    Copy-Item $labels $path
}

if ($video) {
    $path = "tmp/movie.mp4"
    Copy-Item $video $path
}

if (Test-Path $xml) {
    $path = "tmp/model.xml"
    Copy-Item $xml $path
}

if (Test-Path $bin) {
    $path = "tmp/model.bin"
    Copy-Item $bin $path
} else {
    Write-Host "Need a .bin file that is the same name as the .xml file and in the same place."
    Exit 2
}

# Put together the command, based on whether we want webcam or not.
$appcmd = "./mock_eye_app --device=CPU --parser=" + $parser + " --weights=" + $weightfile + " --xml=" + $xmlfile + " --show"
if ($video) {
    $appcmd += " --video_in=" + $videofile
}

if ($labels) {
    $appcmd += " --labels=" + $labelfile
}

# Should we debug?
$dockercmd = ""
$debug_docker_cmd = ""
$buildtype = ""
if ($debugmode) {
    $dockercmd = "mock-eye-module-debug bash -c `""
    $appcmd = "gdb --args " + $appcmd
    $debug_docker_cmd = "-it"
    $buildtype = "Debug"
} else {
    $dockercmd =  "openvino/ubuntu18_runtime:2021.1 bash -c `""
    $debug_docker_cmd = "-t"
    $buildtype = "Release"
}

$dockercmd += "source /opt/intel/openvino/bin/setupvars.sh && "
$dockercmd += "mkdir -p build && "
$dockercmd += "cd build && "
$dockercmd += "cmake -DCMAKE_BUILD_TYPE=" + ${buildtype} + " .. && "
$dockercmd += "make -j4 && "
$dockercmd += $appcmd + "`""

# Set up the windowing system
$display = ${ipaddr} + ":0"

$tmppath = $(Resolve-Path tmp).Path
$docker_run_cmd = "docker run --privileged --rm -e DISPLAY=${display} -v ${tmppath}:/home/openvino/tmp -w /home/openvino/tmp ${debug_docker_cmd} $dockercmd"
Write-Host ${docker_run_cmd}
Invoke-Expression "& $docker_run_cmd"