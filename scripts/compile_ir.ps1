# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

<#
  .SYNOPSIS
  Compiles the given XML and BIN files into BLOB format.

  .DESCRIPTION
  Compiles the OpenVINO IR format model into the Myriad X blob format.

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
    [Alias('x')][Parameter(Mandatory=$true)][ValidateScript({Test-Path $_ -PathType "Leaf"})][string]$xml
)

# Exit on errors
$ErrorActionPreference = "Stop"

# Create a $bin variable analagous to $xml
$bin = [io.path]::GetDirectoryName($xml) + "/" + [io.path]::GetFileNameWithoutExtension($xml) + ".bin"

# Now create the tmp directory and copy everything into it
New-Item -Type Directory -Name tmp

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

# This is the command we will run inside the container
$dockercmd =  "source /opt/intel/openvino/bin/setupvars.sh && "
$dockercmd += "/opt/intel/openvino_2021/deployment_tools/inference_engine/lib/intel64/myriad_compile "
$dockercmd += "-m /home/openvino/tmp/model.xml "
$dockercmd += "-o /home/openvino/tmp/model.blob "
$dockercmd += "-VPU_NUMBER_OF_SHAVES 8 "
$dockercmd += "-VPU_NUMBER_OF_CMX_SLICES 8 "
$dockercmd += "-ip U8 "
$dockercmd += "-op FP32"

$tmppath = $(Resolve-Path tmp).Path
$docker_run_cmd = "docker run --privileged --rm -v ${tmppath}:/home/openvino/tmp -w /home/openvino/tmp  openvino/ubuntu18_dev:2021.1 /bin/bash -c `"$dockercmd`""
Write-Host ${docker_run_cmd}
Invoke-Expression "& $docker_run_cmd"