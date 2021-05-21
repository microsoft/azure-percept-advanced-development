# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

<#
  .SYNOPSIS
  Prepares the device for native compilation by copying over the source code and stopping the
  running azureeyemodule and IoT Edge Runtime.

  .DESCRIPTION
  SCPs the azureeyemodule/app code over to an Azure Percept DK and stops the azureeyemodule
  and IoT Edge Runtime there. This will prepare the device for native building.
  This is useful for debugging a custom azureeyemodule (which is necessary when you are bringing your own AI model).

  .PARAMETER Ip
  The IP address of the Percept DK.

  .PARAMETER Username
  The username for the Percept DK.

  .PARAMETER Source
  Path to the azureeyemodule/app folder.

  .INPUTS
  None. You cannot pipe objects into this script.

  .OUTPUTS
  None.
#>

param (
    [Alias('i')][Parameter(Mandatory=$true)][string]$IpAddr,
    [Alias('u')][Parameter(Mandatory=$true)][string]$UserName,
    [Alias('s')][Parameter(Mandatory=$true)][ValidateScript({Test-Path $_ -PathType "Container"})][string]$SourceCode
)

# Define an SSH function
function ExecuteRemoteCmd {
    param (
        [string]$Cmd,
        [switch]$UseSudo
    )

    $HostAndIp = $UserName + "@" + $IpAddr
    if ($UseSudo) {
        $cmd = "sudo " + $Cmd
        Write-Host "ssh" $HostAndIp -t $cmd
        ssh $HostAndIp -t $cmd
    } else {
        Write-Host "ssh" $HostAndIp $Cmd
        ssh $HostAndIp $Cmd
    }
}

# Exit on errors
$ErrorActionPreference = "Stop"

# Remove the previous tmp folder (mkdir first so we don't complain if it doesn't exist)
ExecuteRemoteCmd -Cmd ("mkdir -p /home/" + $UserName + "/tmp")
ExecuteRemoteCmd -UseSudo -Cmd ("rm -r /home/" + $UserName + "/tmp")

# SCP the source code over
scp -r $SourceCode ($UserName + "@" + $IpAddr + ":~/tmp")

# Kill the IoT Edge Daemon
$iotedgestatus = ssh ($UserName + "@" + $IpAddr) ("systemctl status iotedge | grep \`"inactive (dead)\`"")
if (!$iotedgestatus) {
    ExecuteRemoteCmd -UseSudo -Cmd ("systemctl stop iotedge")
}

# Stop the AzureEyeModule if it is running
ExecuteRemoteCmd -Cmd ("docker stop azureeyemodule")

Write-Host "Code is on the device and the device is all prepared."
Write-Host "SSH over to the device and run"
Write-Host "
    docker run --rm \
                -v /dev/bus/usb:/dev/bus/usb \
                -v ~/tmp:/tmp \
                -w / \
                -it \
                --device-cgroup-rule='c 189:* rmw' \
                -p 8554:8554 \
                mcr.microsoft.com/azureedgedevices/azureeyemodule:preload-devkit /bin/bash
    "
Write-Host ""
Write-Host ""
Write-Host "Once inside the Docker container, run the following commands:"
Write-Host "
    1. cd /tmp
    2. mkdir build
    3. cd build
    4. cmake -DOpenCV_DIR=/eyesom/build/install/lib64/cmake/opencv4 -DmxIf_DIR=/eyesom/mxif/install/share/mxIf -DCMAKE_PREFIX_PATH=/onnxruntime/install/lib64 -DCMAKE_BUILD_TYPE=Debug ..
    5. make -j4
"
Write-Host "Then you can run ./inference"
Write-Host "You can use vi to make minor changes to the code (such as adding debug statements) and then recompile and retest as needed."