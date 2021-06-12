# This script SCPs the source code over to the device and
# gets the device ready for native building and debugging.
set -e

USAGE="$0 <device-username> <device-ip> <path/to/modules/azureeyemodule/app>"

# Check that we have the right number of args
if [ $# -ne 3 ]; then
    echo $USAGE
    exit 1
fi

# Check that $3 is a path to a directory
if [ ! -d "$3" ]; then
    echo $USAGE
    echo "$3 is not a directory."
    exit 2
fi

# SSH command
SSH="ssh $1@$2"

# Remove the previous tmp folder (mkdir it first so we don't complain if it doesn't exist)
$SSH mkdir -p /home/"$1"/tmp
$SSH -t sudo rm -r /home/"$1"/tmp

# Remote copy the source code over to the device
scp -r "$3" "$1"@"$2":~/tmp

# Kill the iotedge daemon
IOTEDGE_STATUS=$($SSH systemctl status iotedge | grep "inactive (dead)") || true
if [ -z "$IOTEDGE_STATUS" ]; then
    echo "Killing iotedge..."
    $SSH -t sudo systemctl stop iotedge
    echo "Done"
fi

# Stop the running azureeyemodule container (but ignore the error if there isn't one running already)
$SSH docker stop azureeyemodule

echo "Device is all prepared. Source code is on the device."
echo "SSH over to the device and run"
echo "
    docker run --rm \
                -v /dev/bus/usb:/dev/bus/usb \
                -v ~/tmp:/tmp \
                -w / \
                -it \
                --device-cgroup-rule='c 189:* rmw' \
                -p 8554:8554 \
                mcr.microsoft.com/azureedgedevices/azureeyemodule:preload-devkit /bin/bash
    "
echo ""
echo ""
echo "Once inside the Docker container, run the following commands:"
echo "
    1. cd /tmp
    2. mkdir build
    3. cd build
    4. cmake -DOpenCV_DIR=/eyesom/build/install/lib64/cmake/opencv4 -DmxIf_DIR=/eyesom/mxif/install/share/mxIf -DCMAKE_PREFIX_PATH=/onnxruntime/install/lib64 -DCMAKE_BUILD_TYPE=Debug ..
    5. make -j
"
echo "Then you can run ./inference"
echo "You can use vi to make minor changes to the code (such as adding debug statements) and then recompile and retest as needed."