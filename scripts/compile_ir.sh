USAGE="$0 <path/to/model.xml> <path/to/model.bin>"

# Check that we have the right number of args
if [ $# -ne 2 ]; then
    echo $USAGE
    exit 1
fi

# Check that $1 is a path to a file
if [ ! -f "$1" ]; then
    echo "$USAGE"
    echo "Need a path to a model.xml, but $1 is not a file."
    exit 2
fi

# Check that $2 is a path to a file
if [ ! -f "$2" ]; then
    echo "$USAGE"
    echo "Need a path to a model.bin, but $2 is not a file."
    exit 3
fi

# Copy the model into the working directory
mkdir -p mnt
cp "$1" mnt/model.xml
cp "$2" mnt/model.bin

# Run the compilation in the Docker image.
docker run --rm -v /dev/bus/usb:/dev/bus/usb --device-cgroup-rule='c 189:* rmw' -v `realpath mnt`:/blah openvino/ubuntu18_dev:2021.1 /bin/bash -c "source /opt/intel/openvino/bin/setupvars.sh && ./deployment_tools/inference_engine/lib/intel64/myriad_compile \
    -m /blah/model.xml \
    -o /blah/model.blob \
    -VPU_NUMBER_OF_SHAVES 8 \
    -VPU_NUMBER_OF_CMX_SLICES 8 \
    -ip U8 \
    -op FP32"

#    -iop \"image_tensor:U8, image_info:FP32\" \ <------- use this for Faster RCNN ResNet50 because it has two different input tensors, and they have different formats.