# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Compiles and then runs the mock azureeyemodule application inside the docker image.
# If you want to use Debug mode, you must run `docker build . -t mock-eye-module-debug` from the mock-eye-module folder.

USAGE="$0
         [--debug/-g] (debug using GDB instead of running the program)
         [--device/-d=('NCS2'|'GPU'|'CPU')] (default is CPU)
         [--labels/-l=<path/to/labels.txt>] (only some models require this)
         [--parser/-p=<name of parser>] (see application for allowed ones)
         [--video/-v=<path/to/video>] (if not given, we use webcam)
         [--weights/-w=<path/to/.bin>] (if not given, we use example/model.bin)
         [--xml/-x=<path/to/.xml>] (if not given, we use example/model.xml)"

# Make sure that we are in the right directory by making sure that there is an example folder here
if [ ! -d "example" ]; then
    echo "You must run this script from the mock-eye-module directory."
    exit 1
fi

# Defaults
DEBUG="false"
DEVICE="CPU"
LABELPATH=""
PARSER="ssd100"
VIDEOPATH=""
WEIGHTPATH=""
XMLPATH=""

# Non-args
LABELFILE="/home/openvino/tmp/labels.txt"
VIDEOFILE=""
WEIGHTFILE="/home/openvino/tmp/model.bin"
XMLFILE="/home/openvino/tmp/model.xml"

# Parse args
for i in "$@"
do
case $i in
    -h|--help)
        echo "$USAGE"
        exit 0
        ;;
    -d=*|--device=*)
        DEVICE="${i#*=}"
        shift
        ;;
    -g|--debug)
        DEBUG="true"
        shift
        ;;
    -l=*|--labels=*)
        LABELPATH="${i#*=}"
        shift
        ;;
    -p=*|--parser=*)
        PARSER="${i#*=}"
        shift
        ;;
    -v=*|--video=*)
        VIDEOPATH="${i#*=}"
        VIDEOFILE="/home/openvino/tmp/movie.mp4"
        shift
        ;;
    -w=*|--weights=*)
        WEIGHTPATH="${i#*=}"
        shift
        ;;
    -x=*|--xml=*)
        XMLPATH="${i#*=}"
        shift
        ;;
    *)
        # Unknown option
        echo "$USAGE"
        exit 0
    ;;
esac
done

# Make the tmp directory, then copy everything into it
mkdir -p tmp
cp -r kernels tmp/
cp -r modules tmp/
cp main.cpp tmp/
cp CMakeLists.txt tmp/
cp example/* tmp/

# If given a video file, let's verify it exists and then copy it into tmp/
if [ ! -z "$VIDEOPATH" ] && [ ! -f "$VIDEOPATH" ]; then
    echo "--video should either not be given, or should point to a file. Given $VIDEOPATH"
    exit 4
elif [ ! -z "$VIDEOPATH" ]; then
    cp "$VIDEOPATH" tmp/movie.mp4
fi

# If given an XML file, let's verify it exists and then copy it into tmp/
if [ ! -z "$XMLPATH" ] && [ ! -f "$XMLPATH" ]; then
    echo "--xml should either not be given, or should point to a file. Given $XMLPATH"
    exit 5
elif [ ! -z "$XMLPATH" ]; then
    cp "$XMLPATH" tmp/model.xml
fi

# If given a weights file, let's verify it exists and then copy it into tmp/
if [ ! -z "$WEIGHTPATH" ] && [ ! -f "$WEIGHTPATH" ]; then
    echo "--weights should either not be given, or should point to a file. Given $WEIGHTPATH"
    exit 6
elif [ ! -z "$WEIGHTPATH" ]; then
    cp "$WEIGHTPATH" tmp/model.bin
fi

# Same for labels
if [ ! -z "$LABELPATH" ] && [ ! -f "$LABELPATH" ]; then
    echo "--labels should either not be given, or should point to a file. Given $LABELPATH"
    exit 7
elif [ ! -z "$LABELPATH" ]; then
    cp "$LABELPATH" tmp/labels.txt
fi

# Put together the command, based on whether we want webcam or not, and based on whether we need a labelfile or not
CMD="./mock_eye_app --device=$DEVICE --parser=$PARSER --weights=$WEIGHTFILE --xml=$XMLFILE --show"
if [ ! -z "$VIDEOPATH" ]; then
    CMD="$CMD --video_in=$VIDEOFILE"
fi

if [ "$PARSER" != "openpose" ]; then
    CMD="$CMD --labels=$LABELFILE"
fi

# Should we debug?
if [ "$DEBUG" == "true" ]; then
    CMD="gdb --args $CMD"
    DEBUG_DOCKER_CMD="-it"
    BUILDTYPE="Debug"
    DOCKERIMG="mock-eye-module-debug"
else
    DEBUG_DOCKER_CMD="-t"
    BUILDTYPE="Release"
    DOCKERIMG="openvino/ubuntu18_runtime:2021.1"
fi

echo "$CMD"

# Accept a local connection to Xserver
xhost +local:$(id -un) > /dev/null

# Docker command needs -i if we are debugging
docker run --rm \
            -e DISPLAY=$DISPLAY \
            -v /dev/bus/usb:/dev/bus/usb \
            -v `realpath tmp`:/home/openvino/tmp \
            -w /home/openvino/tmp \
            --device=/dev/video0:/dev/video0 \
            --device=/dev/dri:/dev/dri \
            --device-cgroup-rule='c 189:* rmw' \
            --network=host \
            --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
            "$DEBUG_DOCKER_CMD" \
            "$DOCKERIMG" bash -c \
                "source /opt/intel/openvino/bin/setupvars.sh && \
                mkdir -p build && \
                cd build && \
                cmake -DCMAKE_BUILD_TYPE=$BUILDTYPE .. && \
                make && \
                $CMD"

# Put the Xserver permission back to normal
xhost - > /dev/null
