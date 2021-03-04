# Mock Eye Module

This directory contains all the code that we use to build and test a mock azureeyemodule application
using C++ and OpenVINO. The theory is that if the model works here, it should be easy to get working
on the Azure Percept DK.

## Scripts

Building and running this application use some scripts in the scripts directory. The compile.sh or compile.ps1
script does not require any dependencies other than Docker.

However, **if you are running Windows**, there is some set up you will have to do for the compile_and_run.ps1 script.

Install an X window server. I recommend [VcXsrv](https://sourceforge.net/projects/vcxsrv/), which can be installed
via Chocolatey with `choco install vcxsrv`. Now launch it with XLaunch, and configure it:

* Choose Multiple Windows
* Start No Client
* Make sure to click the box next to "Disable access control"

**Regardless of what OS you are running on**, you will also need to procure an OpenVINO IR for the model you are using.
For the ones in the Open Model Zoo, you can use the Workbench for downloading it, see [here](../scripts/run_workbench.sh).
If you are bringing a custom model, you can also use the same tool to convert from many common formats to OpenVINO IR.

## Building

I recommend using Docker for this. To do so, you can simply use the scripts found in the `scripts` directory.

To just make sure the application compiles (without actually running it), you can simply run `./scripts/compile.sh`
or `./scripts/compile.ps1`. This should pull the appropriate Docker image if you don't already have it.

## Running

If you would like to compile and run the mock application, you can use `./scripts/compile_and_test.sh` or
`./scripts/compile_and_test.ps1` along with whatever arguments. See `./scripts/compile_and_test.sh --help` for help.

Note that you will need either a webcam or an mp4 file. You will also need to get a neural network in OpenVINO IR
format. You can use the OpenVINO workbench for this if you are getting it from the OpenVINO model zoo.
See [here](../scripts/run_workbench.sh) for an example of how to run it. You can use that tool to download
whatever model you want from the Intel Open Model Zoo.

## Architecture and Extending

The architecture of the mocke eye module is quite simple. There is an enum in `modules/parser.hpp` with each
type of model that the mock application currently accepts. This is a subset of the models that the azureeyemodule
accepts, and this is on purpose to keep this application small and easy to use for porting to the Percept DK.

In `main.cpp`, there is a switch statement that looks at the value passed in on command line, matches it with one
of the available parsers and then hands control over to the parser.

If you want to use this for porting a new model to the Percept DK, you will want to take a look at the example parsers
and reference them when writing your own. See the [tutorials](../tutorials/README.md) in this repository for thorough
guidance.

When extending this application, please note a few folders:

* `kernels`: This folder contains all the OpenCV G-API kernels. You can, though you don't have to, add your kernels here.
* `modules/objectdetection`: This folder contains all the parsers for the object detectors in this mock application.
* `modules/openpose`: When extending this with something other than another object detector, you will likely want to
  create a new folder (and add it to the CMake file) and put all source code related to it inside that folder. In
  the case of Open Pose, which has very complicated post-processing logic, there are several files in this folder.
  This serves as an example for how you might want to structure your code, but try not to feel overwhelmed by its complexity,
  you don't need to understand anything about the logic in this folder to extend this application for your own model.
