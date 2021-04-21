# Mock Eye Module

**Please note!** The experiences in this repository should be considered to be in **preview/beta**.
Significant portions of these experiences are subject to change without warning. **No part of this code should be considered stable**.

This directory contains all the code that we use to build and test a mock azureeyemodule application
using C++ and OpenVINO. The theory is that if the model works here, it should be easy to get working
on the Azure Percept DK.

> **_NOTE:_** The Mock Eye Module runs on your CPU (or if you are running Linux, you can use a Neural Compute Stick 2 as well).
              Just because a neural network works on your 128 GB RAM, Intel i7, doesn't mean that it is going to work satisfactorily
              on the Myriad X VPU and ARM chip. Keep this in mind when porting!

## Scripts

Building and running this application uses some scripts in the scripts directory. The compile.sh or compile.ps1
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
type of model that the mock application currently accepts. To limit code duplication between this and the azureeyemodule,
we only support SSD here. But the point is for you to use this tiny application as a sandbox for developing your
AI model on a PC before porting it to the Azure Percept DK.

In `main.cpp`, there is a switch statement that looks at the value passed in on command line, matches it with one
of the available parsers (which is currently just SSD) and then hands control over to the parser.

If you want to use this for porting a new model to the Percept DK, you will want to take a look at the example parser
and reference it when writing your own. See the [tutorials](../tutorials/README.md) in this repository for thorough
guidance.

When extending this application, please note a few folders:

* `kernels`: This folder contains all the OpenCV G-API kernels. You can, though you don't have to, add your kernels here.
* `modules/objectdetection`: This folder contains all the parsers for the object detectors in this mock application.
