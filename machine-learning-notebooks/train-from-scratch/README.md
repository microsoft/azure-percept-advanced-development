# Train From Scratch

This directory contains an example AML notebook to show you how to train up a U-Net from scratch
and then download it onto the Azure Percept DK. 

This example U-Net model showcases volumetric detection of fresh produce using bananas.  An example video showing the output is available here: https://youtu.be/qnF1QyiF4p0

60% of all fresh food stocked at retail will spoil and is discarded, resulting in an annual $161 billion loss (source: USDA). Azure Percept can be used to generate time series analytical data of fresh produce inventory levels, allowing retailers to create more efficient supply chains with less safety stock. In turn this reduces CO2 emissions by less transportation of fresh food, and less fertilizer needed in the soil. This AI model calculates the size (in pixels) of the fresh produce in the bin. Since we know the static size of the empty bin, it is trivial to then calculate the total area occupied by the fresh produce as measured over time.  Although this example is trained for bananas, it can be used for any irregular shaped object.

**Please Note!**

1. This directory borrows heavily from https://github.com/milesial/Pytorch-UNet, which is under GPLv3.0. This directory
   and any works derived from it are therefore also under GPLv3.0. Use this as an example for how to do development with
   the Azure Percept DK, but be aware of the license restrictions, which are different from those in the rest of this repository.
   See the LICENSE file in this directory.
1. One exception to the above license is everything in the data directory, which is released under C-UDA. See the DATA_LICENSE file.
1. This notebook shows you how to train a U-Net neural network from scratch to detect bananas using semantic segmentation.
   But it does not show you how to add support for a custom neural network to the azureeyemodule. A tutorial for that is coming
   soon. But in the meantime, you can take a look at azureeyemodule/app/model/binaryunet.cpp for how this neural network is
   integrated into the Percept DK's azureeyemodule (the IoT Edge module that does inference using the camera and VPU accelerator).
