# Machine Learning Notebooks

**Please note!** The experiences in this repository should be considered to be in **preview/beta**.
Significant portions of these experiences are subject to change without warning. **No part of this code should be considered stable**.
Although this device is in public preview, if you are using these advanced experiences, we would like your feedback. If you have not
[registered for *private* preview](https://go.microsoft.com/fwlink/?linkid=2156349), please consider doing so before using this functionality
so that we can better track who is using it and how, and so that you can have a say in the changes we make to these experiences.

This folder contains some IPython notebooks that show you how to bring a model that you train yourself to the device.
Use these instructions as examples to train up a model and bring it to the device so that you don't have to rely on
the training that the models have already been through (which will usually be too general for your use case).

**Note**: These notebooks use models that have already been implemented in the azureeyemodule. If you want to bring a completely custom
model to the device, you will need to add support for your custom model into the azureeyemodule and deploy the custom azureeyemodule
to your device. If you port a well-known model that others will likely want to use, please consider opening a pull request!
See the [azureeyemodule](../azureeyemodule/README.md) folder for instructions.

## Transfer Learning using Single Shot Detector

In this [Jupyter notebook](transfer-learning/transfer-learning-using-ssd.ipynb), you can see how to do transfer learning to bring a custom SSD network to
the device. There are two tutorial options to guide you through working with the notebook:

- [Cloud development](transfer-learning/transfer-learning-cloud.md): in this tutorial, you will run the notebook in the [Azure Machine Learning Portal](https://ml.azure.com)
  with a remote compute instance.

- [Local development](transfer-learning/transfer-learning-local.md): in this tutorial, you will run the notebook locally within VS Code, but using a remote compute instance.

**Note**: This notebook is a little out-dated. It uses TensorFlow 1.x, rather than 2.x, and it doesn't use AML features or Docker to manage dependencies.
There is work planned to improve this notebook, but it still works, and for now, please use it as an example of how you might go about doing transfer
learning for the device.
