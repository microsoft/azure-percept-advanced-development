# Getting started with machine learning development using VS Code + Jupyter Notebooks on AzureML

**Please note!** The experiences in this repository should be considered to be in **preview/beta**.
Significant portions of these experiences are subject to change without warning. **No part of this code should be considered stable**.

This article walks you through the process of setting up an Azure Machine Learning workspace, creating a compute instance,
setting up a Visual Studio Code development environment on your local machine, and running the cells of a Jupyter notebook within VS Code.

The [notebook](transfer-learning-using-ssd.ipynb) performs transfer learning using a pre-trained TensorFlow model (MobileNet-SSD-V2-Lite) on AzureML
in Python with a custom dataset to detect bowls.

The notebook shows how to start from [COCO](https://cocodataset.org/#home), filter it down to only the class of interest (bowls), and then
convert it into the appropriate format. Alternatively, you could use the open-source [VoTT 2](https://github.com/microsoft/VoTT) labeling tool to create
and label bounding boxes in the PASCAL VOC format.

After retraining the model on the custom dataset, the model can be deployed to your Azure Percept DK using the module twin update method.
You may then check model inferencing by viewing the live RTSP stream from your devkit. Both model retraining and deployment are
performed within the notebook through your remote compute instance.

## Prerequisites

- [Azure Machine Learning Subscription](https://azure.microsoft.com/en-us/free/services/machine-learning/)
- Azure Percept Development Kit with Azure Eye SoM connected
- OOBE (out of box experience) completed

## Download the Repository

Clone this repository or download the ZIP file.

## Create an Azure Machine Learning workspace and remote compute instance

1. Go to the [Azure Machine Learning Portal](https://ml.azure.com).
1. Select your directory, Azure subscription, and Machine Learning workspace from the drop down menus and click **Get started**.

    ![ml_portal_get_started](article_images/ml_portal_get_started.png)

    If you don’t have an Azure Machine Learning workspace yet, click **Create a new workspace**. In the new browser tab, do the following:

    1. Select your Azure subscription.
    1. Select your resource group.
    1. Enter a name for your workspace.
    1. Select your region.
    1. Select your workspace edition.
    1. Click **Review and create**.
    1. On the next page, review your selections and click **Create**.

        ![workspace_review_and_create](article_images/workspace_review_and_create.png)

    Please allow a few minutes for workspace creation. After the workspace creation is complete, you will see green check marks next to
    your resources and **Your deployment is complete** at the top of the Machine Learning Services overview page.

    ![workspace_creation_complete](article_images/workspace_creation_complete.png)

    Once your workspace creation is complete, return to the machine learning portal tab and click **Get started**.

1. In the machine learning workspace homepage, click **Compute** on the left-hand pane.

1. If there is no existing compute instance, create a new CPU or GPU compute by
   clicking on **New**. In the **New compute instance** window, enter a **Compute name**, choose a **Virtual machine type**,
   and select a **Virtual machine size**. Click **Create**.

    ![new_compute_instance](article_images/new_compute_instance.png)

    Once you click **Create**, it takes a few minutes to create the compute instance. Your **Compute** status will display a green circle and **\<your compute name> - Running** after compute creation is complete. This compute instance runs the Jupyter server and will be leveraged for this tutorial.

## Visual Studio Code development environment setup

1. Install the required tools.

    1. Option 1:

        Use the [Dev Tools Pack Installer](https://github.com/microsoft/Percept-Docs/blob/main/docs/dev-tools-installer.md) to install the following packages:

        - Visual Studio Code
        - Python 3.5, 3.6, or 3.7
        - Miniconda3
        - Intel OpenVino Toolkit 2020.2

        Note: The Intel OpenVino Toolkit is listed as an optional package in the Dev Tools Pack Installer, but it is required for working
        with the Azure Percept DK's camera module.

    1. Option 2:

        If you would prefer not to use the Dev Tools Pack Installer, manually install the
        following packages by clicking the links below and following the specified download and install guidelines:

        - [Visual Studio Code](https://code.visualstudio.com/)
        - [Python 3.5, 3.6, or 3.7](https://www.python.org/)
        - [Miniconda3](https://docs.conda.io/en/latest/miniconda.html)
        - [Intel OpenVino Toolkit 2021.1](https://docs.openvinotoolkit.org/)

    Note: If you already have the full Anaconda distribution installed, you don't need to install Miniconda. Alternatively, if you'd prefer not to use
    Anaconda or Miniconda, you can create a Python virtual environment and install the packages needed for running your AI model development using pip.

1. Launch Visual Studio Code.
1. Set up a data science environment:

    - Option 1 - Connect to an existing or new machine learning remote compute instance which already has the curated ML
      packages. **This is the option we will be using in this tutorial**.

    - Option 2 - Set up a conda environment on a local machine.
   	    1. Open an Anaconda or Miniconda command prompt and run the following command to create an environment named **myenv** with the specified packages:

            `conda create -n myenv python=3.7 pandas jupyter seaborn scikit-learn keras tensorflow=1.15`

            For additional information about creating and managing Anaconda environments, see the [Anaconda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

1. Create a VS Code workspace:

    1. Create a folder in a convenient location to serve as your Visual Studio Code workspace. Name it **myworkspace**.
    1. Open **myworkspace** in Visual Studio Code by clicking **File** > **Open Folder** > **Select Folder**.
    1. In the file explorer, navigate to and select the [transfer-learning-using-ssd.ipynb file](transfer-learning-using-ssd.ipynb)
       from your local copy of this GitHub repository. Copy this notebook file to the myworkspace folder.

## Azure integration with Visual Studio Code

1. If you haven't already, sign up for the [free or paid version of Azure Machine Learning](https://aka.ms/AMLFree).

1. Install the following Azure extensions for VS Code:
    - [Azure Machine Learning extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.vscode-ai).
    - Python Insiders extension. In VS Code, go to **View** -> **Command Palette**. In the command palette, enter and select **Python: Switch to Insiders Daily Channel**.
    - [Azure Account extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode.azure-account). Reload your window in VS Code when prompted.
    - [Azure IoT Hub Toolkit extension](https://marketplace.visualstudio.com/items?itemName=vsciot-vscode.azure-iot-toolkit).
    - [Azure IoT Edge extension](https://marketplace.visualstudio.com/items?itemName=vsciot-vscode.azure-iot-edge).

1. Sign in to your Azure Account within Visual Studio Code to provision resources and run workloads on Azure.
    1. Open the command palette in Visual Studio Code by selecting **View** > **Command Palette** from the menu bar.
    1. Enter **Azure: Sign In** into the command palette to start the login process.

        ![azure_sign_in](article_images/transfer_learning_azure_sign_in.png)

    1. Activate the Python extension and Azure account by clicking the Azure icon on the left-hand side of VS Code.

        ![azure_icon](article_images/azure_icon.png)

    1. Open the transfer-learning-using-ssd.ipynb notebook from the **myworkspace** folder.
    1. Open the command palette. Enter and select **Python: specify local or remote Jupyter server for connections**.
    1. You should see a list of connection options to choose from. Select **Azure ML Compute Instances**.

        ![azure_ml_compute_instances](article_images/azure_ml_compute_instances.png)

    1. Follow the guided prompts to choose a subscription, workspace, and remote compute instance.
       Select the workspace and remote compute instance created earlier in this tutorial. You also have the option of creating a new compute instance.
    1. After selecting a compute instance, you’ll be prompted to reload your VS Code window. Once you reload, select
       the **Python 3.6 - AzureML** kernel. You can now run any of the cells in your notebook. Running a notebook cell will instantiate the
       connection between your notebook and your compute instance. The notebook toolbar will be updated to reflect the compute instance you’re working on.

## Working with the notebook

You are now ready to run the notebook to train your custom bowl detector in VS Code and deploy it to your devkit. Make sure to run
each cell of the notebook individually as some of the cells require input parameters before executing the script.
Cells that require input parameters may be edited directly in the notebook. To run a cell, click the play icon on the left side of the cell:

![run_cell](article_images/run_cell1.png)

## Provide Feedback

After completing the advanced tools experiences, please provide feedback
via this [questionnaire](https://forms.office.com/Pages/ResponsePage.aspx?id=v4j5cvGGr0GRqy180BHbRzoJxrXKT0dEvfQyxsA0h8lUMzE0V0pCTFU4UUVSS0xTRUtNT0hZSEs1Ry4u).
Your feedback will help us continue to fine-tune and improve the advanced tools experience.

For additional Azure Machine Learning service example notebooks, please visit this [repo](https://github.com/Azure/MachineLearningNotebooks/tree/2aa7c53b0ce84e67565d77e484987714fdaed36e/how-to-use-azureml)
