# Additional Information and Troubleshooting

## Information

- To create a decent model with transfer learning (used here) you'll likely need >100 images per class.  This may take some time to label so be prepared to spend an hour or two on that task.

## Troubleshooting

1. SSH into device not working. Sometimes, upon reboot, the IP address of the Percept DK will change.  Increment the current known IP address's last number by 1 and try again.  To assign a static IP, set the MAC address of the device on your local router/network to a local IP of your choosing.

2. The Azure ML Labeling project is not refreshing even though I added more images.  To add more images to the Azure ML labeling project: 1) upload the images to the same Datastore 2) in the labeling project, under "Details", go to "Incremental refresh (optional)" and uncheck, save, recheck, save (this will force it to find the new images in the Storage container).  Then label as usual.

3. Windows does not have a Python package available.  Try going to the [Unofficial Windows Binaries for Python Extension Packages](https://www.lfd.uci.edu/~gohlke/pythonlibs/), downloading the wheel (.whl) file, and using `python -m pip install <name of .whl file>` in your activated Python environment.  You may have to install the packages found in `requirements.txt` one by one.

4. JupyterLab throws error on Windows.  If you get "ImportError: DLL load failed while importing win32api: The specified module could not be found", try: `conda install pywin32`.

5. Trouble mounting the local folder in the docker container on Windows.  For guidance on how to mount the current directory to a Linux docker container see this [how-to-guide](https://rominirani.com/docker-on-windows-mounting-host-directories-d96f3f056a2c).

6. Poor model performance even with 100s of images per class.  Data science and ML are iterative disciplines especially in the realm of deep learning.  In the `3.Train_with_AzureML.ipynb`, you may need to try out different values for the hyperparameters `batch_size` and `learning_rate` set in the ML/train config file.  For example, if you a lot of images (100's), try a bigger `batch_size` (e.g. for 200 images you could try `batch_size=8`) or if you have very few images, try a smaller `batch_size`.  Another hyperparameter to play with is the `learning_rate`.  Try values an order of magnitude higher or lower and check for improvements in the mean average precision (mAP) values found in the `70_driver_log.txt` of the Azure ML run, under "Outputs + logs" tab.  For lower values of learning rates, extend the number of `epochs` when submitting the training experiment.  Hyperparameters are a more complex ML topic and a detailed explanation is out of scope for this guide, but you can certainly research and learn more as you go.