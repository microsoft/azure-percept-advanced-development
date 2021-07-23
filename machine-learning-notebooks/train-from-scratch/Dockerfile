# cf. https://github.com/Azure/AzureML-Containers
FROM mcr.microsoft.com/azureml/intelmpi2018.3-cuda10.0-cudnn7-ubuntu16.04

# opencv-python dependency
RUN apt-get install -y libgl1-mesa-dev

# Fix for AML not finding the right libffi version
RUN ln -s /opt/miniconda/lib/libffi.so.7 /usr/lib/x86_64-linux-gnu/libffi.so.7