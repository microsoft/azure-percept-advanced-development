# Using Azure Functions to decrypt and send retraining data to a Custom Vision project for retraining

- **Please note!** Azure Percept currently supports AI model protection as a private preview feature.  
- Portions of this code base are subject to change without notice.

Please consider taking our [Product survey](https://go.microsoft.com/fwlink/?linkid=2156573) to help us improve Azure Percept Model and Data Protection features based on your IoT Edge background and goals.

This sample demonstrates how to create an Azure Function to decrypt the Azure Percept MM encrypted retraining data and upload it to a Custom Vision project for model retraining. 

## Prerequisites

To run the sample, you need:

* [Install Visual Studio Code](https://code.visualstudio.com)
* [Config Visual Studio Code for Azure Functions development](https://docs.microsoft.com/en-us/azure/azure-functions/create-first-function-vs-code-python)
* [Setup Custom Vision project and get project info](https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/quickstarts/object-detection?tabs=visual-studio&pivots=programming-language-python)
* Service Principal (AZURE_CLIENT_ID, AZURE_CLIENT_SECRET and AZURE_TENANT_ID) which was recorded during Azure Percept Model Management (MM) service deployment.

## Steps

### 1. Launch Visual Studio Code, create a new Azure Function project with the following settings:
* Language : Python
* Python interpreter: Python 3.6.x, 3.7.x 3.8.x are supported
* Template: Azure Blob Storage trigger
* Select a storage account: choose your Azure Percept MM service storage account (i.e. testmmmodels)
* Blob storage path to be monitored: data

### 2. Add the following environment variables in local.settings.json according to your Azure Percept MM service and Custom Vision project:
```
"AZURE_CLIENT_ID": "", 
"AZURE_CLIENT_SECRET": "",
"AZURE_TENANT_ID": "",
"mm_server_url": "",
"mm_storage_account": "",
"mm_telemetry_storage_container": "data",
"custom_vision_endpoint": "",
"custom_vision_training_key": "",
"custom_vision_project_id": ""
```
For example: 
```
"AZURE_CLIENT_ID": "33e5...",
"AZURE_CLIENT_SECRET": "c383...",
"AZURE_TENANT_ID": "72f9...",
"mm_server_url": "https://test-mm.westus2.cloudapp.azure.com",
"mm_storage_account": "testmmmodels",
"mm_telemetry_storage_container": "data",
"custom_vision_endpoint": "https://cvdemo.cognitiveservices.azure.com/",
"custom_vision_training_key": "4240...",
"custom_vision_project_id": "2253..."
```
### 3. Grant the Service Principal account as "Storage Blob Data Reader" role to your Azure Percept MM storage account (defined as "mm_storage_account").   

### 4. Add the following dependencies in requirements.txt:
```
azure-identity
azure-storage-blob
azure-cognitiveservices-vision-customvision
sczpy
```
### 5. Update ```__init__.py``` with the following code:

```python
import logging
import os

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.identity import DefaultAzureCredential
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry
from msrest.authentication import ApiKeyCredentials
import azure.functions as func

import sczpy

# Secure locker config
server_url = os.environ["mm_server_url"]
storage_name = os.environ["mm_storage_account"]
storage_container = os.environ["mm_telemetry_storage_container"]
storage_url = f"https://{storage_name}.blob.core.windows.net"

# Custom vision project config
custom_vision_endpoint = os.environ["custom_vision_endpoint"]
training_key = os.environ["custom_vision_training_key"]
project_id = os.environ["custom_vision_project_id"]

logging.info(f"Create data folder.")
data_dir = '/tmp'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

logging.info(f"Initialize sczpy client.")
client = sczpy.SCZClient(server_url)

logging.info(f"Initialize custom vision project.")
credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(custom_vision_endpoint, credentials)
project = trainer.get_project(project_id)

def create_data_storage_client(blob_name):
    store_credential = DefaultAzureCredential()
    return BlobClient(storage_url, container_name=storage_container, blob_name=blob_name, credential=store_credential)

def main(myblob: func.InputStream):
    logging.info(f"Blob trigger function processed. Blob name: {myblob.name}")

    # Download data file to local
    blob_name = myblob.name.replace('data/', '')
    storage_client = create_data_storage_client(blob_name)
    blob_prop = storage_client.get_blob_properties()
    model_name = blob_prop.metadata['model_name'] 
    model_version = blob_prop.metadata['model_version']

    download_file = os.path.join(data_dir,  blob_name)
    decrypted_file = os.path.join(data_dir, blob_name + '.dec.jpg')
    image_list = []

    # Decrypt data
    with open(download_file, "wb") as encypted_data:
        download_stream = storage_client.download_blob()
        encypted_data.write(download_stream.readall())
        client.decrypt(model_name, model_version, download_file, decrypted_file)
    
    # Upload data to custom vision project
    with open(decrypted_file, "rb") as image_contents:
        image_list.append(ImageFileCreateEntry(name=blob_name, contents=image_contents.read()))

    upload_result = trainer.create_images_from_files(project.id, ImageFileCreateBatch(images=image_list))
    if not upload_result.is_batch_successful:
        logging.info(f"Image batch upload failed.")
        for image in upload_result.images:
            logging.info(f"Image status: {image.status}")
    else:
        logging.info(f"Image batch upload succeeded.")

    # Clean up temp files
    os.remove(download_file)
    os.remove(decrypted_file)
```

### 6. Debug the project locally to make sure the retraining data can be fetched from the Azure Percept MM storage blob, decrypted and uploaded to Custom Vision project as "Untagged" images.

### 7. Deploy the project to your Azure subscription and upload the local settings to cloud.
