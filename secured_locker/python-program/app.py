#Copyright (c) Microsoft Corporation. All rights reserved.
#Licensed under the MIT License.

import sczpy
import os

os.environ["AZURE_CLIENT_ID"] = ""
os.environ["AZURE_CLIENT_SECRET"] = ""
os.environ["AZURE_TENANT_ID"] = ""
server_url = ""

model_name = "my-model"
model_version = "v1"
client = sczpy.SCZClient(server_url)
client.register_model(model_name, model_version)
client.encrypt(model_name, model_version, "model.txt", "model.txt.enc")
client.decrypt(model_name, model_version, "model.txt.enc", "model.decrypted.txt")
client.upload_model(model_name, model_version, "model.txt.enc")
client.download_model(model_name, model_version, "downloaded.txt.enc")
client.decrypt(model_name, model_version, "downloaded.txt.enc", "downloaded.decrypted.txt")
