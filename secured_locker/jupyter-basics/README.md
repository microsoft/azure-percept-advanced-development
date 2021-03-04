# sczpy-basics.ipynb

- **Please note!** Azure Percept currently supports AI model protection as a private preview feature.  
- Portions of this code base are subject to change without notice.

Please consider taking our [Product survey](https://go.microsoft.com/fwlink/?linkid=2156573) to help us improve Azure Percept Model and Data Protection features based on your IoT Edge background and goals.

This Notebook uses Azure Percept MM Python sample code to preform a series of model operations:
> * Encryption (encrypts ```model.txt``` to ```model.txt.enc```)
> * Decryption (decrypts ```model.txt.enc``` to ``` model.decrypted.txt```)
> * Upload (uploads ```model.txt.enc```)
> * Download (downloads to ``` downloaded.txt.enc``` and then decryptes to ``` downloaded.decrypted.txt```)

1.	Before testing, you need to update the environment variables ```AZURE_CLIENT_ID```, ```AZURE_CLIENT_SECRET``` and ```AZURE_TENANT_ID``` to match with your service principal credential. Then, you need to update the ```server_url``` to point to your Azure Percept MM server endpoint. For example:
    ```
    %env AZURE_CLIENT_ID=555d...
    %env AZURE_CLIENT_SECRET=6da3...
    %env AZURE_TENANT_ID=72f9...
    server_url="https://test-mm.westus2.cloudapp.azure.com"
    ```
2.	Run all cells in the Notebook.

3.	Observe a few files get created under the application folder:
    ```bash
    model.txt.enc
    model.decrypted.txt
    downloaded.txt.enc
    downloaded.decrypted.txt
    ```
