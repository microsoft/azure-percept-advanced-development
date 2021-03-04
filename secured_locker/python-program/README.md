# Using Azure Percept MM Python code in your Python program

- **Please note!** Azure Percept currently supports AI model protection as a private preview feature.  
- Portions of this code base are subject to change without notice.

Please consider taking our [Product survey](https://go.microsoft.com/fwlink/?linkid=2156573) to help us improve Azure Percept Model and Data Protection features based on your IoT Edge background and goals.

This sample program uses Azure Percept MM Python code to preform a series of model operations:
> * Encryption (encrypts ```model.txt``` to ```model.txt.enc```)
> * Decryption (decrypts ```model.txt.enc``` to ```model.decrypted.txt```)
> * Upload (uploads ```model.txt.enc```)
> * Download (downloads to ``` downloaded.txt.enc``` and then decryptes to ```downloaded.decrypted.txt```)

1.	Before testing, you need to update app.py with the environment variables ```AZURE_CLIENT_ID```, ```AZURE_CLIENT_SECRET``` and ```AZURE_TENANT_ID``` to match with your service principal credential. Then, you need to update the ```server_url``` to point to your Azure-Percept-SMM server endpoint. For example:
    ```python
    os.environ["AZURE_CLIENT_ID"] = "555d..."
    os.environ["AZURE_CLIENT_SECRET"] = "6da3..."
    os.environ["AZURE_TENANT_ID"] = "72f9..."
    server_url = "https://test-mm.westus2.cloudapp.azure.com"
    ```
2.	Run the program.

    ```bash
    python app.py
    ```
3.	Observe a few files get created under the application folder:
    ```bash
    model.txt.enc
    model.decrypted.txt
    downloaded.txt.enc
    downloaded.decrypted.txt
    ```
> **NOTE:** You'll see a few certificate warnings. These are caused by the self-signed certificate used by Azure-Percept-SMM server deployment. You can ignore these warnings, or replace the certificate with a trusted certificate.

```bash
InsecureRequestWarning: Unverified HTTPS request is being made to host 'test-mm.westus2.cloudapp.azure.com'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
```
