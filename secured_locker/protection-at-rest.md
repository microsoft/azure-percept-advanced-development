# AI model protection at rest

- **Please note!** Azure Percept currently supports AI model protection as a private preview feature.  
- Portions of this code base are subject to change without notice.

Please consider taking our [Product survey](https://go.microsoft.com/fwlink/?linkid=2156573) to help us improve Azure Percept Model and Data Protection features based on your IoT Edge background and goals.

Azure Percept MM can encrypt AI models before saving them to the device. Each AI model version has its unique key, which is stored in Azure Key Vault. For a client to retrieve the secured key, it needs to authenticate with the Azure Percept MM service using a Service Principal that has been granted reading access to the Key Vault.

The encrypted model files can be embedded into a Docker container or dynamically retrieved through the Azure Percept MM sample code.

> **NOTE:** At the time of writing, Azure Percept MM doesn’t support customer-supplied keys.
