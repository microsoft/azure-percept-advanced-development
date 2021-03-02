# AI Model Protection in transit

- **Please note!** Azure Percept currently supports AI model protection as a private preview feature.  
- Portions of this code base are subject to change without notice.

Please consider taking our [Product survey](https://go.microsoft.com/fwlink/?linkid=2156573) to help us improve Azure Percept Model and Data Protection features based on your IoT Edge background and goals.

Azure Percept MM transfers AI models and  keys over the TLS (Transport Layer Security) channel after successfully authenticating the client.

When you deploy the Azure Percept MM service using the deployment script, the script generates a self-signed certificate to be associated with the Azure Application Gateway instance that serves as the front door of the Azure Percept MM service. Azure Application Gateway performs TLS termination and talks to the Azure Percept MM service over a private network in plain text (HTTP).

It's also possible to configure end-to-end SSL, as introduced [here](https://docs.microsoft.com/en-us/azure/application-gateway/end-to-end-ssl-portal).

> **NOTE:** In future versions, the Azure Percept MM will use a device-specific certificate to provide additional protection; when the device requests encrypted models or secret keys, it will send its corresponding public key to Azure Percept MM, and Azure Percept MM will use the public key to encrypt the payload before it sends it back to the client over TLS.
