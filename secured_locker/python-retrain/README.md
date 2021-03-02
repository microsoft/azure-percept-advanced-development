# Re-training 

- **Please note!** Azure Percept currently supports AI model protection as a private preview feature.  
- Portions of this code base are subject to change without notice.

Please consider taking our [Product survey](https://go.microsoft.com/fwlink/?linkid=2156573) to help us improve Azure Percept Model and Data Protection features based on your IoT Edge background and goals.

Azure Percept MM helps to you to trigger automatic re-training pipelines. Instead of creating its own telemetry collection system, Azure Percept MM is integrated with [OpenTelemetry]( https://opentelemetry.io/) that supports a big ecosystem of distributed tracing and metrics collection/analysis systems, including [Prometheus]( https://prometheus.io/), [Jaeger]( https://www.jaegertracing.io/), [Zipkin]( https://zipkin.io/), and many others. It also has SDKs in popular programming languages such as C#, Go and Python.

Azure Percept MM SDK provides two OpenTelemetry exporters - ```SCZSpanExporter``` and ```SCZMeticsExporter``` that can be used to monitor distributed tracing span or metrics, respectively. You can associate policies to these exporters to decide when additional training data should be collected, encrypted, and uploaded to Azure Percept MM training data repository. For example, you may set up a trigger to upload a current image if the confidence score is lower than a threshold (which means the model is less certain about the content).

Once the data is uploaded to data repository, you can use services such as [Azure Logic Apps]( https://azure.microsoft.com/en-us/services/logic-apps/) to pick up the data and trigger the retraining pipeline, as shown in [this sample]( https://docs.microsoft.com/en-us/azure/machine-learning/how-to-trigger-published-pipeline).

> **NOTE:** At the time of writing, the triggering policy is fixed to fire if:
> * a traced event named ```inference```
> * The event has a property named ```confidence``` that is lower than 90 percent
> * The event has associated ```model_name```, ```model_version``` attributes
> * The event has an associcated ```file_ref``` attribute. This is the file to be uploaded
> 
> A flexible policy engine is in the plan for future versions.

1.	Before testing, you need to update the environment variables ```AZURE_CLIENT_ID```, ```AZURE_CLIENT_SECRET``` and ```AZURE_TENANT_ID``` to match with your service principal credential. Then, you need to update the ```server_url``` to point to your Azure-Percept-SMM server endpoint. For example:
    ```python
    os.environ["AZURE_CLIENT_ID"] = "555d..."
    os.environ["AZURE_CLIENT_SECRET"] = "6da3..."
    os.environ["AZURE_TENANT_ID"] = "72f9..."
    server_url = "https://test-mm.westus2.cloudapp.azure.com"
    ```
2.	Run the program.

    ```bash
    pip install -r requirements.txt
    python app.py
    ```
3. [Optional] Launch Jaeger container (see Appendix).	
4. Observe new data files are uploaded to your Azure-Percept-SMM storage account, under the ```data``` container.

    > **NOTE:** ```data``` container will be replaced by model-specific containers in future versions.
5. [Optional] Observe Jaeger dashboard at ```http://localhost:16686```.

## Appendix - use Jaegar

To launch an all-in-one Jaegar container that includes the dashboard, use the following command: 
```bash
docker run -p 16686:16686 -p 6831:6831/udp jaegertracing/all-in-one
```
Once the container is launched, you can browse its portal at ```http://localhost:16686```.

## Appendix - use Prometheus

To launch a Prometheus container, use the following command:

```
docker run --net=host -v /tmp/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus --log.level=debug --config.file=./prometheus.yml
```

> **NOTE:** This will not work with WSL on Windows 10 because of networking limitations.
