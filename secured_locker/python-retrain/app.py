#Copyright (c) Microsoft Corporation. All rights reserved.
#Licensed under the MIT License.

from opentelemetry import trace
from opentelemetry import metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleExportSpanProcessor,
    BatchExportSpanProcessor
)
import time
import random 
from opentelemetry.exporter import jaeger
import sczpy
import os

os.environ["AZURE_CLIENT_ID"] = ""
os.environ["AZURE_CLIENT_SECRET"] = ""
os.environ["AZURE_TENANT_ID"] = ""
server_url = ""


def main():
    model_name = "my-model"
    model_version = "v1"
    client = sczpy.SCZClient(server_url)

    trace.set_tracer_provider(TracerProvider())

    jaeger_exporter = jaeger.JaegerSpanExporter(
        service_name="my-device",
        agent_host_name="localhost",
        agent_port=6831,
    )

    trace.get_tracer_provider().add_span_processor(
        BatchExportSpanProcessor(jaeger_exporter)
    )
    trace.get_tracer_provider().add_span_processor(
        SimpleExportSpanProcessor(sczpy.SCZSpanExporter(client))
    )
    tracer = trace.get_tracer(__name__)

    while True:
        with tracer.start_as_current_span("inference") as inference:
            inference.set_attribute('device', 'my-device')
            inference.add_event('inference', {
                "confidence": random.randint(80, 101),
                "model_name": model_name,
                "model_version": model_version,
                "file_ref": "dummy_data.txt"
            })
        time.sleep(3)

if __name__ == '__main__':
    main()
