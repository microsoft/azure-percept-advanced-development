# Control script for a training run
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core import Dataset

if __name__ == "__main__":
    ws = Workspace.from_config()
    dataset = Dataset.get_by_name(workspace=ws, name='bananas_dataset')

    experiment = Experiment(workspace=ws, name='bananas-experiment')

    config = ScriptRunConfig(
        source_directory='.',
        script='train.py',
        compute_target='gpu1',
        arguments=[
            '--data-path', dataset.as_named_input('input').as_mount(),
            '--output-path', './outputs',
            '--epochs', 3,
            '--batch-size', 2,
            '--learning-rate', 0.001,
            '--scale', 0.5,
            '--to-bgr'
        ],
    )
    # set up the training environment
    env = Environment.from_conda_specification(
        name='train-env',
        file_path='./train-env.yml'
    )
    # use a customized docker image 
    env.docker.base_image = None
    env.docker.base_dockerfile = "./Dockerfile" 
    config.run_config.environment = env

    run = experiment.submit(config)
    aml_url = run.get_portal_url()
    print("")
    print("Submitted to compute cluster. Click link below:")
    print(aml_url)

    #run.wait_for_completion(show_output=True)
    
