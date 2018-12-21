import azureml.core
from azureml.core import Workspace, Run, Experiment, Datastore
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
from azureml.core.runconfig import CondaDependencies, RunConfiguration
from azureml.core.runconfig import DEFAULT_CPU_IMAGE

from azureml.telemetry import set_diagnostics_collection

from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineData, StepSequence

import pandas as pd

import json

ws = Workspace.from_config()
print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep = '\n')

experiment_name =  'pred-maint-automl' # choose a name for experiment
project_folder = '.' # project folder

experiment=Experiment(ws, experiment_name)
print("Location:", ws.location)
output = {}
output['SDK version'] = azureml.core.VERSION
output['Subscription ID'] = ws.subscription_id
output['Workspace'] = ws.name
output['Resource Group'] = ws.resource_group
output['Location'] = ws.location
output['Project Directory'] = project_folder
output['Experiment Name'] = experiment.name
pd.set_option('display.max_colwidth', -1)
pd.DataFrame(data=output, index=['']).T

set_diagnostics_collection(send_diagnostics=True)

print("SDK Version:", azureml.core.VERSION)

cd = CondaDependencies.create(pip_packages=["pandas", "azureml-train-automl", "pyculiarity", "pytictoc"])

# Runconfig
amlcompute_run_config = RunConfiguration(conda_dependencies=cd)
amlcompute_run_config.environment.docker.enabled = False
amlcompute_run_config.environment.docker.gpu_support = False
amlcompute_run_config.environment.docker.base_image = DEFAULT_CPU_IMAGE
amlcompute_run_config.environment.spark.precache_packages = False


# create AML compute
aml_compute_target = "aml-compute"
try:
    aml_compute = AmlCompute(ws, aml_compute_target)
    print("found existing compute target.")
except:
    print("creating new compute target")
    
    provisioning_config = AmlCompute.provisioning_configuration(vm_size = "STANDARD_D2_V2",
                                                                idle_seconds_before_scaledown=1800,
                                                                min_nodes = 0, 
                                                                max_nodes = 4)    
    aml_compute = ComputeTarget.create(ws, aml_compute_target, provisioning_config)
    aml_compute.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
    
print("Azure Machine Learning Compute attached")

# get pointer to default blob store
def_blob_store = Datastore(ws, "workspaceblobstore")
print("Blobstore's name: {}".format(def_blob_store.name))


# Naming the intermediate data as anomaly data and assigning it to a variable
anomaly_data = PipelineData("anomaly_data", datastore=def_blob_store)
print("Anomaly data object created")


anom_detect = PythonScriptStep(name="anomaly_detection",
                               # script_name="anom_detect.py",
                               script_name="code/anom_detect.py",
                               arguments=["--output_directory", anomaly_data],
                               outputs=[anomaly_data],
                               compute_target=aml_compute, 
                               source_directory=project_folder,
                               allow_reuse=True,
                               runconfig=amlcompute_run_config)
print("Anomaly Detection Step created.")


automl_train = PythonScriptStep(name="automl_train",
                                # script_name="automl_train.py", 
                                script_name="code/automl_train.py", 
                                arguments=["--input_directory", anomaly_data],
                                inputs=[anomaly_data],
                                compute_target=aml_compute, 
                                source_directory=project_folder,
                                allow_reuse=True,
                               runconfig=amlcompute_run_config)
print("AutoML Training Step created.")


steps = [anom_detect, automl_train]
print("Step lists created")

pipeline = Pipeline(workspace=ws, steps=steps)
print ("Pipeline is built")

pipeline.validate()
print("Pipeline validation complete")

pipeline_run = Experiment(ws, 'PdM_pipeline').submit(pipeline, regenerate_outputs=True)
print("Pipeline is submitted for execution")

pipeline_run.wait_for_completion()
print("Pipeline execution completed")


# Writing the run id to /aml_config/run_id.json
run_id = {}
run_id['run_id'] = pipeline_run.id
run_id['experiment_name'] = pipeline_run.experiment.name
with open('code/aml_config/run_id.json', 'w') as outfile:
  json.dump(run_id,outfile)
