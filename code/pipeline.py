import azureml.core
from azureml.core import Workspace, Run, Experiment, Datastore
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget

from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineData, StepSequence


ws = Workspace.from_config()

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



# create AML compute
aml_compute_target = "aml-compute"
try:
    aml_compute = AmlCompute(ws, aml_compute_target)
    print("found existing compute target.")
except:
    print("creating new compute target")
    
    provisioning_config = AmlCompute.provisioning_configuration(vm_size = "STANDARD_D2_V2",
                                                                min_nodes = 1, 
                                                                max_nodes = 4)    
    aml_compute = ComputeTarget.create(ws, aml_compute_target, provisioning_config)
    aml_compute.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
    
print("Azure Machine Learning Compute attached")

# get pointer to default blob store
def_blob_store = Datastore(ws, "workspaceblobstore")
print("Blobstore's name: {}".format(def_blob_store.name))


# Naming the intermediate data as anomaly data and assigning it to a variable
anomaly_data = PipelineData("anomaly_data",datastore=def_blob_store)
print("Anomaly data object created")


anom_detect = PythonScriptStep(name="anomaly_detection",
                               script_name="anom_detect.py",
                               arguments=["--output_data", anomaly_data],
                               outputs=[anomaly_data],
                               compute_target=aml_compute, 
                               source_directory=project_folder,
                               allow_reuse=True)
print("Anomaly Detection Step created.")


automl_train = PythonScriptStep(name="automl_train",
                                script_name="automl_train.py", 
                                arguments=["--input_data", anomaly_data],
                                inputs=[anomaly_data],
                                compute_target=aml_compute, 
                                source_directory=project_folder,
                                allow_reuse=True)
print("AutoML Training Step created.")


steps = [anom_detect, automl_train]
print("Step lists created")

pipeline = Pipeline(workspace=ws, steps=steps)
print ("Pipeline is built")

pipeline1.validate()
print("Pipeline validation complete")

pipeline_run = Experiment(ws, 'PdM_pipeline').submit(pipeline, regenerate_outputs=True)
print("Pipeline is submitted for execution")
