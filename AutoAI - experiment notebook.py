#!/usr/bin/env python
# coding: utf-8

# ## <span style="color:darkblue">IBM Watson AutoAI - Generated SDK Notebook v1.14.4</span>
# 
# 
# This notebook contains the steps and code to demonstrate support of AutoAI experiments in Watson Machine Learning service. It introduces Python SDK commands for data retrieval, training experiments, persisting pipelines, testing pipelines, refining pipelines, and scoring the resulting model.
# 
# **Note:** Notebook code generated using AutoAI will execute successfully. If code is modified or reordered, there is no guarantee it will successfully execute. For details, see: <a href="https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/autoai-notebook.html">Saving and Auto AI experiment as a notebook</a>
# 

# Some familiarity with Python is helpful. This notebook uses Python 3.7 and `ibm_watson_machine_learning` SDK.
# 
# 
# ## Notebook goals
# 
# The learning goals of this notebook are:
# -  Defining an AutoAI experiment
# -  Training AutoAI models 
# -  Comparing trained models
# -  Deploying the model as a web service
# -  Online deployment and score the trained model
# -  Scoring the model to generate predictions.
# 
# 
# 
# ## Contents
# 
# This notebook contains the following parts:
# 
# **[Setup](#setup)**<br>
# &nbsp;&nbsp;[Package installation](#install)<br>
# &nbsp;&nbsp;[Watson Machine Learning connection](#connection)<br>
# **[Experiment configuration](#configuration)**<br>
# &nbsp;&nbsp;[Experiment metadata](#metadata)<br>
# **[Working with completed AutoAI experiment](#work)**<br>
# &nbsp;&nbsp;[Get fitted AutoAI optimizer](#get)<br>
# &nbsp;&nbsp;[Pipelines comparison](#comparison)<br>
# &nbsp;&nbsp;[Get pipeline as scikit-learn pipeline model](#get_pipeline)<br>
# &nbsp;&nbsp;[Inspect pipeline](#inspect_pipeline)<br>
# &nbsp;&nbsp;&nbsp;&nbsp;[Visualize pipeline model](#visualize)<br>
# &nbsp;&nbsp;&nbsp;&nbsp;[Preview pipeline model as python code](#preview)<br>
# **[Deploy and Score](#scoring)**<br>
# &nbsp;&nbsp;[Working with spaces](#working_spaces)<br>
# **[Running AutoAI experiment with Python SDK](#run)**<br>
# **[Clean up](#cleanup)**<br>
# **[Next steps](#next_steps)**<br>
# **[Copyrights](#copyrights)**

# <a id="setup"></a>
# # Setup

# <a id="install"></a>
# ## Package installation
# Before you use the sample code in this notebook, install the following packages:
#  - ibm_watson_machine_learning,
#  - autoai-libs.
# 

# In[1]:


get_ipython().system('pip install ibm-watson-machine-learning | tail -n 1')
get_ipython().system('pip install -U autoai-libs | tail -n 1')


# <a id="configuration"></a>
# # Experiment configuration

# <a id="metadata"></a>
# ## Experiment metadata
# This cell defines the metadata for the experiment, including: training_data_reference, training_result_reference, experiment_metadata.

# In[2]:


# @hidden_cell
from ibm_watson_machine_learning.helpers import DataConnection, S3Connection, S3Location

training_data_reference = [DataConnection(
    connection=S3Connection(
        api_key='f4kPKHTWcZ-s8bANuAxYy2hmZ5rabyc5Y90LNoJM1hRa',
        auth_endpoint='https://iam.bluemix.net/oidc/token/',
        endpoint_url='https://s3.eu-geo.objectstorage.softlayer.net'
    ),
        location=S3Location(
        bucket='creditcardfraudpredictionusingibm-donotdelete-pr-qdxryymophsglr',
        path='fraud_dataset.csv'
    )),
]
training_result_reference = DataConnection(
    connection=S3Connection(
        api_key='f4kPKHTWcZ-s8bANuAxYy2hmZ5rabyc5Y90LNoJM1hRa',
        auth_endpoint='https://iam.bluemix.net/oidc/token/',
        endpoint_url='https://s3.eu-geo.objectstorage.softlayer.net'
    ),
    location=S3Location(
        bucket='creditcardfraudpredictionusingibm-donotdelete-pr-qdxryymophsglr',
        path='auto_ml/7fb173b8-8364-4122-9c74-212c395e91b9/wml_data/0bea83c5-c9d7-4953-adbc-7d9b378a661a/data/automl',
        model_location='auto_ml/7fb173b8-8364-4122-9c74-212c395e91b9/wml_data/0bea83c5-c9d7-4953-adbc-7d9b378a661a/data/automl/pre_hpo_d_output/Pipeline1/model.pickle',
        training_status='auto_ml/7fb173b8-8364-4122-9c74-212c395e91b9/wml_data/0bea83c5-c9d7-4953-adbc-7d9b378a661a/training-status.json'
    ))


# In[3]:


experiment_metadata = dict(
   prediction_type='classification',
   prediction_column='Fraud_Risk',
   test_size=0.1,
   scoring='accuracy',
   project_id='3f1878b3-49ad-401a-b3a0-3fa8b5375639',
   deployment_url='https://eu-gb.ml.cloud.ibm.com',
   csv_separator=',',
   random_state=33,
   max_number_of_estimators=2,
   daub_include_only_estimators=None,
   training_data_reference=training_data_reference,
   training_result_reference=training_result_reference,
   positive_label=1
)


# <a id="connection"></a>
# ## Watson Machine Learning connection
# 
# This cell defines the credentials required to work with the Watson Machine Learning service.
# 
# **Action** Please provide IBM Cloud apikey following [docs](https://cloud.ibm.com/docs/account?topic=account-userapikey).

# In[4]:


api_key = '918Qbg9vw65-heOPvtePjYEDaJ6JmdqprdUZIpp_5-JY'


# In[6]:


wml_credentials = {
    "apikey": api_key,
    "url": experiment_metadata['deployment_url']
}


# <a id="work"></a>
# 
# 
# # Working with completed AutoAI experiment
# 
# This cell imports the pipelines generated for the experiment so they can be compared to find the optimal pipeline to save as a model.

# <a id="get"></a>
# 
# 
# ## Get fitted AutoAI optimizer

# In[7]:


from ibm_watson_machine_learning.experiment import AutoAI

pipeline_optimizer = AutoAI(wml_credentials, project_id=experiment_metadata['project_id']).runs.get_optimizer(metadata=experiment_metadata)


# Use `get_params()`- to retrieve configuration parameters.

# In[8]:


pipeline_optimizer.get_params()


# <a id="comparison"></a>
# ## Pipelines comparison
# 
# Use the `summary()` method to list trained pipelines and evaluation metrics information in
# the form of a Pandas DataFrame. You can use the DataFrame to compare all discovered pipelines and select the one you like for further testing.

# In[9]:


summary = pipeline_optimizer.summary()
best_pipeline_name = list(summary.index)[0]
summary


# You can visualize the scoring metric calculated on a holdout data set.

# In[10]:


summary[f"holdout_{experiment_metadata['scoring'].replace('neg_','')}"].plot();


# <a id="get_pipeline"></a>
# ### Get pipeline as scikit-learn pipeline model
# 
# After you compare the pipelines, download and save a scikit-learn pipeline model object from the
# AutoAI training job.
# 
# **Tip:** If you want to get a specific pipeline you need to pass the pipeline name in:
# ```
# pipeline_optimizer.get_pipeline(pipeline_name=pipeline_name)
# ```

# In[11]:


pipeline_model = pipeline_optimizer.get_pipeline()


# Next, check features importance for selected pipeline.

# In[12]:


pipeline_optimizer.get_pipeline_details()['features_importance']


# **Tip:** If you want to check all model evaluation metrics-details, use:
# ```
# pipeline_optimizer.get_pipeline_details()
# ```

# <a id="inspect_pipeline"></a>
# ## Inspect pipeline

# <a id="visualize"></a>
# ### Visualize pipeline model
# 
# Preview pipeline model stages as a graph. Each node's name links to a detailed description of the stage.
# 

# In[13]:


pipeline_model.visualize()


# <a id="preview"></a>
# ### Preview pipeline model as python code
# In the next cell, you can preview the saved pipeline model as a python code.  
# You will be able to review the exact steps used to create the model.
# 
# **Note:** If you want to get sklearn representation add following parameter to `pretty_print` call: `astype='sklearn'`.

# In[14]:


pipeline_model.pretty_print(combinators=False, ipython_display=True)


# <a id="scoring"></a>
# ## Deploy and Score
# 
# In this section you will learn how to deploy and score the model as a web service.

# <a id="working_spaces"></a>
# ### Working with spaces
# 
# In this section you will specify a deployment space for organizing the assets for deploying and scoring the model. If you do not have an existing space, you can use [Deployment Spaces Dashboard](https://dataplatform.cloud.ibm.com/ml-runtime/spaces?context=cpdaas) to create a new space, following these steps:
# 
# - Click **New Deployment Space**.
# - Create an empty space.
# - Select Cloud Object Storage.
# - Select Watson Machine Learning instance and press **Create**.
# - Copy `space_id` and paste it below.
# 
# **Tip**: You can also use the SDK to prepare the space for your work. Learn more [here](https://github.com/IBM/watson-machine-learning-samples/blob/master/notebooks/python_sdk/instance-management/Space%20management.ipynb).
# 
# **Action**: assign or update space ID below

# ### Deployment creation

# In[15]:


target_space_id = "06a92013-4b5b-44f7-8d77-0d3a5d6cbb40"

from ibm_watson_machine_learning.deployment import WebService
service = WebService(source_wml_credentials=wml_credentials,
                     target_wml_credentials=wml_credentials,
                     source_project_id=experiment_metadata['project_id'],
                     target_space_id=target_space_id)
service.create(
model=best_pipeline_name,
metadata=experiment_metadata,
deployment_name='Best_pipeline_webservice'
)


# Use the `print` method for the deployment object to show basic information about the service: 

# In[16]:


print(service)


# To show all available information about the deployment use the `.get_params()` method:

# In[17]:


service.get_params()


# ### Scoring of webservice
# You can make scoring request by calling `score()` on the deployed pipeline.

# If you want to work with the web service in an external Python application,follow these steps to retrieve the service object:
# 
#  - Initialize the service by `service = WebService(wml_credentials)`
#  - Get deployment_id by `service.list()` method
#  - Get webservice object by `service.get('deployment_id')` method
# 
# After that you can call `service.score()` method.

# ### Deleting deployment
# <a id="cleanup"></a>
# You can delete the existing deployment by calling the `service.delete()` command.
# To list the existing web services, use `service.list()`.

# <a id="run"></a>
# 
# ## Running AutoAI experiment with Python SDK

# Rerun experiment with Python SDK.

#  - Go to your COS dashboard.
#  - In Service credentials tab, click New Credential.
#  - Add the inline configuration parameter: `{“HMAC”:true}`, click Add.
# This configuration parameter adds the following section to the instance credentials, (for use later in this notebook):
# ```
# cos_hmac_keys”: {
#       “access_key_id”: “***“,
#       “secret_access_key”: “***”
#  }
#  ```

# If you want to run AutoAI experiment using python API change following cells to `code` cells.
# 
# **Action** Please provide cos credentials.
from ibm_watson_machine_learning.experiment import AutoAI

experiment = AutoAI(wml_credentials, project_id=experiment_metadata['project_id'])# @hidden_cell
cos_hmac_keys = {
    "access_key_id": "PLACE YOUR ACCESS KEY ID",
    "secret_access_key": "PLACE YOUR SECRET ACCESS KEY"
  }# @hidden_cell
cos_api_key = "PLACE YOUR API KEY"OPTIMIZER_NAME = "custom_name"
# The experiment settings were generated basing on parameters set on UI.
from ibm_watson_machine_learning.helpers import DataConnection, S3Connection, S3Location

training_data_reference=pipeline_optimizer.get_data_connections()

training_result_reference = [DataConnection(
    connection=S3Connection(
        api_key=cos_api_key,
        auth_endpoint='https://iam.bluemix.net/oidc/token/',
        endpoint_url='https://s3.eu-geo.objectstorage.softlayer.net',
        access_key_id = cos_hmac_keys['access_key_id'],
        secret_access_key = cos_hmac_keys['secret_access_key']
    ),
        location=S3Location(
        bucket='creditcardfraudpredictionusingibm-donotdelete-pr-qdxryymophsglr',
        path='fraud_dataset.csv'
    )),
]

# In[18]:


pipeline_optimizer = experiment.optimizer(
    name=OPTIMIZER_NAME,
    prediction_type=experiment_metadata['prediction_type'],
    prediction_column=experiment_metadata['prediction_column'],
    scoring=experiment_metadata['scoring'],
    daub_include_only_estimators=experiment_metadata['daub_include_only_estimators'],
    test_size=experiment_metadata['test_size'],
    csv_separator=experiment_metadata['csv_separator'],
    positive_label=experiment_metadata['positive_label'])

pipeline_optimizer.fit(training_data_reference=training_data_reference,
                       training_results_reference=training_result_reference[0],
                       background_mode=False)
# 
# <a id="next_steps"></a>
# # Next steps
# 
# #### [Online Documentation](https://www.ibm.com/cloud/watson-studio/autoai)

# <a id="copyrights"></a>
# ### Copyrights
# 
# Licensed Materials - Copyright © 2021 IBM. This notebook and its source code are released under the terms of the ILAN License.
# Use, duplication disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
# 
# **Note:** The auto-generated notebooks are subject to the International License Agreement for Non-Warranted Programs  
# (or equivalent) and License Information document for Watson Studio Auto-generated Notebook (License Terms),  
# such agreements located in the link below. Specifically, the Source Components and Sample Materials clause  
# included in the License Information document for Watson Studio Auto-generated Notebook applies to the auto-generated notebooks.  
# 
# By downloading, copying, accessing, or otherwise using the materials, you agree to the <a href="http://www14.software.ibm.com/cgi-bin/weblap/lap.pl?li_formnum=L-AMCU-BHU2B7&title=IBM%20Watson%20Studio%20Auto-generated%20Notebook%20V2.1">License Terms</a>  
# 
# ___
