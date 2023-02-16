# Build an ML Pipeline for Short-Term Rental Prices in NYC
A Udacity Project on MLOps

Done by: kk

Date: Feb 2023

---

## W&B project link
https://wandb.ai/klangkk/nyc_airbnb

## GitHub project link
https://github.com/KlangKK/build-ml-pipeline-for-short-term-rental-prices

---

## Table of contents

- [Project Brief](#project-brief)
- [Starter Kit Boilerplate code](#starter-kit-boilerplate-code)
- [README](#readme)
- [Tools used](#tools-used)
- [Environment Setup (Conda)](#environment-setup-conda)
- [Parameters Configuration (Hydra)](#parameters-configuration-hydra)
- [Cookie Cutter](#cookie-cutter)
- [MLflow running the entire pipeline or just a selection of steps](#MLflow-running-the-entire-pipeline-or-just-a-selection-of-steps)
- [Weights and Biases API key](#Weights-and-Biases-API-key)
- [Overview of Design](#Overview-of-Design)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Data cleaning](#data-cleaning)
- [Data checking](#data-checking)
- [Data splitting](#data-splitting)
- [Train Pipeline](#train-pipeline)
- [Select the best model](#select-the-best-model)
- [Test](#test)
- [Release the pipeline](#release-the-pipeline)
- [Train the model on a new data sample](#train-the-model-on-a-new-data-sample)
- [License](#License)

---

## Project Brief

The project is for a property management company renting rooms and properties for short periods of time on various rental platforms. 

The aim is to estimate the typical price for a given property based 
on the price of similar properties. 

New data is received in bulk every week. 

The model needs to be retrained with the same cadence, necessitating an end-to-end pipeline that can be reused.

This project is to build such a pipeline.

---

## Starter Kit Boilerplate code
The original project instructions and starter kit can be retrieved from [here](https://github.com/udacity/build-ml-pipeline-for-short-term-rental-prices.git) 

---

## README
This README is crafted to document important steps of the project.

---

## Tools used
- Conda for environment management
- Hydra for configuration management
- Cookie Cutter for creating boilerplate code stubs for new pipeline components
- MLflow for reproduction and management of pipeline processes
- Weights and Biases for artifact and execution tracking
- Pandas for data analysis
- Scikit-Learn for data modeling

---

## Environment Setup (Conda)

1. Clone the git repo
```bash
git clone https://github.com/KlangKK/build-ml-pipeline-for-short-term-rental-prices

```
2. Go into repo
```bash
cd build-ml-pipeline-for-short-term-rental-prices
```
3. Make sure to have conda installed and ready, then create a new environment using the ``environment.yml``
file provided in the root of the repository and activate it:

```bash
> conda env create -f environment.yml
> conda activate nyc_airbnb_dev
```

---

## Parameters Configuration (Hydra)
NOTE: Any parameter should NOT be hardcoded when writing the pipeline. All the parameters should be accessed from the configuration file.

As usual, the parameters controlling the pipeline are defined in the ``config.yaml`` file defined in the root of the starter kit. 

Hydra is used to manage this configuration file. 

This file is only read by the ``main.py`` script (i.e., the pipeline) and its content is available with the ``go`` function in ``main.py`` as the ``config`` dictionary. 

For example, the name of the project is contained in the ``project_name`` key under the ``main`` section in the configuration file. It can be accessed from the ``go`` function as ``config["main"]["project_name"]``.

---

## Cookie Cutter
A cookie cutter template has been provided. It can be used to create boilerplate code stubs for new pipeline components.

Run the cookiecutter and enter the required information, and a new component will be created including the `conda.yml` file, the `MLproject` file as well as the script.

These can then be modified as needed, instead of starting from scratch.

For example:

```bash
> cookiecutter cookie-mlflow-step -o src

step_name [step_name]: basic_cleaning
script_name [run.py]: run.py
job_type [my_step]: basic_cleaning
short_description [My step]: This steps cleans the data
long_description [An example of a step using MLflow and Weights & Biases]: Performs basic cleaning on the data and save the results in Weights & Biases
parameters [parameter1,parameter2]: parameter1,parameter2,parameter3
```

This will create a step called ``basic_cleaning`` under the directory ``src`` with the following structure:

```bash
> ls src/basic_cleaning/
conda.yml  MLproject  run.py
```

The files created can be modified:
- the script (``run.py``), 
- the conda environment (``conda.yml``) and 
- the project definition (``MLproject``).

The script ``run.py`` will receive the input parameters ``parameter1``, ``parameter2``,``parameter3`` and it will be called like:

```bash
> mlflow run src/step_name -P parameter1=1 -P parameter2=2 -P parameter3="test"
```

---

## MLflow running the entire pipeline or just a selection of steps
In order to run the pipeline, execute the following at the base folder:

```bash
>  mlflow run .
```
This will run the entire pipeline.

When developing it is useful to be able to run one step at the time. 

The `main.py` is written so that the steps are defined at the top of the file, in the ``_steps`` list, and can be selected by using the `steps` parameter on the command line:

For example, to run only the ``download`` step: 

```bash
> mlflow run . -P steps=download
```

For example, to run the ``download`` and the ``basic_cleaning`` steps:
```bash
> mlflow run . -P steps=download,basic_cleaning
```

Any other parameter in the configuration file can be overriden using the Hydra syntax, by providing it as a ``hydra_options`` parameter. 

For example, to set the parameter modeling -> random_forest -> n_estimators to 10 and etl->min_price to 50:

```bash
> mlflow run . \
  -P steps=download,basic_cleaning \
  -P hydra_options="modeling.random_forest.n_estimators=10 etl.min_price=50"
```
---

## Weights and Biases API key
Log in to Weights & Biases and get API key from W&B by going to 
[https://wandb.ai/authorize](https://wandb.ai/authorize) and click on the + icon (copy to clipboard), then paste the API key into this command:

```bash
> wandb login [your API key]
```

A message similar to should be displayed:
```
wandb: Appending key for api.wandb.ai to your netrc file: /home/[your username]/.netrc
```

---

## Overview of Design
![[images/flowchart.png]](images/flowchart.png)

---

## Exploratory Data Analysis (EDA)
The scope of this section is to get an idea of how the process of an EDA works in the context of pipelines, during the data exploration phase. In a real scenario, a lot more time will be spent in this phase, but in this project, the bare minimum will be done.

## Data cleaning
The data processing done as part of the EDA will be transferred to a new ``basic_cleaning`` step that starts from the ``sample.csv`` artifact and create a new artifact ``clean_sample.csv`` with the cleaned data.

## Data checking
After the cleaning, it is a good practice to put some tests that verify that the data does not contain surprises. One of the tests will compare the distribution of the current data sample with a reference, to ensure that there is no unexpected change. Therefore, a "reference dataset" is defined first.The latest artifact ``clean_sample.csv`` on W&B is tagged as the reference dataset. 

## Data splitting
Use the provided component called ``train_val_test_split`` to extract and segregate the test set and training set.

## Train Pipeline
Use the provided component called ``train_random_forest`` to train and save the interference pipeline as a mlflow.sklearn model in the directory "random_forest_dir" and upload to W&B. Feature importance is plotted and uploaded to W&B. Metrics summary is calculated and uploaded to W&B too.

## Select the best model
Go to W&B and select the best performing model. Consider the Mean Absolute Error as our target metric and choose the model with the lowest MAE. 

## Test
Use the provided step ``test_regression_model`` to test your production model against the test set. 

## Release the pipeline
A first production release ``1.0.0`` has been made on GitHub. 
Another production release ``1.0.1`` has also been made on GitHub. 

## Train the model on a new data sample
The release can be run using ``mlflow`` without any other pre-requisite. Train the model on a new sample of data that the company received (``sample2.csv``)

---

## License

[License](LICENSE.txt)

