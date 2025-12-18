# Project Scoping

Short-term air temperature forecasts based on historical and recent weather observations, developed using an MLOps Continuous learning approach. 

## Problem and expected results

People are not happy with meteo predictions, so we want to create a better solution that predicts short term temperature evolution with actual and historical meteo data.

## Stakeholders

- The people near the prediction area (users)
- Us: the managers and technical people
- Swiss meteo: the data provider
- The teachers: evaluate our work

## Requirements

Meteo data on a certain location, a prediction model, a UI where users can visualise predictions

## Machine Learning problem

We need a regression model that uses time-series data to make a time-series prediction.

We need to collect data and infer in batches.

To avoid data drift issue, continuously re-train the model with new data. Also track the model performance in production to see how it evolves.

### Input

temperature, pressure, wind, sun, rain in time-series format

### Output

Short term temperature evolution in time-serie format

### Requirements

A cloud to store our data and our model and run the pipeline.
A webpage to see the weather predictions and model performance.

## Development steps

Develop all the project steps with MLOPS in mind.

- EDA: Create a first model on a notebook with downloaded historical data
- In parallel, set up the production pipeline: collect data in batches, infer in batches, display predictions, evaluate performance
- Deploy the model inside the pipeline do it automatically with CI/CD
- Implement continuous learning iterations for the production model:
  - **Iteration 1:** every Monday, retrieve the past week’s data and **retrain** the model on the updated dataset
  - **Iteration 2:** use **Learning without Forgetting (LwF)** with a **replay buffer** to reduce catastrophic forgetting during updates

## Repository structure
Key folders/files (as currently present): :contentReference[oaicite:5]{index=5}

## Repository structure

```text
MachLeData_Miss_Meteo/
├── .dvc/
├── .github/
├── data/
├── kubernetes/
├── model/
├── notebook/
├── src/
├── .dvcignore
├── .env
├── .gitignore
├── README.md
├── dvc.lock
├── dvc.yaml
├── params.yaml
├── requirements-freeze.txt
└── requirements.txt
```

# gcloud

Need to setup and source environment variables
```bash
$ export GCP_K8S_CLUSTER_ZONE="..."
$ export GCP_K8S_CLUSTER_NAME="..."
```

## Setup gcloud provider from our local machine

Create the Kubernetes cluster (takes several minutes)
```bash
$ gcloud container clusters create \
    --machine-type=e2-standard-2 \
    --num-nodes=2 \
    --zone=$GCP_K8S_CLUSTER_ZONE \
    $GCP_K8S_CLUSTER_NAME
```

Merge credential with local kubectl config
```bash
$ gcloud container clusters get-credentials $GCP_K8S_CLUSTER_NAME --zone $GCP_K8S_CLUSTER_ZONE
```

Apply our configuration to the cluster
```bash
$ kubectl apply -f kubernetes/deployment.yaml -f kubernetes/service.yaml
```

Access the model
```bash
$ kubectl describe services air-temperature-regressor
```

# Disable gcloud from local machine

Delete the cluster
```bash
$ gcloud container clusters delete --zone $GCP_K8S_CLUSTER_ZONE $GCP_K8S_CLUSTER_NAME
```

## Quickstart

1) Create & activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
````
2) Install dependencies:
```` bash
pip install -r requirements.txt
pip install -r requirements-freeze.txt
````
3) (Optional) Run the pipeline
```` bash
dvc repro
````