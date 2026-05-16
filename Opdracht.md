# Final Project

## Renewable Energy Forecasting

### Deadline

The deadline for this project is **May 26th 2026** (28/05/2026).

The deadline for the peer evaluation is **June 2nd 2026** (02/06/2026).

### Learning Goals

With this project, you will:

- Deploy an ML-model.
- Set up virtualisation by containerizing an AI-application.
- Use a versioning system for data, models and code.
- Apply orchestration for model development.
- Make sure your ML experiments are reproducible, organised and optimized.
- Apply DevOps principles in an AI-context.

At the end of this project you will have an automated workflow, where a model is generated and deployed as both a **Web Service** (on-demand forecast API) and a **Batch Service** (scheduled inference and monitoring pipeline).

### Context

The renewable energy transition creates grid instability: solar and wind production are weather-dependent and inherently difficult to predict. Grid operators, energy traders, and prosumers all need reliable short-term forecasts to make balancing decisions.

In this project you will build an end-to-end ML system that predicts **solar and/or wind energy production (in MW) for the Antwerp region over the next 24 hours**, using weather forecast data as input.

You will use the data from your **Data Engineering** course. It combines multiple sources:

| Sheet          | Sources                                             | Features                              |
| -------------- | --------------------------------------------------- | ------------------------------------- |
| **wind**       | Open Meteo ECMWF, Geo.be, Kaggle (Uccle, Antwerpen) | Wind speed (km/h) per hour            |
| **zon**        | Open Meteo ECMWF, Geo.be, Kaggle (Uccle)            | Solar radiation (W/m²) per hour       |
| **productie**  | Energie Vlaanderen, Elia                            | Solar & wind production (MW) per hour |
| **consumptie** | Energie Vlaanderen, Elia, Kaggle                    | Grid load & consumption (MW) per hour |

The ECMWF model also provides **multi-day ahead forecasts** (hourly), meaning your deployed API can make real predictions using live forecast data — not just backtests.

### Assignment: Renewable Energy Production Forecasting

For the project, you will build an end-to-end ML project. You will need to:

- [ ] Use the dataset provided by the Data Engineering course (see [Dataset](#dataset))
  - _Join the weather sheets (`wind`, `zon`) with the production targets (`productie`) on the `tijd` column. You may choose to predict solar production, wind production, or both._
- [ ] Describe the problem and your application in the `README.md` of your repository
  - _Explain what is predicted, what the inputs and outputs of your model are, and why this is useful_
- [ ] Train a model on that dataset tracking your experiments
- [ ] Create a model training pipeline
- [ ] Deploy the model as a **web service**
  - _A REST API that accepts weather forecast data as input and returns predicted energy production (MW) for the next 24 hours_
- [ ] Deploy the model as a **batch service**
  - _A scheduled Prefect pipeline that periodically fetches fresh ECMWF weather forecasts, runs inference, stores predictions, and compares them against newly available Elia actuals_
- [ ] Monitor the performance of your model
  - _Use the batch service output to compute error metrics over time (e.g. RMSE), visualise them in Grafana, and trigger a retraining flow when a threshold is exceeded_
- [ ] Follow the best practices

### Technologies

| Subject                      | package/software   |
| ---------------------------- | ------------------ |
| **Experiment tracking tool** | MLFlow             |
| **Workflow orchestration**   | Prefect            |
| **Monitoring**               | Evidently, Grafana |

### Evaluation

#### Requirements (Best practices)

- [ ] There are no unit tests (-1 point)
- [ ] Did not follow the Coding Guidelines (-1 point)
- [ ] There are no pre-commit hooks (-1 point)

#### Peer Review

You need to **evaluate 3 projects** of your peers. You can get 2 points for each evaluation.

<table><thead><tr><th width="163"></th><th>0 points</th><th>1 points</th><th>2 points</th></tr></thead><tbody><tr><td><strong>Peer Evaluation 1</strong></td><td>No peer evaluation was done, or the peer evaluation was sub-par</td><td>The peer evaluation was created, but the evaluation was not or barely motivated</td><td>The peer evaluation was created, the evaluation was well motivated</td></tr><tr><td><strong>Peer Evaluation 2</strong></td><td>No peer evaluation was done, or the peer evaluation was sub-par</td><td>The peer evaluation was created, but the evaluation was not or barely motivated</td><td>The peer evaluation was created, the evaluation was well motivated</td></tr><tr><td><strong>Peer Evaluation 3</strong></td><td>No peer evaluation was done, or the peer evaluation was sub-par</td><td>The peer evaluation was created, but the evaluation was not or barely motivated</td><td>The peer evaluation was created, the evaluation was well motivated</td></tr></tbody></table>

#### Rubrik

<table><thead><tr><th width="164" align="right">Subject</th><th>0 points</th><th>1 points</th><th>2 points</th></tr></thead><tbody><tr><td align="right"><strong>Problem description</strong></td><td>The problem is not described</td><td>The problem is described but shortly or not clearly</td><td>The problem is well described: it is clear what is predicted, what the inputs and outputs are, and why this forecast is useful</td></tr><tr><td align="right"><strong>Experiment tracking &#x26;</strong><br><strong>model registry</strong></td><td>No experiment tracking or model registry</td><td>Experiments are tracked or models are registered in the registry</td><td>Both experiment tracking and model registry are used</td></tr><tr><td align="right"><strong>Workflow orchestration</strong></td><td>No workflow orchestration</td><td>Basic workflow orchestration</td><td>Fully worked out and deployed workflow</td></tr><tr><td align="right"><strong>Model deployment</strong></td><td>Model is not deployed</td><td>Model is deployed as a web service or batch service, but only locally</td><td>Both a web service and a batch service are implemented, containerized, and could be deployed to cloud</td></tr><tr><td align="right"><strong>Model monitoring</strong></td><td>No model monitoring, or Evidently and Grafana are not implemented or not working correctly</td><td>The batch service computes prediction error metrics (e.g. RMSE) by comparing predictions against Elia actuals, and reports them to Evidently and Grafana</td><td>The batch service reports to Evidently and Grafana, and automatically triggers a retraining workflow when a defined metric threshold is violated</td></tr><tr><td align="right"><strong>Reproducibility</strong></td><td>No instructions on how to run the code at all, the data is missing</td><td>Some instructions are there, but they are not complete OR instructions are clear and complete, the code works, but the data is missing</td><td>Instructions are clear, it's easy to run the code, and it works. The versions for all the dependencies are specified.</td></tr></tbody></table>
