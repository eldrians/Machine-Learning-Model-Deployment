# Machine Learning Model Deployment - Model Tracking Using MLflow

## Use Case Summary

### Objective Statement
* Get an insight into how much the company spends on the marketing budget
* Get an insight into how much sales based on the marketing budget
* Get an understanding about the correlation between marketing budget and sales
* Get an understanding about marketing strategy or planning based on the model prediction
* Create models to predict sales with simple linear regression
* Deploy the model using MLflow deployment

### Challenges
* There is no explanation for each column![a6027e34-e129-4f1c-a6e0-4c4af6614db0](https://user-images.githubusercontent.com/91566708/202836577-293ae765-d317-485b-9cd8-270046e3fdb3.jpg)


### Methodology/ Analytic Technique
* Descriptive analysis
* Graph analysis
* Machine learning model using Simple Linear Regression
* Deployment using MLflow

### Business Benefit
* Help Business to make decision about marketing strategy based on the prediction

### Expected Outcome
* Get to know about how much the company spend on marketing budget
* Get to know about how much sales based on marketing budget
* Get to know about the correlation between marketing budget and sales
* Recommendation based on model prediction
* Making models to predict sales
* Deploy the model using MLflow

## Business Understanding

A marketing budget is the sum of money a company assigns to marketing projects (paid advertising, sponsored content, etc.) aimed at product promotion. It helps startups and established companies manage resources efficiently and achieve business goals.

This case has some business questions using the data:

  * How much does the company spends on marketing budget?
  * How much sales are based on the marketing budget?
  * Is there any correlation between marketing budget and sales?
  * How about the recommendation based on the model prediction?
  
## Data Understanding

### Source Data
 
* https://www.kaggle.com/code/devzohaib/simple-linear-regression/notebook
* 200 rows
* 2 columns

### Data Dictionary

* `TV`, Marketing budget for selling TV
* `Sales`, Total sales budget

## Data Preparation

import library : Pandas, numpy, matplotlib, seaborn, sklearn, and mlflow

## Data Cleansing

Data is clean, the data type  is all correct and there is no missing value

## Exploratory Data Analysis

#### Statistic Numerical Data

![image](https://user-images.githubusercontent.com/91566708/202832643-fff298e5-287e-4374-8a14-95b287a03866.png)

From the analysis of tvmarketing.csv data which contains `200 lines`, the average cost spent on the TV marketing budget is `$147.04` with the lowest cost spending `$0.7` and the highest cost of `$296.4`. Also, the average sales are `$14.02` with the lowest being `$1.6` and the highest is `$27`.

#### Distribution of Marketing Budget

![image](https://user-images.githubusercontent.com/91566708/202832698-cfd9f466-745f-47f9-85ea-589096bd7a6a.png)

Based on the graph above distribution of marketing budget 'TV' has the highest density of marketing Budget which is `$200` dollars.

#### Distribution of Sales

![image](https://user-images.githubusercontent.com/91566708/202832725-c26cb0d2-8ab1-4387-981a-44a2b1d758f3.png)

Based on the data above, we can see that 'Sales' are mostly or have the highest density at values of `$10-15`

#### Correlation between TV and Sales

![image](https://user-images.githubusercontent.com/91566708/202832774-97a10727-b048-45b8-99fc-69b70d790e69.png)

Based on the graph, we can see that the Sales and Marketing Budget of TV has a correlation, where the higher the marketing costs of the TV, the greater the selling price of the TV. However, from the graph above there is also data that is known to have a low selling price but high marketing costs.

## Feature Engineering

#### MinMaxScaler

![image](https://user-images.githubusercontent.com/91566708/202832975-2fa50f95-db07-45ab-9ea1-1402a9f10bdb.png)

We duplicate dataset to do scaling using MinMaxScaler. Features in the data have different value ranges. By normalizing the data using MinMaxScaler, the data will have the same scale and the model can learn faster and the accuracy of the model can increase.

## Hyperparameter Tuning

we used Hyperparameter Tuning to get best model for parameter and hopefully that can impact score RSME, MAE, MAPE, and R2 at evaluate model. from the Hyperparameter Tuning, we get :

![image](https://user-images.githubusercontent.com/91566708/202836595-02b34f00-8319-4e93-90e0-9e89f434acd6.png)

that is best model parameter for simple linear regression from Hyperparameter Tuning.

## Preprocessing Modeling

![image](https://user-images.githubusercontent.com/91566708/202833030-a4f5decc-e647-4dd4-9160-fed7b06215c9.png)

From the graph, it is known that there is a positive relationship between the train data and the predictive data, the red dot indicates the train data which is close to the blue line (predictive data). This shows that the model used is good enough for sales predictions from the tvmarketing dataset.

## Evaluate Model

![image](https://user-images.githubusercontent.com/91566708/202833073-82f233be-a6f3-4218-aa60-5c11f975e981.png)

Based on the results of the evaluation that has been carried out, we can see that the MAE value is 2.44, the smaller the MAE value, the more accurate the model used. The MAPE value is 0.18 or 18%, this figure indicates the percentage error of the predictions made, the smaller the percentage error value in MAPE, the more accurate the forecasting results, for MAPE is 18% it is known that the predictions made are still classified as good for prediction. The RMSE value is 3.19, RMSE indicates that the variation in values produced by a forecast model is close to the variation in the observed value. RMSE measures how different the sets of values are. The smaller the RMSE value, the closer the predicted and detected values are. 
The value of R2 is 0.67 or 67%, R square has a value between 0 – 1 provided that the closer to number one means the better, R2 has a value of 0.67, meaning that 67% of the distribution of the dependent variable can be explained by the independent variable. The remaining 33% cannot be explained by independent variables or can be explained by variables outside the independent variables (component errors).

## Comparationn Evaluate Model

![image](https://user-images.githubusercontent.com/91566708/202835613-9a746be0-11f0-4a5d-89b5-c37b5d15775b.png)

we try to get result from evaluate model with 2 different process :
1. `without`  Hyperparameter Tuning,  do simmple linear regression with default parameter.
2. `with` Hyperparameter Tuning, do simple linear regression with best model parameter.

but the conclusion that we get there is no difference so whether or not using hyperparameters is good. The model that was performed by hyperparameter tuning showed no difference in values for MAE, MAPE, and RSME, which means that the use of hyperparameters in this dataset does not have to be done.

## Model Deployment

the result from this model deployment is same like evaluate model that we made before, so this model  deployment with MLflow are succed.
![image](https://user-images.githubusercontent.com/91566708/202835699-ceafb945-7240-43aa-be6a-d5fa745798ff.png)

this  is the UI from ML flow that we already deploy the model to the platform
![image](https://user-images.githubusercontent.com/91566708/202835711-978bed41-83bd-432e-8e86-a62647c4506e.png)

## Result

* From the analysis of tvmarketing data which contains `200 lines`, the average cost spent on the TV marketing budget is `$147.04` with the lowest cost spending `$0.7` and the highest cost `$296.4`, and the average sales is `$14.02` with a lowest is `$1.6` and the highest is `$27`.
* Based on the graph above distribution of marketing budget 'TV' has the highest density of marketing Budget which is `$200 dollars`
* Based on the data above, we can see that 'Sales' are mostly or have the highest density at values of `$10-15`
Based on the graph above, we can see that the Sales and Marketing Budget of TV has a correlation, where the higher the marketing costs , the greater the sales. However, from the graph above there is also data that has high marketing costs but has low sales
* From the graph above we can see that there is a positive relationship between the train data and the predictive data, the red dot indicates the train data which is close to the blue line (predictive data). This shows that the model used is good enough for sales predictions from the tvmarketing dataset.
* Based on the results of the evaluation that has been carried out, we can see that the MAE value is `2.44`, the smaller the MAE value, the more accurate the model used. The MAPE value is `0.18` or `18%`, this figure indicates the percentage error of the predictions made, the smaller the percentage error value in MAPE, the more accurate the forecasting results, for MAPE is 18% it is known that the predictions made are still classified as good for prediction. The RMSE value is `3.19`, RMSE indicates that the variation in values produced by a forecast model is close to the variation in the observed value. RMSE measures how different the sets of values are. The smaller the RMSE value, the closer the predicted and detected values are. The value of R2 is `0.67` or `67%`, R square has a value between 0 – 1 provided that the closer to number one means the better, R2 has a value of 0.67, meaning that 67% of the distribution of the dependent variable can be explained by the independent variable. The remaining 33% cannot be explained by independent variables or can be explained by variables outside the independent variables (component errors).
* There is no difference in the evaluation value of the data that we do the hyperparameter tuning and not
* The model that was performed by hyperparameter tuning showed no difference in values for MAE, MAPE, and RSME, which means that the use of hyperparameters in this dataset does not have to be done.


## Recommendation

Companies can `optimize marketing budget costs for marketing in order to get high profits`, optimizing marketing costs does not have to use large costs because it can be seen from the data that there is a large marketing budget but low selling prices

## Deployment

MLflow is a platform to streamline machine learning development, including tracking experiments, packaging code into reproducible runs, and sharing and deploying models. MLflow offers a set of lightweight APIs that can be used with any existing machine learning application or library (TensorFlow, PyTorch, XGBoost, etc), wherever you currently run ML code (e.g. in notebooks, standalone applications or the cloud). MLflow's current components are:

  * MLflow Tracking: An API to log parameters, code, and results in machine learning experiments and compare them using an interactive UI.
  * MLflow Projects: A code packaging format for reproducible runs using Conda and Docker, so you can share your ML code with others.
  * MLflow Models: A model packaging format and tools that let you easily deploy the same model (from any ML library) to batch and real-time scoring on platforms such as Docker, Apache Spark, Azure ML and AWS SageMaker.
  * MLflow Model Registry: A centralized model store, set of APIs, and UI, to collaboratively manage the full lifecycle of MLflow Models.

In the above data the use of mlflow is very useful for tracking the above model to predict the MAPE, MAE, RMSE and R2 Scores from implementing Linear Regression on this data. Model tracking is also to maintain model consistency if new data is entered




