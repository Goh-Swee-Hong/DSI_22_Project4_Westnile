# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png)

# West Nile Virus Prediction
## Predict West Nile virus in mosquitos across the city of Chicago
## Table of Contents:

### Part 1
1. Background 
1. Problem Statement 
1. Executive Summary
1. Data Cleaning  
1. Exploratory Data Analysis 

### Part 2
6. Final Dataset Preparation for Modelling
1. Modeling    
    1. Base Case
1. Fine Tuning 
    1. Model 1
    1. Model 2
    1. Model 3
    1. Model 4
    1. Model 5
    1. Final Model
1. Cost Analysis
1. Conclusion and Recommendations 
1. References and Data Sources 

## 1. Background
Due to the recent epidemic of West Nile Virus in the City of Chicago, the Chicago Department of Public Health had set up a surveillance and control system. Traps are laid across the city and mosquitoes in the trap are tested for the virus every week from late spring to end of fall.

## 2. Problem Statement
Using the data collected by CDC on the traps, their past spraying data and the weather data from 2 Chicago weather stations, our goal is to build a model to make predictions on the locations of outbreaks of West Nile Virus in Chicago. Using this prediction, we will conduct a cost-benefit analysis on the spraying of pesticide in different locations. We will present our findings to the members of CDC.

## 3. Executive Summary

#### Data cleaning: train set
<p>Starting with the train set, we checked and found that there weren't any null values in the set. Next, we added traps found in test set but not in train set into train set. We then labeled the mosquito species numerically and reformatted the date column for use in EDA later.</p>

#### Date cleaning: weather set
<p>Null values were represented in different ways like 'M', '-', ' ', 'T' and '  T'. We found all of them and imputed them with appropriate values accordingly. Finally, we combined and took the average values of the 2 station readings into 1 for use in EDA later.</p>

#### EDA
<p>Referencing from the Culex lifecycle, we lagged our weather data by 7 days when merging with train set to account for the period when the mosquitoes are still in egg/larva/pupa stage. We grouped our features into location, temperature, water and wind and plotted them against keys features number of mosquitoes and virus present. Features are then selected and dropped according to the correlation between one another and their correlation with the 2 key features. We then further explored the data by plotting the 2 key features, in count or percentage, against other features to end up with 177 features to go into modeling.</p>

#### Final dataset preparation for modeling
<p>Using knowledge gained from outside research on impact of climate variables and Culex lifecycle, we experimented and found the sweet spot for rolling specific weather features to finalize the dataset for further fine tuning and modeling. A sample submission returned a Kaggle score of 0.72682 confirmed that the dataset is ready to go into the modeling.
</p>

#### Modelling
<p>We passed through the data in the following models:
    * K Nearest Neighbors
    * Random Forests
    * Naive Bayes
    * Decision Tree
    * Logistic Regression
We SMOTE-ed the training data and used imbalance pipe to fine tune details. After running several iterations, these were the results:
    
| Model                                     | Kaggle Test Score | Best Params                                                                                                   |
|--------------------------------------|-------------------|---------------------------------------------------------------------------------------------------------------|
| Logistic Regression           | 0.735              | {'C': 10, 'max_iter': 100, 'penalty': 'l1', 'solver': 'liblinear'} |                                                          |
| Random Forest Classifier | 0.729	              | {'rt__max_depth': 15, 'rt__max_features': 0.1, 'rt__min_samples_leaf': 5, 'rt__min_samples_split': 3}|
| Multinomial Naive Bayes      | 0.713	              | {'nb__alpha': 0.001} |
| Decision Tree	           | 0.609	             | {'dt__max_depth': 4, 'dt__min_samples_leaf': 3, 'dt__min_samples_split': 10}                                            |
| K Nearest Neighbours                 | 0.539	              | {'knn__metric': 'manhattan', 'knn__n_neighbors': 40}                                  |    
</p>


#### Model Conclusion
<p>Based on the models that we have tested, Logistic Regression is deemed to be the best model to predict the locations of West Nile Virus outbreaks in Chicago. This model has the highest Kaggle ROC score out of all the models that were tested.

Logistic Regression tends to perform well when the dataset is linearly separable and when there is average or no multicollinearity between independent variables. Additionally, the number of mosquitos is likely to follow a poisson distribution which the sigmoid logit link mapping perform well on. Moreover, the features that were used in the model were carefully selected via EDA and Lasso regularization. The relevant weather features were also rolled according to the Culex lifecycle. Thus, enabling us to get the best result from Logistic Regression. 

The top features for the Logistic Regression model include Tavg_roll6, DewPoint_roll2, PrecipTotal_roll5, PrecipTotal_roll2 and Stn Pressure. Weather features, specifically, the rolled weather features turned out to be some of the important features for the model. This observation can also be seen across all the other models that were tested. Hence, this confirms that rolling the relevant weather variables is a good strategy and that doing so helps to strengthen the result.
</p>

#### Cost Benefit Analysis of Spraying
Based on looking at several key dates, for example, 7 September 2011, 25 July 2013 and 8 August 2013, we can see that the effect of spraying is not consistent. On 7 September 2011, the areas that were sprayed saw a significant drop in the average number of mosquitos and WNV rate. However, on 8 August 2013, the areas that were sprayed saw a huge increase in the WNV rate despite spraying while the average number of mosquitos in that area only marginally dropped.

To estimate the cost and benefit of spraying, we propose a simple equation covered below.

#### a. Data Dictionary

| Variable           | Explaination                                                   |
| ------------------ | -------------------------------------------------------------- |
| pipiens/restuans   | Dummy variable for species pipiens/restuans                    |
| restuans           | Dummy variable for species restuans                            |
| pipiens            | Dummy variable for species pipiens                             |
| salinarius         | Dummy variable for species salinarius                          |
| territans          | Dummy variable for species territans                           |
| 6                  | Dummy variable for June                                        |
| 7                  | Dummy variable for July                                        |
| 8                  | Dummy variable for August                                      |
| 9                  | Dummy variable for September                                   |
| 10                 | Dummy variable for October                                     |
| TXXX               | Respective dummy variables for trap with trap id TXXX          |
| StnPressure        | Average station pressure (inches of HG)                        |
| ResultSpeed        | Average wind speed in miles/hour                               |
| ResultDir          | Resultant wind direction up to tens of degrees                 |
| Dewpoint           | Average dew point                                              |
| Dewpoint\_rollX    | Average dew point rolled by X number of days                   |
| Tavg               | Average temperature in Fahrenheit                              |
| Tavg\_rollX        | Average temperature in Fahrenheit rolled by X number of days   |
| PrecipTotal        | Total precipitation for 24hr period                            |
| PrecipTotal\_rollX | Total precipitation for 24hr period rolled by X number of days |
| WnvPresent         | Dummy for presence of West Nile Virus                          |

## A. Model Conclusion

Based on the models that we have tested, Logistic Regression is deemed to be the best model to predict the locations of West Nile Virus outbreaks in Chicago. This model has the highest Kaggle ROC score out of all the models that were tested.

Logistic Regression tends to perform well when the dataset is linearly separable and when there is average or no multicollinearity between independent variables. Additionally, the number of mosquitos is likely to follow a poisson distribution which the sigmoid logit link mapping perform well on. Moreover, the features that were used in the model were carefully selected via EDA and Lasso regularization. The relevant weather features were also rolled according to the Culex lifecycle. Thus, enabling us to get the best result from Logistic Regression. 

The top features for the Logistic Regression model include Tavg_roll6, DewPoint_roll2, PrecipTotal_roll5, PrecipTotal_roll2 and Stn Pressure. Weather features, specifically, the rolled weather features turned out to be some of the important features for the model. This observation can also be seen across all the other models that were tested. Hence, this confirms that rolling the relevant weather variables is a good strategy and that doing so helps to strengthen the result.

## B. Cost Benefit Analysis Conclusion

Based on looking at several key dates, for example, 7 September 2011, 25 July 2013 and 8 August 2013, we can see that the effect of spraying is not consistent. On 7 September 2011, the areas that were sprayed saw a significant drop in the average number of mosquitos and WNV rate. However, on 8 August 2013, the areas that were sprayed saw a huge increase in the WNV rate despite spraying while the average number of mosquitos in that area only marginally dropped.

To estimate the cost and benefit of spraying, we propose a simple equation (this is mentioned in detail in the Costs and Benefits Equation subsection above). 

### Total cost of spraying

Based on our equation, the total cost of spraying is dependent on 3 factors. These are:

- cost of spraying itself
- accuracy of prediction
- efficacy of prediction

We only have data for the accuracy of prediction (which is based on our model). Based on our research, there is no data for the specific cost of spraying in the Chicago areas. The relevant data that was found was the amount of budget allocated to Environmental Health in 2013, which was $1,191,811. This budget is used to perform routine and complaint-generated inspections of facilities to ensure the City's ordinances related to environmental hazards are enforced. The budget also covers coodination of mosquito surveillance and control activities.

### Potential Economic Impact of Spraying

Based on our equation, the potential economic impact of not spraying is dependent on 3 factors. These are:

- mosquito infection rate
- population density
- medical cost per case

Based on our research, we only have the value for the average medical cost per case, which amounts to $21,000.  

This value is derived from the total cost estimation of $778 million in health care expenditures and lost productivity based on the 37,088 WNV disease cases reported to CDC from 1999 through 2012.

### Conclusion

As some of the values, like explicit cost of spraying is not available, we used our proposed equation to calculate the cost and benefit of spraying after inputing actual and also fictitious numbers. Based on our calculation, the total cost of spraying is only slightly above the potential economic impact of not spraying. In such cases, it would be up to the Department of Health to make a call. 

One important thing to note is that our simple calculation do not include various intangible benefits and costs that are associated to spraying and West Nile Virus. 

## C. Recommendations

There are other "cheaper" ways for the government to help reduce the spread of WNV. The Chicago budget in 2013 showed $9,000,000 dedicated to community engaged care. This covers promoting health through education, policy and service. The government can ramp up education on West Nile Virus and how to prevent mosquito breeding. They can educate the public on preventing stagnant water, applying insect repellent, wearing the proper attire to reduce the chances of getting bitten by mosquitos.

Other measures which they have already undertaken but can place more emphasis on is the reporting of dead birds (a key in transmission of WNV), reporting on tall grass and weeds.
