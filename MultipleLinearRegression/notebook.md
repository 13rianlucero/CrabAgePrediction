

# CrabAgePrediction

**CLASS:**  `CPSC-483 Machine Learning Section-02`
**LAST UPDATE:**  `May 5, 2022`
**PROJECT NAME:** `Crab Age Prediction`
**PROJECT GROUP:**
| Name                       | Email                                 | Student              |
| -------------------------- | ------------------------------------- | -------------------- |
| Brian Lucero               | 13rianlucero@csu.fullerton.edu        | Undergraduate        |
| Justin Heng                | justinheng@csu.fullerton.edu          | Graduate             |

**PROJECT PAPER:** 
----------------------------------------------------------------------------------------------------------------------------------

## Overview
1. **Abstract**
    - Paper Summary
2. **Introduction**
    - The problem 
    - Why it's important 
    - Key method and strategy
3. **Background**
    - Technologies & ideas used to build our method
4. **Methods**
    - Approach to solving the problem
    - Key contributions
        - from Justin
        - from Brian
5. **Experiments**
    - Description of ML process workflow 
        - Featuring the project source code to compliment the experiment process description 
6. **Conclusion**
    - Summary of contributions & results
    - Future work
7. **References**
    - All of our project resources

## Abstract
> Machine learning can be used to predict the age of crabs. It can be more accurate than simply weighing a crab to estimate its age. Several different models can be used, though support vector regression was found to be the most accurate in this experiment.


## Introduction
> Crab is very tasty and many countries of the world import huge amounts of crabs for consumption every year. The main benefits of crab farming are, labor cost is very low, production cost is comparatively lower and they grow very fast. Commercial crab farming business is developing the lifestyle of the people of coastal areas. By proper care and management we can earn more from crab farming business than shrimp farming. You can raise mud crabs in two systems. Grow out farming and fattening systems. For a commercial crab farmer knowing the right age of the crab helps them decide if and when to harvest the crabs. Beyond a certain age, there is negligible growth in crab's physical characteristics and hence, it is important to time the harvesting to reduce cost and increase profit.

## Background
**Technologies used:**
> - K-Nearest Neighbours (KNN) - Machine Learning Model
> - Multiple Linear Regression - Machine Learning Model
> - Support Vector Machine (SVM) - Machine Learning Model
> - Feature Selection & Representation
> - Evaluation on variety of methods
> - Method Selection
> - Parameter Tuning
> - Classifier Evaluation
> - Train-Test Split
> - Cross Validation




----------------------------------------------------------------------------------------------------------------------------------





**Dataset/Data Preprocessing**
> The dataset that is being used was taken from Kaggle [1]. It contains over 1000 samples and nine features each. The features are "Sex", "Length", "Diameter", "Height", "Weight", "Shucked Weight", "Viscera Weight", "Shell Weight", and "Age". Fortunately, all of the data was present and no values were missing. In the case that values were missing, that specific data point could be taken out in order to avoid any errors during calculations. Since "Sex" had a value of either "M" for male, "F" for female, and "I" for indeterminate, conversions were necessary in order to give the feature a numerical value. Male was given a numerical value of 1, female was given 2, and indeterminate was given 1.5. These values were stored into a new feature called "SexValue".

> To perform feature selection, the Pearson correlation coefficient was found for each of the eight values in relation to "Age". The results are in the table below.

| SexValue       | 0.0337 |
| -------------- | ------ |
| Length         | 0.555  |
| Diameter       | 0.574  |
| Height         | 0.552  |
| Weight         | 0.539  |
| Shucked Weight | 0.419  |
| Viscera Weight | 0.501  |
| Shell Weight   | 0.625  |

<sub>Table 1. Pearson correlation coefficients</sub>

[1] --- https://www.kaggle.com/datasets/sidhus/crab-age-prediction  

<p align="center">
    <img src="https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white" alt=""/>
</p>

