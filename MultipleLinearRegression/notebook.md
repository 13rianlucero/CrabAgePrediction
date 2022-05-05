# sCrabAgePrediction

**CLASS:**  `CPSC-483 Machine Learning Section-02`

**LAST UPDATE:**  `May 5, 2022`

**PROJECT NAME:** `Crab Age Prediction`

**PROJECT GROUP:**

| Name         | Email                          | Student       |
| ------------ | ------------------------------ | ------------- |
| Brian Lucero | 13rianlucero@csu.fullerton.edu | Undergraduate |
| Justin Heng  | justinheng@csu.fullerton.edu   | Graduate      |

**PROJECT PAPER:**   [Here](https://github.com/13rianlucero/CrabAgePrediction/blob/main/FirstDraft/Crab%20Age%20Prediction%20Paper.pdf)

**PROJECT GITHUB REPOSITORY:** [Here](https://github.com/13rianlucero/CrabAgePrediction)

---

# Overview

> ## **1. Abstract**
>
> `Paper Summary:`
>
> Machine learning can be used to predict the age of crabs. It can be more accurate than simply weighing a crab to estimate its age. Several different models can be used, though support vector regression was found to be the most accurate in this experiment.`
>
>> ## **2. Introduction**
>>
>> - The problem
>> - Why it's important
>> - Key method and strategy
>>   | The Problem                                                                                                                                                                                                                     | Why it's important?                                                                                                                                                       | Our Solution Strategy                                                                                                                                                                                                       |
>>   | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
>>   | It is quite difficult to determine a crab's age due to their molting cycles which happen throughout their whole life.<br /><br />Essentially, the failure to harvest at an ideal age, increases cost and crab lives go to waste | Beyond a certain age, there is negligible growth in crab's physical characteristics and hence, it is important to time the harvesting to reduce cost and increase profit. | Prepare crab data and use it to train several machine learning models.<br /><br />Thus, given certain physcial chraracteristics and the corresponding values, the ML models will accurately determine the age of the crabs. |
>>
>>> ## **3. Background**
>>>
>>> - `Technologies & ideas used to build our method`
>>>
>>>> ## **4. Methods**
>>>>
>>>> - `Approach to solving the problem`
>>>> - `Key contributions`
>>>>   - `from Justin`
>>>>   - `from Brian`
>>>>
>>>>> ## **5. Experiments**
>>>>>
>>>>> - `Description of ML process workflow`
>>>>>   - `Featuring the project source code to compliment the experiment process description`
>>>>>
>>>>>> ## **6. Conclusion**
>>>>>>
>>>>>> - `Summary of contributions & results`
>>>>>> - `Future work`
>>>>>>
>>>>>>> ## **7. References**
>>>>>>>
>>>>>>> - `All of our project resources`
>>>>>>>
>>>>>>
>>>>>
>>>>
>>>
>>
>

| SexValue       | 0.0337 |
| -------------- | ------ |
| Length         | 0.555  |
| Diameter       | 0.574  |
| Height         | 0.552  |
| Weight         | 0.539  |
| Shucked Weight | 0.419  |
| Viscera Weight | 0.501  |
| Shell Weight   | 0.625  |

`<sub>`Table 1. Pearson correlation coefficients `</sub>`

---

[1] --- https://www.kaggle.com/datasets/sidhus/crab-age-prediction

<p align="center">
    <img src="https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white" alt=""/>
</p>

For a commercial crab farmer knowing the right age of the crab helps them decide if and when to harvest the crabs. Beyond a certain age, there is negligible growth in crab's physical characteristics and hence, it is important to time the harvesting to reduce cost and increase profit.
