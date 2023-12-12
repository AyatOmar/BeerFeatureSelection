![](UTA-DataScience-Logo.png)

# A Data Science-Driven Exploration of Beer Chemistry

* **One Sentence Summary** This project explores the differentiation of beer styles techniques such as PCA, ANOVA, and Random Forest based on LC-QTOF-MS data. 

## Overview
Utilizing the data collected from the LC-QTOF-MS and employing machine learning techniques and multivariate analysis the study seeks to identify  and provide insights into the unique characteristics of each beer style. The project applies Principal Component Analysis (PCA) for dimensionality reduction, Analysis of Variance (ANOVA) for feature selection, and Random Forest for classification. The final step involves evaluating the performance of the Random Forest model in accurately classifying beer styles based on selected features.

## Summary of Workdone
Conducted ANOVA on selected features to pinpoint those with notable variations across beer classes (ANOVA.ipynb). Top 10 features that are found to be significant in differntiating 

There is a notebook (NN attempt.ipynb) where neural network was implemented using Keras to classify beer samples into different styles based on their chemical profiles. The model achieved a test accuracy of 86.67%. (one or more hidden layers with activation functions, and an output layer with a softmax activation function for multi-class classification. The specific configuration and hyperparameters would be detailed in the code, which can be referenced in the project repository for more information.)

### Data

* Data:
  * Type: For example
    * Input: medical images (1000x1000 pixel jpegs), CSV file: image filename -> diagnosis
    * Input: CSV file of features, output: signal/background flag in 1st column.
  * Size: How much data?
  * Instances (Train, Test, Validation Split): how many data points? Ex: 1000 patients for training, 200 for testing, none for validation

#### Preprocessing / Clean up

* Describe any manipulations you performed to the data.

#### Data Visualization

Show a few visualization of the data and say a few words about what you see.

### Problem Formulation

Input / Output: LC-QTOF-MS features / Beer styles.
Models: PCA, ANOVA, Random Forest.
Loss, Optimizer, other Hyperparameters: N/A.

### Training

* Describe the training:
  * How you trained: Training involved applying PCA, ANOVA, and Random Forest sequentially.
  * How did training take.
  * Training curves (loss vs epoch for test/train).
  * How did you decide to stop training.
  * Any difficulties? How did you resolve them?

### Performance Comparison

* Clearly define the key performance metric(s).
* Show/compare results in one table.
* Show one (or few) visualization(s) of results, for example ROC curves.

### Conclusions

* State any conclusions you can infer from your work. Example: LSTM work better than GRU.

### Future Work

* Exploring additional methods for feature reduction.
* Extending the analysis to a larger dataset.

## How to reproduce results

* In this section, provide instructions at least one of the following:
   * Reproduce your results fully, including training.
   * Apply this package to other data. For example, how to use the model you trained.
   * Use this package to perform their own study.
* Also describe what resources to use for this package, if appropirate. For example, point them to Collab and TPUs.

### Overview of files in repository

* There are many notebooks, most of them were drafts/previous attempts and I did not want to lose in case I wanted to go back on an idea
* The three main notebooks to look at
   * Anova.ipynb : applying ANOVA to the dataset, however dimension reduction needs to be done
   * NN attempt.ipynb: This was my first attempt at applying a neural network using keras to the dataset, it worked but I have not finished working on it. 
   * TSNE attempt.ipynb: this was the start of applying TSNE to do visualization(identifying clusters in the data)
* These are notebooks that you can disregard (they are for me to reference to, I will be deleting them when I no longer reference to them and incorperate them in the main notebooks, so they will be removed by the end of the project)
   * ANOVA attempt3.ipynb
   * Data Processing.ipynb
   * attempt 1.ipynb
   * attempt 2.ipynb


### Software Setup
* List all of the required packages.
* If not standard, provide or point to instruction for installing the packages.
* Describe how to install your package.

### Data

* Point to where they can download the data.
* Lead them through preprocessing steps, if necessary.

### Training

* Describe how to train the model

#### Performance Evaluation

* Describe how to run the performance evaluation.


## Citations

* Provide any references.







