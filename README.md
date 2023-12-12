![](UTA-DataScience-Logo.png)

# A Data Science-Driven Exploration of Beer Chemistry

* **One Sentence Summary** This project explores the differentiation of beer styles techniques such as PCA, ANOVA, and Random Forest based on LC-QTOF-MS data. 

## Overview
Utilizing the data collected from the LC-QTOF-MS and employing machine learning techniques and multivariate analysis the study seeks to identify  and provide insights into the unique characteristics of each beer style. The project applies Principal Component Analysis (PCA) for dimensionality reduction, Analysis of Variance (ANOVA) for feature selection, and Random Forest for classification. The final step involves evaluating the performance of the Random Forest model in accurately classifying beer styles based on selected features.

## Summary of Workdone
Conducted ANOVA on selected features to pinpoint those with notable variations across beer classes (ANOVA.ipynb). Top 10 features that are found to be significant in differntiating 

There is a notebook (NN attempt.ipynb) where neural network was implemented using Keras to classify beer samples into different styles based on their chemical profiles. The model achieved a test accuracy of 86.67%.

### Data

* Data:
    * Input: excel sheet obtained from chemistry lab containing 10815 rows x 122 columns (6 different types of beers, water and QCs run in triplicates)
  * Size: 10815 features for 71 samples
  * Instances (Train, Test, Validation Split): 10815 features for 71 samples.
 
    <img width="557" alt="beer pic" src="https://github.com/AyatOmar/BeerFeatureSelection/assets/111785493/89acd48b-2313-4c5e-b8b3-69ab5bf33829">


#### Preprocessing / Clean up
* 51 columns containing details pertaining to the chemistry aspect of the dataset was removed, this is not needed for the data analysis 
* Abbreviation Column: A column containing abbreviations for beer samples, providing a concise representation for each sample.
* Class Column: A column indicating the beer class or style to which each sample belongs (e.g., IPA, Blonde, Stout).
* QC Column: A quality control column, marking whether a sample is a quality control sample.
  
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

* The data was obtained from Dr. Kevin Schug's Analytical Chemistry lab, Chemistry and Biochemistry Department at UTA

### Training

* Describe how to train the model

#### Performance Evaluation

* Dimensionality of the Dataset: The beer dataset has a high dimensionality with 10,000 features and a relatively small number of samples (71). This high-dimensional space poses challenges for both feature selection and machine learning models.
* ANOVA (Analysis of Variance) was employed as a feature selection method to identify significant features that exhibit statistically significant differences across different beer classes. However, due to the high dimensionality, ANOVA alone may not be sufficient.
* Principal Component Analysis (PCA), a technique for reducing the dimensionality of the dataset. PCA transforms the original features into a lower-dimensional space, capturing the most significant variance. This reduction can be beneficial for subsequent analyses.
* While aneural network implemented using Keras yielded acceptable results, there are concerns about the suitability of neural networks for datasets with high dimensionality and limited samples. The risk of overfitting and computational complexity?

## Citations

* Provide any references.







