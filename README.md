# spam-message_detection-

# SMS Spam Classification

This project aims to build a machine learning model to classify SMS messages as either spam or ham (not spam). The dataset used is the "spam.csv" dataset.

## Table of Contents
1. [Data Cleaning](#data-cleaning)
2. [Exploratory Data Analysis (EDA)](#eda)
3. [Text Preprocessing](#text-preprocessing)
4. [Model Building](#model-building)
5. [Model Evaluation](#model-evaluation)
6. [Model Improvement](#model-improvement)
7. [Voting Classifier](#voting-classifier)
8. [Conclusion](#conclusion)

## Data Cleaning <a name="data-cleaning"></a>

The dataset contains SMS messages with columns 'target' and 'text'. The 'Unnamed: 2', 'Unnamed: 3', and 'Unnamed: 4' columns were dropped as they contained missing values. The 'target' column was encoded as 0 for ham and 1 for spam. Duplicate rows were also removed.

## Exploratory Data Analysis (EDA) <a name="eda"></a>

EDA involved analyzing the distribution of text lengths, word counts, sentence counts, and creating word clouds to visualize the most common words in spam and ham messages.

## Text Preprocessing <a name="text-preprocessing"></a>

Text preprocessing steps included converting text to lowercase, tokenization, removing special characters, punctuation, and stop words. Porter stemming was also applied to reduce words to their root forms.

## Model Building <a name="model-building"></a>

Various classifiers were trained and evaluated, including Gaussian Naive Bayes, Multinomial Naive Bayes, Support Vector Classifier, Random Forest, and more. Accuracy using cross val score and precision were used as evaluation metrics.

## Model Evaluation <a name="model-evaluation"></a>

The trained models were evaluated using accuracy and precision scores, and the results were visualized using bar plots.

## Model Improvement <a name="model-improvement"></a>

Different techniques were applied to improve model performance, such as changing max_features in TfIdf, scaling features, and including the number of characters in messages as an additional feature.

## Voting Classifier

A voting classifier and stacking classifier were implemented to combine the predictions of multiple models, resulting in improved accuracy and precision.

## Conclusion <a name="conclusion"></a>

This project demonstrated the process of building a SMS spam classification model, starting from data cleaning and preprocessing to model building, evaluation, and improvement. The final model achieved high accuracy and precision, effectively distinguishing between spam and ham messages.

------

