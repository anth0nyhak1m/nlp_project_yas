# nlp_project_yas
> CAPP 30254 ML Final Project

## Authors: 
* Anthony Hakim
* Sasha Filippova
* Yifu Hou

## Project Descripion:
In this project our team designed 2 Natural Language Processing machine learning models to classify fake news articles using only article titles. For our baseline model we use a logistic regression model and TF-IDF
techniques to classify fake news articles with 94% accuracy. We also apply a pre-trained BERT model for classification, and discover that the more complex model preforms 
with lower accuracy.

## Directory:

* **baseline_model.ipynb**: TF-IDF logistic regression training and testing.
* **classification.ipynb**: Final BERT model hyperparameter tuning, training and testing.
* **original_bert.ipynb**: Baseline BERT model training and testing.
* **util.py**: file of helper functions to preprocess data.
* **final_presentation**: final presentation of results.

## Data Visualization:
![image](https://user-images.githubusercontent.com/36241004/192418924-b4464c23-090a-4929-b3b0-668176a6f528.png)

## Data Source:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?select=True.csv
