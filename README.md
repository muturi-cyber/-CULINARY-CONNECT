# CULINARY CONNECT- A Smart Recipe Recommendation and Nutrition Analysis Platform 

---

![image link](https://github.com/muturi-cyber/Recipe-Intelligence-/blob/main/pexels-janetrangdoan-1132047.jpg)

---

#### Authors:
Adrianna Ndubi

Caroline Kimani

Dennis Muuo

Fiona Amugune

Daniel Karue

---

## PROJECT OVERVIEW

Food is essential to culture, and African cuisine is rich in diversity, flavors, and traditions. However,
discovering authentic African recipes tailored to personal preferences can be challenging, especially for
users unfamiliar with regional variations. This project aims to bridge that gap by developing an AI-powered
Recipe Recommendation System and Sentiment Analysis Model.

---

## BUSINESS UNDERSTANDING

---

### Business Objectives

1. The primary objective is to create a recipe Recommendation system, a Sentiment Analysis Model to enhance user satisfaction through personalized recipe suggestions and improved engagement across diverse culinary experiences. 

2. Improve decision-making with data-driven insights by identifying popular recipes and trends through sentimental analysis, understanding the pain points of customers eg ingredient availability, etc., and  optimizing recipe suggestions based on positive and negative feedback. 

3. Increasing revenue and platform growth by driving traffic to food blogs, e-commerce stores, and cooking classes. Boosting engagement for advertisers, chefs, and food businesses on the platform.


### Stakeholders

- **Platform Users:** Seek personalized and diverse recipe recommendations.
- **Developers & Data Science Team:** Responsible for model development and chatbot integration.
- **Business Analysts:** Utilize insights to identify popular trends and user preferences.


### EDA Integration
1. What are the most common ingredients used across all cuisines?
2. How does the nutritional content vary across different cuisines?
3. Which cuisines receive the most positive/negative feedback?

---

## DATA UNDERSTANDING
The dataset used in this project is sourced from scraping websites

Columns in the various datasets:

#### International Recipes Data

name; Name of the recipe

id; Unique identifier for the recipe

minutes; Time required to prepare the recipe (in mins)

tag; Tags associated with the recipe

steps; Step-by-step instructions for preparation

ingredients; List of ingredients used

calories; Caloric content per serving

total fat (PDV); Total fat content as a percentage of daily value

sugar (PDV); Sugar content as a percentage of daily value

sodium (PDV); Sodium content as a percentage of daily value

protein (PDV); Protein content as a percentage of daily value

saturated fat (PDV); Saturated fat content as a percentage of daily value

carbohydrates (PDV); Carbohydrate content as a percentage of daily value


#### Nutrition Data

name; Name of the food item

serving_size; Size of the serving (in grams or milliliters)

calories; Caloric content per serving

total_fat; Total fat content (in grams)

saturated_fat; Saturated fat content (in grams)

cholesterol; Cholesterol content (in mg)

sodium; Sodium content (in mg)

potassium; Potassium content (in mg)

saturated_fatty_acids; Total saturated fatty acids (in grams)

#### Interactions Data

user_id; Unique identifier for the user

recipe_id; Unique identifier for the recipe

date; Date of interaction

rating; User's rating for the recipe

review; User's review or feedback


---

## DATA CLEANING
#### 1. Checked for missing values
#### 2.Dropping the irrelevant columns
#### 3.Checked for duplicate values
#### 4. Cleaning of some columns

---

## EXPLORATORY DATA  ANALYSIS


### What are the most common ingredients used across all cuisines?


#### 1. Foreign cuisine

 
![image link](https://github.com/muturi-cyber/Recipe-Intelligence-/blob/main/top%2010%20ingdnts%20in%20foreign%20cuisine.png)


The most common ingredient in Foreign cuisine is Salt.


#### 2. African cuisine


![image link](https://github.com/muturi-cyber/Recipe-Intelligence-/blob/main/top%2010%20ingdnt%20african%20cuisine.png)


The most common ingredient in African cuisine is Salt.


#### Correlation between different Nutrients


![image link](https://github.com/muturi-cyber/Recipe-Intelligence-/blob/main/corr%20btn%20diff%20nutrients.png)


- Fat content is a strong predictor of calorie count.
 
- High-fat foods typically contain more saturated fat and fatty acids.

- Sodium-rich foods tend to have less potassium which may impact dietary balance. 


#### Top 10 Healthiest African Recipes


![image link](https://github.com/muturi-cyber/Recipe-Intelligence-/blob/main/top%2010%20healthiest%20african%20recipies.png)


#### Top 10 Healthiest International Recipes


![image link](https://github.com/muturi-cyber/Recipe-Intelligence-/blob/main/top%2010%20healthiest%20internl%20recipies.png)


## MODELING

### 1. RECOMMENDATION SYSTEM

Objective:
 
Provide personalized recipe suggestions to users based on preference, past interactions or similarities.
 
Modeling Techniques used:
 
Content-Based Filtering:
It utilizes TF-IDF or cosine similarity to recommend recipes based on ingredient overlap.
 
Collaborative Filtering:
Leverages user-item interaction matrix for personalized suggestions.
 
Evaluation Metrics:

Content-Based Model Results:

MSE: 0.1214

RMSE: 0.3484

- The model shows low prediction error, indicating good accuracy.

- Predictions deviate by ~0.35 units on average.

- Content-based filtering performs well for personalized recipe recommendations

### 2. SENTIMENTAL ANALYSIS MODEL

Objective: 

Classify user reviews into positive, neutral or negative sentiments, to understand user preferences and improve engagement
 
Modeling Techniques Used: 

Data Preprocessing: 
Cleaning text data by removing stopwords, punctuation, and special characters.

Tokenization and vectorization using TF-IDF or Word Embeddings for feature extraction.

Model Selection and Training:  

Logistic Regression: It’s used to predict the sentiment category.

Evaluation Metrics:

Accuracy - 96 %

### How Recommendation and sentiment analysis work together.

recommendation System - suggests recipes based on calories, ingredients and cooking time.
Sentiments Analysis - Analyzes user reviews to understand satisfaction and improve recommendations.

Feedback Loop :

- Positive reviews - Boosts recipe ranking.
- Negative reviews - Lower ranking or adjustment
- Continuous Improvement - enhances user satisfaction and engagement

---

## CONCLUSION

The Sentiment Analysis Model provides valuable insights into customer opinions, allowing businesses to improve their offerings. However,
refining text cleaning and model tuning can improve accuracy.

The Recommendation System successfully suggests recipes but can be enhanced with personalized filtering and user feedback loops.

Implementing these models in a real-world food app can boost user engagement, improve satisfaction, and increase retention rates.

---

## RECOMMENDATION

### Marketing Focus

Promote highly rated and calorie-dense recipes to attract food enthusiasts.

Highlight these recipes in advertisements, social media, and promotional campaigns.

### User Insights & Review Analysis

Analyze negative reviews across different rating ranges to identify recurring issues.

Focus on reviews from 2013-2018 to detect significant shifts in sentiment and trends.

### Health-Conscious Options

Introduce low-calorie and nutrient-rich alternatives to cater to health-conscious consumers.

---

## NEXT STEP

Build a Chatbot for User Engagement
 
Chatbot Features:
Provide recipe suggestions based on user mood (using sentiment analysis results).
Answer nutrition-related questions (calories, ingredients, dietary preferences).
Support voice commands for an interactive experience.














