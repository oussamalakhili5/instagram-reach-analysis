# 📱 Instagram Reach Analysis

## 🎯 Project Objective
Comprehensive analysis of Instagram post reach to understand the factors influencing engagement and predict future post performance.

## 📊 Methodology (KDD Process)
1. **Data Cleaning** : Data cleaning and preparation (removing duplicates and missing values)
2. **EDA** : Exploratory Data Analysis with visualizations (distributions, correlations, word clouds)
3. **Clustering** : Post segmentation with K-Means (3 clusters identified)
4. **Prediction** : Reach forecasting with Random Forest

## 🛠️ Technologies Used
- Python 3.x
- Pandas & NumPy for data manipulation
- Matplotlib & Seaborn for visualization
- Scikit-learn for Machine Learning
- WordCloud for text analysis

## 📈 Key Results

### Post Segmentation (K-Means)
| Cluster | Number of Posts | Average Impressions | Characteristic |
|---------|----------------|---------------------|-----------------|
| 0 | 82 | 4,336 | Low-Performing Posts |
| 1 | 17 | 9,491 | Average Posts |
| 2 | 3 | 29,003 | Viral Posts |

### Predictive Model Performance (Random Forest)
- **R² Score** : 0.847 (84.7% of variance explained)
- **Mean Absolute Error (MAE)** : 1,448 impressions

### Most Influential Variables
1. Likes
2. Saves
3. Profile Visits
4. Shares
5. Comments

## 💡 Recommendations for Content Creators
- **Increase Likes** : the variable most correlated with impressions
- **Encourage Saves** : 2nd most important variable
- **Optimize Profile Visits** : strong impact on virality
- **Target Cluster 2** : viral posts get 6x more impressions

## 📁 Project Structure
 ├── data/ # Raw data
 ├── notebooks/ # Jupyter Notebook
 ├── images/ # Generated visualizations
 └── README.md # Documentation 
