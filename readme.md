# Darcy Davis
### Fintech Bootcamp - Module 10 Homework Assignment

The Python notebook for consideration is named [crypto_investments.ipynb](https://github.com/darcy5d/Fintech_Module_10_Homework_Assignment/blob/main/crypto_investments.ipynb) and located in my repo named [Fintech_Module_10_Homework_Assignment](https://github.com/darcy5d/Fintech_Module_10_Homework_Assignment).

# Cryptocurrency Clustering with K-Means and PCA

## Table of Contents

1. [Overview](#overview)
2. [Technologies](#technologies)
3. [Questions & Answers](#questions--answers)
4. [Elbow Method](#elbow-method)
5. [K-Means Clustering](#k-means-clustering)
6. [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
7. [Visualizations](#visualizations)
8. [Conclusions](#conclusions)

---

## Overview

This project aims to segment different cryptocurrencies into clusters based on various features. We explore two different approaches: one using the original feature set and the other using Principal Component Analysis (PCA) for dimensionality reduction.

---

## Technologies

- Python
- Pandas
- Scikit-learn
- hvPlot

---

## Questions & Answers

### What is the best value for `k` when using the original data?

The best value for `k` when using the original data is 4, as determined by the elbow method.

### What is the best value for `k` when using the PCA data?

The best value for `k` when using the PCA data is also 4. This finding aligns with the optimal k-value found using the original data, adding confidence to our choice.

### What is the impact of using fewer features to cluster the data using K-Means?

The impact of using fewer features (via PCA) on the K-means clustering is generally minimal in this situation. Though we are losing dimentionality, using PCA analysis we can still yield accurrate clustering outcomes (as we still capture most of the variance in the data).

---

## Elbow Method

The elbow method was used to find the optimal number of clusters (`k`). This involved plotting the inertia for a range of `k` values and identifying the "elbow" point where the inertia starts to decrease at a slower rate.


## K-Means Clustering

K-Means clustering was applied twice:

- Once on the original dataset.
- Once on the dataset after applying PCA.


## Principal Component Analysis (PCA)

PCA was performed to reduce the dimensions of the dataset, thus speeding up the clustering process without sacrificing much information.

![Alt text](https://github.com/darcy5d/Fintech_Module_10_Homework_Assignment/blob/main/images/PCA_components_3D.png?raw=true "PCA Clustering presented in three dimensions") 

## Visualizations

We visualized the elbow curves and the clustered points for both the original and PCA datasets.

![Alt text](https://github.com/darcy5d/Fintech_Module_10_Homework_Assignment/blob/main/images/elbows.png?raw=true "Elbow curves")

![Alt text](https://github.com/darcy5d/Fintech_Module_10_Homework_Assignment/blob/main/images/cluster_comparison_small.png?raw=true "Different cluster formats")

---

## Conclusions

The project successfully clustered cryptocurrencies into 4 distinct groups both using the original dataset and after applying PCA. These findings can be used for further financial analysis and decision-making processes.

---
