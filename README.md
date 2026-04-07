# 24ADI003_24BAD020_EXP-07


SCENARIO 1 – CLUSTERING USING K-MEANS:

Problem Statement:
Group customers/data points into clusters based on similarity using K-Means clustering.

Dataset (Kaggle – Public)
Mall Customer Segmentation Dataset Dataset Link: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python


Discrription of the Code:
K-Means clustering is applied to divide customers into distinct groups based on similarity. The dataset is first loaded and preprocessed by handling missing values and selecting relevant features. To ensure accurate clustering, the data is normalized using standard scaling. The optimal number of clusters is determined using the Elbow Method, which analyzes the relationship between the number of clusters and inertia. After selecting the best value of K, the K-Means algorithm is applied to assign each customer to a specific cluster. The model is evaluated using inertia and silhouette score to measure compactness and separation of clusters. Visualizations such as the elbow curve, cluster scatter plot, and centroid representation are used to better understand the grouping of customers. K-Means performs hard clustering, meaning each data point belongs to only one cluster, and it works best when clusters are well-separated and spherical in shape.




SCENARIO 2 – CLUSTERING USING GMM

Problem Statement
Cluster data using Gaussian Mixture Models to capture probabilistic cluster membership.

Dataset (Kaggle – Public)
Mall Customer Segmentation Dataset Dataset Link: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python


Discrription of the Code:
Gaussian Mixture Model (GMM) is used to perform clustering with a probabilistic approach. Similar preprocessing steps are followed, including scaling of features. The optimal number of clusters is selected using AIC (Akaike Information Criterion) and BIC (Bayesian Information Criterion), which help in choosing the best model. GMM is trained using the Expectation-Maximization (EM) algorithm, which estimates the parameters of the data distribution. Unlike K-Means, GMM provides soft clustering, where each data point is assigned a probability of belonging to each cluster. Final cluster labels are assigned based on the highest probability. The model is evaluated using log-likelihood, silhouette score, AIC, and BIC. Visualizations include probability distribution plots, contour plots showing cluster density, and comparison graphs between K-Means and GMM. GMM is more flexible than K-Means as it can model overlapping and elliptical clusters.
