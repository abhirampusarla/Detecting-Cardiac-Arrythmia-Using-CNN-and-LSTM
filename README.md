## Abstract

The present research endeavors to examine and categorize electrocardiogram (ECG) patterns through the application of sophisticated clustering, feature extraction, and classification methodologies aimed at identifying cardiac arrhythmias. The INCART 2-lead Arrhythmia Database,MIT-BIH Arrhythmia Database,  MIT-BIH Supraventricular Arrhythmia Database and Sudden Cardiac Death Holter Database served as the testing ground for the proposed analytical framework. This investigation amalgamates machine learning (ML) models, deep learning (DL) frameworks, and unsupervised clustering techniques to classify arrhythmia patterns and assess their distribution, variability, and clinical relevance. The proposed pipeline is very generalizable and can be used across similar ECG datasets with minimal changes to effectively be used towards addressing cardiac arrhythmias amongst different populations.


## Introduction

Electrocardiograms are very crucial in determining and diagnosing the existence of cardiac abnormalities. Among them, an arrhythmia is a maladjusted rhythm of the heart that can lead to death if not treated. In the project, the INCART 2-lead Arrhythmia Database was utilised to classify different types of arrhythmia from ECG recordings.

Clustering ECG patterns into meaningful groups with their features, most relevant feature extraction, and clinical insights are to be identified both with unsupervised and supervised methods by scaling up and robustly extending the pipeline for three new datasets.
Background
Arrhythmia refers to a heart condition associated with irregular heartbeats, either too slow or too fast, or chaotic. Early detection is paramount in preventing serious cardiovascular events since untreated arrhythmias can cause severe complications including strokes, cardiac arrests, or heart failure.

## Project Objective

This project aims at achieving the correct classification of ECG readings as normal or arrhythmic using ML techniques. A strong classification model can be developed with this, and clinicians as well as medical devices could rapidly and accurately assess the heart health from ECG signals, thus allowing timely intervention and better patient outcomes.

## Datasets Used
In order to improve the model's robustness and generalisability, we employed four separate datasets:
INCART 2-lead Arrhythmia Database: This two-lead ECG data set is a collection that contains normal and arrhythmic patterns.
MIT-BIH Arrhythmia Database & MIT-BIH Arrhythmia Supra ventricular Database: This dataset is the most commonly used for arrhythmia detection and includes records of several types of arrhythmias taken in the clinical environment.
Sudden Cardiac Death Holter Database: This is a collection of long-term ECG recordings of patients who experienced sudden cardiac death during the recordings.

## Data Wrangling and Model Training
### 1. Preprocessing and Feature Scaling
Data Preprocessing: Columns not required for clustering and classification were excluded, and numeric values were normalized using MinMaxScaler to ensure uniformity in feature magnitudes.
Reshaping for CNN: Data was reshaped to a 3D format for convolutional layers, treating each instance as a time series signal.
### 2. Deep Learning for Feature Extraction
A Convolutional Neural Network (CNN) was constructed with layers for convolution, pooling, and fully connected layers to extract high-level features.
CNN Architecture:
Conv1D layers with ReLU activation to extract time-series signal features.
MaxPooling1D layers to downsample feature maps and reduce dimensionality.
Dense layers to model complex feature interactions and reduce features to a 32-dimensional vector.
### 3. Clustering and Visualization
### KMeans Clustering:
Unsupervised clustering to group ECG patterns into 5 clusters, chosen based on clinical interpretations of ECG types.
PCA for Dimensionality Reduction
Reduced the extracted features to 2D for visualizing cluster separations.
Heatmaps and Distributions:
Heatmaps for label distributions across clusters, bar plots for variance analysis, and scatter plots to show relationships between cluster size and feature variance.
### 4. Supervised Learning for Feature Importance
Random Forest Classifier:
A supervised ensemble model used to rank feature importance for diagnosing arrhythmia types.
Feature correlation heatmaps helped identify interdependencies among features and potential redundancy.
### 5. Clustering Evaluation Metrics
#### Silhouette Score: Measures intra-cluster cohesion and inter-cluster separation.
#### Davies-Bouldin Index: Measures cluster compactness and distinctness.
#### Calinski-Harabasz Index: Evaluates the dispersion ratio between and within clusters.

## Domain Knowledge
### 1. Electrocardiogram Data
ECG datasets include time-series signals of electrical cardiac activity. Arrhythmias can be classified into:
Normal Sinus Rhythm (N): The normal cardiac rhythm.
Ventricular Ectopic Beats (VEB): A premature beat originating from the ventricles.
Supraventricular Ectopic Beats: Extrasystoles originating above the ventricles.
Fused Beats (F): Overlapping or fusion of normal and abnormal impulses.
Mixed/Unclear (Q): Ambiguous patterns requiring further analysis.
### 2. Dataset Overview
Each record in the INCART 2-lead Database combines features that reflect signal amplitudes, temporal intervals, and frequencies. The dataset is annotated for supervised learning experiments and used in an unannotated format for clustering tasks.
### 3. Feature Extraction and Clustering Importance
The categorization of arrhythmias is essential for recognizing patterns that may not be readily apparent through conventional diagnostic methods.
Feature extraction with CNNs ensures to capture high-level patterns that would help with clustering and with classification tasks.

## Accuracy and Evaluation
The study contains three add-on datasets and is each treated to the same approach applied above. Of interest are:
### 1. Clustering Outcomes
Cluster Summary: Each cluster appears to reflect the different distributions of signal characteristics, such as predominantly dominant VEBs in one and more mixed beat in another.
Feature Variance: Variances describe the spread of the characteristics within each of the clusters.
### 2. Grading Guidelines
Silhouette Score: the clusters are well-separated.
Davies-Bouldin Index Low values indicate well-defined or tightly packed clusters.
Calinski-Harabasz Index: Elevated scores indicate effective dispersion in clustering.
### 3. Random Forest Classification
Accuracies across datasets (average of 99.2% for primary dataset, ~98.5% for others) underline the robust classification ability.
Key characteristics, such as heart rate variability and RR intervals, contributed greatly to improving the prediction accuracy.


Dataset / Different scores	INCART 2-lead Arrhythmia Database	MIT-BIH Arrhythmia Database	MIT-BIH Supraventricular Arrhythmia	 Sudden Cardiac Death Holter Database
Silhouette Score	0.6681	0.6678	0.6682	0.6337
Davies-Bouldin Index	0.4680	0.4689	0.4679	0.7154
Calinski-Harabasz Index	1288228.8179	739060.7383	1351732.0097	1163339.1445
Challenges
Imbalanced labels in clusters needed careful analysis to avoid bias.
High dimensionality of features required PCA for interpretability.
Processing of time-series data in CNNs required significant computational efforts and considerable preprocessing.
Health-Related Outcomes
These help detect weak patterns in ECG signals and improve the diagnosis of arrhythmias.
These help clinicians identify complex patterns of fusion or mixed beats.

### Scalability:
The pipeline can apply to many ECG datasets and is versatile.

### Analytical Depth:
Uses self-learning methods (KMeans) and supervised techniques (Random Forest) to adequately analyze together. 
