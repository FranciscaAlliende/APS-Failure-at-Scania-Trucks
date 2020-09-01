# APS-Failure-at-Scania-Trucks

### Data
Source: APS Failure at Scania Trucks Data Set from UCI Machine Learning Repository available [here](https://archive.ics.uci.edu/ml/datasets/APS+Failure+at+Scania+Trucks)

Notebook: [Data description](https://github.com/FranciscaAlliende/APS-Failure-at-Scania-Trucks/blob/master/Data_Description.ipynb)


### Goal
Reduce the false negatives as much as possible, having false positives under control.

Minimize cost =  c1 * False Positives + c2 * False Negatives with c1 = 10, and c2 = 500

### First Approach: Sampling techniques
- Handle null values on training  and test set 
- Apply a sampling technique to the training set
- Train the models with balanced data
- The models are the regular versions of RF, k-NN and SVM
- Predict and evaluate

Notebooks
* [SMOTE](https://github.com/FranciscaAlliende/APS-Failure-at-Scania-Trucks/blob/master/SMOTE.ipynb)
* [Downsampling](https://github.com/FranciscaAlliende/APS-Failure-at-Scania-Trucks/blob/master/Downsampling.ipynb)

### Second Approach: Cost-Sensitive Learning Models
- Divide the training set into training and validation sets
- Handle null values on training, validation and test set 
- Find the right hyparameters for each of the mentioned models in order to reduce the false negatives as much as possible, having false positives under control.
- Train the models
- Predict and evaluate

Notebooks:
* [Weighted Random Forest](https://github.com/FranciscaAlliende/APS-Failure-at-Scania-Trucks/blob/master/Weighted_Random_Forest.ipynb)
* [Modified k-NN]() to do
* [Cost-sensitive SVM]() to do



