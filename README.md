# APS-Failure-at-Scania-Trucks

### Data
Source: APS Failure at Scania Trucks Data Set from UCI Machine Learning Repository available [here](https://archive.ics.uci.edu/ml/datasets/APS+Failure+at+Scania+Trucks)

Notebook: [Data description](https://github.com/FranciscaAlliende/APS-Failure-at-Scania-Trucks/blob/master/Data_Description.ipynb)

### Goal
Minimize cost =  c1 * False Positives + c2 * False Negatives with c1 = 10, and c2 = 500

### First Approach: Sampling techniques
- Handle null values on training  and test set 
- Apply a sampling technique to the training set
- Train the models with balanced data
- The models are the regular versions of RF, LR and SVM
- Predict and evaluate

Notebooks
* [SMOTE](https://github.com/FranciscaAlliende/APS-Failure-at-Scania-Trucks/blob/master/SMOTE.ipynb)
* [Downsampling](https://github.com/FranciscaAlliende/APS-Failure-at-Scania-Trucks/blob/master/Downsampling.ipynb)

### Second Approach: Cost-Sensitive Learning Models
- Divide the training set into training and validation sets
- Handle null values on training, validation and test set 
- Find the right hyparameters for each of the models in order to reduce the false negatives as much as possible, having false positives under control.
- Train the models
- Predict and evaluate

Notebooks:
* [Cost Sentive Learning: Random Forest](https://github.com/FranciscaAlliende/APS-Failure-at-Scania-Trucks/blob/master/Cost_Sentive_Learning_Random_Forest.ipynb)
* [Cost Sentive Learning: SVM](https://github.com/FranciscaAlliende/APS-Failure-at-Scania-Trucks/blob/master/Cost_Sentive_Learning_SVM.ipynb) 
