# APS-Failure-at-Scania-Trucks

Data: APS Failure at Scania Trucks Data Set from UCI Machine Learning Repository available [here](https://archive.ics.uci.edu/ml/datasets/APS+Failure+at+Scania+Trucks)

[Data description Notebook](https://github.com/FranciscaAlliende/APS-Failure-at-Scania-Trucks/blob/master/Data_Description.ipynb)


#### First Approach: Cost-Sensitive Learning Models
- Divide the training set into training and validation sets
- Handle null values on training, validation and test set 
- Find the right hyparameters for each of the mentioned models in order to reduce the false negatives as much as possible, having false positives under control.
- Train the models
- Predict and evaluate

Notebooks:
* [Weighted Random Forest](https://github.com/FranciscaAlliende/APS-Failure-at-Scania-Trucks/blob/master/Weighted_Random_Forest.ipynb)
* [Modified k-NN]()
* [Weighted Logistic Regression]()
* [Cost-sensitive SVM]()

#### Second Approach: Sampling techniques
- Handle null values on training  and test set 
- Apply a sampling technique to the training set
- Train the models with balanced data
- The model used, will be (regular versions of RF, k-NN, LR and SVM)
- Predict and evaluate

Notebooks
* [SMOTE]()
* [Downsampling]()

