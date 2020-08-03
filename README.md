# APS-Failure-at-Scania-Trucks
1st case Master Thesis Machine Learning Techniques for Predictive Maintenance in Industry


#### Pipeline

STEPS:
1. basic_prepro
 - input: raw dataset
 - output: basic preprocessesed dataset
2. null_values
 - input: basic preprocessesed train + test sets
 - output: y_train, X_train, y_test, X_test, wihout null values
 * hyperparameter. t: % of null values for a feature to be consideR: set at 50%
3. unbalanced classes (downsampling or SMOTE)
 - input: y, X unbalance pair
 - output: y, X balanced pair
 * hyperparameter: SMOTE number of neighbors: set at 100
 4. dimensionality reduction (Random Forest or PCA)
 - input: X_train, y_train after unbalanced class and X_test after null_values
 - output: X_train and X_test with the selected features
 * hyperparameters:
    - Random Forest: n_estimators: set at 100 and g: feature importance (gini) threshold, set after vizualizing the corresponding graph at 0.02 for downsampling and 0.015 for upsampling.  
    - PCA: nc: number of components, set at 80
