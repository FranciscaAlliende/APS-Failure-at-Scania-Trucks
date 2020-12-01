import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

def data_prepro(set):
    # replace in class column: pos = 1; neg = 0
    set['class'] = set['class'].map({'pos': 1, 'neg': 0})

    # replace the na and nan values with np.NaN
    set.replace(to_replace=['na','nan'],value = np.NaN,inplace=True)

    return set

def null_values(train, test, t):
  # split the dataset
    y_train = train["class"]
    X_train = train.drop("class", axis=1)

    y_test = test["class"]
    X_test = test.drop("class", axis=1)
   
    # drop the features with more than t% of null values on the train set
    X_train = X_train.loc[:, X_train.isnull().mean() <= t]
    # remove those columns from the test set
    X_test = X_test[X_train.columns]

    # fill the remaining null values with the mean of the corresponding feature
    imp = SimpleImputer(strategy = "median")
    T = pd.DataFrame(imp.fit_transform(X_train))
    T.columns = X_train.columns
    T.index = X_train.index
    X_train = T

    T = pd.DataFrame(imp.fit_transform(X_test))
    T.columns = X_test.columns
    T.index = X_test.index
    X_test = T
  
    return y_train, X_train, y_test, X_test

def standardize(X_train, X_test):
    transformer = RobustScaler().fit(X_train)
    
    X_trainS = transformer.transform(X_train)
    X_testS = transformer.transform(X_test)
    
    X_trainS = pd.DataFrame(data = X_trainS, columns = X_train.columns)
    X_testS = pd.DataFrame(data = X_testS, columns = X_test.columns)
    return X_trainS, X_testS

def feature_selection_with_plots(X_train, y_train):
    # initial point
    print("Current N of features:", len(X_train.columns))
    print(" ")

    # removing features with zero variance
    print("\033[1m" + "Remove features with zero variance" + "\033[0m")
    selector = VarianceThreshold()
    selector.fit_transform(X_train)
    selected_columns = X_train.columns[(selector.get_support())]
    print("N of dropped columns:", len(set(X_train.columns) - set(selected_columns)))

    X_train = X_train[selected_columns]
    print("Current N of features:", len(X_train.columns))

    # Tree-based feature selection
    clf = ExtraTreesClassifier(n_estimators=50, random_state=333)
    clf = clf.fit(X_train, y_train)

    # feature importance
    feature_importance = clf.feature_importances_.ravel()
    feature_names = X_train.columns
    data_tuples = list(zip(feature_names, feature_importance))
    features = pd.DataFrame(data_tuples, columns=["feature_names", "feature_importance"])

    # plot top n features sorted by feature importance
    n = 30
    fe = features.sort_values(["feature_importance"], ascending=False).reset_index(drop=True)
    fe = fe.head(n)
    fe = fe.sort_values(["feature_importance"], ascending=True).reset_index(drop=True)

    fig = plt.figure(figsize = [12,7])
    ax = fig.add_axes([0,0,1,1])

    data = fe["feature_importance"].values
    names = fe["feature_names"].values
    y_pos = np.arange(len(names))

    plt.barh(y_pos, data, color = "darkgreen")
    plt.yticks(y_pos, names)

    plt.title("Top "+str(n)+ " features")
    plt.xlabel("feature importance")
    plt.ylabel("column name")
    plt.savefig("figures/Top"+str(n)+ "features.png", bbox_inches = "tight")
    plt.show()
    
    print("\033[1m" + "Tree-based feature selection" + "\033[0m")
    selector = SelectFromModel(clf, prefit=True)
    selected_columns = X_train.columns[(selector.get_support())]
    print("N of dropped columns:", len(set(X_train.columns) - set(selected_columns)))

    X_train = X_train[selected_columns]
    print("Current N of features:", len(X_train.columns))

    corr = abs(X_train.corr())

    plt.figure(figsize=(12,12))
    sns.heatmap(corr, square = True)
    plt.title("Correlation Matrix after tree-based feature selection", fontsize = 15)
    plt.savefig("figures/cm_after_1stFS.png", bbox_inches = "tight")
    plt.show()
    
    # drop columns highly correlated between each-other and choose the one with higher feature importance
    print("\033[1m" + "Drop highly correlated features" + "\033[0m")
    correlations = []
    feature_tuples = []
    for col in X_train.columns:
        for row in X_train.columns:
            correlation = corr.loc[row, col]
            if row == col:
                pass
            elif (col, row) in feature_tuples:
                pass
            elif correlation >= 0.7:
                correlations.append(correlation)
                feature_tuples.append((row, col))

    drop = []
    for tup in feature_tuples:
        f0 = tup[0]
        f1 = tup[1]
        imp_f0 = features[features["feature_names"] == f0]["feature_importance"].values
        imp_f1 = features[features["feature_names"] == f1]["feature_importance"].values
        if imp_f0 <= imp_f1:
            drop.append(f0)
        else:
            drop.append(f1)
    drop = set(drop)

    print("N of dropped features:", len(drop))

    selected_columns = list(set(X_train.columns) - set(drop))
    X_train = X_train[selected_columns]

    print("Current N of features:", len(X_train.columns))

    corr = abs(X_train.corr())
    plt.figure(figsize=(12,12))
    sns.heatmap(corr, square = True, annot = True, fmt = ".2")
    plt.title("Final Correlation Matrix", fontsize = 15)
    plt.savefig("figures/cm_after_2ndFS.png", bbox_inches = "tight")
    plt.show()
    
    return X_train

def feature_selection(X_train, y_train):
    # initial point
    selector = VarianceThreshold()
    selector.fit_transform(X_train)
    selected_columns = X_train.columns[(selector.get_support())]
    
    X_train = X_train[selected_columns]
    
    # Tree-based feature selection
    clf = ExtraTreesClassifier(n_estimators=50, random_state=333)
    clf = clf.fit(X_train, y_train)
    
    feature_importance = clf.feature_importances_.ravel()
    feature_names = X_train.columns
    data_tuples = list(zip(feature_names, feature_importance))
    features = pd.DataFrame(data_tuples, columns=["feature_names", "feature_importance"])
    
    selector = SelectFromModel(clf, prefit=True)
    selected_columns = X_train.columns[(selector.get_support())]
    
    X_train = X_train[selected_columns]
    
    corr = abs(X_train.corr())
    
    # drop columns highly correlated between each-other and choose the one with higher feature importance
    correlations = []
    feature_tuples = []
    for col in X_train.columns:
        for row in X_train.columns:
            correlation = corr.loc[row, col]
            if row == col:
                pass
            elif (col, row) in feature_tuples:
                pass
            elif correlation >= 0.7:
                correlations.append(correlation)
                feature_tuples.append((row, col))

    drop = []
    for tup in feature_tuples:
        f0 = tup[0]
        f1 = tup[1]
        imp_f0 = features[features["feature_names"] == f0]["feature_importance"].values
        imp_f1 = features[features["feature_names"] == f1]["feature_importance"].values
        if imp_f0 <= imp_f1:
            drop.append(f0)
        else:
            drop.append(f1)
    drop = set(drop)

    selected_columns = list(set(X_train.columns) - set(drop))
    X_train = X_train[selected_columns]
    
    return X_train

def scatter_2features(X_train, y_train, title):
    d = {"class": y_train, "cn_001": X_train["cn_001"], "by_000": X_train["by_000"]}
    V = pd.DataFrame(d)
    
    Vcn_1 = V[V["class"] == 1]["cn_001"].values
    Vcn_0 = V[V["class"] == 0]["cn_001"].values

    Vby_1 = V[V["class"] == 1]["by_000"].values
    Vby_0 = V[V["class"] == 0]["by_000"].values
    
    # plot
    plt.scatter(Vcn_1, Vby_1, color = 'green') # minority class
    plt.scatter(Vcn_0, Vby_0 , color = 'black') # mayority class
    plt.xlabel("cn_001")
    plt.ylabel("by_000")
    plt.legend(('minority class', 'majority class'))
    plt.title(title)
    plt.savefig("figures/"+title+".png", bbox_inches = "tight")
    plt.show()

def PCA_4vis(X_train, y_train, title):
    # PCA 2 components
    pca = PCA(n_components=2)
    pc = pca.fit_transform(X_train)
    
    # to dataframe
    pc_df = pd.DataFrame(data = pc, columns = ["pc1", "pc2"])
    pc_df["class"] = y_train
    pc_df = pc_df[["class", "pc1", "pc2"]]
    
    pc1_1 = pc_df[pc_df["class"] == 1]["pc1"].values
    pc1_0 = pc_df[pc_df["class"] == 0]["pc1"].values

    pc2_1 = pc_df[pc_df["class"] == 1]["pc2"].values
    pc2_0 = pc_df[pc_df["class"] == 0]["pc2"].values
    
    # plot
    plt.scatter(pc1_1, pc2_1, color = 'red') # minority class
    plt.scatter(pc1_0, pc2_0 , color = 'blue') # mayority class
    plt.xlabel("principal component 1")
    plt.ylabel("principal component 2")
    plt.legend(('minority class', 'majority class'))
    plt.title(title)
    plt.savefig("figures/"+title+".png", bbox_inches = "tight")
    plt.show()
    
def resample_with_plots(technique, name, X_train, y_train):
        print("\033[1m" + name + "\033[0m")
    
        X_res, y_res = technique.fit_resample(X_train, y_train)
    
        print("class distribution after "+ name)
        print(sorted(Counter(y_res).items()))
    
        f.scatter_2features(X_res, y_res, "Class distribution after "+ name)
        f.PCA_4vis(X_res, y_res, "Class distribution after "+ name)
    
        return X_res, y_res

def resample(technique, X_train, y_train): 
        X_res, y_res = technique.fit_resample(X_train, y_train)
        print("-----------resample - class distribution after sampling:", sorted(Counter(y_res).items()))
        return X_res, y_res

def evaluation_metrics(y_pred, y_test, X_test, clf, c1, c2, modelid):
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    cost = c1*fp + c2*fn
    print("Total cost:", cost)
    print()
    
    f1s = f1_score(y_test, y_pred)
    pres = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("F1 score" , round(f1s, 3))
    print("Precision: ", round(pres, 3))
    print("Recall", round(recall, 3))
    print()
    print("False positives: ",fp)
    print("False negatives: ",fn)
    print()
 
    print("Confusion matrix, without normalization")
    plot_confusion_matrix(clf, X_test, y_test, values_format = "d", cmap = "Greens")
    plt.savefig("figures/cm"+modelid+".png", bbox_inches = "tight")
    plt.show()  

    print("     Normalized confusion matrix")
    plot_confusion_matrix(clf, X_test, y_test, cmap = "Blues",  normalize="true")
    plt.savefig("figures/cm_normalized"+modelid+".png", bbox_inches = "tight")
    plt.show() 

    return cost, f1s, pres, recall

def final_steps(X_train, y_train, X_test, y_test, algo, parameters, score, modelid):
    # grid search
    clf = GridSearchCV(estimator=algo, param_grid=parameters, scoring=score, n_jobs=-1, refit=True, error_score=0)
    print("-----------cross-validation: grid search")
    # train
    clf.fit(X_train, y_train)
    print("-----------training")
    # predict
    y_pred = clf.predict(X_test)
    print("-----------prediction")
    print()

    print("Best parameters set found on development set:")
    print(clf.best_params_)
    print()
    print("\033[1m"+"Model evaluation"+"\033[0m")
    print()
    cost, f1s, pres, recall = evaluation_metrics(y_pred, y_test, X_test, clf, 10, 500, modelid)
    return cost, f1s, pres, recall

def final_pipeline(train, test, sample_technique, algorithm, parameters,score, modelid):
    print(modelid)
    # basic prepocessing
    """format classes as pos: 1, neg: 0
    convert na in NaN values"""
    train = data_prepro(train)
    test = data_prepro(test)
    print("-----------basic preprocessing")
    
    # null values
    """drop features with more than t% of NaN on the trainset
    fill the remaining nulls with the mean of the column"""
    [y_train, X_train, y_test, X_test] = null_values(train, test, t=0.5)
    print("-----------null values")
    
    # standarize
    """RobustScaler, mean = 0, var = 1 robust against outliers
    the fit is only on the train set, and applied to both sets"""
    X_train, X_test = standardize(X_train, X_test)
    print("-----------standarization")
    
    # feature selection
    """drop zero-variance features, select relevant features with tree-based algorithm
    drop highly correlated features, select the same features for the test set"""
    X_train = feature_selection(X_train, y_train)
    X_test = X_test[X_train.columns]
    print("-----------feature selection")

    # sample tecnique
    """applied only on the train set"""
    X_train, y_train = resample(sample_technique, X_train, y_train)
    
    cost, f1s, pres, recall = final_steps(X_train, y_train, X_test, y_test,algorithm, parameters, score,  modelid)