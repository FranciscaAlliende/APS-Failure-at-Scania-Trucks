# -*- coding: utf-8 -*-
"""Evaluation Metrics.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15cCeTJzCib37k4iDP40BzpzmovMsYHlO
"""

# Setup
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import f1_score, precision_score, recall_score

def evaluation_metrics(y_pred, y_test, X_test, clf, c1, c2):
  tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
  print("False positives: ",fp)
  print("False negatives: ",fn)
  cost = c1*fp + c2*fn
  print("Total cost:", cost)

  print("Confusion matrix, without normalization")
  plot_confusion_matrix(clf, X_test, y_test, values_format = "d", cmap = "Greens")  
  plt.show()  

  print("     Normalized confusion matrix")
  plot_confusion_matrix(clf, X_test, y_test, cmap = "Blues",  normalize="true")  
  plt.show() 

  disp = plot_precision_recall_curve(clf, X_test, y_test)
  disp.ax_.set_title('Precision-Recall curve')
  plt.show()

  f1s = f1_score(y_test, y_pred)
  pres = precision_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred)

  print("f1 score" , round(f1s, 3))
  print("precision: ", round(pres, 3))
  print("recall", round(recall, 3))

  return cost, f1s, pres, recall