# -*- coding: utf-8 -*-
"""basic_prepro.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MOmQXAdZADrZLlVHI-3CJ3O4vF09Mh91
"""

import pandas as pd
import numpy as np

def data_prepro(set):

  """ basic data prepropcesing common for all braches, """

  # replace in class column: pos = 1; neg = 0
  set['class'] = set['class'].map({'pos': 1, 'neg': 0})

  # replace the na and nan values with np.NaN
  set.replace(to_replace=['na','nan'],value = np.NaN,inplace=True)

  return set