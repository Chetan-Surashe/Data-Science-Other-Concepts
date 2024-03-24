# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 16:37:11 2023

@author: Chetan Surashe
"""

import numpy as np
from numpy import array
from scipy.linalg import svd

A=array([[1,0,0,0,2],[0,0,3,0,0],[0,0,0,0,0],[0,4,0,0,0]])

print(A)
U,d,vt=svd(A)
print(U)
print(d)
print(vt)
print(np.diag(d))

#svd applying to dataset
import pandas as pd
data=pd.read_excel("C:/Data Set/University_Clustering.xlsx")
data.head()
data=data.iloc[:,2:]
#remove non numeric data

from sklearn.decomposition import TruncatedSVD
svd=TruncatedSVD(n_components=3)
svd.fit(data)
result=pd.DataFrame(svd.transform(data))
result.head()
result.columns='pc0','pc1','pc2'
result.head()

#scatter diagram

import matplotlib.pylab as plt
plt.scatter(x=result.pc0,y=result.pc1)