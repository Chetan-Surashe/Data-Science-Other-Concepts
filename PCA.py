# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 15:36:13 2023

@author: Chetan Surashe
"""
import pandas as pd
import numpy as np
uni1=pd.read_excel("C:/1-maths/University_Clustering.xlsx")
uni1.describe()
uni1.info()
uni=uni1.drop(["State"],axis=1)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

#considering only numerical data
uni.data=uni.iloc[:, 1:]

#normalizing the numerical data
uni_normal = scale(uni.data)
uni_normal

pca=PCA(n_components =6)
pca_values=pca.fit_transform(uni_normal)

var = pca.explained_variance_ratio_
var

#PCA weights
#pca.components_
#pca.components__[0]

#cumulative variance
var1 = np.cumsum(np.round(var, decimals=4)*100)
var1

#variance plot for PCA components obtained
plt.plot(var1, color= "red")

#PCA scores
pca_values


pca_data =pd.DataFrame(pca_values)
pca_data.columns="comp0","comp1","comp2","comp3","comp4","comp5"
final=pd.cocat([uni.Univ ,pca_data.iloc[:, 0:3]], axis=1)

#this is Univ  column of uni data frame

#scatter diagram

import matplotlib.pylab as plt
ax =final.plot(x="comp0",y="comp1",kind="scatter",figsize=(12,8))
final[['comp0','comp1','Univ']].apply(lambda x: ax.text(*x), axis=1)







