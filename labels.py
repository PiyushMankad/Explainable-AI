# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 14:58:55 2020

@author: Piyush
"""
import pandas as pd

data = pd.read_csv(r"D:\Intelligent Systems\Dissertation ####\Explainable-AI\subdata.csv").iloc[:,0]
labels = data.unique()

