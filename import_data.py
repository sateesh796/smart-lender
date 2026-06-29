import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.tree import decisiontreeclassifier
from sklearn.ensemble import  GradientBoostingClassifier
from sklearn .neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
import imblearn
from sklearn.model_selection import train_test_split
from sklearn .preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score
#import the dataset
data = pd.read_csv('/Users/sarayu/Downloads/loan_prediction.csv')


