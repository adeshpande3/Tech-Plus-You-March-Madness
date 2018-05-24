############################## IMPORTS ##############################

from __future__ import division
import sklearn
import pandas as pd
import numpy as np
import collections
import os.path
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn import tree
from sklearn.cross_validation import cross_val_score
from keras.utils import np_utils
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import sys
from sklearn.ensemble import GradientBoostingRegressor
import math
import csv
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV
import urllib
from sklearn.svm import LinearSVC
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from datetime import datetime

############################## LOAD TRAINING SET ##############################

if os.path.exists("Data/xTrain.npy") and os.path.exists("Data/yTrain.npy"):
	xTrain = np.load("Data/xTrain.npy")
	yTrain = np.load("Data/yTrain.npy")
else:
	print ('We need a training set!')
	sys.exit()

def loadTeamVectors(years):
	listDictionaries = []
	for year in years:
		curVectors = np.load("Data/TeamVectors/" + str(year) + "TeamVectors.npy").item()
		listDictionaries.append(curVectors)
	return listDictionaries

############################## LOAD CSV FILES ##############################

teams_pd = pd.read_csv('Data/Teams.csv')

############################## TRAIN MODEL ##############################

model = GradientBoostingRegressor(n_estimators=100, max_depth=5)

categories=['Wins','PPG','PPGA','PowerConf','3PG', 'APG','TOP','Conference Champ','Tourney Conference Champ',
           'Seed','SOS','SRS', 'RPG', 'SPG', 'Tourney Appearances','National Championships','Location']
accuracy=[]
X_train, X_test, Y_train, Y_test = train_test_split(xTrain, yTrain)
startTime = datetime.now() # For some timing stuff
results = model.fit(X_train, Y_train)
preds = model.predict(X_test)

preds[preds < .5] = 0
preds[preds >= .5] = 1
localAccuracy = np.mean(preds == Y_test)
print ("The accuracy is", localAccuracy)

############################## TEST MODEL ##############################

def predictGame(team_1_vector, team_2_vector, home, modelUsed):
    diff = [a - b for a, b in zip(team_1_vector, team_2_vector)]
    diff.append(home)
    return modelUsed.predict([diff])[0]

print ("======= Now it's your turn! =======")
while (True):
	team1 = raw_input('Enter the name of the 1st college team followed by the year (e.g UCLA 2017): ')
	team2 = raw_input('Enter the name of the 2nd college team followed by the year (e.g Duke 2017): ')

	team1Name = team1[:team1.find(" ")]
	team1Year = int(team1[team1.find(" ") + 1:])

	team2Name = team2[:team2.find(" ")]
	team2Year = int(team2[team2.find(" ") + 1:])

	years = [team1Year, team2Year]
	teamVectors = loadTeamVectors(years)
	year1Vectors = teamVectors[0]
	year2Vectors = teamVectors[1]

	team1Vector = year1Vectors[int(teams_pd[teams_pd['TeamName'] == team1Name].values[0][0])]
	team2Vector = year2Vectors[int(teams_pd[teams_pd['TeamName'] == team2Name].values[0][0])]
	prediction = predictGame(team1Vector, team2Vector, 0, model)

	print ("Probability that {0} wins is {1}".format(team1Name, prediction))