import pandas as pd
import numpy as np
import re
from sklearn import cross_validation as cv
from sklearn.linear_model import LogisticRegression

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


'''*********************** Helper Functions ***********************'''

def get_title(name):
	'''retrieves title (ie. Mrs., Mr.) from name string'''
	title = re.search('[A-Za-z]+\.', name)
	try:
		return title.group(0)
	except:
		return ''

'''********** Converting non-numerical columns to numbers **********'''

#Converts the Sex column 
train.loc[train['Sex'] == 'female', "Sex"] = 1
train.loc[train['Sex'] == 'male', "Sex"] = 0
test.loc[test['Sex'] == 'female', "Sex"] = 1
test.loc[test['Sex'] == 'male', "Sex"] = 0

#Converts the Embarked column
train.loc[train['Embarked'] == 'S', 'Embarked'] = 0
train.loc[train['Embarked'] == 'C', 'Embarked'] = 1
train.loc[train['Embarked'] == 'Q', 'Embarked'] = 2
test.loc[test['Embarked'] == 'S', 'Embarked'] = 0
test.loc[test['Embarked'] == 'C', 'Embarked'] = 1
test.loc[test['Embarked'] == 'Q', 'Embarked'] = 2


'''********************* Filling missing Data *********************'''

#fills missing ages for males using male median age'''
train.loc[train['Sex'] == 0, 'Age'] = train.loc[train['Sex'] == 0, 'Age'].fillna(train[train['Sex'] == 0]['Age'].median())
test.loc[test['Sex'] == 0, 'Age'] = test.loc[test['Sex'] == 0, 'Age'].fillna(test[test['Sex'] == 0]['Age'].median())

#fills missing ages for females using female median age'''
train.loc[train['Sex'] == 1, 'Age'] = train.loc[train['Sex'] == 1, 'Age'].fillna(train[train['Sex'] == 1]['Age'].median())
test.loc[test['Sex'] == 1, 'Age'] = test.loc[test['Sex'] == 1, 'Age'].fillna(test[test['Sex'] == 1]['Age'].median())

#fills missing embarked locations using the most common: 0'''
train["Embarked"] = train["Embarked"].fillna(0)
test["Embarked"] = test["Embarked"].fillna(0)

#fills other potentially missing columns
test['Fare'] = test['Fare'].fillna(test['Fare'].median())
test['Pclass'] = test['Pclass'].fillna(test['Pclass'].median())
test['SibSp'] = test['SibSp'].fillna(test['SibSp'].median())
test['Parch'] = test['Parch'].fillna(test['Parch'].median())


'''************************** Prediction **************************''' 


alg = LogisticRegression(random_state = 1)
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

alg.fit(train[predictors], train['Survived'])

predictions = alg.predict(test[predictors])

submission = pd.DataFrame({
	'PassengerId' : test['PassengerId'],
	'Survived' : predictions
	})

submission.to_csv('kaggle.csv', index = False)