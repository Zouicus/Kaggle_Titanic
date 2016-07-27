import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn import cross_validation as cv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as rf

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


def agegroup(age):
	'''creates age groups'''
	agegp = [9, 18, 30, 40, 50, 60, 100]
	for i in agegp:
		if age <= i:
			return agegp.index(i)
			break


def noble(title):
	'''separates into nobility and commoners by title'''
	tage = [['Mlle.', 'Ms.', 'Mme', 'Miss.', 'Mr.', 'Mrs.', 'Dr.', 'Rev.'], ['Countess.', 'Jonkheer.','Capt.', 'Col.', 'Sir.', 'Lady.', 'Major.', 'Don.', 'Dona.']]
	for i in tage:
		if title in i:
			return tage.index(i)
			break



def feature_engineer(df):
	#creates the titles column
	df['Title'] = df['Name'].apply(get_title)

	#creates the family size column
	df['FSize'] = df['Parch'] + df['SibSp'] + 1

	#creates noble column
	df['Noble'] = df['Title'].apply(noble)
	df['Noble'] = df['Noble'].fillna(1)

	#Converts the Sex column to numerical values
	df.loc[df['Sex'] == 'female', "Sex"] = 1
	df.loc[df['Sex'] == 'male', "Sex"] = 0
	
	#Converts the Embarked column to numerical values
	df.loc[df['Embarked'] == 'S', 'Embarked'] = 0
	df.loc[df['Embarked'] == 'C', 'Embarked'] = 1
	df.loc[df['Embarked'] == 'Q', 'Embarked'] = 2
	
	#fills missing ages for males using male median age'''
	df.loc[df['Sex'] == 0, 'Age'] = df.loc[df['Sex'] == 0, 'Age'].fillna(df[df['Sex'] == 0]['Age'].median())

	#fills missing ages for females using female median age'''
	df.loc[df['Sex'] == 1, 'Age'] = df.loc[df['Sex'] == 1, 'Age'].fillna(df[df['Sex'] == 1]['Age'].median())

	#fills missing embarked locations using the most common: 0'''
	df["Embarked"] = df["Embarked"].fillna(0)

	#fills other potentially missing columns
	df['Fare'] = df['Fare'].fillna(df['Fare'].median())
	df['Pclass'] = df['Pclass'].fillna(df['Pclass'].median())
	df['SibSp'] = df['SibSp'].fillna(df['SibSp'].median())
	df['Parch'] = df['Parch'].fillna(df['Parch'].median())

	#creates age group column
	df['AgeGp'] = df['Age'].apply(agegroup)
	
	#Converts Title column to numerical values
	tls = df['Title'].unique().tolist()
	for i in tls:
		df.loc[df['Title'] == i, 'Title'] = tls.index(i)

	return df


train = feature_engineer(train)
test = feature_engineer(test)


'''************************** Prediction **************************''' 
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", 'FSize', 'AgeGp', 'Noble', 'Title']

alg = LogisticRegression(random_state = 1)
#alg = rf(random_state = 1, n_estimators = 1555, min_samples_split = 4, min_samples_leaf = 3)
scores = cv.cross_val_score(alg, train[predictors], train['Survived'])
print(np.mean(scores))

alg.fit(train[predictors], train['Survived'])

predictions = alg.predict(test[predictors])

submission = pd.DataFrame({
	'PassengerId' : test['PassengerId'],
	'Survived' : predictions
	})

submission.to_csv('kaggle.csv', index = False)

