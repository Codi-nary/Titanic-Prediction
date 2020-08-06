import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot  as plt
from sklearn import preprocessing as pr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('/Users/navin.jain/Desktop/PycharmProjects/Titanic/train.csv')
test = pd.read_csv('/Users/navin.jain/Desktop/PycharmProjects/Titanic/test.csv')

data = data[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked',
             'Survived']]
data['Train'] = 1
test['Train'] = 0
test['survived'] = np.nan

data_combined = pd.concat([data, test])

data = data.rename(columns={'Pclass': 'Ticket Class', 'SibSp': 'Siblings', 'Parch': 'Parents'})

data_description = data.describe()

data.describe().columns

data.astype({'Ticket Class': 'object', 'Survived': 'object'}).dtypes

num_columns = data[['Age', 'Siblings', 'Parents', 'Fare']]

cat_columns = data[['Ticket Class', 'Survived', 'Sex', 'Cabin', 'Embarked', 'Ticket']]

# plot numeric values for distribution
# Fare, Parents and siblings are not normalized

for i in num_columns:
    sns.distplot(num_columns[i])
    plt.show()

# plot categorical variables for distribution
# There are a lot of categories in Cabin and Ticket

for i in cat_columns:
    sns.barplot(cat_columns[i].value_counts().index, cat_columns[i].value_counts()).set_title(i)
    plt.show()

# heat map/ Pair plot for correlation

sns.pairplot(num_columns)
plt.show()
num_columns.corr()

# extract the title

t = []
for i in data['Name']:
    a = i.split(',')[1].split('.')[0].strip()
    t.append(a)
data['Title'] = t

data['Title'].value_counts()

# Data cleaning by combining into main titles

# Desc of the other titles
Other_title = data.loc[data['Title'].isin(['Ms', 'Don', 'Capt', 'Mme', 'Jonkheer', 'Sir',
                                           'Lady', 'the Countess', 'Mile', 'Col', 'Major', 'Jonkheer', 'Mlle'
                                              , 'Rev', 'Dr'])].sort_values(by='Title')

Miss = ['Ms', 'Mme']
Mr = ['Capt', 'Col', 'Don', 'Rev', 'Major', 'Sir', 'Jonkheer']
Mrs = ['the Countess', 'Mme', 'Lady', 'Mlle']

data['Title'] = data['Title'].replace(Miss, 'Miss')
data['Title'] = data['Title'].replace(Mr, 'Mr')
data['Title'] = data['Title'].replace(Mrs, 'Mrs')

data['Title'] = np.where((data['Title'] == 'Dr') & (data['Sex'] == 'female'), 'Mrs', data['Title'])
data['Title'] = np.where((data['Title'] == 'Dr') & (data['Sex'] == 'male'), 'Mr', data['Title'])

data.query('Sex=="male" & Title=="Dr"')

# Mean Age by Titles
means = data.groupby('Title')['Age'].mean()

# fill Nan by mean values
data['Age'] = np.where((np.isnan(data['Age'])) & (data['Title'] == 'Master'), means[0], data['Age'])
data['Age'] = np.where((np.isnan(data['Age'])) & (data['Title'] == 'Miss'), means[1], data['Age'])
data['Age'] = np.where((np.isnan(data['Age'])) & (data['Title'] == 'Mr'), means[2], data['Age'])
data['Age'] = np.where((np.isnan(data['Age'])) & (data['Title'] == 'Mrs'), means[3], data['Age'])

# redefine numeric and categorical values
num_columns = data[['Age', 'Siblings', 'Parents', 'Fare']]

cat_columns = data[['Ticket Class', 'Survived', 'Sex', 'Cabin', 'Embarked', 'Ticket']]

# plot numerical data distribution

fig, axs = plt.subplots(ncols=1, figsize=(10, 8))
axs.set_title = ('Original Distributions')

for i in num_columns:
    sns.kdeplot(num_columns[i], ax=axs)
plt.show()

# Normalize the distribution using MinMax Scaler
# ( MinMaxScaler subtracts the column mean from each value and then divides by the range)

mm_scaler = pr.MinMaxScaler()

num_columns = mm_scaler.fit_transform(num_columns)

num_columns = pd.DataFrame(num_columns, columns=['Age', 'Siblings', 'Parents', 'Fare'])

cat_columns = data[['Ticket Class', 'Survived', 'Sex', 'Cabin', 'Embarked', 'Ticket']]

# plot normalized dist
fig, axs = plt.subplots(ncols=1, figsize=(10, 8))
axs.set_title = ('Norm Distributions')

for i in num_columns:
    sns.kdeplot(num_columns[i], ax=axs)
plt.show()

# More feature engineering
# Travelling Alone or not

data['Travelling_alone'] = np.where((data['Parents'] == 0) & data['Siblings'] == 0, 1, 0)

# Cabin first letter
data['Cabin'].astype(str)

data.dtypes

data['Cabin_area'] = np.nan

a = []
for i in data['Cabin']:
    a.append(str(i)[0])

data['Cabin_area'] = a

mean_fare = data.groupby('Cabin_area')['Fare'].mean()

# fill null cabins with the value closes to their mean = F

data['Cabin_area'] = np.where((data['Cabin_area'] == 'n'), 'F', data['Cabin_area'])

# all_data transformation

t = []
for i in data_combined['Name']:
    a = i.split(',')[1].split('.')[0].strip()
    t.append(a)
data_combined['Title'] = t

Other_title = data_combined.loc[data_combined['Title'].isin(['Ms', 'Don', 'Capt', 'Mme', 'Jonkheer', 'Sir',
                                                             'Lady', 'the Countess', 'Mile', 'Col', 'Major', 'Jonkheer',
                                                             'Mlle'
                                                                , 'Rev', 'Dr'])].sort_values(by='Title')

Miss = ['Ms', 'Mme']
Mr = ['Capt', 'Col', 'Don', 'Rev', 'Major', 'Sir', 'Jonkheer']
Mrs = ['the Countess', 'Mme', 'Lady', 'Mlle']

data_combined['Title'] = data_combined['Title'].replace(Miss, 'Miss')
data_combined['Title'] = data_combined['Title'].replace(Mr, 'Mr')
data_combined['Title'] = data_combined['Title'].replace(Mrs, 'Mrs')

data_combined['Title'] = np.where((data_combined['Title'] == 'Dr') & (data_combined['Sex'] == 'female'), 'Mrs',
                                  data_combined['Title'])
data_combined['Title'] = np.where((data_combined['Title'] == 'Dr') & (data_combined['Sex'] == 'male'), 'Mr',
                                  data_combined['Title'])

data_combined.query('Sex=="male" & Title=="Dr"')

# Mean Age by Titles
means = data_combined.groupby('Title')['Age'].mean()

# fill Nan by mean values
data_combined['Age'] = np.where((np.isnan(data_combined['Age'])) & (data_combined['Title'] == 'Master'), means[0],
                                data_combined['Age'])
data_combined['Age'] = np.where((np.isnan(data_combined['Age'])) & (data_combined['Title'] == 'Miss'), means[1],
                                data_combined['Age'])
data_combined['Age'] = np.where((np.isnan(data_combined['Age'])) & (data_combined['Title'] == 'Mr'), means[2],
                                data_combined['Age'])
data_combined['Age'] = np.where((np.isnan(data_combined['Age'])) & (data_combined['Title'] == 'Mrs'), means[3],
                                data_combined['Age'])

# Normalize the distribution using MinMax Scaler
# ( MinMaxScaler subtracts the column mean from each value and then divides by the range)

data_combined = data_combined.rename(columns={'Pclass': 'Ticket Class', 'SibSp': 'Siblings', 'Parch': 'Parents'})

mm_scaler = StandardScaler()

data_combined[['norm_sibling', 'norm_fare', 'norm_Parents', 'norm_age']] = mm_scaler.fit_transform(
    data_combined[['norm_sibling', 'norm_fare', 'norm_Parents', 'norm_age']])

# plot normalized dist
fig, axs = plt.subplots(ncols=1, figsize=(10, 8))
axs.set_title = ('Norm Distributions')

num_columns = data_combined[['norm_age', 'norm_sibling', 'norm_Parents', 'norm_fare']]

for i in num_columns:
    sns.kdeplot(num_columns[i], ax=axs)
plt.show()

# data_combined['norm_fare'] = mm_scaler.fit_transform(np.array(data_combined['Fare']).reshape(-1, 1))
# data_combined['norm_Parents'] = mm_scaler.fit_transform(np.array(data_combined['Parents']).reshape(-1, 1))
# data_combined['norm_age'] = mm_scaler.fit_transform(np.array(data_combined['Age']).reshape(-1, 1))

data_combined['Ticket Class'] = data_combined['Ticket Class'].astype(str)

data_combined['Travelling_alone'] = np.where((data_combined['Parents'] == 0) & data_combined['Siblings'] == 0, 1, 0)

# Cabin fill Na's

data_combined['Cabin_area'] = np.nan

a = []
for i in data_combined['Cabin']:
    a.append(str(i)[0])

data_combined['Cabin_area'] = a

mean_fare = data_combined.groupby('Cabin_area')['Fare'].mean()

# fill null cabins with the value closes to their mean = F

data_combined['Cabin_area'] = np.where((data_combined['Cabin_area'] == 'n'), 'F', data_combined['Cabin_area'])

# drop NA/Embarked

data_combined = data_combined.dropna(subset=['Embarked'])

# Create dummy variables

data_dummy = pd.get_dummies(
    data_combined[['Ticket Class', 'Sex', 'norm_age', 'norm_sibling', 'norm_Parents', 'norm_fare',
                   'Embarked', 'Cabin_area', 'Travelling_alone', 'Title', 'Train']])

# split into train and test

train = data_dummy[data_dummy.Train == 1].drop(['Train'], axis=1)
test = data_dummy[data_dummy.Train == 0].drop(['Train'], axis=1)

test_1 = data_combined[data_combined.Train == 0].drop(['Train'], axis =1)

#  create y train for target feature
y_train = data_combined[data_combined.Train == 1].Survived

# Baseline models
# Naive Bayes
nb = GaussianNB()
cv = cross_val_score(nb, train, y_train, cv=5)
print(cv)
print(cv.mean())

# logistic regression
lr = LogisticRegression(max_iter=2000)
cv = cross_val_score(lr, train, y_train, cv=5)
print(cv)
print(cv.mean())

# decision tree
dt = tree.DecisionTreeClassifier(random_state=1)
cv = cross_val_score(dt, train, y_train, cv=5)
print(cv)
print(cv.mean())

# knn
knn = KNeighborsClassifier()
cv = cross_val_score(knn, train, y_train, cv=5)
print(cv)
print(cv.mean())

# randomforest
rf = RandomForestClassifier(random_state=1)
cv = cross_val_score(rf, train, y_train, cv=5)
print(cv)
print(cv.mean())
print(rf.feature_importances_)

# Voting classifiers
from sklearn.ensemble import VotingClassifier

voting = VotingClassifier(estimators=[('nb', nb), ('lr', lr), ('dt', dt), ('knn', knn), ('rf', rf)], voting='hard')

cv = cross_val_score(voting, train, y_train, cv=5)
print(cv.mean())

# feature selection and testing again
# feature importance

# randomforest
rf = RandomForestClassifier(random_state=1)
rf.fit(train, y_train)
imp_feat = pd.Series(rf.feature_importances_, index=train.columns)
imp_feat.nlargest(26).plot(kind='barh')
considering_feat = imp_feat.nlargest(20)
plt.show()


remove_feat = []
for i , v in considering_feat.iteritems():
    remove_feat.append(i)

# take only top 20 features and train again

for i in train.columns:
    if i not in remove_feat:
        del train[i]

# train the model again
# Naive Bayes
nb = GaussianNB()
cv = cross_val_score(nb, train, y_train, cv=5)
print(cv)
print(cv.mean())

# logistic regression
lr = LogisticRegression(max_iter=2000)
cv = cross_val_score(lr, train, y_train, cv=5)
print(cv)
print(cv.mean())

# decision tree
dt = tree.DecisionTreeClassifier(random_state=1)
cv = cross_val_score(dt, train, y_train, cv=5)
print(cv)
print(cv.mean())

# knn
knn = KNeighborsClassifier()
cv = cross_val_score(knn, train, y_train, cv=5)
print(cv)
print(cv.mean())

# randomforest
rf = RandomForestClassifier(random_state=1)
cv = cross_val_score(rf, train, y_train, cv=5)
print(cv)
print(cv.mean())

# Voting classifiers
from sklearn.ensemble import VotingClassifier

voting = VotingClassifier(estimators=[('nb', nb), ('lr', lr), ('dt', dt), ('knn', knn), ('rf', rf)], voting='hard')

cv = cross_val_score(voting, train, y_train, cv=5)
print(cv.mean())

voting.fit(train,y_train)

# test set with 20 features

for i in test.columns:
    if i not in remove_feat:
        del test[i]

# prediction on test dataset ( there was 1 nan idk why so used .mean to fill that) 
test = test.fillna(test.mean())
test_prediction = voting.predict(test).astype(int)
submission = {'Passenger_id': test_1.PassengerId, 'Survived': test_prediction }
submission = pd.DataFrame(data =submission)

submission.Survived.value_counts()

submission.to_csv('titanic_pred.csv')