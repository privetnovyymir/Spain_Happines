from sklearn.ensemble import VotingClassifier
import pandas as pd
import numpy as np
import asciitable

desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_colwidth', 15)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

happiness_df = pd.DataFrame(asciitable.
                            read('C:/Users/Ignacio'
                                 ' Ojeda/Documents/_NeverBackUp/EMPRESAS/BBVA/dataset_happiness.txt'))
happiness_catnum_df = pd.read_csv('C:/Users/Ignacio Ojeda/Documents/_NeverBackUp/EMPRESAS/BBVA/happiness_catnum_df.csv')
happiness_num_df = pd.read_csv('C:/Users/Ignacio Ojeda/Documents/_NeverBackUp/EMPRESAS/BBVA/happiness_catnum_df.csv')
print(happiness_df.dtypes, happiness_catnum_df.dtypes, happiness_num_df.dtypes)

''' As a classification problem but using VC Ensemble Model to try to power predictions '''
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as kNN


X_test = happiness_num_df[happiness_num_df['D.1_values'].isna()].select_dtypes([np.number]) # 4k obs
X_train = happiness_num_df[happiness_num_df['D.1_values'].notna()].select_dtypes([np.number]) # 36.3k obs
happiness_num_df = happiness_num_df.select_dtypes([np.number]).\
    drop(columns=['N_Entrevista'])
print(happiness_num_df.dtypes)
Xtr, Xtt = X_train, X_test
X_test, X_train = X_test.drop(columns=['D.1_values']).fillna(0), X_train.drop(columns=['D.1_values']).fillna(0)
y_train, y_test = np.array(Xtr['D.1_values']), np.array(Xtt['D.1_values'])
print(X_train.shape, len(y_train), X_test.shape, len(y_test))

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
knn = kNN(n_neighbors=10) # it1 (n=20 OR n=10, error = 0.02), it2 (n=100, error = 0.03)
knn.fit(X_train, y_train)
dt = DecisionTreeClassifier(max_depth=24, min_samples_leaf=0.05) # it1 <md=8, msl=0.15> ~ lightly underfitted,
# itn <md=24, msl=0.05> ~ lightly underfitted
dt.fit(X_train, y_train)

classifiers = [('Logistic Regression', logreg),
               ('K Nearest Neighbours', knn),
               ('Classification Tree', dt)]

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

for clf_name, clf in classifiers:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print('{:s} : {:.2f}'.format(clf_name, acc))
    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print('Confusion matrix:', cnf_matrix)
    print('{:s} - precision : {:.2f}'.format(clf_name, prec))
    print('{:s} - recall : {:.2f}'.format(clf_name, recall))

vc = VotingClassifier(estimators=classifiers)
vc.fit(X_train, y_train)
y_pred_VC = vc.predict(X_test)
print('Voting Classifier accuracy: {:.3f}'.format(accuracy_score(y_test, y_pred_VC)))

import matplotlib.pyplot as plt # sample visual testing

yx = np.arange(len(X_train['D.1_values']))
print('lengh X_train secondary vector:', len(yx))
print('lengh X_train Happiness:', len(X_train['D.1_values']))
plt.scatter(X_train['D.1_values'], yx)
plt.grid()
plt.show()

# Cross Validation process (CV MSE, Train set MSE and Test set MSE):
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error as MSE

# k - Fold method:
mse_cv = - cross_val_score(dt, X_train, y_train, cv=10, # training folds
                           scoring='neg_mean_squared_error',
                           n_jobs=-1) # all CPU cores usage

mse_cv_eval = mse_cv.mean()

y_pred_train_logreg = dt.predict(X_train) # needed for CV <k-Fold>
y_pred_test_logreg = dt.predict(X_test)
mse_train_logreg = MSE(y_train, y_pred_train_logreg)
mse_test_logreg = MSE(y_test, y_pred_test_logreg)

print('-- CV <k-Fold based method :: mse-based> | o/ Logistic Regression --')
print('cv error: {:.2f}'.format(mse_cv_eval))
print('y <non-intercept> training error: {:.2F}'.format(mse_train_logreg))
print('y <intercept> testing error: {:.2f}'.format(mse_test_logreg))

y_pred_train_knn = knn.predict(X_train) # needed for CV <k-Fold>
y_pred_test_knn = knn.predict(X_test)
mse_train_knn = MSE(y_train, y_pred_train_knn)
mse_test_knn = MSE(y_test, y_pred_test_knn)

print('-- CV <k-Fold based method :: mse-based> | o/ kNN --')
print('cv error: {:.2f}'.format(mse_cv_eval))
print('y <non-intercept> training error: {:.2F}'.format(mse_train_knn))
print('y <intercept> testing error: {:.2f}'.format(mse_test_knn))

y_pred_train_dt = dt.predict(X_train) # needed for CV <k-Fold>
y_pred_test_dt = dt.predict(X_test)
mse_train_dt = MSE(y_train, y_pred_train_dt)
mse_test_dt = MSE(y_test, y_pred_test_dt)

print('-- CV <k-Fold based method :: mse-based> | o/ Decision Tree Classifier --')
print('cv error: {:.2f}'.format(mse_cv_eval))
print('y <non-intercept> training error: {:.2F}'.format(mse_train_dt))
print('y <intercept> testing error: {:.2f}'.format(mse_test_dt))

