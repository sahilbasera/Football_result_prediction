import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np

"""
This is a Machine Learning algorithm to predict the outcome of a football game based on stats like half time scores , fouls , shots , cards and odds given by betting companies
like Bet365 . 


"""
#to read csv file and store it in data list
data = pd.read_csv('E0.csv')

# total columns 
n_matches = data.shape[0]

#total rows
n_features = data.shape[1] - 1

n_homewins = len(data[data['FTR'] == 'H'])

win_rate = (float(n_homewins) / n_matches )* 100
print("\n\n\n\n")
print(" This is an algorithm to predict the outcome of a football game from \n the Barclays Premier League \n")
print("\n\n")
print("="*50)
print("The total matches in the dataset are : " + str(n_matches) + "\n")
print("The total features being considered are : " + str(n_features) + "\n")
print("The total home wins for a team are  : " + str(n_homewins) + "\n")
print(" The win rate for a home team is : " + str(win_rate) + "\n")
print("="*50)
print("\n\n\n\n\n")
Y_all = data['FTR'] # stores the match outcome (i.e. win , loss or draw ) [LABELS]
X_all = data.drop(['FTR' , 'FTHG' , 'FTAG'] , 1) #stores everything else [FEATURES]



#print(Y_all)
"""

NOTE : If you wanna print a particular colummn say 'FTR'
Do this:
print(data[['FTR']]
Here always two square brackets are being used
but above in Y_all data[['FTR']] is not done and if you try and print y_all it will show error
this was because it kept showing error that classifier wanted an array in y but got a vector
i.e wanted something like this = [1 , 2 , 3, 4] but got something like this = [[1] , [2] , [3] , [4]]

"""
# used to scale data i.e. make it of uniform and comparable sizes
"""from sklearn.preprocessing import scale

cols =[['FTHG' , 'FTAG' , 'HTHG' , 'HTAG' , 'B365H' , 'B365A']]
for col in cols :
    X_all[col] = scale(X_all[col])
"""
X_all1 = data[[ 'HTR' , 'HS' , 'AS' , 'HST' , 'AST'  , 'HC' ,  'AC' , 'HF' , 'AF' , 'HY' , 'AY' , 'HR' , 'AR' , 'HTHG' , 'HTAG' , 'B365H' , 'B365A' , 'B365D' , 'BWH' , 'BWD' , 'BWA' , 'IWH' , 'IWA' , 'IWD' , 'WHH' , 'WHA' , 'WHD', 'LBH' , 'LBA' , 'LBD' , 'PSH' , 'PSA' , 'PSD' ]]
#Y_all1 = np.asarray(Y_all)


def preprocess_features(X):
 output = pd.DataFrame(index = X.index)

 for col , col_data in X.iteritems():
     if col_data.dtype == object :
      col_data = pd.get_dummies(col_data , prefix = col)

     output = output.join(col_data)
 return output
X_a = preprocess_features(X_all)
#print(X_a)


#print(Y_all)
from sklearn.cross_validation import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X_a , Y_all , test_size = 0.15, random_state = 2)

clf = LogisticRegression()
clf.fit(X_train , Y_train)
print("The actual outcome of the test cases is    : " +str(Y_test) + "\n")
print("The predicted outcome of the test cases is : " +str(clf.predict(X_test)) +"\n")


print("The accuracy of the given algorithm is : " + str(clf.score(X_test , Y_test)))
