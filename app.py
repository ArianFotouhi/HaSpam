import numpy as np
import pandas as pd




df = pd.read_csv('./TextFiles/smsspamcollection.tsv',sep='\t')
df.head()

#showing any possible missing sample
print('missing values: ',df.isnull().sum())

#showing the possbile labels
print('Categories: ',df['label'].unique())

#showing the number of each labels
print('No. of each category: ',df['label'].value_counts())

#visualization
import matplotlib.pyplot as plt
#visualization of text label based on text length
plt.xscale('log')
bins = 1.15**(np.arange(0,50))
plt.hist(df[df['label']=='ham']['length'], bins=bins,alpha=0.8)
plt.hist(df[df['label']=='spam']['length'], bins=bins,alpha=0.8)
plt.legend(('ham','spam'))
plt.show()
#result: usually spams are longer in text compared to hams


#visualization of text label based on text punctuation
plt.xscale('log')
bins = 1.15**(np.arange(0,50))
plt.hist(df[df['label']=='ham']['punct'], bins=bins,alpha=0.8)
plt.hist(df[df['label']=='spam']['punct'], bins=bins,alpha=0.8)
plt.legend(('ham','spam'))
plt.show()
#result: a small tendency of spam towards more punctutation (not a firm inference)

#Data Preprocessing

#features
X = df[['length','punct']]

#labels
y = df['label']

#data split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

#shape of train and test data
print('train data shape:', X_train.shape)
print('test data shape:', X_test.shape)

#ML model 
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(solver='lbfgs')

lr_model.fit(X_train,y_train)

#Accuracy
from sklearn import metrics
predictions = lr_model.predict(X_test)

