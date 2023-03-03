import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('./TextFiles/smsspamcollection.tsv',sep='\t')
df.head()

#showing any possible missing sample
print('missing values: ',df.isnull().sum())

#showing the possbile labels
print('Categories: ',df['label'].unique())

#showing the number of each labels
print('No. of each category: ',df['label'].value_counts())

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