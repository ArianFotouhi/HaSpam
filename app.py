import numpy as np
import pandas as pd

df = pd.read_csv('./TextFiles/smsspamcollection.tsv',sep='\t')
df.head()

#this shows any possible missing sample
print(df.isnull().sum())

