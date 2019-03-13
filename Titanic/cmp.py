import pandas as pd

a = pd.read_csv("cleaned_sub - 1.csv")
b = pd.read_csv("cleaned_sub.csv")

c = pd.concat([a,b],axis=1)
print(c)