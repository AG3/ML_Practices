import pandas as pd

a = pd.read_csv("3-layer-dense-submission.csv")
b = pd.read_csv("rdm-forest-submission.csv")

aa = a.values
bb = b.values
k=0
for i in range(len(aa)):
    if aa[i][1] != bb[i][1]:
        k+=1
print(k)
print(k/len(aa))