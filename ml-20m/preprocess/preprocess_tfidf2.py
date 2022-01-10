# processing title (removing movie year)
import pandas as pd

df = pd.read_csv('/tmp2/b07902053/ml-20m/itdf.csv')

# you can also use df.apply()
df1 = df.copy()
ti = []
for i in df1['title']:
    ti.append(i.split(' (')[0])
df1['title'] = ti

df1.to_csv('/tmp2/b07902053/ml-20m/itdf1.csv', index=False)
print("write to /tmp2/b07902053/ml-20m/itdf1.csv")
