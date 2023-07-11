import pandas as pd 
from keras.models import Sequential
from keras.layers import Dense

dataset=pd.read_csv('books.csv')

x=dataset.iloc[:,[4,11]].values
y=dataset.iloc[:,3].values

print("The values of x are::",x)
print("The values of y are::",y)

model=Sequential()

model.add(Dense(114,input_dim=8,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(53,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.summary()