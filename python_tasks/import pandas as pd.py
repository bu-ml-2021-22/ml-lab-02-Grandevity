import pandas as pd
import matplotlib.pyplot as plt

data = { # python dictionary
    'apples': [3,2,0,1],
    'oranges':[0,3,7,2]
}

purchases_dataframe = pd.DataFrame(data)
print(purchases_dataframe)

dataset = pd.read_csv('data/ml-assessment-dataset.csv')
print(dataset.head(10)) # displays the header and top half of the dataset

print(dataset.tail(10))

print(dataset.shape)
print(dataset.info())
print(dataset.describe())
print(dataset['Class'].describe(),"\n")
print(dataset['Total mark'].describe(),"\n")

print(dataset['Class'].value_counts()) # gets the numbers of distribution of features

dataset['L4 mean mark'].plot(kind="hist", title='Level 4 mean mark distribution');

dataset.plot(kind='scatter', x='L4 mean mark', y='Total mark', title='Level 4 mean mark vs unit mark');

dataset['Class'].value_counts().plot(kind='bar', title='Class distribution');

i = 0
for row in dataset['Total mark']:
    if row > 39:
        i+=1
print("Students passed:",i)
        
    
        



# Data analysis

#6.1
#How many instances are there in the dataset? - 60
#Is the class balance equal? - Yes, 30,30
#Which class is the MOST represented in the dataset? - idk
#How many students passed? - 31


