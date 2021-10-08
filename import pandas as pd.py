import pandas as pd

data = { # python dictionary
    'apples': [3,2,0,1],
    'oranges':[0,3,7,2]
}

purchases_dataframe = pd.DataFrame(data)
print(purchases_dataframe)

dataset = pd.read_csv('data/ml-assessment-dataset.csv')
dataset.head() # displays the header and top half of the dataset
