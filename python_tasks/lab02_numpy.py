import numpy as np
from numpy.lib.financial import npv
from scipy import stats

an_array = np.array([0,1,2,3,4,3,2,1,0,3]) #1 dimensional (vector)
print(an_array)

'''
Creating a matrix, 2D array.
'''
c = np.array([(1,3,3,7), (4,2,4,2)])
print(c)
print(c[0])
print(c[0][1])

'''
Grabbing columns from matrix
'''
print(c[:,0])

for x in range(an_array.size):
    print(an_array[x])


'''
iterate over each element in a matrix
'''
for row in c:
    print(row)

for row in c:
    for cell in row:
        print(cell, end=' ')


for cell in c.flatten():
    print(cell, end=' ')

'''
.shape attribute tells you the rows, columns in a data structure
'''
print(c.shape)

print(np.inf) # infinity
print(np.nan) # not a number

a = np.arange(20) # creates an array from 0 - [n-1] aka 0-19
print(a)

b = np.zeros((2,2))
print(b)

b = np.ones((2,2))
print(b)

b = np.full((4,4), 50) # fills 4x4 matrix with 50 in each element
print(b)

print(np.eye(5)) #5x5 identity matrix

'''
Vector & matrix operations in python.
'''

v = np.arange(4)
print("v     =", v)
print("v + 5 =", v + 5)
print("v - 5 =", v - 5)
print("v * 2 =", v * 2)
print("v / 2 =", v / 2)

print("________________________________")

print(np.amin(an_array)) #minimum value in list

print(np.amax(an_array)) #maximum value in list

print(np.mean(an_array)) #mean of list

print(np.median(an_array))

m = stats.mode(an_array)

print(m)