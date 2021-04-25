#!/usr/bin/env python
# coding: utf-8

# # **Assignment For Numpy**

# Difficulty Level **Beginner**

# 1. Import the numpy package under the name np

# In[2]:


import numpy as np


# 2. Create a null vector of size 10 

# In[3]:


x=np.zeros(10)


# 3. Create a vector with values ranging from 10 to 49

# In[4]:


x=np.arange(10,50)


# 4. Find the shape of previous array in question 3

# In[5]:


x.shape


# 5. Print the type of the previous array in question 3

# In[6]:


x.dtype


# 6. Print the numpy version and the configuration
# 

# In[7]:


print(np.__version__)
print(np.show_config())


# 7. Print the dimension of the array in question 3
# 

# In[8]:


x.ndim


# 8. Create a boolean array with all the True values

# In[9]:


x = np.ones(10,dtype=bool)


# 9. Create a two dimensional array
# 
# 
# 

# In[10]:


x=np.random.randn(10,10)


# 10. Create a three dimensional array
# 
# 

# In[11]:


x=np.random.randn(4,4,4)


# Difficulty Level **Easy**

# 11. Reverse a vector (first element becomes last)

# In[12]:


x = np.arange(10, 50)
x[::-1]


# 12. Create a null vector of size 10 but the fifth value which is 1 

# In[13]:


x= np.arange(10) == 4
x=x * 1


# 13. Create a 3x3 identity matrix

# In[14]:


x=np.identity(3)
x


# 14. arr = np.array([1, 2, 3, 4, 5]) 
# 
# ---
# 
#  Convert the data type of the given array from int to float 

# In[15]:


arr = np.array([1, 2, 3, 4, 5])
arr = arr.astype('float64')


# 15. arr1 =          np.array([[1., 2., 3.],
# 
#                     [4., 5., 6.]])  
#                       
#     arr2 = np.array([[0., 4., 1.],
#      
#                    [7., 2., 12.]])
# 
# ---
# 
# 
# Multiply arr1 with arr2
# 

# In[16]:


arr1 = np.array([[1., 2., 3.],

            [4., 5., 6.]])
arr2 = np.array([[0., 4., 1.],

           [7., 2., 12.]])
arr1*arr2


# 16. arr1 = np.array([[1., 2., 3.],
#                     [4., 5., 6.]]) 
#                     
#     arr2 = np.array([[0., 4., 1.], 
#                     [7., 2., 12.]])
# 
# 
# ---
# 
# Make an array by comparing both the arrays provided above

# In[17]:


arr1 = np.array([[1., 2., 3.],

            [4., 5., 6.]]) 
arr2 = np.array([[0., 4., 1.],

            [7., 2., 12.]])
x=arr2>arr1


# 17. Extract all odd numbers from arr with values(0-9)

# In[18]:


arr=np.arange(0,10)
arr[arr % 2 == 1]


# 18. Replace all odd numbers to -1 from previous array

# In[19]:


x=np.where(arr % 2 == 0,arr,-1)


# 19. arr = np.arange(10)
# 
# 
# ---
# 
# Replace the values of indexes 5,6,7 and 8 to **12**

# In[20]:


arr = np.arange(10)
arr[5:9]=12


# 20. Create a 2d array with 1 on the border and 0 inside

# In[21]:


import numpy as np
x = np.ones((5,5))
x[1:-1,1:-1] = 0


# Difficulty Level **Medium**

# 21. arr2d = np.array([[1, 2, 3],
# 
#                     [4, 5, 6], 
# 
#                     [7, 8, 9]])
# 
# ---
# 
# Replace the value 5 to 12

# In[22]:


arr2d = np.array([[1, 2, 3],

            [4, 5, 6], 

            [7, 8, 9]])
arr2d= np.where(arr2d==5, 12, arr2d)


# 22. arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# 
# ---
# Convert all the values of 1st array to 64
# 

# In[23]:


arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d[0:1,0:1,0:3]=64


# 23. Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it

# In[24]:


x=np.arange(0,10).reshape(2,5)
x[0:1,:]


# 24. Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it

# In[25]:


x=np.arange(0,10).reshape(2,5)
x[1:2,1:2]


# 25. Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

# In[26]:


x=np.arange(0,10).reshape(2,5)
x[:2,2:3]


# 26. Create a 10x10 array with random values and find the minimum and maximum values

# In[27]:


x=np.random.randn(10,10)
np.amax(x)


# In[28]:


np.amin(x)


# 27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# ---
# Find the common items between a and b
# 

# In[29]:


a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
np.intersect1d(a, b)


# 28. a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# 
# ---
# Find the positions where elements of a and b match
# 
# 

# In[30]:


a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
x = np.where(np.in1d(a,b)==True)


# In[31]:


x = np.where(np.in1d(b,a)==True)


# 29.  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will**
# 

# In[32]:


data = np.random.randn(7, 4)
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data[names!='Will']


# 30. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will** and **Joe**
# 
# 

# In[33]:


data = np.random.randn(7, 4)
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data[(names!='Will')& (names!='Joe')]


# Difficulty Level **Hard**

# 31. Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.

# In[34]:


x= np.random.uniform(1,15, size=(5,3))


# 32. Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

# In[35]:


x= np.random.uniform(1,16, size=(2,2,4))


# 33. Swap axes of the array you created in Question 32

# In[36]:


np.swapaxes(x,0,2)


# 34. Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0

# In[37]:


x = np.random.rand(10)
y= np.sqrt(x)
y = np.where(y<0.5,y,0)
y


# 35. Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

# In[38]:


x = np.random.randint(0,12,(10))
y= np.random.randint(0,12,(10))
c=np.maximum(x, y)
c


# 36. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# 
# ---
# Find the unique names and sort them out!
# 

# In[39]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
x=np.unique(names)
y=np.sort(x)


# 37. a = np.array([1,2,3,4,5])
# b = np.array([5,6,7,8,9])
# 
# ---
# From array a remove all items present in array b
# 
# 

# In[40]:


a = np.array([1, 2, 3, 4, 5])
b = np.array([5, 6, 7, 8, 9])

c = np.setdiff1d(a, b)
c


# 38.  Following is the input NumPy array delete column two and insert following new column in its place.
# 
# ---
# sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) 
# 
# 
# ---
# 
# newColumn = numpy.array([[10,10,10]])
# 

# In[45]:


sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]])
C = np.delete(sampleArray, 1, 1)
newColumn = np.array([[10,10,10]])
x = np.insert(C, 1, newColumn, axis=1)
x


# 39. x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# 
# 
# ---
# Find the dot product of the above two matrix
# 

# In[46]:


x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
np.dot(x,y)


# 40. Generate a matrix of 20 random values and find its cumulative sum

# In[49]:


x = np.random.rand(4,5)
np.cumsum(x)


# In[ ]:




