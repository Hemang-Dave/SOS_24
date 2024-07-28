import numpy as np
#creating numpy arrays and specifying datatype of elements
a = np.array([1,2,3,4], dtype='int16') #1D array
b = np.array([[9.0,1.3,4.5,6.7,8.9,6.5,4.3],[5.6,7.7,3.4,2.3,6.7,4.4,3]]) #2D array. Matrix of "shape" 2x3
print(b)
print(b.shape,b.ndim) #prints shape, dimension of b matrix
print(a.dtype,b.dtype) #prints data type of elements in a and b arrays
print()

#To get or traverse through a numpy array, similar as in lists
element = b[0,3] #(row,column) indexing starts from 0
b2 = b[1,:] #2nd row of b matrix
b3 = b[:,2] #3rd column of b matrix
print(element)
print(b2)
print(b3)
print(b[0,1::2])  #[row,startindex:endindex:stepsize]
print()

#3D (n dimensional) example
c= np.array([[[2,3,4],[4,5,6]],[[3,4,6],[7,8,9]]])
print(c)
print(c.shape)
print(c.ndim)
print()

#Initializing special matrices and arrays
#All zeros
z = np.zeros((2,3))
print(z)
#All ones
o = np.ones(a.shape,dtype='int8')
print(o)
#any other number
n = np.full(b.shape,45)
print(n)
#random decimal numbers
dec = np.random.rand(3,4) #3x4 matrix
print(dec)
#random integers
integer = np.random.randint(-2,10,size=(2,4,3)) #3D array
print(integer)
#identity matrix
iden = np.identity(3)
print(iden)

#Repeating an array
arr = np.array([[1,2,3]])
r1 = np.repeat(arr,3,axis=0)
print(arr)
print(r1)
print()

#Mathematics with numpy. NumPy by default performs element wise arithmetic
a1 = np.array([[1,2,3],[4,5,6]])
a2 = a1.copy()
print(a1)
print(a1*3)
print(a1*3 + a2)
print(np.sin(a1)) #prints sin of values present in a
print()

#Linear algebra and statistics with numpy
w= np.ones((2,4))
f = np.full((4,3),2)
mul=np.matmul(w,f)
print(mul)
#Finding determinant
print(np.linalg.det(iden))
#Other functions include finding inverse, trace, determinant etc of matrix
print()

#stats with numpy
stats = np.array([[7,2,3,],[4,5,6]])
print(stats)
print(np.min(stats))
print(np.min(stats,axis = 1))   #single column of the minimum value in each row
print(np.min(stats,axis=0))   #single row of minimum value in each column
print(np.sum(stats))
print(np.sum(stats,axis=0))
print(np.sum(stats,axis=1))
print()

#Reorganizing arrays
print("---Reorganizing arrays---")
before = np.random.randint(2,8,size=(2,4))
print(before)
print()
after1 = before.reshape(1,8)
print(after1)
print()
after2 = before.reshape(4,2)
print(after2)
print()
after3 = before.reshape(2,2,2)
print(after3)
print()

#Vertical and horizontal stack
v1 = np.array([1,2,3,4])
v2 = np.array([5,6,7,8])
ver = np.vstack([v1,v2,v2]) #Whatever order and number you want
print(ver)
h1= np.ones((2,4))
h2 = np.zeros((2,2))
hor = np.hstack((h1,h2))
print(hor)