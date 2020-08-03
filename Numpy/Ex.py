import numpy as p
import matplotlib.pyplot as pt
# l=[[1,2,3],[4,5,6]]
# ar1=p.array(l)
# # div=ar1/2
# # print(type(ar1))
# print(ar1)
# print(div)


# ar1=[1.87,1.87,1.82,1.91,1.90,1.85]
# ar2=[81.65,97.52,95.25,92.98,86.81,88.45]
# ar1_h=p.array(ar1)
# ar2_w=p.array(ar2)
# bmi=ar1_h/ar2_w**2
# pound=ar2_w*2.2
# print(bmi)
# print(pound)

# ar1=p.array([2,3,4,5])
# print(ar1)
# print(ar1.size)
# print(ar1.dtype)
# print(ar1.ndim)
# print(ar1.shape)

# ar2=p.empty((2,4),dtype=int)
# print(ar2)

# ar2=p.zeros((5,5),dtype=int)
# print(ar2)

# ar2=p.ones(5,dtype=int)
# print(ar2)

# ar1=p.arange(10,55,2,dtype=int)
# print(ar1)

# ar1=p.linspace(1,10,num=9,endpoint=False,dtype=int)
# print(ar1)

# ar1=p.eye(5,k=2)
# # help(p.eye)
# print(ar1)

# ar1=p.identity(4,dtype=float)
# print(ar1)



# x=[1,2,3]
# a=p.asarray(x)
# print(a)

# x=(1,2,3)
# a=p.asarray(x)
# print(a)

# x=[(1,2,3),(4,5)]
# a=p.asarray(x)
# print(a)

# ar1=p.random.rand(3,2)
# print(ar1)

# ar1=p.random.randint(3,8,7)
# print(ar1)

# ar1=p.random.randint(3,8,size=(4,5))
# print(ar1)

# ar1=p.random.randint(3,8,7)
# ar2=p.array(ar1)
# print(ar2)

# ar2=p.array([[2,3,5,6,7,8],
# 	        [5,4,1,2,8,9]])
# # print(ar2[0])
# print(ar2[1][0])
# print(ar2[1][1])
# print(ar2[0][5])
# print(ar2[-2][-1])
# print(ar2[-1][1])

# ar2=p.array([2,3,5,6,7,8])
# print(ar2[:4])
# print(ar2[1:4])
# print(ar2[1:4:2])
# print(ar2[:])

# ar2=p.array([[2,3,5,6,7,8],
# 	        [5,4,1,2,8,9]])
# # print(ar2[1:4])
# # print(ar2[1:4,0:3])
# # print(ar2[1:4,0:3:2])
# print(ar2[0:4,1:3])
# print(ar2[1:5,0:4])

# ar1=p.array([[[0,1],[2,3]],
# 	         [[4,5],[6,7]]])
# print(ar1)

# ar2=p.zeros((3,4,4))
# print(ar2)

# ar1=p.array([2,3,5,6,7,8])
# ar3=p.array([2,3,5,6,7,8])
# ar2=p.array([[2,3,5,6,7,8],
# 	        [5,3,4,5,6,7],
#             [5,3,4,5,6,7]])
# # index=p.array([0,3,5])
# # print(ar1.shape)
# # print(ar1.ndim)
# # print(ar1[index])
# # print(ar1[[4,5,1]])
# # print(ar2[[0,1,2,2],[0,3,1,3]])
# # print(ar1[ar1>5])
# # print(ar2[ar2<5])
# # print(ar1+2)
# # print(ar1**2)
# # print(ar1+ar3)
# res=p.add(ar1,ar3)
# print(res)

# ar1=p.array([2,3])
# ar2=p.array([[1,2],[3,4]])
# # print(ar1.shape)
# # print(ar1.ndim)
# # print(ar2.shape)
# # print(ar2.ndim)
# # print(ar1+ar2)

# for x in p.nditer(ar1):
# 	print(x)

# print(ar1.shape)

# ar1=p.zeros((2,4,3))
# a1=p.ndarray.flatten(ar1)
# print(a1)

# a1=p.ndarray.flatten(ar2,)
# print(a1)

# a2=p.ravel(ar2,order='f')
# print(a2)

# a1=p.transpose(ar2)
# print(a1)

# a1=p.transpose(ar2)
# print(a1)

# ar3=p.array([[2,3],[4,5]])
# ar4=p.array([[2,3],[4,5]])
# ar5=p.dot(ar3,ar4)
# print(ar5)

# mt1=p.matrix('1 2;3 4')
# print(mt1)
# mt2=p.matrix([[1,2],[3,4]])
# print(mt2)

# A=p.array([[6,1,1],[4,-2,5],[2,8,7]])
# print("Rank of A:",p.linalg.matrix_rank(A))
# print("\nTrace of A:",p.trace(A))
# print("\nDeterminant of A:",p.linalg.det(A))
# print("\nInverse of A:",p.linalg.inv(A))
# print("\nMatrix A raised to power 3:\n",p.linalg.matrix_power(A,3))

x=[2,3,4,5,6]
y=[2,3,4,5,6]
pt.plot(x,y,'go:')
# # pt.bar(x,y)
# # pt.hist(x,y)
# pt.scatter(x,y)
pt.title("My Graph")
pt.xlabel("X-axis")
pt.ylabel("Y-axis")
pt.show()