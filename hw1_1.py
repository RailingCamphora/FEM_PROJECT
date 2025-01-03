import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


f=lambda x,t:np.sin(x)*(np.cos(t)-np.sin(t))
u=lambda x,t:np.sin(x)*np.cos(t)

T,N=1,80
delta_x=np.pi/N
x=np.linspace(0,np.pi,N+1)

M=int(T/delta_x)+1
delta_t=T/M
r=delta_t/delta_x**2
## construct the accurate results for comparison
compare=np.zeros((M+1,N+1))
for n in range(M+1):
    for j in range(N+1):
        compare[n][j]=u(j*delta_x,n*delta_t)
## calculate the results
result=np.zeros((M+1,N+1))

def tridiag(num_main,num_up,num_down,dim):
    diagnoses=[np.full(dim,num_main),np.full(dim-1,num_up),np.full(dim-1,num_down)]
    sets=[0,1,-1]
    C=diags(diagnoses,sets,format='csr')
    return C

A=tridiag(1+2*r,-r,-r,N+1)
result[0]=np.sin(x)
for n in range(M):
    result[n+1]=spsolve(A,result[n].T).T
    result[n+1][0],result[n+1][-1]=0,0

error=np.abs(compare[M]-result[M])
plt.figure(1)
plt.plot(x,(np.log(error)))
plt.show()


