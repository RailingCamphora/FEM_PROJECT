from matplotlib import pyplot as plt
from scipy.sparse import diags as dg
from scipy.sparse.linalg import spsolve
import numpy as np

def tridiag(num_main,num_up,num_down,dim):
    diagnoses=[np.full(dim,num_main),np.full(dim-1,num_up),np.full(dim-1,num_down)]
    sets=[0,1,-1]
    C=dg(diagnoses,sets,format='csr')
    return C

iteration=50000
res=[]

for n in [49,99,199,399]:
    x_co=np.linspace(0,10*np.pi,n+1)
    x=np.sin(x_co[1:n])
    A=tridiag(2,-1,-1,n-1)
    DL=tridiag(2,0,-1,n-1)
    U=tridiag(0,1,0,n-1)
    b=A.dot(x.T).T

    x0=np.zeros_like(x)
    i=0
    while i<iteration and np.linalg.norm(x-x0)>=np.linalg.norm(b)*1e-8:
        temp=U.dot(x0.T)+b.T
        x0=spsolve(DL,temp).T
        i+=1
    res.append(i)

plt.figure()
plt.plot(res)
plt.show()



#a=tridiag(2,1,3,4)
#print(a.todense())




