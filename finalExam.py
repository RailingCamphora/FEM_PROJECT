import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def tridiag(num_main,num_up,num_down,dim):
    diagnoses=[np.full(dim,num_main),np.full(dim-1,num_up),np.full(dim-1,num_down)]
    sets=[0,1,-1]
    C=diags(diagnoses,sets,format='csr')
    return C

f=lambda x,y:np.sin(x)*np.sin(y)
u_real=lambda x,y,t:np.sin(x)*np.sin(y)*np.cos(np.sqrt(2)*t)
N=[20,40,80,160,320]
error=[]

for n in N:
    x=np.linspace(0,2*np.pi,n+1)
    y=np.linspace(0,2*np.pi,n+1)
    t = np.linspace(0, 1, n**2 + 1)
    dx,dt=x[1]-x[0],t[1]-t[0]
    r=(dt/dx)**2
    print(r)

    X,Y=np.meshgrid(x[1:-1],y[1:-1])
    u=f(X,Y)
    u_pre=f(X,Y)
    differential_matrix=tridiag(-2*r,r,r,n-1)

    ##actually u mean u(t=\delta t), u_pre means u(t=0)
    ## we need u(t=T), the iteration time is n-1
    for i in range(np.size(t)-2):
        temp=u
        u=-u_pre+2*u+differential_matrix.dot(u)+(differential_matrix.dot(u.T)).T
        u_pre=temp
    u1=u_real(X,Y,np.ones_like(X))
    error.append(np.max(np.abs(u-u1)))

print(error)
N=np.log(np.array(N))
error=np.log(np.array(error))
plt.figure()
plt.plot(N,error)
plt.title('the max error in the end')
plt.show()








