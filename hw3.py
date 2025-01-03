import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

f=lambda x:x*(1-x)

N_list=[1,39,79,159,319]
res=[]
for N in N_list:
    x=np.linspace(0,1,N+2)
    x1=list(x)
    dx=1/(N+1)
    x1=[x1[i]-0.5*dx if i%2==1 else x1[i] for i in range(len(x1)) ]
    fx = f(np.array(x1))
    diag_main=np.full(N,8/(3*dx))
    diag_lower=np.array([-2/(3*dx) if (j+1)%2==1 else -2/(dx) for j in range(N-1) ])
    diag_upper=np.array([-2/(3*dx) if (j+1)%2==1 else -2/(dx) for j in range(N-1)])
    diagnoses=[diag_main,diag_upper,diag_lower]
    S=diags(diagnoses,[0,1,-1],format='csr')
    F=np.full(N,2*dx)

    u=spsolve(S,F.T).T
    res.append(np.max(np.abs(u-fx[1:-1])))


plt.figure()
plt.plot(res)
plt.show()




