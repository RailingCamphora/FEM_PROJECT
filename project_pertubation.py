import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import project_function as pf
import time

True_f=lambda x,y:x*y*(1-x)*(1-y)

error=[]
error_L2=[]
time_consuming=[]
N=[9,19,39,79,159]

for n in N:
    x = np.linspace(0, 1, n+2)
    y = np.linspace(0, 1, n+2)
    dx=dy=x[1]-x[0]
    #print(dx)
    X, Y = np.meshgrid(x, y)
    noise1=np.random.uniform(-1,1,(n,n))
    noise2 = np.random.uniform(-1, 1, (n, n))
    X[1:n+1,1:n+1]+=0.05*dx*noise1
    Y[1:n+1,1:n+1]+=0.05*dx*noise2

    tri_elements=[]
    for i in range(n+1):
        for j in range(n+1):
            tri_elements.append([[i,j],[i+1,j],[i+1,j+1]])
            tri_elements.append([[i,j],[i,j+1],[i+1,j+1]])

    K=lil_matrix((n**2,n**2))
    F=np.zeros(n**2)
    # construct K
    for element in tri_elements:
        i1,i2,i3=element
        p1=[X[i1[0],i1[1]],Y[i1[0],i1[1]]]
        p2 = [X[i2[0], i2[1]], Y[i2[0], i2[1]]]
        p3 = [X[i3[0], i3[1]], Y[i3[0], i3[1]]]
        Ke=pf.local_stiff([p1,p2,p3])
        for l1 in range(3):
            if 1<=element[l1][0]<=n and 1<=element[l1][1]<=n:
                for l2 in range(3):
                    if 1 <= element[l2][0] <= n and 1 <= element[l2][1] <= n:
                        K[(element[l1][0]-1)*n+element[l1][1]-1,(element[l2][0]-1)*n+element[l2][1]-1]+=Ke[l1,l2]

        #construct F
        Fe=pf.local_F([p1,p2,p3])

        for l3 in range(3):
            if 1<=element[l3][0]<=n and 1<=element[l3][1]<=n:
                F[(element[l3][0]-1)*n+element[l3][1]-1]+=Fe[l3]
    K=K.tocsr()
    t1=time.time()
    u=spsolve(K,F.T).T
    t2=time.time()
    time_consuming.append(t2-t1)
    u=u.reshape((n,n))
    Tu=True_f(X[1:n+1,1:n+1],Y[1:n+1,1:n+1])
    error.append(np.max(abs(u-Tu)))
    error_L2.append(np.log(np.linalg.norm(u-Tu)/n))



plt.figure(1)
plt.plot(error)
plt.figure()
plt.scatter(np.log(np.array(N)),error_L2)
plt.xlabel('$log(N)$')
plt.ylabel('log error')
plt.show()


print(['&'+str(error_L2[i]) for i in range(np.size(error_L2))])
print('log time cosuming')
print(np.log(np.array(time_consuming)))



