import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve,splu,eigsh
from scipy.sparse.linalg import norm as snorm
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import project_function as pf
import time

True_f=lambda x,y:x*y*(1-x)*(1-y)

error=[]
error_l2=[]
N=[9,19,39,79,159]
N_as_label=np.array(N)

time_cosume=[]
condition_number=[]


for n in N:
    x = np.linspace(0, 1, n+2)
    y = np.linspace(0, 1, n+2)
    dx=dy=x[1]-x[0]
    #print(dx)
    X, Y = np.meshgrid(x, y)


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
    time_cosume.append(t2-t1)
    eigvals_max = eigsh(K, k=1, which='LM', return_eigenvectors=False)
    eigvals_min = eigsh(K, k=1, which='SM', return_eigenvectors=False)
    condition_number.append(eigvals_max/eigvals_min)

    u=u.reshape((n,n))
    Tu=True_f(X[1:n+1,1:n+1],Y[1:n+1,1:n+1])
    error.append(np.max(abs(u-Tu)))
    error_l2.append(np.linalg.norm(u-Tu)/n)


error_l2=np.array(error_l2)

plt.figure('uniform norm error')
plt.plot(error)
plt.show()
plt.figure()
plt.scatter(np.log(N_as_label),np.log(error_l2))
plt.xlabel('log(N)')
plt.ylabel('log(error)')
plt.title('the L2 error')
plt.show()


plt.figure('Time consuming')
plt.scatter(np.log(N_as_label),np.log(np.array(time_cosume)))
plt.ylabel('second')
plt.show()

print(np.log(N_as_label))
print('log $L^2$ error')
print(np.log(error_l2))
print('condition number')
print(np.log(np.array(condition_number)))
print('time cosume')
print(np.log(np.array(time_cosume)))



