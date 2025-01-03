import numpy as np
from matplotlib import pyplot as plt



f=lambda x,t:np.sin(x)*(np.cos(t)-np.sin(t))
u=lambda x,t:np.sin(x)*np.cos(t)

T,N=1,80
x=np.linspace(0,np.pi,N+1)

delta_x=np.pi/N
M=int(T/(0.5*delta_x**2))+1
delta_t=T/M
result=np.zeros((M+1,N+1))
compare=np.zeros((M+1,N+1))
for n in range(M+1):
    for j in range(N+1):
        compare[n][j]=u(j*delta_x,n*delta_t)


result[0]=np.sin(x)
for n in range(M):
    for j in range(1,N):
        result[n+1,j]=result[n,j]+(delta_t/delta_x**2)*(result[n,j+1]-2*result[n,j]+result[n,j-1])+delta_t*f(j*delta_x,n*delta_t)


error=np.abs(compare[M]-result[M])
plt.figure(1)
plt.plot(x,(np.log(error)))
plt.show()