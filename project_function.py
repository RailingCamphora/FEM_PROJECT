import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

## the f corespond to u=xy(1-x)(1-y) is 2(1-y)y+2(1-x)x
f1=lambda x,y:2*(1-x)*x+2*y*(1-y)
f2=lambda x,y:1

def area(x,y):
    """

    :param x: numpy x coordinate
    :param y: numpy y coordiante
    :return:
    """
    return 0.5*np.abs((x[1]-x[0])*(y[2]-y[0])-(x[2]-x[0])*(y[1]-y[0]))

def inner_product(x,y,u=f1):
    """

    :param x: numpy x coordinate
    :param y: numpy y coordiante
    :param u: function
    :return:
    """
    z=np.column_stack((x,y))
    K=area(x,y)
    s1=u(*z[1])+u(*z[2])+u(*z[0])
    s2=u(*((z[1]+z[2])/2))+u(*((z[0]+z[1])/2))+u(*((z[2]+z[0])/2))
    s3=u(*((z[0]+z[1]+z[2])/3))
    return (K/60)*(3*s1+8*s2+27*s3)




def local_stiff(element):
    p1,p2,p3=element
    x=np.array([p1[0],p2[0],p3[0]])
    y=np.array([p1[1],p2[1],p3[1]])
    A=area(x,y)

    b=np.array([p2[1]-p3[1],p3[1]-p1[1],p1[1]-p2[1]])
    c=np.array([p3[0]-p2[0],p1[0]-p3[0],p2[0]-p1[0]])

    Ke=(1/(4*A))*(b[:,None]*b[None,:]+c[:,None]*c[None,:])
    return Ke


def local_F(element,f=f1):
    p1, p2, p3 = element
    x = np.array([p1[0], p2[0], p3[0]])
    y = np.array([p1[1], p2[1], p3[1]])
    #A = area(x, y)
    A1=(p1[0]-p2[0])*(p1[1]-p3[1])-(p1[0]-p3[0])*(p1[1]-p2[1])
    A2=(p2[0]-p3[0])*(p2[1]-p1[1])-(p2[0]-p1[0])*(p2[1]-p3[1])
    A3=(p3[0]-p1[0])*(p3[1]-p2[1])-(p3[0]-p2[0])*(p3[1]-p1[1])


    g1=lambda x,y:(1/A1)*(p2[0]*p3[1]-p3[0]*p2[1]+(p2[1]-p3[1])*x+(p3[0]-p2[0])*y)*f(x,y)
    g2=lambda x,y:(1/A2)*(p3[0]*p1[1]-p1[0]*p3[1]+(p3[1]-p1[1])*x+(p1[0]-p3[0])*y)*f(x,y)
    g3=lambda x,y:(1/A3)*(p1[0]*p2[1]-p2[0]*p1[1] + (p1[1] - p2[1]) * x + (p2[0] - p1[0]) * y)*f(x,y)

    return [inner_product(x,y,g1),inner_product(x,y,g2),inner_product(x,y,g3)]







if __name__=="__main__":
    e=[[0.1,0.1],[0.1,0.2],[0.2,0.2]]
    print(local_stiff(e))
    print(local_F(e))
