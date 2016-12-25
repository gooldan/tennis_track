import scipy.integrate as integ
import numpy as np
from pylab import *
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit, leastsq,minimize,fmin_slsqp
import math

#---------------------------------------------------------------------------------DE
def vector_field(w,t,*args):
    #w=[a1,b1,a2,b2,a3,b3]
    a1 = w[0]
    b1 = w[1]
    a2 = w[2]
    b2 = w[3]
    a3 = w[4]
    b3 = w[5]

    lamb = args[0]
    w_x = args[1]
    w_y = args[2]
    w_z = args[3]

    omega=pow(pow(w_x,2)+pow(w_y,2)+pow(w_z,2),0.5)
    g=9.8
    v = pow((pow(b2, 2) + pow(b1, 2)+pow(b3,2)), 0.5)
    d = 1.21  # плотность воздуха (kg/m^3)
    R = 0.02  # metr
    m = 0.0027  # kg
    # R = 0.033  # радиус мяча
    # m = 0.057  # масса мяча(kg)
    A = np.pi * pow(R, 2)  # площадь поперечного сечения мяча (m2)
    k = d * A / (2 * m)
    if omega>0.001:
        v_spin=omega*R
        Cd = 0.55 + 1 / pow(22.5 + 4.2 * pow(v / v_spin, 2.5), 0.4)
    else: #omega=0
        Cd=0.55
    da1dt = b1
    db1dt = 2*(lamb/m)*w_y*b3-2*(lamb/m)*w_z*b2-k*v*Cd*b1
    da2dt = b2
    db2dt = 2*(lamb/m)*w_z*b1-2*(lamb/m)*w_x*b3-k*v*Cd*b2
    da3dt=b3
    db3dt= -g+2*(lamb/m)*w_x*b2-2*(lamb/m)*w_y*b1-k*v*Cd*b3
    f = [da1dt, db1dt, da2dt, db2dt,da3dt, db3dt]
    return f

def solve_DE(t,w0,params):
    res=integ.odeint(vector_field,w0,t,args=params)
    return [res[:,0],res[:,2],res[:,4]]#x,y,z

#----------------------------------------------------------------------------------------
points=np.array([[-466.11511564958204, -704.47683715457345, 2071.5203628210625],
                 [-264.47018837287834, -672.89956550855482, 2295.8824856324359],
                 [102.6015879328797, -502.5462825027044, 2826.1231886118685],
                 [184.88450491720738, -447.97202805235315, 3007.284961948606],
                 [274.78242723619877, -359.50086086742454, 3114.3027944378236],
                 [495.83405416513318, -44.235289158547545, 3293.948035912043]
                 ])
points=points/1000#metr
d_t=[0.002,0.016,0.002,0.004,0.01]#sec
R = 0.02#metr
m = 0.0027#kg
d = 1.21
A = np.pi * pow(R, 2)
k = A * d / (2 * m)
g=9.8
#--------------------------------------------------------------------------minimize+DE
def min_func2(X):
    lamb=X[0]
    w_x=X[1]
    w_y=X[2]
    w_z=X[3]
    v_x0 = X[4]
    v_y0 = X[5]
    v_z0 = X[6]
    points = np.array([[-466.11511564958204, -704.47683715457345, 2071.5203628210625],
                       [-264.47018837287834, -672.89956550855482, 2295.8824856324359],
                       [102.6015879328797, -502.5462825027044, 2826.1231886118685],
                       [184.88450491720738, -447.97202805235315, 3007.284961948606],
                       [274.78242723619877, -359.50086086742454, 3114.3027944378236],
                       [495.83405416513318, -44.235289158547545, 3293.948035912043]
                       ])
    points = points / 1000  # metr
    #d_t = [0.002, 0.016, 0.002, 0.004, 0.01]  # sec
    t = [0, 0.002, 0.018, 0.02, 0.024]
    xdata = points[:, 0]
    ydata = points[:, 1]
    zdata = points[:, 2]
    x_=np.array(xdata[0:-1]).reshape(shape(np.array(xdata[0:-1]))[0],1)
    y_=np.array(ydata[0:-1]).reshape(shape(np.array(ydata[0:-1]))[0],1)
    z_=np.array(zdata[0:-1]).reshape(shape(np.array(zdata[0:-1]))[0],1)

    #x_t = []
    #y_t = []
    #z_t = []
    #for i in range(len(points) - 1):
    #    x_t.append(abs(xdata[i + 1] - xdata[i]) / d_t[i])
    #    y_t.append(abs(ydata[i + 1] - ydata[i]) / d_t[i])
    #    z_t.append(abs(zdata[i + 1] - zdata[i]) / d_t[i])

    params = (lamb, w_x, w_y, w_z)

    w0 = [x_[0], v_x0, y_[0], v_y0, z_[0], v_z0]
    [xfit, yfit, zfit] = solve_DE(t, w0, params)
    # print("xfit")
    # print(xfit)
    x_fit=np.array(xfit).reshape(shape(np.array(xfit))[0],1)
    y_fit=np.array(yfit).reshape(shape(np.array(yfit))[0],1)
    z_fit=np.array(zfit).reshape(shape(np.array(zfit))[0],1)
    return sum((x_fit-x_)**2)+sum((y_fit-y_)**2)+sum((z_fit-z_)**2)
x0= np.array([3,0.01,0.01,0.1,1,1,1])

res2 = minimize(min_func2, x0, method='SLSQP')
#print("res2:")
#print(res2)
#--------------------------------------------------------------------------leastsq+DE

def leastsq_function2(params,*args):

    lamb = params[0]
    w_x = params[1]
    w_y = params[2]
    w_z = params[3]
    v_x0 = params[4]
    v_y0 = params[5]
    v_z0 = params[6]

    x = args[0]
    y = args[1]
    z = args[2]
    t=args[3]
    #x_t=args[4]
    #y_t = args[5]
    #z_t = args[6]
    arr=[]
    arr.extend(x)
    arr.extend(y)
    arr.extend(z)

    array_fit=[]
    params =(lamb, w_x, w_y, w_z)

    w0=[x[0],v_x0,y[0],v_y0,z[0],v_z0]
    [xx,yy,zz]=solve_DE(t,w0,params)

    xx=xx.tolist()
    yy = yy.tolist()
    zz = zz.tolist()
    array_fit.extend(xx)
    array_fit.extend(yy)
    array_fit.extend(zz)
    return np.array(arr)-np.array(array_fit)

xdata=points[:,0]
ydata=points[:,1]
zdata=points[:,2]

#x_t=[]
#y_t=[]
#z_t=[]

#for i in range(len(points)-1):
#    x_t.append(abs(xdata[i+1]-xdata[i])/d_t[i])
#    y_t.append(abs(ydata[i+1]-ydata[i])/d_t[i])
#    z_t.append(abs(zdata[i+1]-zdata[i])/d_t[i])

tt=[0,0.002,0.018,0.02,0.024,0.034]
#args=(xdata[0:-1],ydata[0:-1],zdata[0:-1],tt[0:-1],x_t,y_t,z_t)
args=(xdata[0:-1],ydata[0:-1],zdata[0:-1],tt[0:-1])
w0=[1, 1, 1, 1,1,1,1]
result2 = leastsq(leastsq_function2, w0,args=args)
print("Functions is in second method"+str(leastsq_function2(result2[0],*args)))
print("lamb,w_x,w_y,w_z in second method:")
print(result2)

#------------------------------------------------------------------------------------found parametres

lamb=result2[0][0]
w_x=result2[0][1]
w_y=result2[0][2]
w_z=result2[0][3]
v0_x=result2[0][4]
v0_y=result2[0][5]
v0_z=result2[0][6]
#lamb=res2.x[0]
#w_x=res2.x[1]
#w_y=res2.x[2]
#w_z=res2.x[3]
#v0_x=res2.x[4]
#v0_y=res2.x[5]
#v0_z=res2.x[6]

omega=pow(w_x**2+w_y**2+w_z**2,0.5)

xdata=points[:,0]
ydata=points[:,1]
zdata=points[:,2]
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(xdata,ydata,zdata,c='r')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')


#v0_x=abs(xdata[0]-xdata[1])/d_t[0]
#v0_y=abs(ydata[0]-ydata[1])/d_t[0]
#v0_z=abs(zdata[0]-zdata[1])/d_t[0]
x0=points[0][0]
y0=points[0][1]
z0=points[0][2]

#-----------------------------------------------------------------------------------------DE

t=linspace(0,0.05,20) #from 0 to time sec
w0=[x0,v0_x,y0,v0_y,z0,v0_z]
params=(lamb,w_x,w_y,w_z)
[x,y,z]=solve_DE(t,w0,params)
ax.plot_wireframe(x,y,z)
plt.show()


