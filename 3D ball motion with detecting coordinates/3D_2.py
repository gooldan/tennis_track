import scipy.integrate as integ
import numpy as np
from pylab import *
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit, leastsq,minimize,fmin_slsqp
import math
#----------------------------------------------------------------------------------------minimize
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
def min_func(X):
    w_x=X[0]
    w_y=X[1]
    w_z=X[2]
    Fx=X[3]
    Fy=X[4]
    Fz=X[5]

    g=9.8
    points = np.array([[-466.11511564958204, -704.47683715457345, 2071.5203628210625],
                       [-394.86649074536876, -741.8546128761177, 2339.2141064146858],
                       [-264.47018837287834, -672.89956550855482, 2295.8824856324359],
                       [100.15327117664276, -482.21723218641796, 2759.8425411735061],
                       [184.77422380764008, -439.59506736128463, 3011.3081404447626],
                       [274.5904214844677, -354.50929756538187, 3112.3186020109724],
                       [501.31091148481494, -44.21705036561778, 3297.7928426550848],
                       [623.25237528942569, 96.926403487757256, 3671.6767120336308]])
    points = points / 1000  # metr
    d_t = [0.001, 0.003, 0.016, 0.002, 0.004, 0.01, 0.002]  # sec
    R = 0.02  # metr
    m = 0.0027  # kg
    xdata = points[:, 0]
    ydata = points[:, 1]
    zdata = points[:, 2]

    x_t = []
    y_t = []
    z_t = []
    vals_x = []
    vals_y = []
    vals_z = []
    v=[]

    # m = 0.057
    # R = 0.033
    d = 1.21
    A = np.pi * pow(R, 2)
    k = A * d / (2 * m)
    for i in range(len(points) - 1):
        x_t.append(abs(xdata[i + 1] - xdata[i]) / d_t[i])
        y_t.append(abs(ydata[i + 1] - ydata[i]) / d_t[i])
        z_t.append(abs(zdata[i + 1] - zdata[i]) / d_t[i])
        v.append(pow(x_t[i]**2+y_t[i]**2+z_t[i]**2,0.5))
    for i in range(len(points) - 2):
        vals_x.append(abs(x_t[i + 1] - x_t[i]) / d_t[i])
        vals_y.append(abs(y_t[i + 1] - y_t[i]) / d_t[i])
        vals_z.append(abs(z_t[i + 1] - z_t[i]) / d_t[i] + g)
    x=np.array(x_t[0:-1]).reshape(shape(np.array(x_t[0:-1]))[0],1) #4x1
    y=np.array(y_t[0:-1]).reshape(shape(np.array(y_t[0:-1]))[0],1)
    z=np.array(z_t[0:-1]).reshape(shape(np.array(z_t[0:-1]))[0],1)
    v=np.array(v[0:-1]).reshape(shape(np.array(v[0:-1]))[0],1)
    x_res=np.array(vals_x).reshape(shape(np.array(vals_x))[0],1) #4x1
    y_res=np.array(vals_y).reshape(shape(np.array(vals_y))[0],1)
    z_res=np.array(vals_z).reshape(shape(np.array(vals_z))[0],1)


    return sum((Fx/m-x_res-k*x*v*(0.55 + 1 / pow(22.5 + 4.2 * pow(v / (R*pow(w_x**2+w_y**2+w_z**2,0.5)), 2.5), 0.4)))**2)+\
           sum((Fy/m-y_res-k*y*v*(0.55 + 1 / pow(22.5 + 4.2 * pow(v / (R*pow(w_x**2+w_y**2+w_z**2,0.5)), 2.5), 0.4)))**2)+\
           sum((Fz/m-z_res-k*z*v*(0.55 + 1 / pow(22.5 + 4.2 * pow(v / (R*pow(w_x**2+w_y**2+w_z**2,0.5)), 2.5), 0.4)))**2)

x0= np.array([1,1,1,1,1,1])
#bnds = ((None, None), (0, None),(0, None),(0, None))
res = minimize(min_func, x0, method='L-BFGS-B')
#res = fmin_slsqp(min_func, x0, bounds=bnds)
print("res:")
print(res)
#-----------------------------------------------------------------------------leastsq
def leastsq_function(params,*args):
    w_x=params[0]
    w_y=params[1]
    w_z=params[2]
    Fx = params[3]
    Fy = params[4]
    Fz = params[5]
    x = args[0]
    y = args[1]
    z = args[2]
    yy = args[3]

    R = 0.02  # metr
    m = 0.0027  # kg
    # m = 0.057
    # R = 0.033
    d = 1.21
    A = np.pi * pow(R, 2)
    k = A * d / (2 * m)
    omega=pow(w_x**2+w_y**2+w_z**2,0.5)
    v_spin=omega*R

    #yfit = np.zeros((len(yy),1))
    yfit=[]

    n=int(len(yy)/3)
    for i in range(n):
        v = pow(x[i] ** 2 + y[i] ** 2 + z[i] ** 2, 0.5)
        if omega > pow(10, -10):
            Cd = 0.55 + 1 / pow(22.5 + 4.2 * pow(v / v_spin, 2.5), 0.4)
        else:
            Cd=0.55
        yfit.append(Fx/m-x[i]*k*v*Cd)

    for i in range(n):
        v = pow(x[i] ** 2 + y[i] ** 2 + z[i] ** 2, 0.5)
        if omega > pow(10, -10):
            Cd = 0.55 + 1 / pow(22.5 + 4.2 * pow(v / v_spin, 2.5), 0.4)
        else:
            Cd=0.55
        yfit.append(Fy/m-y[i]*k*v*Cd)

    for i in range(n):
        v = pow(x[i] ** 2 + y[i] ** 2 + z[i] ** 2, 0.5)
        if omega > pow(10, -10):
            Cd = 0.55 + 1 / pow(22.5 + 4.2 * pow(v / v_spin, 2.5), 0.4)
        else:
            Cd=0.55
        yfit.append(Fz/m-z[i]*k*v*Cd)

    return np.array(yy)-np.array(yfit)

d_t=[0.001,0.003,0.016,0.002,0.004,0.01,0.002]#sec
R = 0.02#metr
m = 0.0027#kg

g=9.8
xdata=points[:,0]
ydata=points[:,1]
zdata=points[:,2]

x_t=[]
y_t=[]
z_t=[]
vals_x=[]
vals_y=[]
vals_z=[]

for i in range(len(points)-1):
    x_t.append(abs(xdata[i+1]-xdata[i])/d_t[i])
    y_t.append(abs(ydata[i+1]-ydata[i])/d_t[i])
    z_t.append(abs(zdata[i+1]-zdata[i])/d_t[i])
for i in range(len(points) - 2):
    vals_x.append(abs(x_t[i+1]-x_t[i])/d_t[i])
    vals_y.append(abs(y_t[i+1]-y_t[i])/d_t[i])
    vals_z.append(abs(z_t[i+1]-z_t[i])/d_t[i]+g)

y=[]
y.extend(vals_x)
y.extend(vals_y)
y.extend(vals_z)

params0 = [1,1,1,1,1,1]
args = (x_t[0:-1],y_t[0:-1],z_t[0:-1],y)

result = leastsq(leastsq_function, params0,args=args)
print("Function is "+str(leastsq_function(result[0],*args)))
print("w_x,w_y,w_z,Fx,Fy,Fz:")
print(result[0])
#--------------------------------------------------------------------------
w_x=result[0][0]
w_y=result[0][1]
w_z=result[0][2]
Fx=result[0][3]
Fy=result[0][4]
Fz=result[0][5]
# w_x=res.x[0]
# w_y=res.x[1]
# w_z=res.x[2]
# Fx=res.x[3]
# Fy=res.x[4]
# Fz=res.x[5]
omega=pow(w_x**2+w_y**2+w_z**2,0.5)
print("omega"+str(omega))

d_t=[0.001,0.003,0.016,0.002,0.004,0.01,0.002]#sec

xdata=points[:,0]
ydata=points[:,1]
zdata=points[:,2]
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(xdata,ydata,zdata,c='r')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

v0_x=abs(xdata[0]-xdata[1])/d_t[0]
v0_y=abs(ydata[0]-ydata[1])/d_t[0]
v0_z=abs(zdata[0]-zdata[1])/d_t[0]
x0=points[0][0]
y0=points[0][1]
z0=points[0][2]

#-----------------------------------------------------------------------------------------DE
def vector_field(w,t,*args):
    #w=[a1,b1,a2,b2,a3,b3]
    a1 = w[0]
    b1 = w[1]
    a2 = w[2]
    b2 = w[3]
    a3 = w[4]
    b3 = w[5]

    w_x = args[0]
    w_y = args[1]
    w_z = args[2]
    Fx=args[3]
    Fy=args[4]
    Fz=args[5]

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
    if omega>pow(10,-10):
        v_spin=omega*R
        Cd = 0.55 + 1 / pow(22.5 + 4.2 * pow(v / v_spin, 2.5), 0.4)
    else: #omega=0
        Cd=0.55
    da1dt = b1
    db1dt = Fx/m-k*v*Cd*b1
    da2dt = b2
    db2dt = Fy/m-k*v*Cd*b2
    da3dt = b3
    db3dt= -g+Fz/m-k*v*Cd*b3
    f = [da1dt, db1dt, da2dt, db2dt,da3dt, db3dt]
    return f

t=linspace(0,0.01,20) #from 0 to time sec

args=(w_x,w_y,w_z,Fx,Fy,Fz)
w0=[x0,v0_x,y0,v0_y,z0,v0_z]
res=integ.odeint(vector_field,w0,t,args=args)

ax.plot_wireframe(res[:,0],res[:,2],res[:,4])
plt.show()
