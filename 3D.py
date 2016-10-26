import scipy.integrate as integ
import numpy as np
from pylab import *
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------3D

def vector_field(w,t):
    #w=[a1,b1,a2,b2,a3,b3]
    a1 = w[0]
    b1 = w[1]
    a2 = w[2]
    b2 = w[3]
    a3 = w[4]
    b3 = w[5]

    omega=0
    g=9.8
    v = pow((pow(b2, 2) + pow(b1, 2)+pow(b3,2)), 0.5)
    d = 1.21  # плотность воздуха (kg/m^3)
    R = 0.033  # радиус мяча
    m = 0.057  # масса мяча(kg)
    A = np.pi * pow(R, 2)  # площадь поперечного сечения мяча (m2)
    k = d * A / (2 * m)
    if omega>0:
        v_spin=omega*R
        #Cl=1/(2+v/v_spin)
        Cd = 0.55 + 1 / pow(22.5 + 4.2 * pow(v / v_spin, 2.5), 0.4)
    if omega<0:
        v_spin = -omega * R
        #Cl = -1 / (2 + v / v_spin)
        Cd = 0.55 + 1 / pow(22.5 + 4.2 * pow(v / v_spin, 2.5), 0.4)
    if omega==0:
        #Cl=0
        Cd=0.55
        Fm=0
    da1dt = b1
    db1dt = -Fm-k*v*Cd*b1
    da2dt = b2
    db2dt = -Fm-k*v*Cd*b2
    da3dt=b3
    db3dt= -g+Fm-k*v*Cd*b3
    f = [da1dt, db1dt, da2dt, db2dt,da3dt, db3dt]
    return f

t = linspace(0, 5, 50)  # from 0 to 3 sec
alpha=np.pi/4 #угол, под которым был произведен удар (rad)
beta=np.pi/10
v0=30 #скорость удара (m/s)
omega=150
x0=0
y0=0
z0=1

# Initial conditions
# a1, a2 are the initial displacements; b1 and b2 are the initial velocities
a1 = x0
a2 = y0
a3 = z0
b1=v0*cos(alpha)*cos(beta)
b2 = v0*sin(beta)*cos(alpha)
b3=v0*sin(alpha)
w0=[a1,b1,a2,b2,a3,b3]

res=integ.odeint(vector_field,w0,t)
print(res)

fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.plot_wireframe(res[:,0],res[:,2],res[:,4])
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()