import numpy as np
from pylab import *
import matplotlib
import scipy.integrate as integ

#------------------------------------------------------------------
#------------------------------------------------------------------2D
def Fd(vx,vy,omega):
    d = 1.21  # плотность воздуха (kg/m^3)
    R = 0.033  # радиус мяча
    if omega>0: #backspin
        v_spin = R * omega
    if omega<0: #topspin
        v_spin = R * (-omega)
    if omega==0: #without spin
        return 0
    v = pow(pow(vx, 2) + pow(vy, 2), 0.5)
    Cd = 0.55 + 1 / pow(22.5 + 4.2 * pow(v / v_spin, 2.5), 0.4)
    A = np.pi * pow(R, 2)  # площадь поперечного сечения мяча (m2)
    return Cd*A*d*pow(v,2)/2
def Fl(vx,vy,omega):
    d = 1.21  # плотность воздуха (kg/m^3)
    R = 0.033  # радиус мяча
    v = pow(pow(vx, 2) + pow(vy, 2), 0.5)
    if omega>0:
        v_spin = R * omega
        Cl = 1 / (2 + v / v_spin)
    if omega<0:
        v_spin = R * (-omega)
        Cl = -1 / (2 + v / v_spin)
    if omega==0:
        return 0
    A = np.pi * pow(R, 2)  # площадь поперечного сечения мяча (m2)
    return Cl*A*d*pow(v,2)/2
def Cd(v,omega):
    R = 0.033  # радиус мяча
    if omega>0:
        v_spin = R * omega
    if omega<0:
        v_spin = R * (-omega)
    if omega==0:
        return 0
    return 0.55 + 1 / pow(22.5 + 4.2 * pow(v / v_spin, 2.5), 0.4)
def Cl(v,omega):
    R = 0.033  # радиус мяча
    if omega>0:
        v_spin = R * omega
        Cl=1 / (2 + v / v_spin)
    if omega<0:
        v_spin = R * (-omega)
        Cl=-1 / (2 + v / v_spin)
    if omega==0:
        Cl=0
    return Cl
t=linspace(0,5,100) #from 0 to 5 sec

theta=np.pi/3 #угол, под которым был произведен удар (rad)
v0=30 #скорость удара (m/s)
omega=150#угловая скорость, сообщенная мячу (rad/sec) = 2Пn, где n-количество оборотов в единицу времени
x0=0 #(m)
y0=10 #(m)

m = 0.057  # масса мяча(kg)
g = 9.8 #(m/c2)
d = 1.21  # плотность воздуха (kg/m^3)
R = 0.033  # радиус мяча(m)
A = np.pi * pow(R, 2)  # площадь поперечного сечения мяча (m2)
vals_forx=np.zeros((3,size(t)))
vals_fory=np.zeros((3,size(t)))
dt=(max(t)-min(t))/size(t)
#omega=(v0*m-m*g*0.0004)*3*sin(theta)/(2*m*R)
print(omega)
for i in range(size(t)):
    if i==0:
        vals_forx[0, 0]=-Fl(v0*cos(theta),v0*sin(theta),omega)*sin(theta)/m-Fd(v0*cos(theta),v0*sin(theta),omega)*cos(theta)/m  # a0x
        vals_fory[0, 0]=-g-Fd(v0*cos(theta),v0*sin(theta),omega)*sin(theta)/m+Fl(v0*cos(theta),v0 * sin(theta),omega)*cos(theta)/m  # a0y
        vals_forx[1, 0]=v0*cos(theta)# v0x
        vals_fory[1, 0]=v0*sin(theta)# v0y
        vals_forx[2, 0]=x0 # x
        vals_fory[2, 0]=y0 # y
    else:
        v=pow(pow(vals_forx[1, i-1],2)+pow(vals_fory[1, i-1],2),0.5)
        vals_forx[1,i]=vals_forx[1,i-1]+vals_forx[0, i-1]*dt #vxi
        vals_fory[1,i]=vals_fory[1,i-1]+vals_fory[0, i-1]*dt #vyi
        vals_forx[2,i]=vals_forx[2,i-1]+vals_forx[1, i-1]*dt #xi
        vals_fory[2,i]=vals_fory[2,i-1]+vals_fory[1, i-1]*dt #yi
        vals_forx[0, i]=-Cd(v,omega)*A*v*d*vals_forx[1,i]/(2*m)-Cl(v,omega)*A*v*d*vals_fory[1,i]/(2*m)#axi
        vals_fory[0, i]=-g-Cd(v,omega)*A*v*d*vals_fory[1,i]/(2*m)+Cl(v,omega)*A*v*d*vals_forx[1,i]/(2*m)#ayi
        if vals_fory[2,i]>0 and vals_fory[2,i]<0.7:
            print(i)
figure()
plot(vals_forx[2,:], vals_fory[2,:], 'k-')
xlabel('x position')
ylabel('y position')
show()

#-----------------------------------------------------------------------------------------------------------

def vector_field(w,t):
    #w=[a1,b1,a2,b2]
    a1 = w[0]
    b1 = w[1]
    a2 = w[2]
    b2 = w[3]

    omega=150
    g=9.8
    v = pow((pow(b2, 2) + pow(b1, 2)), 0.5)
    d = 1.21  # плотность воздуха (kg/m^3)
    R = 0.033  # радиус мяча
    m = 0.057  # масса мяча(kg)
    A = np.pi * pow(R, 2)  # площадь поперечного сечения мяча (m2)
    k = d * A / (2 * m)
    if omega>0:
        v_spin=omega*R
        Cl=1/(2+v/v_spin)
        Cd = 0.55 + 1 / pow(22.5 + 4.2 * pow(v / v_spin, 2.5), 0.4)
    if omega<0:
        v_spin = -omega * R
        Cl = -1 / (2 + v / v_spin)
        Cd = 0.55 + 1 / pow(22.5 + 4.2 * pow(v / v_spin, 2.5), 0.4)
    if omega==0:
        Cl=0
        Cd=0.55
    da1dt = b1
    db1dt = -k*v*Cl*b2-k*v*Cd*b1
    da2dt = b2
    db2dt = k*v*Cl*b1-k*v*Cd*b2-g
    f = [da1dt, db1dt, da2dt, db2dt]
    return f

t = linspace(0, 5, 100)  # from 0 to 5 sec

theta=np.pi/3 #угол, под которым был произведен удар (rad)
v0=30 #скорость удара (m/s)
omega=150
x0=0
y0=10

# Initial conditions
# a1, a2 are the initial displacements; b1 and b2 are the initial velocities
a1 = x0
a2 = y0
b1=v0*cos(theta)
b2 = v0*sin(theta)
w0=[a1,b1,a2,b2]

res=integ.odeint(vector_field,w0,t)
print(res)

figure()
plot(res[:,0], res[:,2], 'k-')
xlabel('x position')
ylabel('y position')
show()