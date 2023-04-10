import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#Дифур
frames = 250
t = np.linspace(0 , 13.5 , frames)

G = 6.67 * 10**(-11)
M = 6 * 10**(24)
m = 4730
k = 0.35
radius_earth = 6378100
radius_atmosphere = radius_earth + 120 * 1000

def diff(w, t):
    global radius_atmosphere
    x, vx, y, vy = w
    if y < radius_atmosphere * (1-(np.sqrt(3)/2)) and y > radius_atmosphere*(-1+(np.sqrt(3)/2)) and x <  radius_atmosphere*(1+(1 - (1/2))) and x > radius_atmosphere*(-1+(1/2)):
        dxdt = vx
        dvxdt = (-((G * m * M * x) / (x**2 + y**2)**1.5) + (k  * vx))/m
        dydt = vy
        dvydt = (-((G * m * M * y) / (x**2 + y**2)**1.5) + (k * vy))/m
        #return dxdt, dvxdt, dydt, dvydt
    else:
        dxdt = vx
        dvxdt = -((G  * M * x) / (x**2 + y**2)**1.5)
        dydt = vy
        dvydt = -((G * M * y) / (x**2 + y**2)**1.5)
        #return dxdt, dvxdt, dydt, dvydt
    return dxdt, dvxdt, dydt, dvydt


x0 = -240 * 1000
vx0 = 8000 * np.cos(60)
y0 = 240 * 1000
vy0 = -8000 * np.sin(60)
w0 = x0, vx0, y0, vy0

sol = odeint(diff, w0, t)

def move(i):
    x= sol[i, 0]
    y= sol[i, 2]
    return x, y

fig, axes = plt.subplots()

#Земля с атмосферой
angle = np.linspace( 0 , 2 * np.pi , 150 ) 

radius_earth = 6378100
radius_atmosphere = radius_earth + 120 * 1000

x_atmosphere = radius_atmosphere* np.cos(angle) + 0.5*radius_earth
y_atmosphere = radius_atmosphere* np.sin(angle) - (np.sqrt(3)*radius_earth)/2
plt.plot(x_atmosphere, y_atmosphere)

x_earth = radius_earth* np.cos(angle) + 0.5*radius_earth
y_earth = radius_earth* np.sin(angle) - (np.sqrt(3)*radius_earth)/2
plt.plot(x_earth, y_earth, color='g')

apparat, = plt.plot([], [], 'o', color = 'k')

def animate(i):
    apparat.set_data(move(i))
    #axes.set_title('Температура КА: ' + sol[i])

ani = FuncAnimation(fig, animate, frames=frames, interval=30)

edge  =  500 * 1000
plt.axis('equal')
plt.xlim(-edge, edge)
plt.ylim(-edge, edge)
#plt.xlim(5*10**6, 6*10**6)
#plt.ylim(3.25*10**6, 4.25*10**6)

ani.save('project_test.gif')
