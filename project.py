import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#Дифур
frames = 250
t= np.linspace(0 , 16 , frames)

G = 6.67 * 10**(-11)
M = 6 * 10**(24)
m = 750
k = 0.35

def diff(w, t):
    x, vx, y, vy = w
    dxdt = vx
    dvxdt = (-((G * m * M * x) / (x**2 + y**2)**1.5) + (k  * vx))/m
    dydt = vy
    dvydt = (-((G * m * M * y) / (x**2 + y**2)**1.5) + (k * vy))/m
    return dxdt, dvxdt, dydt, dvydt

x0 = -240 *10**3
vx0 = 21600 * np.cos(45)
y0 = 240 *10**3
vy0 = 21600 * np.sin(45)
w0 = x0, vx0, y0, vy0

sol = odeint(diff, w0, t)

def move(i):
    x= sol[i, 0]
    y= sol[i, 2]
    return x, y

fig, ax = plt.subplots()

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

apparat, = plt.plot([], [], 'o', color = 'b')

def animate(i):
    apparat.set_data(move(i))

ani = FuncAnimation(fig, animate, frames=frames, interval=30)

edge  = 350 * 10**3
plt.axis('equal')
plt.xlim(-edge, edge)
plt.ylim(-edge, edge)

ani.save('project.gif')
