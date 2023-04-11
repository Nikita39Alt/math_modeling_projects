import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from info import air_temperature
from info import air_heatexchange
from info import air_density

#Дифур
frames = 100
t = np.linspace(0 , 1000 , frames)

G = 6.67 * 10**(-11)
M = 6 * 10**(24)
m = 4730
k = 0.35
radius_earth = 6378100
radius_atmosphere = radius_earth + 120 * 1000

S = 10
S_bok = 11
c_res = 1
c_ka = 897

def diff(w, t):
    global radius_atmosphere
    x, vx, y, vy, T = w
    #if not(y < radius_atmosphere  and y > -radius_atmosphere and x <  radius_atmosphere and x > -radius_atmosphere):
        #dxdt = vx
        #dvxdt = ((G  * M * x) / (x**2 + y**2)**1.5)
        #dydt = vy
        #dvydt = -((G * M * y) / (x**2 + y**2)**1.5)
    #else:
    delta_x = x
    delta_y = y 
    r = np.sqrt(delta_x**2+delta_y**2)
    sina = delta_x / r
    cosa = delta_y / r

    dxdt = vx
    dvxdt = (G * M / (r ** 2) * cosa - vx * np.abs(vx) * air_density(r - radius_earth) * S * c_res / 2 / m) * x
    dydt = vy
    dvydt = -(G * M / (r ** 2) * cosa - vy * np.abs(vy) * air_density(r - radius_earth) * S * c_res / 2 / m) * y
    dTdt = T + np.sqrt(vx ** 2 + vy ** 2) * (air_temperature((x**2 + y**2) - radius_earth) - T) * air_heatexchange(air_temperature((x**2 + y**2) - radius_earth)) * air_density((x**2 + y**2) - radius_earth) * t / c_ka * (S_bok)
    return dxdt, dvxdt, dydt, dvydt, dTdt


x0 = radius_atmosphere * np.cos(60) 
vx0 = 8000 * np.cos(300)
y0 = radius_atmosphere * np.sin(60)
vy0 = -8000 * np.sin(300)
T0 = 2.7
w0 = x0, vx0, y0, vy0, T0

sol = odeint(diff, w0, t)
print(sol)

def move(i):
    x= sol[i, 0]
    y= sol[i, 2]
    return x, y

fig, axes = plt.subplots()

#Земля с атмосферой
angle = np.linspace( 0 , 2 * np.pi , 150 ) 

radius_earth = 6378100
radius_atmosphere = radius_earth + 120 * 1000

x_atmosphere = radius_atmosphere* np.cos(angle)
y_atmosphere = radius_atmosphere* np.sin(angle)
plt.plot(x_atmosphere, y_atmosphere)

x_earth = radius_earth* np.cos(angle)
y_earth = radius_earth* np.sin(angle)
plt.plot(x_earth, y_earth, color='g')

apparat, = plt.plot([], [], 'o', color = 'k')

def animate(i):
    apparat.set_data(move(i))
    #axes.set_title('Температура КА: ' + sol[i])

ani = FuncAnimation(fig, animate, frames=frames, interval=30)

edge  =  radius_atmosphere * 2
plt.axis('equal')
plt.xlim(-edge, edge)
plt.ylim(-edge, edge)

ani.save('project.gif')
