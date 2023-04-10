import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import os

from info import air_temperature
from info import air_heatexchange
from info import air_density

#Дифур

G = 6.67 * 10**(-11)
M = 6 * 10**(24)
m = 4730
k = 0.35
radius_earth = 6378100
radius_atmosphere = radius_earth + 120000

S = 10
S_bok = 11
c_res = 1
c_ka = 897

vel = []
vel2 = []

temp = []
temp2 = []

height = []
height2 = []

inter = 30
a,b,c,d,e = 0,0,0,0,0

def diff(w, t):
    global resistance
    x, vx, y, vy, T = w
    dxdt = vx
    dvxdt = -(G * M * x) / ((x**2 + y**2)**1.5) - vx * np.abs(vx) * S * c_res / 2 / m * air_density(np.sqrt(x**2 + y**2) - radius_earth) * resistance
    dydt = vy
    dvydt = -(G * M * y) / ((x**2 + y**2)**1.5) - vy * np.abs(vy) * S * c_res / 2 / m * air_density(np.sqrt(x**2 + y**2) - radius_earth) * resistance
    dTdt = np.sqrt(vx ** 2 + vy ** 2) * (air_temperature((x**2 + y**2) - radius_earth) - T) * air_heatexchange(air_temperature((x**2 + y**2) - radius_earth)) * air_density((x**2 + y**2) - radius_earth) * t / c_ka * (S_bok)
    if resistance == 1:
        vel.append(np.sqrt(vx**2+vy**2))
        temp.append(T)
        height.append(np.sqrt(x**2+y**2) - radius_earth)
    else:
        vel2.append(np.sqrt(vx**2+vy**2))
        temp2.append(T)
        height2.append(np.sqrt(x**2+y**2) - radius_earth)
        
    if np.sqrt(x**2 + y**2) > radius_earth+100:
        a,b,c,d,e = x,vx,y,vy,T
        return dxdt, dvxdt, dydt, dvydt, dTdt
    else:
        return 0,0,0,0, dTdt

os.system('cls')
choice = input('Атмосфера Земли велика, но практически незаметна на фоне самой планеты. Поэтому вам предлагаются 2 варианта:\nРассмотреть приближение КА к Земле\n или Рассмотреть вход в земную атмосферу\n\nНапишите в терминал 1 или 2 в соответсвтвии с выбранным вами вариантом.\n>> ')

if choice == '2':
    frames = 1000
    t= np.linspace(0 , 750, frames)
    x0 = -926132.2632431979
    vx0 = -7333.474913381629 
    y0 = 6839659.83114655
    vy0 = -4232.321797974888
    T0 = 2.7
    w0 = x0, vx0, y0, vy0, T0
    
    resistance = 1
    sol = odeint(diff, w0, t)
    resistance = 0
    sol2 = odeint(diff, w0, t)
    
    e_xa, e_xb = -3000000,-1000000
    e_ya, e_yb = 6000000,7000000
    
    inter = inter*3
    
else:
    frames = 250
    t= np.linspace(0 , 5000, frames)
    
    x0 = radius_atmosphere*2
    vx0 = 0
    y0 = 0
    vy0 = 7921* np.cos(45)
    T0 = 2.7
    w0 = x0, vx0, y0, vy0, T0

    resistance = 1
    sol = odeint(diff, w0, t)
    resistance = 0
    sol2 = odeint(diff, w0, t)
    
    e_xa, e_xb = -radius_atmosphere*2.5,radius_atmosphere*2.5
    e_ya, e_yb = -radius_atmosphere*2.5,radius_atmosphere*2.5

def move(data,i):
    x= data[i, 0]
    y= data[i, 2]
    return x, y

fig, axes = plt.subplots()

#Земля с атмосферой
angle = np.linspace( 0 , 2 * np.pi , 150 ) 

x_atmosphere = radius_atmosphere* np.cos(angle)
y_atmosphere = radius_atmosphere* np.sin(angle)
plt.plot(x_atmosphere, y_atmosphere, color='b')

x_earth = radius_earth* np.cos(angle)
y_earth = radius_earth* np.sin(angle)
plt.plot(x_earth, y_earth, color='g')

apparat, = plt.plot([], [], 'o', color = 'g')
apparat2, = plt.plot([], [], 'o', color = 'r')

def animate(i):
    apparat.set_data(move(sol,i))
    apparat2.set_data(move(sol2,i))
    #axes.set_title('Температура КА: ' + sol[i])

ani = FuncAnimation(fig, animate, frames=frames, interval=inter)

edge  =  radius_atmosphere * 2
plt.axis('equal')
plt.xlim(e_xa, e_xb)
plt.ylim(e_ya, e_yb)
ani.save('project.gif')



fig, axes = plt.subplots()
plt.plot(np.arange(0,len(vel)), vel, color='g')
plt.plot(np.arange(0,len(vel2)), vel2, color='r')
plt.xlabel('Время (с)')
plt.ylabel('Скорость (м/с)')
plt.title('Изменение скорости при входе в атмосферу Земли')
plt.grid()
plt.savefig('velocity.png')
os.system('start velocity.png')


fig, axes = plt.subplots()
plt.plot(np.arange(0,len(height)), height, color='g')
plt.plot(np.arange(0,len(height2)), height2, color='r')
plt.xlabel('Время (с)')
plt.ylabel('Высота (м)')
plt.title('Спуск КА в атмосфере Земли')
plt.grid()
plt.savefig('height.png')
os.system('start height.png')


fig, axes = plt.subplots()
plt.plot(np.arange(0,len(temp)), temp, color='g')
plt.plot(np.arange(0,len(temp2)), temp2, color='r')
plt.xlabel('Время (с)')
plt.ylabel('Абсолютная температура (К)')
plt.title('Температура КА во сремя спуска в атмосфере Земли')
plt.grid()
plt.savefig('temperature.png')
os.system('start temperature.png')