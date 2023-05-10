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
temp_atm = []
temp2 = []

height = []
height2 = []

inter = 30

def diff(x, vx, y, vy, T):
    global resistance
    dvxdt = -(G * M * x) / ((x**2 + y**2)**1.5) - vx * np.abs(vx) * S * c_res / 2 / m * air_density(np.sqrt(x**2 + y**2) - radius_earth) * resistance
    dvydt = -(G * M * y) / ((x**2 + y**2)**1.5) - vy * np.abs(vy) * S * c_res / 2 / m * air_density(np.sqrt(x**2 + y**2) - radius_earth) * resistance
    dTdt = np.sqrt(vx ** 2 + vy ** 2) * (air_temperature(np.sqrt(x**2 + y**2) - radius_earth) - T) * air_heatexchange(air_temperature(np.sqrt(x**2 + y**2) - radius_earth)) * air_density(np.sqrt(x**2 + y**2) - radius_earth) / c_ka * (S_bok)
    if T + dTdt < air_temperature(np.sqrt(x**2 + y**2) - radius_earth) and air_temperature(np.sqrt(x**2 + y**2) - radius_earth) <= T:
        dTdt = air_temperature(np.sqrt(x**2 + y**2) - radius_earth) - T
    elif T + dTdt > air_temperature(np.sqrt(x**2 + y**2) - radius_earth) and air_temperature(np.sqrt(x**2 + y**2) - radius_earth) >= T:
        dTdt = air_temperature(np.sqrt(x**2 + y**2) - radius_earth) - T
    dTdt += 2*(dvxdt**2 + dvydt**2)/c_ka
    if resistance == 1:
        vel.append(np.sqrt(vx**2+vy**2))
        temp.append(T)
        temp_atm.append(air_temperature(np.sqrt(x**2 + y**2) - radius_earth))
        height.append(np.sqrt(x**2+y**2) - radius_earth)
    else:
        vel2.append(np.sqrt(vx**2+vy**2))
        temp2.append(T)
        height2.append(np.sqrt(x**2+y**2) - radius_earth)
        
    if np.sqrt(x**2 + y**2) > radius_earth+100:
        return x+vx, vx+dvxdt, y+vy, vy+dvydt,T+dTdt
    else:
        return 0,0,0,0,T

os.system('cls')
choice = input('Атмосфера Земли велика, но практически незаметна на фоне самой планеты. Поэтому вам предлагаются 2 варианта:\nРассмотреть приближение КА к Земле\n или Рассмотреть вход в земную атмосферу\n\nНапишите в терминал 1 или 2 в соответсвтвии с выбранным вами вариантом.\n>> ')

if choice == '2':
    frames = 1500
    x = -926132.2632431979
    vx = -7333.474913381629 
    y = 6839659.83114655
    vy = -4232.321797974888
    T = 2.7
    
    resistance = 1
    sol = []
    for i in range(frames):
        if x !=0 and y != 0:
            x,vx,y,vy,T = diff(x,vx,y,vy,T)
            sol.append([x,y])

    frames = 1500
    x = -926132.2632431979
    vx = -7333.474913381629 
    y = 6839659.83114655
    vy = -4232.321797974888
    T = 2.7
    resistance = 0
    sol2 = []
    for i in range(frames):
        if x !=0 and y != 0:
            x,vx,y,vy,T = diff(x,vx,y,vy,T)
            sol2.append([x,y])
    
    e_xa, e_xb = -3000000,-1000000
    e_ya, e_yb = 6000000,7000000
    
    inter = inter*3


def move(data,i):
    if i < len(data):
        x= data[i][0]
        y= data[i][1]
    else:
        x= data[-1][0]
        y= data[-1][1]
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

if choice == '2':
    fig, axes = plt.subplots()
    plt.plot(np.arange(0,len(temp)), temp, color='g')
    plt.plot(np.arange(0,len(temp_atm)), temp2, color='cyan')
    plt.xlabel('Время (с)')
    plt.ylabel('Абсолютная температура (К)')
    plt.title('Температура КА во сремя спуска в атмосфере Земли в сравнении с температурой ISA')
    plt.grid()
    plt.savefig('atmtemp.png')
    os.system('start atmtemp.png')
