import matplotlib.pyplot as plt
import numpy as np
#Земля
angle = np.linspace( 0 , 2 * np.pi , 150 ) 

radius_earth = 6378100
radius_atmosphere = radius_earth + 120 * 1000

x_atmosphere = radius_atmosphere* np.cos(angle) + 0.5*radius_earth
y_atmosphere = radius_atmosphere* np.sin(angle) - (np.sqrt(3)*radius_earth)/2
plt.plot(x_atmosphere, y_atmosphere)

x_earth = radius_earth* np.cos(angle) + 0.5*radius_earth
y_earth = radius_earth* np.sin(angle) - (np.sqrt(3)*radius_earth)/2
plt.plot(x_earth, y_earth, color='g')

edge  = 120000 * 2
plt.axis('equal')
plt.xlim(-edge, edge)
plt.ylim(-edge, edge)

plt.savefig('project')
