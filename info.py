# дифуры
#
# Переменные:
#   a_x, a_y, v_x, v_y  - Копоненты ускорения и скорости
#   x_earth, y_earth    - Координаты Земли
#   x,y                 - Координаты ЛА
#   G                   - Гравитационная постоянная
#   c_res, S, S_bok     - Обтекаемость, площадь основания капсулы, площадь боковой поверхности
#                           По умолчанию (соответственно): как в таблице, 10, 11
#                           Используется для расчета сопротивления и теплового потока.
#   T                   - Термодинамическая температура капсулы. Изначальное T = 2.7 [K]
#   c_ka                - Теплоемкость капсулы. с_ka = 897 [Дж/(кг К)]  Алюминий
#   m                   - Масса КА          m = 4730 [кг]
#   M, R                - Масса и радиус Земли
#
#   t                   - Точность программы (т.е. насколько часто в секундах, она расчитывает данные)
#
#   ОЧЕНЬ ВАЖНЫЙ КОММЕНТАРИЙ: ВСЕ ВЕЛИЧИНЫ ДАНЫ В СИ. КООРДИНАТЫ ЗЕМЛИ И КА ДОЛЖНЫ БЫТЬ ПРИВЕДЕНЫ К СИ
#   Рекомендация:   зная радиус Земли на экране и в жизни, сделай переменную, в которой будет храниться отношение экрана к реальной жизни
#
#               R_earth_screen
#   пусть K = ------------------        ВЗЯТ ПОЛЯРНЫЙ РАДИУС ЗЕМЛИ
#                 6 356 777

import math #библиотека math нужна даже несмотря на то, что она не нравится учителю. Можно и через numpy sqrt (если в numpy такое есть, я просто не помню)

# В ТЕЛО ПРОГРАММЫ
G = 6.67430 * 10 ** (-11)
M = 5.972 * 10**24
R = 6356777
m = 4730

S = 10
S_bok = 11
c_res = 1
c_ka = 897

T = 2.7

a_pressure = [  [0,101300],         [5000,54052],           [10000,26500],
                [15000, 12266],     [20000,5529],           [28000,1616],
                [32000,889],        [40000,287],            [50000,80],
                [60000,22],         [80000,1],              [100000, 3.19*10**-2]
            ]

a_density = [   [0,1.2250],         [5000,0.7365],          [10000,0.4135],
                [15000, 0.1972],    [20000,0.0889],         [28000,0.0251],
                [32000,0.0136],     [40000,4*10**-3],       [50000,1.03*10**-3],
                [60000,3*10**-4],   [80000,1.85*10**-5],    [100000, 5.55*10**-7]
            ]

a_temp    = [   [0,293],            [11000,216.8],          [20000,216.8],
                [32000,228.5],      [48000,270.7],          [53000,270.7],
                [60500, 247],       [80000,198.6],          [90000,198.6],
                [500000,1000],      [1000000,1000]
            ]

a_heatexchange =    [   [0,1013],   [243,1013],     [253,1009],     [263,1009],
                        [273,1005], [333,1005],     [343,1009],     [393, 1009],
                        [433, 1017],[773,1140],     [1000,1100],    [1400,1200]
                    ]

def approx(data,x,default):
    if data[-1][0] < x:
        return default
    elif x <= 0 :
        return 0
    else:
        for i in range(len(data)-1):
            if x == data[i][0]:
                return data[i][1]
            if data[i][0] < x < data[i+1][0]:
                print(data[i][0],data[i+1][0])
                k = (data[i+1][1]-data[i][1]) / (data[i+1][0]-data[i][0])
                b = data[i][1] - k * data[i][0]
                return b + k * (x)

def air_temperature(height):
    return approx(a_temp, height, 2.7)
def air_heatexchange(temperature):
    return approx(a_heatexchange, temperature, 1200)
def air_density(height):
    return approx(a_density, height, 0)
def air_pressure(height):
    return approx(a_pressure, height, 0)

# КОНЕЦ

# В ФУНКЦИЮ ДИФФУРА
'''
x_earth, y_earth, x, y=1,1,2,2           #удалить, это нужно было просто, чтобы питон не ругался, что переменная не объявлена
delta_x = (x_earth - x) # /K
delta_y = (y_earth - y) # /K
r = math.sqrt(delta_x**2+delta_y**2)
sina = delta_x / r
cosa = delta_y / r


a_x = G * M / (r ** 2) * cosa - v_x * abs(v_x) * air_density(r - R) * S * c_res / 2 / m  # *K
a_y = G * M / (r ** 2) * sina - v_y * abs(v_y) * air_density(r - R) * S * c_res / 2 / m  # *K

T += math.sqrt(v_x ** 2 + v_y ** 2) * (air_temperature(r - R) - T) * air_heatexchange(air_temperature(r - R)) * air_density(r - R) * t / c_ka * (S_bok)
# КОНЕЦ
'''