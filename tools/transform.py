import math
from math import cos, sin

def rotateX(_x, _y, _z, degrees):
    radians = math.radians(degrees)
    x = _x
    y = cos(radians) * _y - sin(radians) * _z
    z = sin(radians) * _y + cos(radians) * _z
    return x, y, z

def rotateY(_x, _y, _z, degrees):
    radians = math.radians(degrees)
    x = cos(radians) * _x + sin(radians) * _z
    y = _y
    z = cos(radians) * _z - sin(radians) * _x
    return x, y, z

def rotateZ(_x, _y, _z, degrees):
    radians = math.radians(degrees)
    x = cos(radians) * _x - sin(radians) * _y
    y = sin(radians) * _x + cos(radians) * _y
    z = _z
    return x, y, z

p1 = "scenes/suzanne.obj"
p2 = "scenes/suzanne1.obj"
f1 = open(p1, "r")
f2 = open(p2, "w")
d = f1.readlines()

vl = [line for line in d if line[0] == "v"]
fl = [line for line in d if line[0] == "f"]

for i in range(len(vl)):
    vli = vl[i].split(" ")
    x, y, z = rotateY(float(vli[1]), float(vli[2]), float(vli[3]), 180)
    vl[i] = f"v {x} {y} {z}\n"

f2.write("".join(vl))
f2.write("".join(fl))