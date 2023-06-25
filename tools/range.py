from math import ceil

p1 = "scenes/lucy.obj"
f1 = open(p1, "r")
d = f1.readlines()

vl = [line for line in d if line[0] == "v"]
fl = [line for line in d if line[0] == "f"]

vx = [float(line.split()[1]) for line in vl]
vy = [float(line.split()[2]) for line in vl]
vz = [float(line.split()[3]) for line in vl]

print(f"v: {len(vl)}")
print(f"f: {len(fl)}")

print(f"x: {min(vx)} {max(vx)}")
print(f"y: {min(vy)} {max(vy)}")
print(f"z: {min(vz)} {max(vz)}")

print(f"xm: {(min(vx) + max(vx)) / 2}")
print(f"zm: {(min(vz) + max(vz)) / 2}")

erato_rat = 27360 / 216000

obj_w = abs(min(vx)) + abs(max(vx))
obj_h = abs(min(vy)) + abs(max(vy))
obj_d = abs(min(vz)) + abs(max(vz))

obj_vol  = obj_w * obj_h * obj_d
box_vol  = obj_vol / erato_rat
box_side = box_vol ** (1. / 3.)

print(f"obj vol: { obj_vol }")
print(f"box vol: { box_vol }")
print(f"box side: { box_side }")
print(f"x range: {((min(vx) + max(vx)) / 2) - (box_side / 2)} to {((min(vx) + max(vx)) / 2) + (box_side / 2)}")
print(f"z range: {((min(vz) + max(vz)) / 2) - (box_side / 2)} to {((min(vz) + max(vz)) / 2) + (box_side / 2)}")

faces = [
    [1, 2, 3],
    [3, 4, 1],
    [5, 6, 7],
    [7, 8, 5],
    [4, 3, 7],
    [7, 8, 4],
    [3, 2, 6],
    [6, 7, 3],
    [1, 4, 8],
    [8, 5, 1]
]

for face in faces:
    print(f"f {face[0] + len(vl)} {face[1] + len(vl)} {face[2] + len(vl)}")
