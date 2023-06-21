p1 = "scenes/boxa.obj"
f1 = open(p1, "r")
d = f1.readlines()

vl = [line for line in d if line[0] == "v"]
fl = [line for line in d if line[0] == "f"]

vx = [float(line.split(" ")[1]) for line in vl]
vy = [float(line.split(" ")[2]) for line in vl]
vz = [float(line.split(" ")[3]) for line in vl]

print(len(vl))
print(min(vx), max(vx))
print(min(vy), max(vy))
print(min(vz), max(vz))

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
