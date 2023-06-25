p1 = "scenes/lucy-o.obj"
p2 = "scenes/lucy.obj"

f1 = open(p1, "r")
f2 = open(p2, "w")

d1 = f1.readlines()

vl = [line for line in d1 if line.split(" ")[0] == "v"]
fl = [line for line in d1 if line.split(" ")[0] == "f"]

f2.write("".join(vl))

for i in range(len(fl)):
    fli = fl[i].split()
    
    if "//" in fl[i]:
        i0 = int(fli[1].split("//")[0])
        i1 = int(fli[2].split("//")[0])
        i2 = int(fli[3].split("//")[0])
    elif "/" in fl[i]:
        i0 = int(fli[1].split("/")[0])
        i1 = int(fli[2].split("/")[0])
        i2 = int(fli[3].split("/")[0])
        
    if i0 < 0:
        i0 = len(vl) + i0 + 1
    if i1 < 0:
        i1 = len(vl) + i1 + 1
    if i2 < 0:
        i2 = len(vl) + i2 + 1
        
    fl[i] = f"f {i0} {i1} {i2}\n"

f2.write("".join(fl))
