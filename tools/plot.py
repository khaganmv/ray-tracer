import matplotlib.pyplot as plt
from math import sqrt
from statistics import mean, stdev
import numpy as np

def from_file(path):
    with open(path, "r") as file:
        data = file.readline().split()
        data = [float(entry) for entry in data]
        return data

plt.rcdefaults()
plt.figure(figsize=(8, 8))
fig, ax = plt.subplots()

bunny_cpu = mean(from_file("data/bunny_cpu.txt"))
bunny_gpu = mean(from_file("data/bunny_gpu.txt"))
erato_cpu = mean(from_file("data/erato_cpu.txt"))
erato_gpu = mean(from_file("data/erato_gpu.txt"))
dragon_cpu = mean(from_file("data/dragon_cpu.txt"))
dragon_gpu = mean(from_file("data/dragon_gpu.txt"))
aurelius_cpu = mean(from_file("data/aurelius_cpu.txt"))
aurelius_gpu = mean(from_file("data/aurelius_gpu.txt"))

scenes = ["Stanford Bunny\n(144k)", "Erato\n(412k)", "Stanford Dragon\n(871k)", "Marcus Aurelius\n(1704k)"]
x = np.arange(len(scenes))
cpu = [bunny_cpu, erato_cpu, dragon_cpu, aurelius_cpu]
gpu = [bunny_gpu, erato_gpu, dragon_gpu, aurelius_gpu]

WIDTH = 0.3

plt.bar(x,         cpu, WIDTH, label="CPU", color="r")
plt.bar(x + WIDTH, gpu, WIDTH, label="GPU", color="g")

plt.xticks(x + WIDTH / 2, scenes, fontsize=8)
plt.xlabel("Scenes")
plt.ylabel("Time (Seconds)")
plt.title("Render Times")
plt.legend()
plt.savefig("plots/render.png", dpi=1200)

plt.cla()
plt.clf()

fig, ax = plt.subplots()

speedup = [bunny_gpu/bunny_cpu, erato_gpu/erato_cpu, dragon_gpu/dragon_cpu, aurelius_gpu/aurelius_cpu]
speedup_below = [
    mean(speedup), 
    mean(speedup), 
    mean(speedup), 
    aurelius_gpu/aurelius_cpu
]

plt.bar(x, speedup,       WIDTH, color="r")
plt.bar(x, speedup_below, WIDTH)
ax.axhline(mean(speedup), color="grey", label="Mean")

plt.ylim(0, 5)
plt.xticks(x, scenes, fontsize=8)
plt.xlabel("Scenes")
plt.ylabel("Factor")
plt.title("Speedup")
plt.legend()
plt.savefig("plots/speedup.png", dpi=1200)

print("Bunny CPU mean: " + str(bunny_cpu))
print("Bunny GPU mean: " + str(bunny_gpu))
print("Speedup: " + str(bunny_gpu/bunny_cpu))
print("Erato CPU mean: " + str(erato_cpu))
print("Erato GPU mean: " + str(erato_gpu))
print("Speedup: " + str(erato_gpu/erato_cpu))
print("Dragon CPU mean: " + str(dragon_cpu))
print("Dragon GPU mean: " + str(dragon_gpu))
print("Speedup: " + str(dragon_gpu/dragon_cpu))
print("Aurelius CPU mean: " + str(aurelius_cpu))
print("Aurelius GPU mean: " + str(aurelius_gpu))
print("Speedup: " + str(aurelius_gpu/aurelius_cpu))
