import matplotlib.pyplot as plt
from math import sqrt
from statistics import mean, stdev
import numpy as np

def from_file(path):
    with open(path, "r") as file:
        data = file.readline().split()
        data = [float(entry) for entry in data]
        return data
    
teapot_cpu = mean(from_file("data/teapot_cpu.txt"))
teapot_gpu = mean(from_file("data/teapot_gpu.txt"))

platforms = ["CPU", "GPU"]
y_pos = np.arange(len(platforms))

plt.rcdefaults()
fig, ax = plt.subplots()    

ax.bar(platforms, [teapot_cpu, teapot_gpu], align="center", color=["r", "g"])
ax.set_xticks(y_pos, labels=platforms)
# ax.invert_yaxis()
ax.set_ylabel("Time (Seconds)")
ax.set_title("Render Times")
plt.savefig("plots/render_teapot.png", dpi=1200)

print("Teapot CPU mean: " + str(teapot_cpu))
print("Teapot GPU mean: " + str(teapot_gpu))
print("Speedup: " + str(teapot_cpu/teapot_gpu))
