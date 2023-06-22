import matplotlib.pyplot as plt
from math import sqrt
from statistics import mean, stdev

plt.rcdefaults()
fig, ax = plt.subplots()

platforms = ["CPU", "GPU"]
teapot_cpu = [4.172, 5.072, 4.94, 4.448, 4.491, 5.248, 4.326, 4.056, 3.89, 4.271]
teapot_gpu = [1.832, 1.797, 1.773, 1.773, 1.736, 1.769, 1.741, 1.794, 1.769, 1.751]
suzanne_cpu = [17.632, 16.388, 18.028, 16.906, 16.647, 17.208, 16.197, 18.166, 20.569, 17.074]
suzanne_gpu = [4.194, 4.233, 3.99, 3.993, 3.99, 3.993, 4.037, 4.01, 4.025, 3.971]
bunny_cpu = [95.407, 92.625, 91.566, 93.711, 90.759, 90.324, 89.377, 91.252, 93.399, 92.186]
bunny_gpu = [16.108, 16.381, 15.984, 17.441, 17.731, 16.704, 16.452, 16.764, 16.366, 16.773]
serapis_cpu = [172.865, 173.803, 174.224, 166.878, 168.729, 169.151, 167.748, 166.952, 159.985, 164.973]
serapis_gpu = [28.425, 27.199, 28.15, 29.601, 28.324, 28.22, 28.198, 29.528, 28.153, 28.118]
box_cpu = [19.416, 19.191, 19.646, 18.584, 19.258, 19.155, 19.616, 22.248, 20.167, 21.148]
box_gpu = [8.119, 7.926, 8.019, 8.465, 8.291, 7.976, 7.759, 7.827, 8.253, 8.252]
boxa_cpu = [414.005, 425.796, 410.866, 402.01, 401.535, 401.937, 399.682, 428.186, 428.574, 423.707]
boxa_gpu = [85.125, 84.762, 84.978, 85.61, 84.46, 82.945, 85.696, 83.766, 83.042, 81.385]

d1 = boxa_cpu
d2 = boxa_gpu

error = [stdev(d1) / sqrt(len(d1)), stdev(d2) / sqrt(len(d2))]
print(f"Speedup: {(mean(d1) / mean(d2)):.2f}")

ax.barh(platforms, [mean(d1), mean(d2)], xerr=error, align='center')
ax.invert_yaxis()
ax.set_xlabel('Seconds')
ax.set_title('Cornell Box - Stanford Bunny (512x512)')

plt.savefig("plots/boxa.png")
plt.savefig("plots/boxa.pdf")
