import matplotlib.pyplot as plt

triangles = [6320, 15488, 69630, 88040]
triangles_r = [6330, 69640]
speedup = [2.53, 4.32, 5.52, 5.94]
speedup_r = [2.45, 4.91]

fig, ax = plt.subplots()

ax.plot(triangles_r, speedup_r)
ax.set(xlabel="Triangles", ylabel="Seconds", title="Speedup (Reflective)")

fig.savefig("plots/speedup_r.pdf")
fig.savefig("plots/speedup_r.png")
