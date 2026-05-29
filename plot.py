import matplotlib.pyplot as plt

times = []
scores = []

with open("validation.csv", "r") as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        t, score = line.split(",")

        times.append(float(t))
        scores.append(float(score))

plt.figure(figsize=(7, 5))

plt.plot(
    times,
    scores,
    linewidth=2,
    label="Countdown"
)

plt.xlabel("Relative wall-clock time (hours)")
plt.ylabel("Validation Score")

plt.grid(True, alpha=0.25)

plt.legend(frameon=False)

plt.tight_layout()

plt.savefig("figure_4b.png", dpi=300)
plt.savefig("figure_4b.pdf")

print("Saved figure_4b.png")
print("Saved figure_4b.pdf")

plt.show()