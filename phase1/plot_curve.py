"""Plot the learning curve from training output.

Usage:
    .venv/bin/python phase1/plot_curve.py
"""
import matplotlib.pyplot as plt

# 从训练输出里复制的数据（无 step override 版本，reward ~1248）
steps = [0, 3_276_800, 6_553_600, 9_830_400, 13_107_200,
         16_384_000, 19_660_800, 22_937_600, 26_214_400, 29_491_200]

rewards = [
    643.16,
    1209.06,
    1078.34,
    1233.39,
    1252.15,
    1190.62,
    1241.51,
    1222.78,
    1250.32,
    1253.28,
]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot([s / 1e6 for s in steps], rewards, marker='o', linewidth=2)
ax.set_xlabel("Training Steps (millions)")
ax.set_ylabel("Episode Reward")
ax.set_title("PPO Baseline — Placement Task Learning Curve")
ax.axhline(y=1282.5, color='r', linestyle='--', alpha=0.5, label="Theoretical max (1282.5)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("learning_curve.png", dpi=150)
print("Saved learning_curve.png")
plt.show()
