import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "DejaVu Sans"

# ---------------------------------------------------------------------------
# Chart 1: three-way CPR validation vs. official Putrevu et al. (2023) mosaic
# ---------------------------------------------------------------------------
products = ["SLI - default\n(cpr/)", "GRI - default\n(cpr_gri/)", "GRI - research μ_c\n(cpr_gri/)"]
pearson = [0.079, 0.650, -0.213]
colors = ["#0d47a1", "#1b5e20", "#b71c1c"]

fig, ax = plt.subplots(figsize=(8, 5), dpi=220)
bars = ax.barh(products, pearson, color=colors, height=0.55)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlim(-0.4, 0.8)
ax.set_xlabel("Pearson r  vs. official DFSAR CPR mosaic (Putrevu et al. 2023)", fontsize=11)
ax.set_title("CPR Validation — 3 Computed Products vs. ISRO's Official Mosaic", fontsize=12, fontweight="bold")
for bar, v in zip(bars, pearson):
    x = v + (0.02 if v >= 0 else -0.02)
    ha = "left" if v >= 0 else "right"
    ax.text(x, bar.get_y() + bar.get_height() / 2, f"r = {v:+.3f}", va="center", ha=ha, fontsize=11, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("temp/diagrams/stats_cpr_validation.png", facecolor="white", dpi=220)
plt.close()

# ---------------------------------------------------------------------------
# Chart 2: DPSR area across 5 named craters
# ---------------------------------------------------------------------------
craters = ["Shackleton", "Faustini", "Haworth", "Shoemaker", "Cabeus"]
dpsr_km2 = [0.0000, 1.1904, 0.9560, 0.9752, 0.4432]
colors2 = ["#9e9e9e", "#e65100", "#e65100", "#e65100", "#e65100"]

fig, ax = plt.subplots(figsize=(8, 5), dpi=220)
bars = ax.bar(craters, dpsr_km2, color=colors2, width=0.55)
ax.set_ylabel("DPSR area (km$^2$)", fontsize=11)
ax.set_title("Doubly-Shadowed (DPSR) Area per Crater — O'Brien & Byrne (2022) Method", fontsize=11.5, fontweight="bold")
for bar, v in zip(bars, dpsr_km2):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03, f"{v:.3f}", ha="center", fontsize=10.5, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylim(0, 1.4)
plt.tight_layout()
plt.savefig("temp/diagrams/stats_dpsr_craters.png", facecolor="white", dpi=220)
plt.close()

print("saved stats_cpr_validation.png, stats_dpsr_craters.png")
