import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"

eqs = [
    ("Ice-candidate criterion (PS-8)",
     r"$\mathrm{CPR} > 1 \quad \mathrm{AND} \quad \mathrm{DOP} < 0.13$"),
    ("Circular Polarization Ratio",
     r"$\mathrm{CPR} = \dfrac{\sigma_{SC}}{\sigma_{OC}} = "
     r"\dfrac{\left\langle |S_{HH}-S_{VV}+2jS_{HV}|^2 \right\rangle}"
     r"{\left\langle |S_{HH}+S_{VV}|^2 \right\rangle}$"),
    ("Degree of Polarization (Stokes)",
     r"$\mathrm{DOP} = \dfrac{\sqrt{S_1^2+S_2^2+S_3^2}}{S_0}$"),
    ("DPSR Curvature Correction (O'Brien & Byrne 2022, Eq. A4)",
     r"$\tan(\mu) = \dfrac{R_1\left(R_2-\sqrt{d^2+R_1^2}\right)}{d\,R_2}$"),
    ("Ice Confidence Fusion (8 bands, this project)",
     r"$\mathrm{Score}(p) = \dfrac{\sum_i w_i \cdot \mathrm{norm}_i(p)}{\sum_i w_i^{\,valid}(p)}$"),
]

fig, ax = plt.subplots(figsize=(11, 11.5), dpi=220)
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

y = 0.985
label_gap = 0.045
eq_gap = 0.155
for label, eq in eqs:
    ax.text(0.02, y, label, fontsize=17, va="top", ha="left", color="#0d2a4a",
            fontweight="bold")
    ax.text(0.07, y - label_gap, eq, fontsize=21, va="top", ha="left", color="#111111")
    y -= (label_gap + eq_gap)

fig.patch.set_facecolor("white")
plt.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.01)
plt.savefig("temp/diagrams/key_equations.png", facecolor="white", bbox_inches="tight", pad_inches=0.15)
print("saved key_equations.png")
