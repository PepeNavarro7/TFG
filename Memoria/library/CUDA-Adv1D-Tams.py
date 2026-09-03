import pandas as pd
import matplotlib.pyplot as plt

# ============================
# 1. Datos (actualizados)
# ============================

data = [
    [50,     27.43,    548.6],
    [100,    21.4263,  214.263],
    [200,    28.7236,  143.618],
    [500,    31.0415,  62.083],
    [1000,   30.878,   30.878],
    [2000,   31.2256,  15.6128],
    [5000,   31.2184,  6.24368],
    [10000,  31.1457,  3.11457],
    [15000,  32.7945,  2.1863],
    [20000,  51.7041,  2.585205],
    [30000,  53.4058,  1.780193333],
    [40000,  73.87,    1.84675],
    [50000,  73.4573,  1.469146],
    [75000,  115.031,  1.533746667],
    [100000, 134.525,  1.34525],
    [125000, 179.365,  1.43492],
    [150000, 202.547,  1.350313333],
    [175000, 247.074,  1.411851429],
    [200000, 270.135,  1.350675],
    [250000, 340.207,  1.360828],
    [300000, 411.625,  1.372083333],
    [350000, 484.814,  1.385182857],
    [400000, 542.487,  1.3562175],
    [450000, 615.779,  1.368397778],
    [500000, 687.308,  1.374616],
    [750000, 1044.07,  1.392093333],
    [1000000, 1375.45, 1.37545],
]

df = pd.DataFrame(data, columns=["neqn", "Ts", "ms_neqn"])

# ============================
# 2. Preparar la gráfica
# ============================

fig, ax1 = plt.subplots(figsize=(14, 7))
ax2 = ax1.twinx()  # segundo eje Y

# ============================
# 3. Líneas
# ============================

ax1.plot(df["neqn"], df["Ts"], color="#1f77b4", linewidth=2.5, label="Ts (s)")
ax2.plot(df["neqn"], df["ms_neqn"], color="#ff7f0e", linewidth=2.5, label="ms/neqn")

# ============================
# 4. Escalas logarítmicas
# ============================

ax1.set_xscale("log")
ax2.set_xscale("log")

ax1.set_yscale("log")
ax2.set_yscale("log")

# ============================
# 5. Líneas horizontales suaves
# ============================

ax1.grid(axis='y', linestyle='--', linewidth=0.5, color='gray', alpha=0.4)

# Leyenda centrada arriba dentro del recuadro
lines = [
    plt.Line2D([0], [0], color="#1f77b4", linewidth=3),
    plt.Line2D([0], [0], color="#ff7f0e", linewidth=3)
]
labels = ["T(s)", "ms/neqn"]

ax1.legend(lines, labels,
           fontsize=15,
           loc="upper center",
           bbox_to_anchor=(0.5, 1.0))

# ============================
# 7. Etiquetas y presentación
# ============================

ax1.set_xlabel("neqn [log]", fontsize=15)
ax1.set_ylabel("T(s) [log]", fontsize=15)
ax2.set_ylabel("ms/neqn [log]", fontsize=15)

ax1.tick_params(labelsize=14)
ax2.tick_params(labelsize=14)

plt.tight_layout()
plt.show()
