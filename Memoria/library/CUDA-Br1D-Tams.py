import pandas as pd
import matplotlib.pyplot as plt

# ============================
# 1. Datos
# ============================

data = [
    [200,     18.842,   94.21],
    [400,     19.1643,  47.91075],
    [1000,    17.0844,  17.0844],
    [2000,    17.0971,  8.54855],
    [4000,    17.2015,  4.300375],
    [10000,   17.4387,  1.74387],
    [20000,   22.3018,  1.11509],
    [30000,   23.5805,  0.786016667],
    [40000,   28.2738,  0.706845],
    [60000,   33.4105,  0.556841667],
    [80000,   38.5658,  0.4820725],
    [100000,  43.4037,  0.434037],
    [150000,  68.5434,  0.456956],
    [200000,  89.1621,  0.4458105],
    [250000,  115.643,  0.462572],
    [300000,  141.374,  0.471246667],
    [350000,  173.563,  0.495894286],
    [400000,  195.793,  0.4894825],
    [500000,  246.389,  0.492778],
    [600000,  291.951,  0.486585],
    [700000,  335.235,  0.478907143],
    [800000,  379.273,  0.47409125],
    [900000,  423.521,  0.470578889],
    [1000000, 469.071,  0.469071],
    [1500000, 702.068,  0.468045333],
    [2000000, 927.775,  0.4638875],
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
