import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================
# 1. Datos
# ============================

data = [
    ["RK4",  "OMP Elementos",   1, 2501.84],
    ["RK4",  "OMP Operaciones", 1, 1544.87],
    ["RK4",  "OMP Elementos",   4, 670.504],
    ["RK4",  "OMP Operaciones", 4, 371.93],
    ["RK4",  "OMP Elementos",   16, 197.065],
    ["RK4",  "OMP Operaciones", 16, 180.327],

    ["AB4",  "OMP Elementos",   1, 707.634],
    ["AB4",  "OMP Operaciones", 1, 556.051],
    ["AB4",  "OMP Elementos",   4, 179.559],
    ["AB4",  "OMP Operaciones", 4, 130.993],
    ["AB4",  "OMP Elementos",   16, 56.5174],
    ["AB4",  "OMP Operaciones", 16, 66.1485],

    ["ABM4", "OMP Elementos",   1, 2914.04],
    ["ABM4", "OMP Operaciones", 1, 1772.73],
    ["ABM4", "OMP Elementos",   4, 785.555],
    ["ABM4", "OMP Operaciones", 4, 438.018],
    ["ABM4", "OMP Elementos",   16, 237.236],
    ["ABM4", "OMP Operaciones", 16, 219.918],
]

df = pd.DataFrame(data, columns=["Solver", "Estrategia", "Hebras", "T"])

# ============================
# 2. Orden manual de grupos
# ============================

solvers = ["RK4", "AB4", "ABM4"]
df["Solver"] = pd.Categorical(df["Solver"], categories=solvers, ordered=True)
df = df.sort_values(["Solver", "Hebras"])

# ============================
# 3. Parámetros de espaciado
# ============================

bar_width = 0.35
pair_spacing = 0.05
intra_group_spacing = 0.15
inter_group_spacing = 0.60

# ============================
# 4. Preparar la gráfica
# ============================

hebras = [1, 4, 16]
estrategias = ["OMP Operaciones", "OMP Elementos"]

colores = {
    "OMP Operaciones": "#1f77b4",
    "OMP Elementos": "#ff7f0e"
}

fig, ax = plt.subplots(figsize=(14, 7))

offset = 0
xticks = []
xtick_labels = []
group_centers = []

for solver in solvers:
    df_s = df[df["Solver"] == solver]

    group_start = offset

    for h in hebras:
        df_h = df_s[df_s["Hebras"] == h]

        base = offset

        for i, est in enumerate(estrategias):
            valor = df_h[df_h["Estrategia"] == est]["T"].values[0]

            ax.bar(base + i * (bar_width + pair_spacing),
                   valor,
                   width=bar_width,
                   color=colores[est])

        pareja_center = base + (bar_width + pair_spacing) / 2
        hebra_label = "1 hebra" if h == 1 else f"{h} hebras"

        xticks.append(pareja_center)
        xtick_labels.append(hebra_label)

        offset += (2 * bar_width + pair_spacing) + intra_group_spacing

    group_end = offset - intra_group_spacing
    group_centers.append((group_start + group_end) / 2)

    offset += inter_group_spacing

# ============================
# 5. Líneas horizontales suaves
# ============================

ax.grid(axis='y', linestyle='--', linewidth=0.5, color='gray', alpha=0.4)

# Leyenda manual (fuente aumentada)
handles = [
    plt.Rectangle((0,0),1,1, color=colores["OMP Operaciones"]),
    plt.Rectangle((0,0),1,1, color=colores["OMP Elementos"])
]
labels = ["OMP Operaciones", "OMP Elementos"]

# Leyenda arriba a la derecha, dentro del recuadro
ax.legend(handles, labels,
          fontsize=15,
          loc="upper right")

# ============================
# 7. Etiquetas y presentación
# ============================

ax.set_ylabel("Tiempo (s)", fontsize=15)
ax.set_xticks(xticks)
ax.set_xticklabels(xtick_labels, fontsize=15)

# Etiquetas de grupo arriba
for center, solver in zip(group_centers, solvers):
    ax.text(center, ax.get_ylim()[1] * 1.02, solver,
            ha='center', va='bottom', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.show()
