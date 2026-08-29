import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================
# 1. Datos
# ============================

data = [
    ["OMP Operaciones", 1, 45000, 976.061],
    ["OMP Elementos",   1, 45000, 2151.79],
    ["OMP Operaciones", 4, 45000, 263.081],
    ["OMP Elementos",   4, 45000, 569.693],
    ["OMP Operaciones", 16, 45000, 180.662],
    ["OMP Elementos",   16, 45000, 206.744],

    ["OMP Operaciones", 1, 80000, 1825.86],
    ["OMP Elementos",   1, 80000, 3970.37],
    ["OMP Operaciones", 4, 80000, 466.282],
    ["OMP Elementos",   4, 80000, 1049.35],
    ["OMP Operaciones", 16, 80000, 214.67],
    ["OMP Elementos",   16, 80000, 331.809],

    ["OMP Operaciones", 1, 125000, 2692.79],
    ["OMP Elementos",   1, 125000, 5982.94],
    ["OMP Operaciones", 4, 125000, 675.123],
    ["OMP Elementos",   4, 125000, 1593.61],
    ["OMP Operaciones", 16, 125000, 264.402],
    ["OMP Elementos",   16, 125000, 445.081],
]

df = pd.DataFrame(data, columns=["Tipo", "Hebras", "neqn", "T (s)"])

# ============================
# 2. Orden manual de grupos
# ============================

orden_neqn = [45000, 80000, 125000]

df["neqn"] = pd.Categorical(df["neqn"], categories=orden_neqn, ordered=True)
df = df.sort_values(["neqn", "Hebras", "T (s)"], ascending=[True, True, False])

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

grupos = orden_neqn
hebras = [1, 4, 16]
tipos = ["OMP Operaciones", "OMP Elementos"]

colores = {
    "OMP Operaciones": "#1f77b4",
    "OMP Elementos": "#ff7f0e"
}

fig, ax = plt.subplots(figsize=(14, 7))

offset = 0
xticks = []
xtick_labels = []
group_centers = []

for neqn in grupos:
    df_g = df[df["neqn"] == neqn]

    group_start = offset

    for h in hebras:
        df_h = df_g[df_g["Hebras"] == h]

        base = offset

        for i, tipo in enumerate(tipos):
            valor = df_h[df_h["Tipo"] == tipo]["T (s)"].values[0]

            ax.bar(base + i * (bar_width + pair_spacing),
                   valor,
                   width=bar_width,
                   color=colores[tipo])

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

# ============================
# 6. Leyenda manual (fuente aumentada)
# ============================

handles = [
    plt.Rectangle((0,0),1,1, color=colores["OMP Operaciones"]),
    plt.Rectangle((0,0),1,1, color=colores["OMP Elementos"])
]
labels = ["OMP Operaciones", "OMP Elementos"]
ax.legend(handles, labels, fontsize=16)

# ============================
# 7. Etiquetas y presentación
# ============================

ax.set_ylabel("Tiempo (s)", fontsize=16)
ax.set_xticks(xticks)
ax.set_xticklabels(xtick_labels, fontsize=16)

# Etiquetas de grupo arriba (fuente aumentada)
for center, neqn in zip(group_centers, grupos):
    ax.text(center, ax.get_ylim()[1] * 1.02, f"{neqn//1000}K",
            ha='center', va='bottom', fontsize=18, fontweight='bold')

plt.tight_layout()
plt.show()
