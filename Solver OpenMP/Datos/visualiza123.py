import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

def view(filename):
    x_vals = []
    y_vals = []
    z_vals = []
    matrix_vals = []  # Para el caso C

    tipo = None  # 2 columnas, 3 columnas o N+1 columnas

    with open(filename, 'r') as file:
        for linea in file:
            datos = linea.split()

            # Detectar tipo en la primera línea válida
            if tipo is None:
                if len(datos) == 2:
                    tipo = 2
                elif len(datos) == 3:
                    tipo = 3
                elif len(datos) > 3:
                    tipo = "C"
                else:
                    print(f"Línea ignorada por formato incorrecto: {linea.strip()}")
                    continue

            # Procesar según tipo
            if tipo == 2 and len(datos) == 2:
                x_vals.append(float(datos[0]))
                y_vals.append(float(datos[1]))

            elif tipo == 3 and len(datos) == 3:
                x_vals.append(float(datos[0]))
                y_vals.append(float(datos[1]))
                z_vals.append(float(datos[2]))

            elif tipo == "C" and len(datos) > 3:
                fila = list(map(float, datos))
                x_vals.append(fila[0])
                matrix_vals.append(fila[1:])  # N valores

    if not x_vals:
        print("No se encontraron datos válidos.")
        return

    # -------------------------
    #   CASO A y B (2D)
    # -------------------------
    if tipo in (2, 3):
        plt.figure(figsize=(8, 6))

        if tipo == 2:
            plt.plot(x_vals, y_vals, marker='o', linestyle='-', color='b', label='f(x)')
        else:
            plt.plot(x_vals, y_vals, marker='o', linestyle='-', color='b', label='f1(x)')
            plt.plot(x_vals, z_vals, marker='o', linestyle='-', color='r', label='f2(x)')

        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title(filename)
        plt.legend()
        plt.grid(True)

        plot_file = "plot_" + filename + ".png"
        plt.savefig(plot_file)
        print(f"Gráfica guardada en {plot_file}")
        return

    # -------------------------
    #   CASO C (3D con superficie)
    # -------------------------
    import numpy as np
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Convertir datos a matriz NumPy
    Z = np.array(matrix_vals)[:, 1:]      # matriz N x M
    X = np.array(x_vals)                  # vector N
    M = Z.shape[1]
    Y = np.arange(M)                      # 0,1,2,...,M-1

    # Crear mallas
    X_grid, Y_grid = np.meshgrid(X, Y, indexing='ij')

    # Dibujar superficie
    ax.plot_surface(X_grid, Y_grid, Z, cmap='viridis', edgecolor='none')

    ax.set_xlabel('X')
    ax.set_ylabel('Índice Y')
    ax.set_zlabel('Valor')
    ax.set_title(filename)

    plot_file = "plot_" + filename + ".png"
    plt.savefig(plot_file)
    print(f"Superficie 3D guardada en {plot_file}")



parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
args = parser.parse_args()

view(args.filename)
