import matplotlib.pyplot as plt
import numpy as np
import argparse


def view (filename):
    x_vals = []
    y_vals = []
    z_vals = []

    with open(filename, 'r') as file:
        for linea in file:
            datos = linea.split()
            if len(datos) == 2:
                try:
                    x = float(datos[0])
                    y = float(datos[1])
                    x_vals.append(x)
                    y_vals.append(y)
                except ValueError:
                    print(f"Línea ignorada por formato incorrecto: {linea.strip()}")

            elif len(datos) == 3:
                try:
                    x = float(datos[0])
                    y = float(datos[1])
                    z = float(datos[2])
                    x_vals.append(x)
                    y_vals.append(y)
                    z_vals.append(z)
                except ValueError:
                    print(f"Línea ignorada por formato incorrecto: {linea.strip()}")


    if not x_vals:
        print("No se encontraron datos válidos en el archivo.")
        return

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, marker='o', linestyle='-', color='b', label='f(x)')
    if len(datos) == 3:
        plt.plot(x_vals, z_vals, marker='o', linestyle='-', color='r', label='f2(x)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(filename)
    plt.legend()
    plt.grid(True)
    plot_file="plot_"+filename+".png"
    plt.savefig(plot_file)

parser=argparse.ArgumentParser()
parser.add_argument("filename",type=str)
args=parser.parse_args()

view(args.filename)
