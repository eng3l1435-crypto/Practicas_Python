import pandas as pd

datos = {
    "estudiante": [
        "Ana", "Luis", "María", "Sofía", "Pedro", "Elena", "Jorge", "Carlos",
        "Valentina", "Santiago", "Camila", "Alejandro", "Lucía", "Diego", 
        "Valeria", "Mariana", "Daniel", "Victoria", "David", "Gabriela"
    ],
    "horas_estudio": [
        8, 3, 10, 5, 2, 7, 4, 6,
        3, 9, 2, 7, 5, 8, 1, 4, 7, 3, 9, 5
    ],
    "asistencia": [
        95, 60, 98, 75, 85, 55, 90, 70,
        65, 99, 70, 85, 78, 94, 52, 76, 90, 60, 93, 80
    ],
    "actividades_entregadas": [
        100, 50, 100, 80, 90, 40, 95, 65,
        60, 100, 70, 90, 75, 92, 50, 70, 88, 55, 96, 75
    ],
    "calificacion": [
        92, 58, 96, 72, 81, 50, 88, 64,
        62, 97, 68, 86, 74, 93, 49, 73, 89, 57, 92, 77
    ]
}

df = pd.DataFrame(datos)

X = df[[
    "horas_estudio",
    "asistencia",
    "actividades_entregadas"
]]

y = df["calificacion"]

print("VARIABLES DE ENTRADA: X")
print(X.head(20))

print("\nVARIABLE OBJETIVO: y")
print(y.head(20))

print("\nDimensiones de X:", X.shape)
print("Dimensiones de y:", y.shape)

assert len(X) == len(y), "X e y tienen diferente cantidad de filas"
print("\nLas entradas y el objetivo están alineados.")