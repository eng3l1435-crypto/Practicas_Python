import pandas as pd

datos = {
    "estudiante": [
        "Ana", "Luis", "María", "Carlos", "Sofia",
        "Pedro", "Elena", "Jorge", "Mario", "Dana",
        "Juan", "Luis", "Salem", "Cris","Sebas",
        "Pedro", "Elena", "Jorge", "Mario", "Dana"
    ],
    "horas_estudio": [
        8, 3, 10, 5, 6, 2, 7, 4, 10, 3,
        8, 3, 10, 5, 6, 2, 7, 4, 10, 3
    ],
    "asistencia": [
        95, 60, 98, 75, 85, 55, 90, 70,90,30,
        95, 60, 98, 75, 85, 55, 90, 70,90,30
    ],
    "actividades_entregadas": [
        100, 50, 100, 80, 90, 40, 95, 65,80, 90,
        100, 50, 100, 80, 90, 40, 95, 65, 89, 44
    ],
    "calificaciones": [
        92, 58, 96, 72, 81, 50, 88, 64,75,63,
        92, 58, 96, 72, 81, 50, 88, 64,75,63
    ]
}

# Paso 2: Usar 'df' en lugar de 'dt'
df = pd.DataFrame(datos)

# Paso 3
print("Conjunto de datos:")
print(df)

# Paso 4
print("\nDimensiones:")
print(df.shape)

# Es buena práctica aclarar qué es cada dimensión
print("Total de estudiantes (filas):", df.shape[0]) 
print("Total de variables (columnas):", df.shape[1])

# Paso 5
df.to_csv(
    "analisis.csv",
    index=False,
    encoding="utf-8"
)
print("\nArchivo creado correctamente")
