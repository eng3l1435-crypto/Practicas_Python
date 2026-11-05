estudiantes = {
    "Luis": 3,
    "Carlos": 5,
    "Ana": 8,
    "Maria": 10
}

horas_minimas = int(input("¿Cuántas horas debe estudiar para pasar? "))

def evaluacion(estudiantes, horas_minimas):

    for alumno, horas_alumno in estudiantes.items():
        if horas_alumno >= horas_minimas:
            print(f"Pasó {alumno} con {horas_alumno} hrs de estudio")
        else:
            print(f"No pasó {alumno} con {horas_alumno} hrs de estudio")

evaluacion(estudiantes, horas_minimas)
