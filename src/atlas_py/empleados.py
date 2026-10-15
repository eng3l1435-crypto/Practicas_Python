import itertools

# Datos
empleados = ["Ana", "Luis", "Carlos", "Maria"]
turnos = ["Matutino", "Vespertino", "Nocturno", "Mixto/Rolado"]

preferencias = {
    "Ana": "Matutino",
    "Luis": "Vespertino",
    "Carlos": "Nocturno",
    "Maria": "Matutino"
}

mejor_satisfaccion = -1
mejor_asignacion = None

#Combinaciones posibles
for combo in itertools.permutations(turnos):
    satisfaccion_actual = 0
    
    for emp, turno in zip(empleados, combo):
        if turno == preferencias[emp]:
            satisfaccion_actual += 2 
        elif turno == "Mixto/Rolado":
            satisfaccion_actual += 1  # Turno comodín aceptable
            
    if satisfaccion_actual > mejor_satisfaccion:
        mejor_satisfaccion = satisfaccion_actual
        mejor_asignacion = dict(zip(empleados, combo))

# Mostrar resultado
print("ASIGNACIÓN")
for emp, turno in mejor_asignacion.items():
    print(f"{emp} -> {turno} (Buscaba: {preferencias[emp]})")

print(f"\nPuntaje de satisfacción total: {mejor_satisfaccion}/8")
