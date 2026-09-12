# CASO DE ESTUDIO:
# Sistema de autorización para examen

print("=" * 60)
print("     SISTEMA DE AUTORIZACIÓN PARA EXAMEN FINAL")
print("=" * 60)

# ----------------------------------------------------------
# 1. ENTRADA DE DATOS
# ----------------------------------------------------------

try:
    asistencia = float(input("Porcentaje de asistencia: "))
    promedio = float(input("Promedio: "))
except ValueError:
    print("Error: la asistencia y el promedio deben ser números.")
    exit()

proyecto = input("¿Entregó el proyecto? (si/no): ").lower()
autorizacion = input("¿Tiene autorización especial? (si/no): ").lower()
adeudos = input("¿Tiene adeudos pendientes? (si/no): ").lower()
lista = input("¿Aparece en la lista de autorizados? (si/no): ").lower()


# ----------------------------------------------------------
# 2. CONVERTIMOS LOS DATOS EN PROPOSICIONES
# ----------------------------------------------------------

P = asistencia >= 80          # asistencia suficiente
Q = promedio >= 7             # promedio aprobatorio
R = proyecto == "si"          # entregó el proyecto
S = autorizacion == "si"      # tiene autorización especial
T = adeudos == "no"           # NO tiene adeudos pendientes
U = lista == "si"             # aparece en la lista


# ----------------------------------------------------------
# 3 A 9. OPERACIONES LÓGICAS
# ----------------------------------------------------------

negacion_P = not P

conjuncion = P and Q

disyuncion = Q or S

condicional = (not P) or Q

bicondicional = P == Q

expresion = (P and Q) or S

resultado_final = expresion and T and U


# ----------------------------------------------------------
# 10. REPORTE
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("              VALORES DE LAS PROPOSICIONES")
print("=" * 60)

print("P - Asistencia suficiente (>= 80):", P)
print("Q - Promedio aprobatorio (>= 7):", Q)
print("R - Proyecto entregado:", R)
print("S - Autorización especial:", S)
print("T - Sin adeudos pendientes:", T)
print("U - Aparece en la lista:", U)


print("\n" + "-" * 60)
print("                    OPERACIONES")
print("-" * 60)

print("NEGACIÓN         ¬P =", negacion_P)
print("CONJUNCIÓN     P ∧ Q =", conjuncion)
print("DISYUNCIÓN     Q ∨ S =", disyuncion)
print("CONDICIONAL    P → Q =", condicional)
print("BICONDICIONAL  P ↔ Q =", bicondicional)
print("PARÉNTESIS (P ∧ Q) ∨ S =", expresion)


print("\n" + "-" * 60)
print("                     RESULTADO")
print("-" * 60)

if expresion:
    print("Requisitos académicos: CUMPLE")
else:
    print("Requisitos académicos: NO CUMPLE")


print()

print(
    "Decisión final ((P ∧ Q) ∨ S) ∧ T ∧ U =",
    resultado_final
)

print()


if resultado_final:
    print("El alumno PUEDE presentar el examen.")

else:
    print("El alumno NO puede presentar el examen.")

    if not expresion:
        print(
            "Motivo: no cumple asistencia y promedio, "
            "ni tiene autorización."
        )

    if not T:
        print("Motivo: tiene adeudos pendientes.")

    if not U:
        print("Motivo: no aparece en la lista de autorizados.")