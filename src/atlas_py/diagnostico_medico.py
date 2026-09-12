# SISTEMA EXPERTO DE DIAGNÓSTICO

print("SISTEMA DE DIAGNÓSTICO")
print()

fiebre = input("¿Tiene fiebre? (s/n): ").lower()
tos = input("¿Tiene tos? (s/n): ").lower()
dolor = input("¿Tiene dolor de garganta? (s/n): ").lower()
dificultad_respirar = input("¿Tiene dificultad para respirar? (s/n): ").lower()
dolor_pecho = input("¿Tiene dolor en el pecho? (s/n): ").lower()

# Convertimos las respuestas en proposiciones lógicas
P = fiebre == "s"
Q = tos == "s"
R = dolor == "s"
S = dificultad_respirar == "s"
T = dolor_pecho == "s"

# REGLAS DE DECISIÓN

if S and T:
    diagnostico = "Se detectaron señales de alerta. Se recomienda valoración médica inmediata."

elif S:
    diagnostico = "La dificultad para respirar requiere valoración profesional."

elif T:
    diagnostico = "El dolor en el pecho requiere valoración profesional."

elif P and Q and R:
    diagnostico = "Posible infección respiratoria con varios síntomas."

elif P and Q:
    diagnostico = "Posible infección respiratoria."

elif Q and R:
    diagnostico = "Posible irritación o infección de vías respiratorias."

elif P and R:
    diagnostico = "Posible proceso infeccioso."

elif P:
    diagnostico = "Se detectó fiebre. Se recomienda vigilar la temperatura y evolución."

elif Q:
    diagnostico = "Se detectó tos. Se recomienda vigilar su evolución."

elif R:
    diagnostico = "Se detectó dolor de garganta. Se recomienda vigilar su evolución."

else:
    diagnostico = "No se identificó un patrón con los síntomas proporcionados."

# RESULTADO

print()
print("=" * 50)
print("RESULTADO DEL DIAGNÓSTICO")
print("=" * 50)
print(diagnostico)