import random
from datetime import datetime

print("Bienvenido al sistema de diagnóstico")
print()

# DATOS DEL USUARIO

usuario = input("Ingrese su nombre de usuario: ")
nombre = input("Ingrese su nombre completo: ")
direccion = input("Ingrese su dirección: ")

# Diccionario para los dispositivos
dispositivos = {
    "1": "PC",
    "2": "LAPTOP",
    "3": "SERVIDOR",
    "4": "TABLET"
}

opcion = input(
    "Seleccione el dispositivo "
    "[1] PC [2] LAPTOP [3] SERVIDOR [4] TABLET: "
)

dispositivo = dispositivos.get(opcion, "Desconocido")

print(f"Dispositivo seleccionado: {dispositivo}")

# DATOS DEL REPORTE

numero_reporte = random.randint(1, 1000)
fecha_hora = datetime.now()

# DICCIONARIO DE RESPUESTAS

respuestas = {}

print("\n--- INICIO DEL DIAGNÓSTICO ---\n")

# PREGUNTAS

respuestas["electricidad"] = input(
    "¿Tiene electricidad? (s/n): "
).lower()

if respuestas["electricidad"] == "n":
    input(
        "Revise la entrada de electricidad "
        "y presione Enter para continuar..."
    )


respuestas["conector"] = input(
    "¿Ha conectado su cable al conector? (s/n): "
).lower()

if respuestas["conector"] == "n":
    input(
        "Conecte el cable al conector "
        "y presione Enter para continuar..."
    )


respuestas["luz_vecinos"] = input(
    "¿Sus vecinos tienen luz? (s/n): "
).lower()

if respuestas["luz_vecinos"] == "n":
    input(
        "Revise su medidor de luz "
        "y presione Enter para continuar..."
    )


respuestas["recibo"] = input(
    "¿Pagó su recibo de luz? (s/n): "
).lower()

if respuestas["recibo"] == "n":
    input(
        "Pague o verifique su recibo de luz "
        "y presione Enter para continuar..."
    )


respuestas["enciende"] = input(
    "¿La computadora enciende? (s/n): "
).lower()

if respuestas["enciende"] == "n":
    input(
        "Revise la alimentación de la computadora "
        "y presione Enter para continuar..."
    )


respuestas["boton_encendido"] = input(
    "¿Ya probó presionando el botón de encendido? (s/n): "
).lower()

if respuestas["boton_encendido"] == "n":
    input(
        "Presione el botón de encendido "
        "y después presione Enter para continuar..."
    )


respuestas["cargada"] = input(
    "¿Su computadora está cargada? (s/n): "
).lower()

if respuestas["cargada"] == "n":
    input(
        "Conecte y cargue su computadora "
        "y presione Enter para continuar..."
    )


respuestas["imagen"] = input(
    "¿La computadora muestra imagen? (s/n): "
).lower()

if respuestas["imagen"] == "n":
    input(
        "Revise la pantalla o monitor "
        "y presione Enter para continuar..."
    )


respuestas["fondo_negro"] = input(
    "¿La pantalla muestra un fondo negro? (s/n): "
).lower()

if respuestas["fondo_negro"] == "s":
    diagnostico = "Se debe revisar la pantalla, sistema de video o sistema operativo."

else:
    diagnostico = "El sistema aparentemente funciona correctamente."

# REPORTE FINAL

print("\n")
print("*" * 100)
print("REPORTE DE DIAGNÓSTICO")
print("*" * 100)

print(f"Reporte: REP-{numero_reporte:04d}")
print(f"Fecha: {fecha_hora.strftime('%Y-%m-%d %H:%M:%S')}")

print("\n--- DATOS DEL USUARIO ---")

print(f"Usuario: {usuario}")
print(f"Nombre: {nombre}")
print(f"Dirección: {direccion}")
print(f"Dispositivo: {dispositivo}")

print("\n--- RESPUESTAS DEL DIAGNÓSTICO ---")

print(f"Tiene electricidad: {respuestas['electricidad']}")
print(f"Cable conectado: {respuestas['conector']}")
print(f"Vecinos tienen luz: {respuestas['luz_vecinos']}")
print(f"Recibo de luz pagado: {respuestas['recibo']}")
print(f"Computadora enciende: {respuestas['enciende']}")
print(f"Probó botón de encendido: {respuestas['boton_encendido']}")
print(f"Computadora cargada: {respuestas['cargada']}")
print(f"Muestra imagen: {respuestas['imagen']}")
print(f"Presenta fondo negro: {respuestas['fondo_negro']}")

print("\n--- DIAGNÓSTICO ---")

print(diagnostico)

print("*" * 100)