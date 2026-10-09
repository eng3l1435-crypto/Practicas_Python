import os
import random
from dotenv import load_dotenv
from pymongo import MongoClient
from urllib.parse import quote_plus

# --- CARGAR VARIABLES DE ENTORNO ---
load_dotenv()

usuario = quote_plus(os.getenv("MONGO_USER"))
password = quote_plus(os.getenv("MONGO_PASSWORD"))
cluster = os.getenv("MONGO_CLUSTER")
nombre_db = os.getenv("MONGO_DB")
nombre_coleccion = os.getenv("MONGO_COLLECTION")

uri = f"mongodb+srv://{usuario}:{password}@{cluster}/"

def determinar_accion(temperatura, humedad):
  """Replica exactamente la lógica de decisión del agente de climatización."""
  if temperatura > 30 and humedad > 70:
    return "Encender aire acondicionado (Modo Deshumidificador)"
  elif temperatura > 30:
    return "Encender ventilador"
  elif temperatura < 18:
    return "Encender calefacción"
  else:
    return "Mantener sistema apagado"


def generar_datos_prueba():
  print("Conectando a MongoDB para generar registros de prueba...")
  try:
    cliente = MongoClient(uri)
    db = cliente[nombre_db]
    coleccion = db[nombre_coleccion]
  except Exception as e:
    print(f"❌ Error al conectar con la base de datos: {e}")
    return

  registros = []
  print("Generando 100 registros combinados...")

  for _ in range(100):
    # Generamos temperaturas aleatorias realistas (ej. entre 10°C y 40°C)
    temperatura = round(random.uniform(10.0, 40.0), 1)
    # Generamos humedades aleatorias (ej. entre 20% y 90%)
    humedad = round(random.uniform(20.0, 90.0), 1)

    # Obtenemos la acción aplicando las reglas del agente
    accion = determinar_accion(temperatura, humedad)

    documento = {
        "temperatura": temperatura,
        "humedad": humedad,
        "accion": accion,
    }
    registros.append(documento)

  # Insertar masivamente en MongoDB
  try:
    resultado = coleccion.insert_many(registros)
    print(
        f"insertado exitoso {len(resultado.inserted_ids)}"
        f" registros en la colección '{nombre_coleccion}'!"
    )
  except Exception as e:
    print(f"❌ Error al insertar los datos: {e}")


if __name__ == "__main__":
  generar_datos_prueba()