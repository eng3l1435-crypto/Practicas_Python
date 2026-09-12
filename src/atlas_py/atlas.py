import os
from dotenv import load_dotenv
from pymongo import MongoClient
from urllib.parse import quote_plus


load_dotenv()

usuario = quote_plus(os.getenv("MONGO_USER"))
password = quote_plus(os.getenv("MONGO_PASSWORD"))
cluster = os.getenv("MONGO_CLUSTER")
nombre_db = os.getenv("MONGO_DB")
nombre_coleccion = os.getenv("MONGO_COLLECTION")

uri = f"mongodb+srv://{usuario}:{password}@{cluster}/"

cliente = MongoClient(uri)
db = cliente[nombre_db]
coleccion = db[nombre_coleccion]

print("Conectado al Atlas")

animales = {
    "tipo": "Perro",
}

resultado = coleccion.insert_one(animales)

print("Documento insertado correctamente")
print("ID:", resultado.inserted_id)