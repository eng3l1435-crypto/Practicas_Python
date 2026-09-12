from pymongo import MongoClient

# Conexión con MongoDB
cliente = MongoClient("mongodb://localhost:27017")
# Seleccionar base de datos
db = cliente["escuela"]
# Seleccionar colección
alumnos = db["alumnos"]
print("Conectado")

# Documento a insertar
alumno = {
    "nombre": "Angel",
    "edad": 19,
    "carrera": "Desarrollo de software",
}

# Insertar documento
resultado = alumnos.updateMany(alumno)

print("Dato insertado correctamente")
print("ID:", resultado.inserted_id)