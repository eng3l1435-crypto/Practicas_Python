# ============================================================
# API BÁSICA UTILIZANDO SOLAMENTE PYTHON (CON GET Y PUT)
# Sin ningun framework
# ============================================================

from http.server import BaseHTTPRequestHandler, HTTPServer
import json

# Simulamos una pequeña base de datos en memoria con un usuario inicial
BASE_DATOS_MEMORIA = {
    "1": {"id": "1", "nombre": "Carlos", "tipo": "Oficios Técnicos"}
}

class MiAPI(BaseHTTPRequestHandler):

    # ========================================================
    # MÉTODO GET: Para consultar datos
    # ========================================================
    def do_GET(self):
        if self.path == "/usuario/1":
            respuesta = BASE_DATOS_MEMORIA["1"]
            self._enviar_json(200, respuesta)
        else:
            self._enviar_json(404, {"error": "Ruta GET no encontrada"})

    # ========================================================
    # MÉTODO PUT: Para actualizar/reemplazar un recurso por completo
    # ========================================================
    def do_PUT(self):
        if self.path == "/usuario/1":
            # 1. Obtenemos el tamaño del contenido enviado en la cabecera
            longitud_contenido = int(self.headers.get('Content-Length', 0))
            
            # 2. Leemos los datos binarios del cuerpo (body)
            cuerpo_datos = self.rfile.read(longitud_contenido)
            
            try:
                # 3. Convertimos los bytes a un diccionario de Python
                nuevos_datos = json.loads(cuerpo_datos.decode('utf-8'))
                
                # 4. Actualizamos el registro en nuestra "base de datos"
                BASE_DATOS_MEMORIA["1"] = nuevos_datos
                
                respuesta = {
                    "mensaje": "Recurso actualizado exitosamente con PUT",
                    "usuario_actualizado": nuevos_datos
                }
                self._enviar_json(200, respuesta)
                
            except json.JSONDecodeError:
                self._enviar_json(400, {"error": "El cuerpo de la petición debe ser un JSON válido"})
        else:
            self._enviar_json(404, {"error": "Ruta PUT no encontrada"})

    # Método auxiliar para no repetir código al enviar respuestas JSON
    def _enviar_json(self, codigo_estado, datos):
        contenido = json.dumps(datos, ensure_ascii=False).encode("utf-8")
        self.send_response(codigo_estado)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(contenido)))
        self.end_headers()
        self.wfile.write(contenido)


# ============================================================
# INICIAR EL SERVIDOR
# ============================================================

if __name__ == "__main__":
    servidor = HTTPServer(("127.0.0.1", 8000), MiAPI)
    print("API disponible en http://127.0.0.1:8000")
    print("Métodos implementados: GET (/usuario/1) y PUT (/usuario/1)")
    print("Presiona CTRL+C para detener el servidor.")
    servidor.serve_forever()