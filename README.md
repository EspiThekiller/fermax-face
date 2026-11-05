# 👤 Fermax Face Recognition Service

Microservicio de reconocimiento facial en tiempo real de alto rendimiento optimizado para **CPU**, diseñado específicamente para integrarse con videoporteros (como Fermax Meet / Duox) y sistemas domóticos como **Home Assistant** o cualquier servidor con Docker.

Impulsado por **FastAPI**, **InsightFace** (modelo `buffalo_s`) y **ONNX Runtime**, ofrece una precisión biométrica superior con consumo mínimo de recursos y panel web interactivo.

---

## ✨ Características Principales

- **⚡ Inferencia CPU Ultrarrápida:** Utiliza el modelo ONNX `buffalo_s` con detección y extracción de embeddings faciales de 512 dimensiones en milisegundos sin requerir GPU dedicada.
- **🔐 Seguridad de Nivel Empresarial:**
  - **Contraseña de primer arranque aleatoria:** Al iniciar por primera vez genera una contraseña aleatoria de 10 caracteres mostrada una única vez en los registros.
  - **Obligatoriedad de cambio de credenciales:** En el primer inicio en el panel web se exige eliminar/reemplazar el usuario temporal `admin` por un usuario y contraseña definitivos.
  - **Tokens de API con SHA-256:** Generación de tokens identificados por nombre para Home Assistant u otros servicios. Los tokens se muestran una sola vez y se guardan como hashes SHA-256.
  - **Protección contra fuerza bruta:** Bloqueo temporal ante múltiples intentos fallidos de autenticación.
- **🖥️ Panel Web Completo e Intuitivo:**
  - Registro de nuevas personas (subida simple o masiva de fotos con drag & drop).
  - Probador en vivo: sube una foto o fotograma y comprueba la coincidencia, nivel de confianza, tiempo de respuesta y coordenadas del rostro detectado.
  - Histórico de capturas de 3 días con depuración automática (>72h).
  - **Módulo de auto-entrenamiento:** Permite asignar fotogramas del historial a personas existentes para enriquecer su perfil biométrico con fotos en condiciones reales.
- **🐳 100% Contenedorizado:** Despliegue en un solo comando mediante Docker / Docker Compose.

---

## 🚀 Instalación y Despliegue en 1 Solo Comando (Sin git clone)

No necesitas clonar el repositorio ni tener `git` instalado. En cualquier máquina con Docker, ejecuta simplemente:

```bash
curl -sSL https://raw.githubusercontent.com/EspiThekiller/fermax-face/main/deploy.sh | bash
```
*(O alternativamente: `bash <(curl -sSL https://raw.githubusercontent.com/EspiThekiller/fermax-face/main/deploy.sh)`)*

El asistente interactivo:
1. Descarga automáticamente el proyecto en la carpeta que elijas (por defecto `./fermax-face`).
2. Te pide por pantalla de forma interactiva las rutas y opciones deseadas (o pulsa `[ENTER]` para aceptar los valores por defecto).
3. Levanta y monta el contenedor con Docker Compose o Docker Run.
4. Comprueba la salud del servicio y te muestra el enlace web y las credenciales iniciales de administrador.

---

## 🛠️ Despliegue Local (Si ya tienes el repositorio descargado)

Si ya estás dentro de la carpeta del proyecto, ejecuta:

```bash
chmod +x deploy.sh
./deploy.sh
```

---

## 🛠️ Despliegue Manual con Docker Compose

1. Clona este repositorio:
   ```bash
   git clone https://github.com/tu-usuario/fermax-face.git
   cd fermax-face
   ```

2. Configura las variables en `.env` (o copia desde el ejemplo):
   ```bash
   cp .env.example .env
   ```

3. Levanta el servicio:
   ```bash
   docker compose up -d --build
   ```

4. Observa los registros para obtener la contraseña inicial de administración:
   ```bash
   docker logs fermax-face-service
   ```

5. Abre tu navegador en `http://<IP_SERVIDOR>:8000` e inicia sesión con:
   - **Usuario:** `admin`
   - **Contraseña:** *(la mostrada en los registros)*

---

## ⚙️ Variables de Entorno

| Variable | Descripción | Valor por Defecto |
| :--- | :--- | :--- |
| `PORT` | Puerto de escucha en el host | `8000` |
| `DATA_PATH` | Ruta en el host montada en `/app/data` | `./data` |
| `KNOWN_FACES_PATH` | Ruta en el host montada en `/app/known_faces` | `./known_faces` |
| `SIMILARITY_THRESHOLD` | Umbral de similitud para match (0.0 a 1.0) | `0.65` |
| `DET_SIZE` | Tamaño de ventana de detección de rostros | `320` |

---

## 📡 Endpoints de la API REST

### Reconocimiento Facial
- **`POST /api/v1/recognize`**
  - **Headers:** `X-API-Key: <token>` o `Authorization: Bearer <token>`
  - **Body:** `multipart/form-data` con campo `file` (imagen JPG/PNG) y opcional `threshold` (float).
  - **Respuesta:**
    ```json
    {
      "detected": true,
      "match": true,
      "name": "Nombre",
      "candidate_name": "Nombre",
      "confidence": 0.7421,
      "det_score": 0.9982,
      "threshold_used": 0.65,
      "box": [120, 80, 240, 210],
      "faces_count": 1,
      "execution_time_ms": 42.15
    }
    ```

### Gestión de Personas
- **`GET /api/v1/people`**: Lista todas las personas registradas y número de fotos.
- **`POST /api/v1/register`**: Registra una foto para una persona.
- **`POST /api/v1/register-bulk`**: Registra múltiples fotos de una vez.
- **`DELETE /api/v1/people/{name}`**: Elimina una persona y sus fotos asociadas.

### Histórico y Auto-entrenamiento
- **`GET /api/v1/history`**: Obtiene las capturas recientes (últimas 72h).
- **`POST /api/v1/history/assign`**: Asocia una captura del historial a una persona existente para auto-entrenar el modelo.
- **`DELETE /api/v1/history/{id}`**: Elimina una captura específica.
- **`DELETE /api/v1/history`**: Purga todo el historial.

### Salud del Servicio
- **`GET /health`**: Devuelve estado del servicio y estado de carga del motor InsightFace.

---

## 🏠 Integración con Home Assistant

Crea un Token de API en la pestaña **Seguridad & Tokens** del panel web. Luego, en tu `configuration.yaml` o automatización:

```yaml
rest_command:
  fermax_face_recognize:
    url: "http://<IP_SERVIDOR>:8000/api/v1/recognize"
    method: POST
    headers:
      X-API-Key: "TU_API_TOKEN_AQUI"
    payload: '{{ image_data }}'
```

O vía comando `curl` en shell command:
```bash
curl -X POST "http://<IP_SERVIDOR>:8000/api/v1/recognize" \
  -H "X-API-Key: TU_API_TOKEN_AQUI" \
  -F "file=@/config/www/snapshots/last_doorbell.jpg"
```

---

## 📁 Estructura del Proyecto

```
fermax-face/
├── app/
│   ├── __init__.py
│   ├── main.py          # API FastAPI y Dashboard HTML integrado
│   ├── engine.py        # Motor InsightFace con CPUExecutionProvider
│   ├── security.py      # Autenticación, sesiones y tokens SHA-256
│   └── history.py       # Gestor de historial de 72h y auto-entrenamiento
├── Dockerfile           # Imagen Docker optimizada (python:3.11-slim)
├── docker-compose.yml   # Configuración de Docker Compose
├── deploy.sh            # Asistente interactivo de instalación y arranque
├── requirements.txt     # Dependencias Python
├── .env.example         # Plantilla de variables de entorno
├── .gitignore
├── .dockerignore
├── LICENSE
└── README.md
```

---

## 📄 Licencia

Este proyecto está bajo la licencia [MIT](LICENSE).
