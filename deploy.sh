#!/usr/bin/env bash
# ==============================================================================
# Fermax Face Recognition Service - Script de Despliegue Interactivo
# ==============================================================================

set -e

# Colores para salida interactiva
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Función para lectura interactiva segura (compatible con pipes como 'curl ... | bash')
prompt_read() {
    local prompt_msg="$1"
    local var_name="$2"
    local default_val="$3"
    local input=""

    if [ -t 0 ]; then
        read -rp "$prompt_msg" input
    elif [ -e /dev/tty ]; then
        read -rp "$prompt_msg" input < /dev/tty
    else
        input=""
    fi

    input="${input:-$default_val}"
    printf -v "$var_name" '%s' "$input"
}

clear 2>/dev/null || true

cat << "EOF"
==============================================================================
   ___                                 ___             
  / __\___ _ __ _ __ ___   __ ___  __ / __\__ _  ___ ___ 
 / _\/ _ \ '__| '_ ` _ \ / _` \ \/ // /_/ _` |/ __/ _ \
/ / |  __/ |  | | | | | | (_| |>  </ /__ (_| | (_|  __/
\/   \___|_|  |_| |_| |_|\__,_/_/\_\\____/\__,_|\___\___|
                                                        
     Microservicio de Reconocimiento Facial para Domótica
==============================================================================
EOF

echo -e "${CYAN}Este asistente configurará y levantará el contenedor de Fermax Face.${NC}"
echo -e "${CYAN}Presiona [ENTER] en cualquier opción para aceptar el valor sugerido por defecto.${NC}\n"

# ------------------------------------------------------------------------------
# 1. Comprobación de Requisitos (Docker / Docker Compose / Curl / Tar)
# ------------------------------------------------------------------------------
if ! command -v docker &> /dev/null; then
    echo -e "${RED}[ERROR] Docker no está instalado en este sistema.${NC}"
    echo "Por favor, instala Docker antes de continuar."
    exit 1
fi

COMPOSE_CMD=""
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
elif command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
fi

# ------------------------------------------------------------------------------
# 2. Auto-descarga / Comprobación del Código Fuente (Sin necesidad de git clone)
# ------------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || echo "$PWD")"

if [ ! -f "${SCRIPT_DIR}/Dockerfile" ]; then
    echo -e "${YELLOW}No se han detectado los archivos del proyecto en la carpeta actual.${NC}"
    DEFAULT_INSTALL_DIR="./fermax-face"
    prompt_read "$(echo -e "${BOLD}¿Dónde deseas instalar Fermax Face?${NC} [${GREEN}${DEFAULT_INSTALL_DIR}${NC}]: ")" TARGET_INSTALL_DIR "$DEFAULT_INSTALL_DIR"

    mkdir -p "$TARGET_INSTALL_DIR"
    echo -e "${BLUE}Descargando última versión desde GitHub...${NC}"
    if command -v curl &> /dev/null; then
        curl -sSL "https://github.com/EspiThekiller/fermax-face/archive/refs/heads/main.tar.gz" | tar -xz --strip-components=1 -C "$TARGET_INSTALL_DIR"
    elif command -v wget &> /dev/null; then
        wget -qO- "https://github.com/EspiThekiller/fermax-face/archive/refs/heads/main.tar.gz" | tar -xz --strip-components=1 -C "$TARGET_INSTALL_DIR"
    else
        echo -e "${RED}[ERROR] Se necesita 'curl' o 'wget' para descargar los archivos.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Archivos del proyecto descargados en ${TARGET_INSTALL_DIR}${NC}\n"
    cd "$TARGET_INSTALL_DIR"
    SCRIPT_DIR="$PWD"
fi

# ------------------------------------------------------------------------------
# 3. Configuración Interactiva de Parámetros
# ------------------------------------------------------------------------------

# Puerto
DEFAULT_PORT="8000"
prompt_read "$(echo -e "${BOLD}1. Puerto de escucha en el host${NC} [${GREEN}${DEFAULT_PORT}${NC}]: ")" PORT "$DEFAULT_PORT"

# Directorio de Datos
DEFAULT_DATA="./data"
prompt_read "$(echo -e "${BOLD}2. Ruta en el host para DATOS (seguridad, historial)${NC} [${GREEN}${DEFAULT_DATA}${NC}]: ")" DATA_PATH "$DEFAULT_DATA"

# Directorio de Caras Conocidas
DEFAULT_FACES="./known_faces"
prompt_read "$(echo -e "${BOLD}3. Ruta en el host para FOTOS de personas registradas${NC} [${GREEN}${DEFAULT_FACES}${NC}]: ")" KNOWN_FACES_PATH "$DEFAULT_FACES"

# Umbral de Similitud
DEFAULT_THRESHOLD="0.65"
prompt_read "$(echo -e "${BOLD}4. Umbral de similitud biométrica (0.0 a 1.0)${NC} [${GREEN}${DEFAULT_THRESHOLD}${NC}]: ")" SIMILARITY_THRESHOLD "$DEFAULT_THRESHOLD"

# Tamaño de Detección
DEFAULT_DET_SIZE="320"
prompt_read "$(echo -e "${BOLD}5. Tamaño de ventana de detección (320 = CPU rápido, 640 = alta resolución)${NC} [${GREEN}${DEFAULT_DET_SIZE}${NC}]: ")" DET_SIZE "$DEFAULT_DET_SIZE"

# Acción de construcción de imagen
echo -e "\n${BOLD}6. Opciones de imagen Docker:${NC}"
echo "   1) Construir imagen localmente desde el Dockerfile (Recomendado)"
echo "   2) Usar imagen local existente 'fermax-face-service:latest'"
prompt_read "$(echo -e "Selecciona una opción [${GREEN}1${NC}]: ")" BUILD_OPT "1"

# Método de orquestación
ORCH_METHOD="compose"
if [ -z "$COMPOSE_CMD" ]; then
    echo -e "${YELLOW}[AVISO] Docker Compose no encontrado. Se usará 'docker run' directo.${NC}"
    ORCH_METHOD="run"
else
    echo -e "\n${BOLD}7. Método de despliegue:${NC}"
    echo "   1) Docker Compose (${COMPOSE_CMD})"
    echo "   2) Docker Run directo"
    prompt_read "$(echo -e "Selecciona una opción [${GREEN}1${NC}]: ")" INPUT_ORCH "1"
    if [ "$INPUT_ORCH" = "2" ]; then
        ORCH_METHOD="run"
    fi
fi

# ------------------------------------------------------------------------------
# 4. Resumen y Confirmación
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}--- Resumen de Configuración ---${NC}"
echo -e "  - Carpeta de Proyecto:    ${GREEN}${SCRIPT_DIR}${NC}"
echo -e "  - Puerto:                 ${GREEN}${PORT}${NC}"
echo -e "  - Directorio Datos:       ${GREEN}${DATA_PATH}${NC}"
echo -e "  - Directorio Fotos:       ${GREEN}${KNOWN_FACES_PATH}${NC}"
echo -e "  - Umbral Similitud:       ${GREEN}${SIMILARITY_THRESHOLD}${NC}"
echo -e "  - Tamaño Detección:       ${GREEN}${DET_SIZE}${NC}"
echo -e "  - Método de Despliegue:   ${GREEN}${ORCH_METHOD}${NC}"
echo -e "--------------------------------\n"

prompt_read "$(echo -e "¿Deseas proceder con el despliegue? (${BOLD}S/n${NC}): ")" CONFIRM "s"
if [[ ! "$CONFIRM" =~ ^[sSyY]$ ]]; then
    echo -e "${YELLOW}Despliegue cancelado por el usuario.${NC}"
    exit 0
fi

# ------------------------------------------------------------------------------
# 5. Creación de Directorios y Fichero .env
# ------------------------------------------------------------------------------
echo -e "\n${BLUE}[1/4] Creando directorios y configuraciones...${NC}"
mkdir -p "$DATA_PATH" "$KNOWN_FACES_PATH"

ENV_FILE="${SCRIPT_DIR}/.env"

cat > "$ENV_FILE" << EOF
PORT=${PORT}
DATA_PATH=${DATA_PATH}
KNOWN_FACES_PATH=${KNOWN_FACES_PATH}
SIMILARITY_THRESHOLD=${SIMILARITY_THRESHOLD}
DET_SIZE=${DET_SIZE}
EOF
echo -e "${GREEN}✓ Fichero .env generado con éxito en ${ENV_FILE}${NC}"

# ------------------------------------------------------------------------------
# 6. Construcción / Preparación de Imagen
# ------------------------------------------------------------------------------
IMAGE_NAME="fermax-face-service:latest"
CONTAINER_NAME="fermax-face-service"

if [ "$BUILD_OPT" = "1" ]; then
    echo -e "\n${BLUE}[2/4] Construyendo imagen Docker '${IMAGE_NAME}'...${NC}"
    docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"
    echo -e "${GREEN}✓ Imagen construida correctamente.${NC}"
else
    echo -e "\n${BLUE}[2/4] Utilizando imagen existente '${IMAGE_NAME}'...${NC}"
fi

# ------------------------------------------------------------------------------
# 7. Lanzamiento del Contenedor
# ------------------------------------------------------------------------------
echo -e "\n${BLUE}[3/4] Levantando contenedor...${NC}"

# Si el contenedor ya existe o está en ejecución, detenerlo
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${YELLOW}Deteniendo contenedor anterior '${CONTAINER_NAME}'...${NC}"
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
    docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
fi

if [ "$ORCH_METHOD" = "compose" ]; then
    cd "$SCRIPT_DIR"
    $COMPOSE_CMD down >/dev/null 2>&1 || true
    $COMPOSE_CMD up -d
else
    # Docker volume para modelos InsightFace
    docker volume create fermax-face_insightface_models >/dev/null 2>&1 || true

    docker run -d \
        --name "$CONTAINER_NAME" \
        --restart unless-stopped \
        -p "${PORT}:8000" \
        -v "${DATA_PATH}:/app/data" \
        -v "${KNOWN_FACES_PATH}:/app/known_faces" \
        -v fermax-face_insightface_models:/root/.insightface/models \
        -e DATA_DIR=/app/data \
        -e KNOWN_FACES_DIR=/app/known_faces \
        -e SIMILARITY_THRESHOLD="$SIMILARITY_THRESHOLD" \
        -e DET_SIZE="$DET_SIZE" \
        "$IMAGE_NAME"
fi

# ------------------------------------------------------------------------------
# 8. Verificación de Estado y Healthcheck
# ------------------------------------------------------------------------------
echo -e "\n${BLUE}[4/4] Verificando salud del servicio...${NC}"
printf "Esperando a que la API esté lista"

READY=false
for i in {1..20}; do
    if curl -s "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q '"status":"ok"'; then
        READY=true
        break
    fi
    printf "."
    sleep 2
done
echo ""

if [ "$READY" = true ]; then
    echo -e "\n${GREEN}==============================================================================${NC}"
    echo -e "${GREEN}✓ ¡FERMAX FACE SE HA DESPLEGADO Y ESTÁ FUNCIONANDO CORRECTAMENTE!${NC}"
    echo -e "${GREEN}==============================================================================${NC}"
else
    echo -e "\n${YELLOW}[AVISO] El contenedor arrancó pero el endpoint de salud aún no responde.${NC}"
    echo -e "InsightFace puede tardar unos segundos extra en descargar el modelo en su primer inicio."
fi

# Detectar IP del host para mostrar enlace
HOST_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
HOST_IP="${HOST_IP:-localhost}"

echo -e "\n${BOLD}🌐 Acceso al Panel Web:${NC}"
echo -e "   URL: ${CYAN}http://${HOST_IP}:${PORT}${NC}  (o http://localhost:${PORT})"

# Revisar si hay banner de contraseña inicial en los logs
FIRST_BOOT_PW=$(docker logs "$CONTAINER_NAME" 2>&1 | grep "CONTRASEÑA:" | head -n 1 | awk '{print $NF}')
if [ -n "$FIRST_BOOT_PW" ]; then
    echo -e "\n${YELLOW}------------------------------------------------------------------------------${NC}"
    echo -e "${YELLOW}🔐 CREDENCIALES INICIALES DE ADMINISTRADOR DETECTADAS:${NC}"
    echo -e "   Usuario:    ${BOLD}admin${NC}"
    echo -e "   Contraseña: ${BOLD}${FIRST_BOOT_PW}${NC}"
    echo -e "   ${YELLOW}(Recuerda que el panel te pedirá cambiar estas credenciales en el primer login)${NC}"
    echo -e "${YELLOW}------------------------------------------------------------------------------${NC}"
fi

echo -e "\nPara ver los registros en tiempo real: ${CYAN}docker logs -f ${CONTAINER_NAME}${NC}\n"
