import os
import re
import zipfile
import subprocess
import hashlib
from pathlib import Path
from html.parser import HTMLParser

import chromadb
from sentence_transformers import SentenceTransformer
import requests
from tqdm import tqdm

# -------------------------------------------------
# CONFIGURACIÓN
# -------------------------------------------------

ZIP_PATH = "unity_docs.zip"
EXTRACT_PATH = "docs"
DB_PATH = "./unity_db"

CHUNK_SIZE = 800       # caracteres por chunk
CHUNK_OVERLAP = 150    # solapamiento entre chunks para no perder contexto

MODEL_NAME = None
OLLAMA_PORT = 11434
OLLAMA_URL = None      # se construye tras seleccionar puerto

# -------------------------------------------------
# LIMPIEZA DE HTML
# -------------------------------------------------

class HTMLStripper(HTMLParser):
    """Extrae solo el texto de un HTML, ignorando tags y scripts."""

    def __init__(self):
        super().__init__()
        self.reset()
        self._fed = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "nav", "header", "footer"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "header", "footer"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            self._fed.append(data)

    def get_text(self):
        text = " ".join(self._fed)
        # Eliminar espacios múltiples y saltos de línea excesivos
        text = re.sub(r"\s+", " ", text)
        return text.strip()


def clean_html(raw_html: str) -> str:
    stripper = HTMLStripper()
    try:
        stripper.feed(raw_html)
        return stripper.get_text()
    except Exception:
        # Fallback: regex básico
        clean = re.sub(r"<[^>]+>", " ", raw_html)
        clean = re.sub(r"\s+", " ", clean)
        return clean.strip()


def clean_text(text: str, is_html: bool) -> str:
    if is_html:
        text = clean_html(text)
    # Eliminar líneas que solo tienen puntuación o están vacías
    lines = [l.strip() for l in text.splitlines()]
    lines = [l for l in lines if len(l) > 20]
    return "\n".join(lines)

# -------------------------------------------------
# IDIOMA
# -------------------------------------------------

LANGUAGE = "es"

def select_language():
    global LANGUAGE

    print("\n=== Selecciona idioma / Select language ===")
    print("1. Español")
    print("2. English")
    print("3. Français")

    choice = input("Opción: ").strip()

    if choice == "1":
        LANGUAGE = "es"
    elif choice == "2":
        LANGUAGE = "en"
    elif choice == "3":
        LANGUAGE = "fr"
    else:
        print("Opción inválida, usando español por defecto.")
        LANGUAGE = "es"

# -------------------------------------------------
# PUERTO DE OLLAMA
# -------------------------------------------------

def select_port():
    global OLLAMA_PORT, OLLAMA_URL

    print("\n=== Puerto de Ollama ===")
    port_input = input("Puerto de Ollama [por defecto 11434]: ").strip()

    if port_input == "":
        OLLAMA_PORT = 11434
    else:
        try:
            OLLAMA_PORT = int(port_input)
        except ValueError:
            print("Puerto inválido, usando 11434 por defecto.")
            OLLAMA_PORT = 11434

    OLLAMA_URL = f"http://localhost:{OLLAMA_PORT}/api/generate"
    print(f"Usando Ollama en puerto {OLLAMA_PORT}")

# -------------------------------------------------
# MODELOS
# -------------------------------------------------

AVAILABLE_MODELS = {
    "1": "llama3.2",
    "2": "llama3.1",
    "3": "qwen2.5",
    "4": "mistral",
    "5": "glm4:latest",
}

def check_model_exists(model_name):
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True
        )
        return model_name.split(":")[0] in result.stdout
    except Exception:
        return False


def pull_model(model_name):
    print(f"\nDescargando modelo {model_name}... (puede tardar varios minutos)")
    subprocess.run(["ollama", "pull", model_name])
    print("Modelo listo.")


def select_model():
    global MODEL_NAME

    print("\n=== Selecciona modelo de lenguaje ===")
    for key, name in AVAILABLE_MODELS.items():
        print(f"{key}. {name}")
    print("O escribe directamente el nombre del modelo (ej: phi3, gemma2...)")

    choice = input("Opción: ").strip()

    if choice in AVAILABLE_MODELS:
        MODEL_NAME = AVAILABLE_MODELS[choice]
    elif choice != "":
        MODEL_NAME = choice
    else:
        MODEL_NAME = "llama3.2"

    print(f"Modelo seleccionado: {MODEL_NAME}")

    if not check_model_exists(MODEL_NAME):
        download = input(f"El modelo '{MODEL_NAME}' no está instalado. ¿Descargarlo ahora? (s/n): ").strip().lower()
        if download in ("s", "y", "si", "sí", "yes"):
            pull_model(MODEL_NAME)
        else:
            print("Continuando sin descargar. El modelo debe estar disponible en Ollama.")

# -------------------------------------------------
# SYSTEM PROMPT
# -------------------------------------------------

def load_system_prompt():
    try:
        with open("system_prompt.txt", "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return (
            "Eres un asistente experto en Unity Game Engine. "
            "Responde de forma clara, precisa y con ejemplos de código cuando sea útil. "
            "Usa solo la documentación proporcionada como contexto."
        )

SYSTEM_PROMPT_BASE = load_system_prompt()

LANG_INSTRUCTIONS = {
    "es": "Responde siempre en español.",
    "en": "Always respond in English.",
    "fr": "Réponds toujours en français.",
}

def get_system_prompt():
    lang_instruction = LANG_INSTRUCTIONS.get(LANGUAGE, "Responde siempre en español.")
    return f"{SYSTEM_PROMPT_BASE}\n\n{lang_instruction}"

# -------------------------------------------------
# EMBEDDINGS
# -------------------------------------------------

print("Cargando modelo de embeddings... (solo la primera vez puede tardar)")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------------------------
# VECTOR DB
# -------------------------------------------------

chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(
    "unity_docs",
    metadata={"hnsw:space": "cosine"}   # distancia coseno, más precisa para texto
)

# -------------------------------------------------
# UTILIDADES
# -------------------------------------------------

def db_exists() -> bool:
    """Verifica que la DB existe y tiene documentos indexados."""
    if not os.path.exists(DB_PATH):
        return False
    try:
        count = collection.count()
        return count > 0
    except Exception:
        return False


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Divide el texto en chunks con solapamiento.
    El solapamiento evita perder contexto en los bordes de cada chunk.
    """
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + size, text_len)
        chunk = text[start:end].strip()

        if len(chunk) > 50:   # descartar chunks demasiado pequeños
            chunks.append(chunk)

        if end == text_len:
            break

        start += size - overlap

    return chunks


def make_id(source: str, index: int) -> str:
    """Genera un ID único y reproducible para cada chunk."""
    raw = f"{source}_{index}"
    return hashlib.md5(raw.encode()).hexdigest()

# -------------------------------------------------
# ZIP
# -------------------------------------------------

def extract_zip_if_needed():
    if os.path.exists(EXTRACT_PATH):
        existing = list(Path(EXTRACT_PATH).rglob("*"))
        if len(existing) > 5:
            print(f"ZIP ya extraído ({len(existing)} archivos encontrados), se omite extracción.")
            return

    if not os.path.exists(ZIP_PATH):
        print(f"\nERROR: No se encontró el archivo '{ZIP_PATH}'.")
        print("Coloca el ZIP de la documentación de Unity en la misma carpeta que este script.")
        exit(1)

    print(f"\nExtrayendo '{ZIP_PATH}'...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        members = zip_ref.infolist()
        for member in tqdm(members, desc="Extrayendo", unit="archivo"):
            zip_ref.extract(member, EXTRACT_PATH)

    total = len(list(Path(EXTRACT_PATH).rglob("*")))
    print(f"Extracción completada. {total} archivos en '{EXTRACT_PATH}'.")

# -------------------------------------------------
# CARGA DE ARCHIVOS
# -------------------------------------------------

SUPPORTED_EXTENSIONS = {".txt", ".md", ".html", ".htm"}

def load_files() -> list[dict]:
    """
    Carga todos los archivos de texto/html de forma recursiva.
    Limpia el HTML antes de devolverlo.
    """
    docs = []
    all_files = [
        f for f in Path(EXTRACT_PATH).rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not all_files:
        print(f"\nERROR: No se encontraron archivos .txt/.md/.html en '{EXTRACT_PATH}'.")
        print("Verifica que el ZIP contiene documentación de Unity.")
        exit(1)

    print(f"\nArchivos encontrados: {len(all_files)}")

    is_html_ext = {".html", ".htm"}

    for f in tqdm(all_files, desc="Leyendo archivos", unit="archivo"):
        try:
            raw = f.read_text(encoding="utf-8", errors="ignore")
            is_html = f.suffix.lower() in is_html_ext
            cleaned = clean_text(raw, is_html)

            if len(cleaned) < 100:   # ignorar archivos casi vacíos
                continue

            docs.append({"text": cleaned, "source": str(f)})
        except Exception as e:
            pass   # ignorar archivos no legibles

    print(f"Documentos válidos cargados: {len(docs)}")
    return docs

# -------------------------------------------------
# INDEXACIÓN
# -------------------------------------------------

def build_index(docs: list[dict]):
    """
    Indexa todos los documentos en ChromaDB.
    - Genera chunks con solapamiento
    - Codifica embeddings en batch para mayor velocidad
    - Omite IDs ya existentes para permitir reindexación parcial
    """
    print("\nConstruyendo índice de vectores...")

    all_chunks = []
    all_sources = []
    all_ids = []

    print("Dividiendo documentos en chunks...")
    for doc in tqdm(docs, desc="Chunking", unit="doc"):
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            chunk_id = make_id(doc["source"], i)
            all_chunks.append(chunk)
            all_sources.append(doc["source"])
            all_ids.append(chunk_id)

    print(f"Total de chunks generados: {len(all_chunks)}")

    # Filtrar IDs ya existentes en la DB (por si se reindexó parcialmente)
    existing_ids = set()
    try:
        existing = collection.get(ids=all_ids)
        existing_ids = set(existing["ids"])
    except Exception:
        pass

    new_chunks = []
    new_sources = []
    new_ids = []
    for chunk, source, cid in zip(all_chunks, all_sources, all_ids):
        if cid not in existing_ids:
            new_chunks.append(chunk)
            new_sources.append(source)
            new_ids.append(cid)

    if not new_chunks:
        print("Todos los chunks ya estaban indexados.")
        return

    print(f"Chunks nuevos a indexar: {len(new_chunks)}")

    # Codificar en batches para no saturar la RAM
    BATCH_SIZE = 128
    total_batches = (len(new_chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_i in tqdm(range(total_batches), desc="Indexando", unit="batch"):
        start = batch_i * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(new_chunks))

        batch_chunks = new_chunks[start:end]
        batch_sources = new_sources[start:end]
        batch_ids = new_ids[start:end]

        # Calcular embeddings del batch completo de una vez (mucho más rápido)
        embeddings = embedding_model.encode(
            batch_chunks,
            show_progress_bar=False,
            batch_size=32
        ).tolist()

        collection.add(
            embeddings=embeddings,
            documents=batch_chunks,
            metadatas=[{"source": s} for s in batch_sources],
            ids=batch_ids
        )

    print(f"\nIndexación completada. Total de chunks en DB: {collection.count()}")

# -------------------------------------------------
# BÚSQUEDA
# -------------------------------------------------

def search(query: str, k: int = 6) -> list[str]:
    """Busca los k chunks más relevantes para la consulta."""
    q_emb = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k
    )
    return results["documents"][0]

# -------------------------------------------------
# OLLAMA
# -------------------------------------------------

def check_ollama_running() -> bool:
    try:
        r = requests.get(f"http://localhost:{OLLAMA_PORT}", timeout=3)
        return True
    except Exception:
        return False


def ask_ollama(question: str, context: str) -> str:
    prompt = f"""{get_system_prompt()}

=== DOCUMENTACIÓN DE UNITY (contexto) ===

{context}

=== FIN DEL CONTEXTO ===

Pregunta: {question}

Respuesta:"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,      # más determinista para documentación técnica
                    "num_predict": 1024,
                }
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.Timeout:
        return "Error: El modelo tardó demasiado en responder. Prueba con un modelo más pequeño."
    except requests.exceptions.ConnectionError:
        return f"Error: No se pudo conectar con Ollama en el puerto {OLLAMA_PORT}. ¿Está ejecutándose?"
    except Exception as e:
        return f"Error inesperado: {e}"

# -------------------------------------------------
# PIPELINE
# -------------------------------------------------

def ask(question: str) -> str:
    chunks = search(question)
    context = "\n\n---\n\n".join(chunks)
    return ask_ollama(question, context)

# -------------------------------------------------
# MAIN
# -------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("   Unity Docs AI — Asistente Local con RAG")
    print("=" * 50)

    select_language()
    select_port()
    select_model()

    if not check_ollama_running():
        print(f"\n⚠  AVISO: No se detectó Ollama en el puerto {OLLAMA_PORT}.")
        print("Asegúrate de que Ollama está ejecutándose antes de hacer preguntas.")
        input("Pulsa Enter para continuar de todas formas...")

    print("\nPreparando documentación...")
    extract_zip_if_needed()

    if not db_exists():
        print("\nPrimera ejecución: creando base de datos vectorial.")
        print("Esto puede tardar varios minutos dependiendo del tamaño de la documentación.\n")
        docs = load_files()
        build_index(docs)
    else:
        count = collection.count()
        print(f"Base de datos encontrada ({count} chunks indexados). Cargando directamente.")

    print("\n" + "=" * 50)
    print("Asistente listo. Escribe 'exit' para salir.")
    print("=" * 50 + "\n")

    while True:
        try:
            q = input("Pregunta sobre Unity: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaliendo...")
            break

        if not q:
            continue

        if q.lower() in ("exit", "quit", "salir"):
            print("¡Hasta luego!")
            break

        print("\nBuscando en la documentación...")
        answer = ask(q)
        print("\n" + answer + "\n")
        print("-" * 50 + "\n")
