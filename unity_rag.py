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
# TEXTOS / INTERNACIONALIZACIÓN
# -------------------------------------------------
# Todos los textos que se imprimen por consola viven aquí, organizados
# por clave y por idioma. Para mostrar un texto se usa la función t(key, **kwargs),
# que selecciona el idioma actual (LANGUAGE) y aplica formato si hace falta.

TEXTS = {
    "select_language_title": {
        "es": "\n=== Selecciona idioma / Select language ===",
        "en": "\n=== Selecciona idioma / Select language ===",
        "fr": "\n=== Selecciona idioma / Select language ===",
        "ru": "\n=== Selecciona idioma / Select language ===",
    },
    "select_language_opt1": {
        "es": "1. Español",
        "en": "1. Español",
        "fr": "1. Español",
        "ru": "1. Español",
    },
    "select_language_opt2": {
        "es": "2. English",
        "en": "2. English",
        "fr": "2. English",
        "ru": "2. English",
    },
    "select_language_opt3": {
        "es": "3. Français",
        "en": "3. Français",
        "fr": "3. Français",
        "ru": "3. Français",
    },
    "select_language_opt4": {
        "es": "4. Русский (наречие старинное)",
        "en": "4. Русский (наречие старинное)",
        "fr": "4. Русский (наречие старинное)",
        "ru": "4. Русский (наречие старинное)",
    },
    "select_language_prompt": {
        "es": "Opción: ",
        "en": "Option: ",
        "fr": "Choix : ",
        "ru": "Избери: ",
    },
    "select_language_invalid": {
        "es": "Opción inválida, usando español por defecto.",
        "en": "Invalid option, using Spanish by default.",
        "fr": "Option invalide, utilisation de l'espagnol par défaut.",
        "ru": "Несть таковой опции, изберется язык гишпанский по обычаю.",
    },
    "ollama_port_title": {
        "es": "\n=== Puerto de Ollama ===",
        "en": "\n=== Ollama Port ===",
        "fr": "\n=== Port d'Ollama ===",
        "ru": "\n=== Врата (порт) Ollama ===",
    },
    "ollama_port_prompt": {
        "es": "Puerto de Ollama [por defecto 11434]: ",
        "en": "Ollama port [default 11434]: ",
        "fr": "Port d'Ollama [par défaut 11434] : ",
        "ru": "Врата Ollama [по обычаю 11434]: ",
    },
    "ollama_port_invalid": {
        "es": "Puerto inválido, usando 11434 por defecto.",
        "en": "Invalid port, using 11434 by default.",
        "fr": "Port invalide, utilisation de 11434 par défaut.",
        "ru": "Несть таковых врат, изберутся 11434 по обычаю.",
    },
    "ollama_port_using": {
        "es": "Usando Ollama en puerto {port}",
        "en": "Using Ollama on port {port}",
        "fr": "Utilisation d'Ollama sur le port {port}",
        "ru": "Ollama употребляется чрез врата {port}",
    },
    "select_model_title": {
        "es": "\n=== Selecciona modelo de lenguaje ===",
        "en": "\n=== Select language model ===",
        "fr": "\n=== Sélectionnez le modèle de langage ===",
        "ru": "\n=== Избери модель речения ===",
    },
    "select_model_hint": {
        "es": "O escribe directamente el nombre del modelo (ej: phi3, gemma2...)",
        "en": "Or type the model name directly (e.g., phi3, gemma2...)",
        "fr": "Ou saisissez directement le nom du modèle (ex : phi3, gemma2...)",
        "ru": "Или впиши прямо имя модели (например: phi3, gemma2...)",
    },
    "select_model_prompt": {
        "es": "Opción: ",
        "en": "Option: ",
        "fr": "Choix : ",
        "ru": "Избери: ",
    },
    "select_model_chosen": {
        "es": "Modelo seleccionado: {model}",
        "en": "Selected model: {model}",
        "fr": "Modèle sélectionné : {model}",
        "ru": "Модель избранная: {model}",
    },
    "model_not_installed": {
        "es": "El modelo '{model}' no está instalado. ¿Descargarlo ahora? (s/n): ",
        "en": "The model '{model}' is not installed. Download it now? (y/n): ",
        "fr": "Le modèle '{model}' n'est pas installé. Le télécharger maintenant ? (o/n) : ",
        "ru": "Модель «{model}» не водворена. Стяжати ли ныне? (д/н): ",
    },
    "model_downloading": {
        "es": "\nDescargando modelo {model}... (puede tardar varios minutos)",
        "en": "\nDownloading model {model}... (this may take several minutes)",
        "fr": "\nTéléchargement du modèle {model}... (cela peut prendre plusieurs minutes)",
        "ru": "\nСтяжается модель {model}... (сие может продлиться неколико минут)",
    },
    "model_ready": {
        "es": "Modelo listo.",
        "en": "Model ready.",
        "fr": "Modèle prêt.",
        "ru": "Модель уготована.",
    },
    "model_continue_without_download": {
        "es": "Continuando sin descargar. El modelo debe estar disponible en Ollama.",
        "en": "Continuing without downloading. The model must be available in Ollama.",
        "fr": "Poursuite sans téléchargement. Le modèle doit être disponible dans Ollama.",
        "ru": "Шествуем далее без стяжания. Модель долженствует быти наличною во Ollama.",
    },
    "main_title_bar": {
        "es": "=" * 50,
        "en": "=" * 50,
        "fr": "=" * 50,
        "ru": "=" * 50,
    },
    "main_title": {
        "es": "   Unity Docs AI — Asistente Local con RAG",
        "en": "   Unity Docs AI — Local Assistant with RAG",
        "fr": "   Unity Docs AI — Assistant Local avec RAG",
        "ru": "   Unity Docs AI — Местный Помощник с RAG",
    },
    "ollama_not_detected": {
        "es": "\n⚠  AVISO: No se detectó Ollama en el puerto {port}.",
        "en": "\n⚠  WARNING: Ollama was not detected on port {port}.",
        "fr": "\n⚠  AVIS : Ollama n'a pas été détecté sur le port {port}.",
        "ru": "\n⚠  ВЕСТЬ: Ollama не обретена на вратах {port}.",
    },
    "ollama_not_running_hint": {
        "es": "Asegúrate de que Ollama está ejecutándose antes de hacer preguntas.",
        "en": "Make sure Ollama is running before asking questions.",
        "fr": "Assurez-vous qu'Ollama est en cours d'exécution avant de poser des questions.",
        "ru": "Удостоверися, яко Ollama действует, прежде неже вопрошати.",
    },
    "press_enter_continue": {
        "es": "Pulsa Enter para continuar de todas formas...",
        "en": "Press Enter to continue anyway...",
        "fr": "Appuyez sur Entrée pour continuer quand même...",
        "ru": "Нажми Ввод, дабы продолжити купно...",
    },
    "preparing_docs": {
        "es": "\nPreparando documentación...",
        "en": "\nPreparing documentation...",
        "fr": "\nPréparation de la documentation...",
        "ru": "\nГотовится поучение (документация)...",
    },
    "first_run_creating_db": {
        "es": "\nPrimera ejecución: creando base de datos vectorial.",
        "en": "\nFirst run: creating vector database.",
        "fr": "\nPremière exécution : création de la base de données vectorielle.",
        "ru": "\nПервое тщание: созидается векторная сокровищница данных.",
    },
    "first_run_may_take": {
        "es": "Esto puede tardar varios minutos dependiendo del tamaño de la documentación.\n",
        "en": "This may take several minutes depending on the size of the documentation.\n",
        "fr": "Cela peut prendre plusieurs minutes selon la taille de la documentation.\n",
        "ru": "Сие может продлиться неколико минут, смотря по обилию поучения.\n",
    },
    "db_found": {
        "es": "Base de datos encontrada ({count} chunks indexados). Cargando directamente.",
        "en": "Database found ({count} chunks indexed). Loading directly.",
        "fr": "Base de données trouvée ({count} fragments indexés). Chargement direct.",
        "ru": "Сокровищница обретена ({count} частей внесено). Загружается прямо.",
    },
    "assistant_ready_bar": {
        "es": "=" * 50,
        "en": "=" * 50,
        "fr": "=" * 50,
        "ru": "=" * 50,
    },
    "assistant_ready": {
        "es": "Asistente listo. Escribe 'exit' para salir.",
        "en": "Assistant ready. Type 'exit' to quit.",
        "fr": "Assistant prêt. Tapez 'exit' pour quitter.",
        "ru": "Помощник уготован. Впиши 'exit', дабы изыти.",
    },
    "ask_question_prompt": {
        "es": "Pregunta sobre Unity: ",
        "en": "Question about Unity: ",
        "fr": "Question sur Unity : ",
        "ru": "Вопроси о Unity: ",
    },
    "exiting": {
        "es": "\nSaliendo...",
        "en": "\nExiting...",
        "fr": "\nSortie...",
        "ru": "\nИсходим...",
    },
    "farewell": {
        "es": "¡Hasta luego!",
        "en": "Goodbye!",
        "fr": "Au revoir !",
        "ru": "Прощай покамест!",
    },
    "searching_docs": {
        "es": "\nBuscando en la documentación...",
        "en": "\nSearching the documentation...",
        "fr": "\nRecherche dans la documentation...",
        "ru": "\nИщется во поучении...",
    },
    "separator_bar": {
        "es": "-" * 50 + "\n",
        "en": "-" * 50 + "\n",
        "fr": "-" * 50 + "\n",
        "ru": "-" * 50 + "\n",
    },
    "zip_already_extracted": {
        "es": "ZIP ya extraído ({count} archivos encontrados), se omite extracción.",
        "en": "ZIP already extracted ({count} files found), skipping extraction.",
        "fr": "ZIP déjà extrait ({count} fichiers trouvés), extraction ignorée.",
        "ru": "Свиток (ZIP) уже распакован ({count} хартий обретено), распаковка минуется.",
    },
    "zip_not_found": {
        "es": "\nERROR: No se encontró el archivo '{zip_path}'.",
        "en": "\nERROR: File '{zip_path}' was not found.",
        "fr": "\nERREUR : Le fichier '{zip_path}' est introuvable.",
        "ru": "\nПОГРЕШНОСТЬ: Не обретена хартия «{zip_path}».",
    },
    "zip_not_found_hint": {
        "es": "Coloca el ZIP de la documentación de Unity en la misma carpeta que este script.",
        "en": "Place the Unity documentation ZIP in the same folder as this script.",
        "fr": "Placez le ZIP de la documentation Unity dans le même dossier que ce script.",
        "ru": "Положи свиток поучения Unity во едину храмину со сим писанием.",
    },
    "zip_extracting": {
        "es": "\nExtrayendo '{zip_path}'...",
        "en": "\nExtracting '{zip_path}'...",
        "fr": "\nExtraction de '{zip_path}'...",
        "ru": "\nРаспаковуется «{zip_path}»...",
    },
    "zip_extracting_desc": {
        "es": "Extrayendo",
        "en": "Extracting",
        "fr": "Extraction",
        "ru": "Распаковуется",
    },
    "zip_extracting_unit": {
        "es": "archivo",
        "en": "file",
        "fr": "fichier",
        "ru": "хартия",
    },
    "zip_extraction_complete": {
        "es": "Extracción completada. {total} archivos en '{extract_path}'.",
        "en": "Extraction complete. {total} files in '{extract_path}'.",
        "fr": "Extraction terminée. {total} fichiers dans '{extract_path}'.",
        "ru": "Распаковка свершена. {total} хартий во «{extract_path}».",
    },
    "files_not_found": {
        "es": "\nERROR: No se encontraron archivos .txt/.md/.html en '{extract_path}'.",
        "en": "\nERROR: No .txt/.md/.html files found in '{extract_path}'.",
        "fr": "\nERREUR : Aucun fichier .txt/.md/.html trouvé dans '{extract_path}'.",
        "ru": "\nПОГРЕШНОСТЬ: Не обретены хартии .txt/.md/.html во «{extract_path}».",
    },
    "files_not_found_hint": {
        "es": "Verifica que el ZIP contiene documentación de Unity.",
        "en": "Verify that the ZIP contains Unity documentation.",
        "fr": "Vérifiez que le ZIP contient la documentation Unity.",
        "ru": "Удостоверися, яко свиток содержит поучение Unity.",
    },
    "files_found": {
        "es": "\nArchivos encontrados: {count}",
        "en": "\nFiles found: {count}",
        "fr": "\nFichiers trouvés : {count}",
        "ru": "\nХартий обретено: {count}",
    },
    "reading_files_desc": {
        "es": "Leyendo archivos",
        "en": "Reading files",
        "fr": "Lecture des fichiers",
        "ru": "Чтутся хартии",
    },
    "reading_files_unit": {
        "es": "archivo",
        "en": "file",
        "fr": "fichier",
        "ru": "хартия",
    },
    "valid_docs_loaded": {
        "es": "Documentos válidos cargados: {count}",
        "en": "Valid documents loaded: {count}",
        "fr": "Documents valides chargés : {count}",
        "ru": "Хартий годных загружено: {count}",
    },
    "building_index": {
        "es": "\nConstruyendo índice de vectores...",
        "en": "\nBuilding vector index...",
        "fr": "\nConstruction de l'index vectoriel...",
        "ru": "\nСозидается указатель векторный...",
    },
    "chunking_docs": {
        "es": "Dividiendo documentos en chunks...",
        "en": "Splitting documents into chunks...",
        "fr": "Division des documents en fragments...",
        "ru": "Разделяются хартии на части...",
    },
    "chunking_desc": {
        "es": "Chunking",
        "en": "Chunking",
        "fr": "Fragmentation",
        "ru": "Разделение",
    },
    "chunking_unit": {
        "es": "doc",
        "en": "doc",
        "fr": "doc",
        "ru": "хартия",
    },
    "total_chunks_generated": {
        "es": "Total de chunks generados: {count}",
        "en": "Total chunks generated: {count}",
        "fr": "Total des fragments générés : {count}",
        "ru": "Частей всего сотворено: {count}",
    },
    "all_chunks_already_indexed": {
        "es": "Todos los chunks ya estaban indexados.",
        "en": "All chunks were already indexed.",
        "fr": "Tous les fragments étaient déjà indexés.",
        "ru": "Все части уже бяху внесены.",
    },
    "new_chunks_to_index": {
        "es": "Chunks nuevos a indexar: {count}",
        "en": "New chunks to index: {count}",
        "fr": "Nouveaux fragments à indexer : {count}",
        "ru": "Частей новых ко внесению: {count}",
    },
    "indexing_desc": {
        "es": "Indexando",
        "en": "Indexing",
        "fr": "Indexation",
        "ru": "Внесение",
    },
    "indexing_unit": {
        "es": "batch",
        "en": "batch",
        "fr": "lot",
        "ru": "связка",
    },
    "indexing_complete": {
        "es": "\nIndexación completada. Total de chunks en DB: {count}",
        "en": "\nIndexing complete. Total chunks in DB: {count}",
        "fr": "\nIndexation terminée. Total des fragments dans la BD : {count}",
        "ru": "\nВнесение свершено. Частей всего во сокровищнице: {count}",
    },
    "loading_embedding_model": {
        "es": "Cargando modelo de embeddings... (solo la primera vez puede tardar)",
        "en": "Loading embedding model... (this may only take time on the first run)",
        "fr": "Chargement du modèle d'embeddings... (peut prendre du temps seulement la première fois)",
        "ru": "Загружается модель вложений (embeddings)... (токмо в первый раз продлится)",
    },
    "ollama_timeout_error": {
        "es": "Error: El modelo tardó demasiado en responder. Prueba con un modelo más pequeño.",
        "en": "Error: The model took too long to respond. Try a smaller model.",
        "fr": "Erreur : Le modèle a mis trop de temps à répondre. Essayez un modèle plus petit.",
        "ru": "Погрешность: Модель замедлила со ответом. Попытай модель меньшую.",
    },
    "ollama_connection_error": {
        "es": "Error: No se pudo conectar con Ollama en el puerto {port}. ¿Está ejecutándose?",
        "en": "Error: Could not connect to Ollama on port {port}. Is it running?",
        "fr": "Erreur : Impossible de se connecter à Ollama sur le port {port}. Est-il en cours d'exécution ?",
        "ru": "Погрешность: Не удалося соединитися со Ollama на вратах {port}. Действует ли она?",
    },
    "ollama_unexpected_error": {
        "es": "Error inesperado: {error}",
        "en": "Unexpected error: {error}",
        "fr": "Erreur inattendue : {error}",
        "ru": "Погрешность нечаянная: {error}",
    },
}


def t(key: str, **kwargs) -> str:
    """Devuelve el texto correspondiente a `key` en el idioma actual (LANGUAGE),
    aplicando formato con los argumentos dados si los hay."""
    entry = TEXTS.get(key, {})
    text = entry.get(LANGUAGE, entry.get("es", key))
    if kwargs:
        try:
            return text.format(**kwargs)
        except Exception:
            return text
    return text

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

    print(t("select_language_title"))
    print(t("select_language_opt1"))
    print(t("select_language_opt2"))
    print(t("select_language_opt3"))
    print(t("select_language_opt4"))

    choice = input(t("select_language_prompt")).strip()

    if choice == "1":
        LANGUAGE = "es"
    elif choice == "2":
        LANGUAGE = "en"
    elif choice == "3":
        LANGUAGE = "fr"
    elif choice == "4":
        LANGUAGE = "ru"
    else:
        print(t("select_language_invalid"))
        LANGUAGE = "es"

# -------------------------------------------------
# PUERTO DE OLLAMA
# -------------------------------------------------

def select_port():
    global OLLAMA_PORT, OLLAMA_URL

    print(t("ollama_port_title"))
    port_input = input(t("ollama_port_prompt")).strip()

    if port_input == "":
        OLLAMA_PORT = 11434
    else:
        try:
            OLLAMA_PORT = int(port_input)
        except ValueError:
            print(t("ollama_port_invalid"))
            OLLAMA_PORT = 11434

    OLLAMA_URL = f"http://localhost:{OLLAMA_PORT}/api/generate"
    print(t("ollama_port_using", port=OLLAMA_PORT))

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
    print(t("model_downloading", model=model_name))
    subprocess.run(["ollama", "pull", model_name])
    print(t("model_ready"))


def select_model():
    global MODEL_NAME

    print(t("select_model_title"))
    for key, name in AVAILABLE_MODELS.items():
        print(f"{key}. {name}")
    print(t("select_model_hint"))

    choice = input(t("select_model_prompt")).strip()

    if choice in AVAILABLE_MODELS:
        MODEL_NAME = AVAILABLE_MODELS[choice]
    elif choice != "":
        MODEL_NAME = choice
    else:
        MODEL_NAME = "llama3.2"

    print(t("select_model_chosen", model=MODEL_NAME))

    if not check_model_exists(MODEL_NAME):
        download = input(t("model_not_installed", model=MODEL_NAME)).strip().lower()
        if download in ("s", "y", "si", "sí", "yes", "д", "да"):
            pull_model(MODEL_NAME)
        else:
            print(t("model_continue_without_download"))

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
    "ru": "Отвечай всегда по-русски, языком возвышенным и старинным.",
}

def get_system_prompt():
    lang_instruction = LANG_INSTRUCTIONS.get(LANGUAGE, "Responde siempre en español.")
    return f"{SYSTEM_PROMPT_BASE}\n\n{lang_instruction}"

# -------------------------------------------------
# EMBEDDINGS
# -------------------------------------------------

print(t("loading_embedding_model"))
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
            print(t("zip_already_extracted", count=len(existing)))
            return

    if not os.path.exists(ZIP_PATH):
        print(t("zip_not_found", zip_path=ZIP_PATH))
        print(t("zip_not_found_hint"))
        exit(1)

    print(t("zip_extracting", zip_path=ZIP_PATH))
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        members = zip_ref.infolist()
        for member in tqdm(members, desc=t("zip_extracting_desc"), unit=t("zip_extracting_unit")):
            zip_ref.extract(member, EXTRACT_PATH)

    total = len(list(Path(EXTRACT_PATH).rglob("*")))
    print(t("zip_extraction_complete", total=total, extract_path=EXTRACT_PATH))

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
        print(t("files_not_found", extract_path=EXTRACT_PATH))
        print(t("files_not_found_hint"))
        exit(1)

    print(t("files_found", count=len(all_files)))

    is_html_ext = {".html", ".htm"}

    for f in tqdm(all_files, desc=t("reading_files_desc"), unit=t("reading_files_unit")):
        try:
            raw = f.read_text(encoding="utf-8", errors="ignore")
            is_html = f.suffix.lower() in is_html_ext
            cleaned = clean_text(raw, is_html)

            if len(cleaned) < 100:   # ignorar archivos casi vacíos
                continue

            docs.append({"text": cleaned, "source": str(f)})
        except Exception as e:
            pass   # ignorar archivos no legibles

    print(t("valid_docs_loaded", count=len(docs)))
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
    print(t("building_index"))

    all_chunks = []
    all_sources = []
    all_ids = []

    print(t("chunking_docs"))
    for doc in tqdm(docs, desc=t("chunking_desc"), unit=t("chunking_unit")):
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            chunk_id = make_id(doc["source"], i)
            all_chunks.append(chunk)
            all_sources.append(doc["source"])
            all_ids.append(chunk_id)

    print(t("total_chunks_generated", count=len(all_chunks)))

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
        print(t("all_chunks_already_indexed"))
        return

    print(t("new_chunks_to_index", count=len(new_chunks)))

    # Codificar en batches para no saturar la RAM
    BATCH_SIZE = 128
    total_batches = (len(new_chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_i in tqdm(range(total_batches), desc=t("indexing_desc"), unit=t("indexing_unit")):
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

    print(t("indexing_complete", count=collection.count()))

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
        return t("ollama_timeout_error")
    except requests.exceptions.ConnectionError:
        return t("ollama_connection_error", port=OLLAMA_PORT)
    except Exception as e:
        return t("ollama_unexpected_error", error=e)

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
    print(t("main_title_bar"))
    print(t("main_title"))
    print(t("main_title_bar"))

    select_language()
    select_port()
    select_model()

    if not check_ollama_running():
        print(t("ollama_not_detected", port=OLLAMA_PORT))
        print(t("ollama_not_running_hint"))
        input(t("press_enter_continue"))

    print(t("preparing_docs"))
    extract_zip_if_needed()

    if not db_exists():
        print(t("first_run_creating_db"))
        print(t("first_run_may_take"))
        docs = load_files()
        build_index(docs)
    else:
        count = collection.count()
        print(t("db_found", count=count))

    print("\n" + t("assistant_ready_bar"))
    print(t("assistant_ready"))
    print(t("assistant_ready_bar") + "\n")

    while True:
        try:
            q = input(t("ask_question_prompt")).strip()
        except (EOFError, KeyboardInterrupt):
            print(t("exiting"))
            break

        if not q:
            continue

        if q.lower() in ("exit", "quit", "salir"):
            print(t("farewell"))
            break

        print(t("searching_docs"))
        answer = ask(q)
        print("\n" + answer + "\n")
        print(t("separator_bar"))
