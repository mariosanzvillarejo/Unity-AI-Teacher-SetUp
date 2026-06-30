# Unity Docs AI — Asistente Local con RAG

Un chatbot que responde preguntas sobre Unity usando **tu propia copia de la documentación oficial**, completamente local. No necesita internet para funcionar, no envía tus preguntas a ningún servidor externo.

**¿Qué es RAG?** (Retrieval-Augmented Generation) Es una técnica que permite a un modelo de IA buscar información relevante en una base de datos propia antes de responder, en lugar de depender solo de lo que "sabe" de su entrenamiento.

\---

## Requisitos del sistema

* **Sistema operativo:** Windows 10/11, macOS, o Linux
* **Python:** 3.10 o superior
* **RAM:** Mínimo 8 GB (16 GB recomendado)
* **Espacio en disco:** \~5 GB (documentación + modelos de IA)
* **Ollama instalado** (ver instrucciones abajo)

\---

## Paso 1 — Instalar Ollama

Ollama es el programa que ejecuta los modelos de lenguaje localmente en tu ordenador.

### Windows

1. Ve a [https://ollama.com/download](https://ollama.com/download)
2. Descarga el instalador para Windows (`.exe`)
3. Ejecuta el instalador y sigue los pasos
4. Cuando termine, Ollama se ejecutará automáticamente en segundo plano
5. Puedes verificar que funciona abriendo una terminal (`cmd` o PowerShell) y escribiendo:

```
   ollama --version
   ```

Si ves un número de versión, está bien instalado.

### macOS

1. Ve a [https://ollama.com/download](https://ollama.com/download)
2. Descarga el archivo `.dmg` para macOS
3. Ábrelo y arrastra Ollama a la carpeta Aplicaciones
4. Abre Ollama desde el Launchpad o desde Aplicaciones
5. Verifica en el terminal:

```
   ollama --version
   ```

   ### Linux

   Abre una terminal y ejecuta:

   ```bash
curl -fsSL https://ollama.com/install.sh | sh

   curl -fsSL https://ollama.com/install.sh | sh

   ```

   Ollama se instalará y se configurará como servicio del sistema.

   \\---

   ## Paso 2 — Instalar Python

   Si ya tienes Python 3.10 o superior instalado, salta este paso.

   ### Comprobar si tienes Python instalado

   Abre una terminal y escribe:

   ```

   python --version

   ```

   o

   ```

   python3 --version

   ```

   Si ves `Python 3.10.x` o superior, ya lo tienes.

   ### Si no lo tienes

1. Ve a \[https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Descarga la versión más reciente
3. \*\*En Windows:\*\* durante la instalación, marca la casilla \*\*"Add Python to PATH"\*\* (muy importante)
4. Completa la instalación

   \\---

   ## Paso 3 — Preparar la carpeta del proyecto

1. Descarga o clona este proyecto en tu ordenador
2. La estructura de carpetas debe quedar así:

   ```

   unity\_rag/
├── main.py
├── requirements.txt
├── system\_prompt.txt          ← (opcional, se crea automáticamente)
├── unity\_docs.zip             ← DEBES colocar aquí el ZIP de la documentación
├── docs/                      ← (se crea automáticamente al extraer el ZIP)
└── unity\_db/                  ← (se crea automáticamente al indexar)

   ```

   ### ¿Dónde consigo el ZIP de la documentación de Unity?

1. Ve a \[https://docs.unity3d.com](https://docs.unity3d.com)
2. En la página de documentación, busca la opción de descarga offline (suele estar en el pie de página o en "Download Documentation")
3. Descarga el ZIP y \*\*colócalo en la carpeta del proyecto con el nombre exacto `unity\\\_docs.zip`\*\*

   \\---

   ## Paso 4 — Instalar las dependencias de Python

   Abre una terminal \*\*dentro de la carpeta del proyecto\*\* y ejecuta:

   ```bash
pip install -r requirements.txt
```

   Si tienes varios Python instalados y el comando anterior no funciona, prueba:

   ```bash
pip3 install -r requirements.txt
```

   Esto descargará e instalará todas las librerías necesarias. Puede tardar unos minutos.

   \---

   ## Paso 5 — Ejecutar el asistente

   ### Asegúrate de que Ollama está ejecutándose

* **Windows/macOS:** Debería estar corriendo en segundo plano si lo instalaste. Puedes comprobarlo buscando el icono de Ollama en la barra del sistema.
* **Linux:** Ejecuta en una terminal: `ollama serve` (déjala abierta)

  ### Ejecutar el script

  En la terminal, dentro de la carpeta del proyecto:

  ```bash
python main.py

  python main.py

  ```

  o en macOS/Linux:

  ```bash
python3 main.py
```

  ### Primera ejecución

  El programa te hará varias preguntas:

1. **Idioma:** elige en qué idioma quieres las respuestas (1 = Español)
2. **Puerto de Ollama:** pulsa Enter para usar el puerto por defecto (11434), o escribe otro si lo cambiaste
3. **Modelo de lenguaje:** elige qué modelo quieres usar

|Opción|Modelo|RAM recomendada|Velocidad|
|-|-|-|-|
|1|llama3.2|8 GB|Rápido|
|2|llama3.1|8 GB|Medio|
|3|qwen2.5|8 GB|Medio|
|4|mistral|6 GB|Rápido|
|5|glm4|8 GB|Medio|

Si el modelo no está instalado, el programa te preguntará si quieres descargarlo automáticamente.

4. **Indexación:** la primera vez, el programa leerá toda la documentación y la indexará en una base de datos vectorial. Esto puede tardar **entre 5 y 30 minutos** dependiendo del tamaño de la documentación y la velocidad de tu ordenador. Solo ocurre la primera vez.

   Cuando veas el mensaje `Asistente listo`, ya puedes hacer preguntas.

   \---

   ## Uso

   Una vez que el asistente está listo, simplemente escribe tu pregunta y pulsa Enter:

   ```
Pregunta sobre Unity: ¿Cómo funciona el sistema de físicas de Unity?

   Pregunta sobre Unity: ¿Cómo funciona el sistema de físicas de Unity?

   Pregunta sobre Unity: ¿Qué es un ScriptableObject y cuándo debo usarlo?

   Pregunta sobre Unity: ¿Cómo puedo hacer que un objeto siga al jugador con NavMesh?

   ```

   Para salir, escribe `exit` o `salir` y pulsa Enter.

   \\---

   ## Personalizar el prompt del sistema

   Puedes crear un archivo `system\\\_prompt.txt` en la carpeta del proyecto para personalizar cómo responde el asistente. Por ejemplo:

   ```

   Eres un asistente experto en Unity especializado en juegos 2D con enfoque en principiantes.
Siempre explica los conceptos paso a paso y proporciona ejemplos de código comentados en C#.
Si no encuentras la respuesta en la documentación, dilo claramente.

   ```

   Si no existe el archivo, se usará un prompt por defecto.

   \\---

   ## Preguntas frecuentes

   ### La indexación tarda mucho, ¿es normal?

   Sí. La primera vez el programa lee todos los archivos HTML/Markdown de la documentación, los divide en fragmentos, calcula su representación matemática (embeddings) y los guarda. Dependiendo del tamaño de la documentación esto puede tomar entre 5 y 45 minutos. Las siguientes ejecuciones cargan directamente la base de datos y son instantáneas.

   ### El modelo responde muy lento

   Prueba con un modelo más pequeño (llama3.2 o mistral). Si tu ordenador tiene tarjeta gráfica NVIDIA, Ollama la usará automáticamente para acelerar las respuestas.

   ### El asistente dice que no sabe la respuesta

   Puede ser que esa información no esté en el ZIP de documentación que descargaste, o que la pregunta sea demasiado específica. Intenta reformularla o pregunta de forma más general.

   ### Error: "No se pudo conectar con Ollama"

\* Asegúrate de que Ollama está ejecutándose
\* En Windows/macOS, busca el icono en la bandeja del sistema
\* En Linux, ejecuta `ollama serve` en una terminal separada
\* Verifica que el puerto sea correcto (por defecto 11434)

  ### ¿Puedo usar un modelo que no aparece en la lista?

  Sí. Cuando el programa te pida elegir modelo, escribe directamente el nombre del modelo que quieras usar (por ejemplo: `phi3`, `gemma2`, `deepseek-r1`). Puedes ver todos los modelos disponibles en \[https://ollama.com/library](https://ollama.com/library).

  ### ¿Puedo reindexar la documentación?

  Sí. Borra la carpeta `unity\\\_db/` y vuelve a ejecutar el script. Volverá a indexar todo desde cero.

  \\---

  ## Archivo requirements.txt

  El archivo `requirements.txt` debe contener:

  ```

   chromadb
sentence-transformers
requests
tqdm

   ```

  \\---

  ## Qué hace cada archivo

|Archivo|Descripción|
|-|-|
|`main.py`|Script principal del asistente|
|`requirements.txt`|Dependencias de Python necesarias|
|`system\\\_prompt.txt`|Instrucciones de comportamiento del asistente (opcional)|
|`unity\\\_docs.zip`|La documentación de Unity que debes descargar tú|
|`docs/`|Carpeta donde se extrae el ZIP (se crea automáticamente)|
|`unity\\\_db/`|Base de datos vectorial con la documentación indexada (se crea automáticamente)|

\\---

## Licencia

Este proyecto es de uso personal y educativo. La documentación de Unity está sujeta a los términos de uso de Unity Technologies.


