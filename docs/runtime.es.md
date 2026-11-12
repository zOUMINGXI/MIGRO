<p align="center">
  <a href="runtime.md">English</a> &nbsp; · &nbsp; <a href="runtime.zh-CN.md">简体中文</a> &nbsp; · &nbsp; <strong>Español</strong>
</p>

# Ejecutar MIGRO

El paquete compartido incluye el núcleo de memoria congelado, las tablas de codificación, los tokenizadores y las interfaces de lectura de Phi/Mistral, junto con la procedencia de los componentes y los avisos de licencia. El código y los recursos propios de MIGRO usan Apache-2.0; los componentes de terceros incluidos conservan sus licencias originales.

## Instalación

La configuración comprobada utiliza Linux, Python 3.10, un compilador C++17 y PyTorch 2.5.1. Instala una versión de PyTorch adecuada para tu hardware. La validación con CUDA utilizó:

~~~bash
git clone https://github.com/zOUMINGXI/MIGRO.git
cd MIGRO
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e '.[runtime,benchmarks]' -c constraints/runtime-tested.txt
~~~

Para las operaciones de memoria en CPU, instala la versión CPU de PyTorch sustituyendo el índice CUDA anterior por `https://download.pytorch.org/whl/cpu`.

## Descargar y verificar los recursos compartidos

~~~bash
export MIGRO_ASSETS="$PWD/migro-assets"
python scripts/setup_runtime.py --assets-dir "$MIGRO_ASSETS"
python scripts/verify_runtime.py --assets-dir "$MIGRO_ASSETS"
~~~

El comando de configuración descarga el paquete público y compila los tres núcleos de codificación en C++. El comando de verificación carga el núcleo real y ambos checkpoints de interfaz, escribe observaciones, guarda y vuelve a cargar el estado, lo transfiere, añade observaciones, lo compacta y restaura los vínculos originales y los metadatos.

[Descargar el paquete](https://github.com/zOUMINGXI/MIGRO/releases/tag/runtime-assets-20261112) · [Fuentes y licencias de los componentes](../THIRD_PARTY_NOTICES.md) · [Procedencia](../PROVENANCE.json)

El manifiesto `ASSETS.json` enumera los archivos necesarios. La exportación conserva los valores y tipos de los tensores e incluye el ámbito de interfaz resuelto. Se han eliminado las rutas locales, las configuraciones de entrenamiento y los registros internos de experimentos.

## Generar respuestas nuevas

La demo generativa necesita modelos base y un modelo de reordenación externos. Descárgalos de sus autores mediante el script de configuración:

~~~bash
python scripts/setup_runtime.py --assets-dir "$MIGRO_ASSETS" --skip-assets \
  --models qwen mistral reranker
MIGRO_DEVICE=cuda python demo/run.py --case demo/starter.json --output trace.json
MIGRO_DEVICE=cuda python demo/serve.py --trace trace.json
~~~

Abre `http://127.0.0.1:18764`. La sesión capturada incluye respuestas del modelo de origen, respuestas con el estado transferido, escritura continuada, solo los mensajes posteriores y reconstrucción del historial completo. Utiliza la página local para editar una sesión y generar respuestas nuevas.

Ambos modelos base necesitan varios GB de almacenamiento. La demo completa Qwen 4B → Mistral 7B se comprobó en una H800 de 80GB. La demo pública del navegador también permite reproducir sesiones registradas.

Para utilizar la interfaz de Phi, añade `--models phi reranker`. Las descargas usan `huggingface_hub` y su configuración habitual de autenticación cuando el autor del modelo la requiere.

Si ya dispones de los modelos, configura sus directorios antes de importar el entorno de ejecución:

~~~bash
export MIGRO_QWEN_MODEL=/path/to/Qwen3-4B-Instruct-2507
export MIGRO_MISTRAL_MODEL=/path/to/Mistral-7B-Instruct-v0.3
export MIGRO_PHI_MODEL=/path/to/Phi-3-mini-4k-instruct
export MIGRO_RERANKER_MODEL=/path/to/ms-marco-MiniLM-L6-v2
~~~

## API de Python

~~~python
from migro.memory import MemoryStore, load_core

core = load_core("cpu")
store = MemoryStore.create(core)
store.observe([{
    "speaker": "Ada", "role": "observed_human", "body": "I moved to Kyoto.",
    "session": "session_1", "date": "1 January, 2026", "dialogue_id": "s1:0"
}])
store.save("state.bin")
receiver = MemoryStore.load(core, "state.bin").handoff()
~~~

`MemoryRuntime.observe()` proporciona escritura nativa con el modelo anfitrión congelado y requiere configurar dicho modelo. Los nombres relativos de los recursos mantienen el formato de los estados guardados. `MIGRO_NATIVE_DIR` permite cambiar la ubicación de los núcleos generados y `MIGRO_PROFILES` selecciona otro JSON de perfiles de lectura.

## Comprobaciones

~~~bash
MIGRO_TEST_ASSETS=1 python -m unittest discover -s tests -v
python -m pip check
~~~

La descarga incluye `ASSETS.json`, `PROVENANCE.json`, `NOTICE` y los archivos de licencia. Las pruebas de estado en CPU y las comprobaciones de generación con modelos reales se resumen en el [registro de validación](../assets/runtime/validation.json).

[Volver al proyecto](../README.es.md)
