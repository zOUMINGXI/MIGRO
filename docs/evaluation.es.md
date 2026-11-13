# Evidencia experimental

Una actualización debe conservar el historial útil y dejar espacio para la siguiente experiencia. Los resultados actuales evalúan la utilidad de la memoria, modelos más grandes, la escritura continuada y el espacio persistente. A continuación se muestran los grupos completados; el [índice de resultados con fecha](../results/README.md) recoge los protocolos y el trabajo pendiente.

### Utilidad de la memoria

| Método | LoCoMo | LongMemEval-S | MemOps | BEAM-1M | MAB · principal |
|---|---:|---:|---:|---:|---:|
| MIGRO | 54.13 | 68.40 | 66.30 | 55.83 | 35.16 |
| PlugMem | 41.89 | 43.20 | 35.09 | 34.11 | 7.76 |
| Mem0 | 52.12 | 56.80 | 52.13 | 52.43 | 32.82 |
| TiMem | 49.13 | 48.40 | 50.66 | 51.90 | 18.90 |
| A-MEM | 36.20 | 35.10 | 39.78 | 33.25 | 16.72 |
| LSTMem | 40.64 | 26.00 | 52.59 | 28.44 | 6.56 |
| TransMem | 45.85 | 37.40 | 62.51 | 28.76 | 22.80 |
| Doc-to-LoRA | 14.96 | 12.80 | 15.35 | 30.20 | 11.62 |

Las puntuaciones están en una escala de 0–100. LoCoMo utiliza un indicador compuesto propio para todas las preguntas (1,986); LongMemEval-S, exactitud auxiliar (500); MemOps, exactitud longitudinal de respuesta (2,006); y BEAM, un agregado por categorías (700). MAB utiliza una media macro de cuatro capacidades definida para este estudio (2,071). Estas métricas no son intercambiables.

**Experimentos independientes de sensibilidad al presupuesto en MAB:** con el mismo límite de 2,000 tokens de origen, MIGRO obtiene **35.14** y TransMem **22.88**; con 8,000, obtienen **40.30** y **28.99**. Estos experimentos se mantienen separados de la tabla principal de MAB del manuscrito. Los demás métodos conservan sus rutas de lectura nativas; igualar el límite de tokens de origen no iguala el cómputo de inferencia.

[Resumen de benchmarks · CSV](../results/performance_snapshot.csv) · [JSON](../results/performance_snapshot.json) · [Definiciones de puntuación](../docs/performance_protocol.md)

### Modelos más grandes, memoria existente

|Modelo Qwen receptor|MIGRO · State Transfer desde 4B|MIGRO · Full-History Rebuilding|
|---|---:|---:|
|8B|55.52|55.68|
|14B|60.37|60.32|
|32B|62.33|61.83|

La cadena heterogénea independiente **Qwen 4B → Ministral 14B → Gemma 26B-A4B** obtiene **57.41 → 66.28** en los dos modelos receptores, con un presupuesto de 2,000 tokens de origen y sin nuevos ejemplos de calibración. Ambos saltos reutilizan el mismo estado inmutable, sin escrituras intermedias. Gemma 26B-A4B indica el tamaño total/activo del modelo MoE. Esta cadena es distinta del experimento histórico con Phi/Mistral que aparece más abajo.

[Escalado con el mismo modelo anfitrión](../results/quality_scaling.csv) · [Resúmenes de migración](../results/migration_snapshot.csv)

<details>
<summary><strong>Todos los métodos · escalado y migración entre modelos Qwen</strong></summary>

| Modelo anfitrión | MIGRO | PlugMem | Mem0 | TiMem | A-MEM | LSTMem | TransMem | Doc-to-LoRA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B-Instruct-2507 | 54.13 | 41.89 | 52.12 | 49.13 | 36.20 | 40.64 | 45.85 | 14.96 |
| Qwen3-8B | 55.68 | 34.86 | 54.88 | 46.19 | 44.70 | 19.70 | 46.39 | 12.77 |
| Qwen3-14B | 60.32 | 40.10 | 54.36 | 50.25 | 40.88 | 16.58 | 49.83 | 12.30 |
| Qwen3-32B | 61.83 | 44.66 | 57.16 | 49.24 | 45.38 | 14.35 | 48.03 | 13.18 |

| Método | Operación | Qwen 8B | Qwen 14B | Qwen 32B |
|---|---:|---:|---:|---:|
| MIGRO | State Transfer | 55.52 | 60.37 | 62.33 |
| PlugMem | State Transfer | 44.41 | 46.10 | 49.15 |
| Mem0 | State Transfer | 53.18 | 57.07 | 57.17 |
| TiMem | Full-History Rebuilding | 46.19 | 50.25 | 49.24 |
| A-MEM | Full-History Rebuilding | 44.70 | 40.88 | 45.38 |
| LSTMem | Full-History Rebuilding | 19.70 | 16.58 | 14.35 |
| TransMem | Full-History Rebuilding | 46.39 | 49.83 | 48.03 |
| Doc-to-LoRA | Full-History Rebuilding | 12.77 | 12.30 | 13.18 |

Los valores de reconstrucción de TransMem en Qwen son las referencias con el mismo modelo anfitrión utilizadas en el perfil de transferencia del manuscrito.

</details>

### Seguir escribiendo después del traspaso

En el estudio de continuación Qwen 4B → Mistral 7B, MIGRO obtiene **45.13** tras añadir la segunda mitad del historial, frente a **36.94** con solo transferencia y **45.19** con reconstrucción completa en el destino. Las tres condiciones se puntúan respecto al mismo historial final. Las preguntas sobre el historial anterior, el nuevo y las que integran ambos se informan por separado para los ocho métodos.

<details>
<summary><strong>Resultados de escritura continuada para todos los métodos</strong></summary>

| Método | Solo memoria de la primera mitad | Escritura continuada | Reconstrucción del historial completo |
|---|---:|---:|---:|
| MIGRO | 36.94 | 45.13 | 45.19 |
| PlugMem | 33.46 | 37.38 | 36.95 |
| Mem0 | 35.59 | 43.89 | 44.37 |
| TiMem | 32.30 | 37.17 | 40.92 |
| A-MEM | 29.67 | 34.01 | 37.87 |
| LSTMem | 11.34 | 11.15 | 11.16 |
| TransMem | 9.60 | 7.04 | 7.04 |
| Doc-to-LoRA | 11.04 | 10.98 | 11.32 |

Todas las condiciones usan las etiquetas del historial final con Mistral como receptor. LSTMem y Doc-to-LoRA utilizan prefijos construidos en el destino; TransMem reconstruye su vista al añadir información. Los demás métodos amplían los estados de prefijo heredados.

</details>

### Acceso entre familias de modelos

El manuscrito sigue la cadena Qwen 4B → Mistral 7B → Phi. Las puntuaciones LoCoMo en los modelos receptores corresponden a la operación indicada para cada método.

| Método | Operación | Mistral 7B | Phi |
|---|---:|---:|---:|
| MIGRO | State Transfer | 45.69 | 49.77 |
| PlugMem | State Transfer | 38.60 | 36.30 |
| Mem0 | State Transfer | 44.06 | 44.72 |
| TiMem | Full-History Rebuilding | 41.22 | 28.03 |
| A-MEM | Full-History Rebuilding | 37.73 | 29.57 |
| LSTMem | Full-History Rebuilding | 11.31 | 16.84 |
| TransMem | Full-History Rebuilding | 7.04 | 18.79 |
| Doc-to-LoRA | Full-History Rebuilding | 11.32 | 14.07 |

[Resumen de migración · CSV](../results/migration_snapshot.csv) · [JSON](../results/migration_snapshot.json) · [Operaciones e interfaces](../docs/migration_protocol.md)

<details>
<summary><strong>Detalles de evaluación e interpretación</strong></summary>

El indicador compuesto LoCoMo se eligió a posteriori. Asigna el mismo peso a las 1,986 preguntas: 1,540 puntuaciones de las categorías 1–4 y 446 de la categoría 5. No es el F1 de tokens oficial de LoCoMo. Las filas de distintos modelos anfitriones no constituyen una clasificación estrictamente controlada de su capacidad.

En la referencia AB de MIGRO, el estado existente contiene un prefijo escrito por Qwen, seguido de una continuación en Phi. Phi utiliza la interfaz identidad seleccionada en el paso 0. Mistral utiliza una interfaz de lectura con 131,072 parámetros, calibrada con 64 ejemplos únicos y públicos de preguntas y respuestas humanas durante 64 actualizaciones del optimizador. Los modelos base y el núcleo compartido de memoria permanecen congelados; las respuestas de LoCoMo no son objetivos de calibración.

PlugMem y mem0 reutilizan sus estados de origen construidos en Qwen para las filas State Transfer. Sus trayectorias de origen difieren de la referencia AB de MIGRO. Para afirmar una pérdida de migración se necesita una referencia reconstruida con el mismo modelo de destino.

LongMemEval-S y MemOps utilizan el protocolo registrado de evaluación auxiliar. BEAM emplea puntuación específica por categoría y evaluación auxiliar cuando hace falta. Son evaluaciones del entorno de investigación, no reproducciones de clasificaciones oficiales. Los métodos de referencia conservan sus rutas de lectura nativas declaradas; el cómputo de inferencia no está igualado.

</details>

### Memoria compacta, crecimiento económico

La calidad y los bytes se miden sobre los mismos estados evaluados. El almacenamiento LoCoMo suma diez estados de conversación; el de BEAM se expresa por historial de un millón de tokens.

| Método | LoCoMo | MB privados / 10 estados | MB en BEAM / historial | 500K–1M bytes/token |
|---|---:|---:|---:|---:|
| MIGRO | 54.13 | 1.115 | 1.857 | 1.809 |
| PlugMem | 41.89 | 92.078 | 82.871 | 79.608 |
| Mem0 | 50.45 | 160.671 | 125.499 | 123.093 |
| TiMem | 47.22 | 10.851 | 6.166 | 6.178 |
| A-MEM | 35.29 | 23.340 | 23.191 | 23.639 |
| LSTMem | 40.64 | 1.871 | 4.371 | 4.370 |
| TransMem | 45.96 | 1.200 | 7.092 | 7.778 |
| Doc-to-LoRA | 14.96 | 283.140 | 56.625 | 28.313 |

MIGRO ocupa una media de **1.857 MB** por historial de un millón de tokens. Al compartir los recursos necesarios entre **1,000 memorias independientes**, la media es de **2.154 MB por memoria**, incluidos los recursos fijos del despliegue. La ventana común para medir el crecimiento es de **500K–1M tokens de origen**.

<details>
<summary><strong>Estimaciones de recursos compartidos · 100–100,000 memorias independientes</strong></summary>

MB medios por memoria de un millón de tokens, incluidos los recursos necesarios compartidos una sola vez:

| Método | 100 memorias | 1,000 | 10,000 | 100,000 |
|---|---:|---:|---:|---:|
| MIGRO | 4.829 | 2.154 | 1.887 | 1.860 |
| PlugMem | 84.237 | 83.008 | 82.885 | 82.873 |
| Mem0 | 127.018 | 125.651 | 125.514 | 125.500 |
| TiMem | 7.532 | 6.303 | 6.180 | 6.167 |
| A-MEM | 24.557 | 23.328 | 23.205 | 23.193 |
| LSTMem | 4.775 | 4.411 | 4.375 | 4.371 |
| TransMem | 24.292 | 8.812 | 7.264 | 7.109 |
| Doc-to-LoRA | 73.465 | 58.309 | 56.794 | 56.642 |

</details>

[Resumen de almacenamiento · CSV](../results/storage_summary.csv) · [JSON](../results/storage_summary.json)

### Mecanismos bajo examen

Los contratos experimentales separan las preguntas sobre el sistema completo de las preguntas sobre cada componente.

|Pregunta de investigación|Comparación controlada|Especificación|
|---|---|---|
|¿Aporta evidencia útil el sistema de memoria?|Memoria completa, BM25, historial reciente y controles sin memoria, con reconstrucción nativa en el destino como referencia.|[Transferencia central](../configs/ablations/core_transfer.json)|
|¿Qué aporta la interfaz del modelo anfitrión?|Composición de fronteras cruzada con calibración congelada.|[Transporte y calibración](../configs/ablations/transport_composition.json)|
|¿Cuál es el equilibrio entre espacio de direcciones y calidad?|Anchor + Residual, solo Anchor y direcciones completas originales.|[Representación de direcciones](../configs/ablations/addresses.json)|
|¿Qué ahorra la codificación persistente?|Selección completa sin pérdida, candidatos literales existentes y un archivo sin comprimir.|[Codificación persistente](../configs/ablations/persistence.json)|

El manuscrito presenta los siguientes controles emparejados. Los valores LoCoMo utilizan la referencia local del mecanismo completo, 54.21, separada del 54.13 de la tabla principal.

| Mecanismo | Mecanismo completo | Control más simple |
|---|---:|---:|
| Selección de evidencia | 54.21 | 50.48 sin asociación al modelo anfitrión; 44.82 Top-K semántico |
| Direcciones semánticas | 54.21 · Anchor + Residual | 51.73 · solo Anchor |
| Composición de fronteras de Mistral | 45.48 | 39.98 sin composición de fronteras |
| Codificación del archivo | 0.870 MB | 0.953 MB codificación sin pérdida existente; 2.381 MB sin comprimir |

Anchor + Residual reduce los bytes de la tabla de direcciones en un **91.1%** respecto a float32, con una diferencia de calidad de **0.46 puntos**. Los controles centrales de transferencia también superan a BM25, las ventanas de historial reciente y la ausencia de memoria; las mejoras informadas de la lectura directa frente a BM25 son de **11.74 puntos en Phi** y **3.93 en Mistral**.
