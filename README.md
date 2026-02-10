# Explorando Capas Convolucionales a Través de Datos y Experimentos

## Descripción del problema

Este proyecto estudia cómo las capas convolucionales afectan el rendimiento de una red neuronal en tareas de clasificación de imágenes. En lugar de tratar las redes neuronales como cajas negras, el objetivo es entender cómo las decisiones arquitectónicas (tamaño de kernel, profundidad, stride, padding, pooling) introducen un sesgo inductivo que aprovecha la estructura espacial de las imágenes.

El enfoque es experimental: primero se construye un modelo base sin convoluciones (solo capas densas), y después se diseña una CNN desde cero, justificando cada decisión de diseño. Finalmente se realiza un experimento controlado donde se varía una sola variable (profundidad de la red) para observar su efecto en el rendimiento.

## Descripción del dataset

Se utiliza CIFAR-10, un dataset clásico en visión por computadora que contiene 60,000 imágenes a color de 32×32 píxeles distribuidas en 10 clases: avión, automóvil, pájaro, gato, ciervo, perro, rana, caballo, barco y camión.

El dataset está perfectamente balanceado (5,000 imágenes de entrenamiento y 1,000 de prueba por clase). Cada imagen tiene 3 canales de color (RGB) y los valores de píxel van de 0 a 255, que se normalizan al rango [0, 1] antes de entrenar. Las etiquetas se codifican en formato one-hot para ser compatibles con la función de pérdida categorical crossentropy.

CIFAR-10 se eligió porque tiene suficiente complejidad para que una red totalmente conectada tenga dificultades (fondos variados, ángulos distintos, iluminaciones diferentes), pero cabe en memoria sin problemas y permite iterar rápido.

## Arquitecturas

### Modelo base (sin convoluciones)

Este modelo aplana la imagen en un vector de 3,072 valores y lo pasa por capas densas. Sirve como punto de referencia.

```
Entrada: 32 × 32 × 3
    │
    ▼
Flatten ─── 3,072 valores
    │
    ▼
Dense(512, ReLU)
    │
    ▼
Dense(256, ReLU)
    │
    ▼
Dense(10, Softmax) ─── 10 probabilidades
```

Parámetros totales: ~1.7M

### CNN (3 bloques convolucionales)

La CNN aprovecha la estructura espacial de la imagen. Cada bloque extrae características a un nivel de abstracción distinto.

```
Entrada: 32 × 32 × 3
    │
    ▼
┌─────────────────────────────────────────┐
│  Bloque 1: características de bajo nivel │
│  Conv2D(32, 3×3) → BN → ReLU            │
│  Conv2D(32, 3×3) → BN → ReLU            │
│  MaxPool(2×2) → Dropout(0.25)            │
│  Salida: 16 × 16 × 32                   │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Bloque 2: texturas y formas simples     │
│  Conv2D(64, 3×3) → BN → ReLU            │
│  Conv2D(64, 3×3) → BN → ReLU            │
│  MaxPool(2×2) → Dropout(0.25)            │
│  Salida: 8 × 8 × 64                     │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Bloque 3: partes de objetos             │
│  Conv2D(128, 3×3) → BN → ReLU           │
│  Conv2D(128, 3×3) → BN → ReLU           │
│  MaxPool(2×2) → Dropout(0.25)            │
│  Salida: 4 × 4 × 128                    │
└─────────────────────────────────────────┘
    │
    ▼
Flatten → Dense(256, ReLU) → Dropout(0.5) → Dense(10, Softmax)
```

Decisiones de diseño: kernels de 3×3 (eficientes y con campo receptivo suficiente), padding same (preserva dimensiones), patrón piramidal de filtros (32 → 64 → 128), Batch Normalization para estabilizar el entrenamiento, y Dropout para regularización.

## Resultados experimentales

### Comparación: modelo base vs CNN

| Modelo | Precisión (test) | Parámetros |
|--------|-----------------|------------|
| Base (solo Dense) | ~50-55% | ~1.7M |
| CNN (3 bloques) | ~75-80% | ~0.9M |

La CNN supera al modelo base con una mejora de aproximadamente 20-25 puntos porcentuales, a pesar de tener menos parámetros.

### Experimento controlado: efecto de la profundidad

Se entrenaron tres variantes de CNN cambiando únicamente el número de bloques convolucionales (1, 2 y 3), manteniendo todo lo demás constante.

| Configuración | Precisión (test) | Parámetros |
|---------------|-----------------|------------|
| CNN 1 bloque | ~65-70% | menor |
| CNN 2 bloques | ~72-76% | medio |
| CNN 3 bloques | ~75-80% | mayor |

Incluso con un solo bloque convolucional, la CNN ya supera al modelo base. Agregar más bloques mejora la precisión, pero con rendimientos decrecientes: el salto de 1 a 2 bloques es más grande que el de 2 a 3. Esto muestra un trade-off claro entre complejidad y rendimiento.

## Interpretación

La CNN supera al modelo base porque su arquitectura incorpora tres suposiciones (sesgo inductivo) que se alinean con la naturaleza de las imágenes.

La primera es localidad: los patrones relevantes en una imagen (bordes, texturas) aparecen en regiones pequeñas y contiguas. Una capa convolucional con kernel de 3×3 mira exactamente eso — ventanas locales — en lugar de procesar toda la imagen de golpe como hace una capa densa.

La segunda es equivarianza a la traslación: un gato es un gato sin importar dónde esté en la imagen. Al aplicar el mismo filtro en todas las posiciones, la red aprende a detectar características independientemente de su ubicación.

La tercera es compartición de pesos: el mismo kernel se reutiliza en toda la imagen. Esto reduce drásticamente los parámetros (de ~1.7M a ~0.9M en nuestro caso) y obliga al modelo a aprender patrones generalizables, lo cual reduce el sobreajuste.

Cuando los datos no tienen estructura espacial local (datos tabulares, grafos, secuencias con dependencias muy largas), estos sesgos dejan de ser útiles y pueden incluso perjudicar el rendimiento. En esos casos conviene usar otras arquitecturas como Transformers, GNNs o modelos de ensemble.

## Estructura del repositorio

```
├── Taller_Redes_neuronales.ipynb          # Notebook principal (CIFAR-10 + CNN)
└── README.md                              # Este archivo
```

## Cómo ejecutar

Se necesita Python 3.11+ con TensorFlow, NumPy, Matplotlib y scikit-learn. Para instalar las dependencias:

```
pip install tensorflow numpy matplotlib scikit-learn
```

Luego abrir `Taller_Redes_neuronales.ipynb` en Jupyter o VS Code y ejecutar las celdas en orden.