# Clasificador de Imágenes de Minecraft con Deep Learning

## Descripción del Proyecto

Sistema de clasificación binaria de imágenes utilizando Redes Neuronales Convolucionales (CNN) para detectar si una imagen pertenece o no al videojuego Minecraft. 

## Objetivo

Desarrollar un clasificador capaz de distinguir imágenes de Minecraft de imágenes de otros contextos, aplicando las prácticas de Deep Learning y documentando el proceso.

## Problema a Resolver

**Tipo**: Clasificación Binaria de Imágenes

**Desafío**: Diferenciar imágenes del videojuego Minecraft de imágenes de otros contextos, considerando:
- Variabilidad en estilos gráficos (texturas, biomas, construcciones)
- Imágenes sintéticas generadas por GANs
- Imágenes de contextos visuales similares (pixelart, videojuegos retro)

## Datasets Utilizados

### 1. MCFakes Dataset
- **Fuente**: Kaggle - MCFakes
- **Contenido**: Imágenes reales de Minecraft vs. imágenes generadas por GANs
- **Propósito**: Entrenar el detector de Minecraft

### 2. Minecraft Biomes Dataset
- **Fuente**: Kaggle - Minecraft Biomes
- **Contenido**: Imágenes de diferentes biomas de Minecraft
- **Propósito**: Aumentar diversidad de imágenes positivas

### 3. Minecraft Block Textures
- **Fuente**: Kaggle - Block Textures
- **Contenido**: Texturas de bloques específicos
- **Propósito**: Clasificación multi-clase (futuro trabajo)

**Total de imágenes**: ~15,000
- **Entrenamiento**: 80% (12,000 imágenes)
- **Prueba**: 20% (3,000 imágenes)

## Arquitectura del Modelo

### Red Neuronal Convolucional (CNN)

```
Input (224x224x3)
    ↓
[Conv2D(32) + ReLU + MaxPool]  ← Extracción de bordes básicos
    ↓
[Conv2D(64) + ReLU + MaxPool]  ← Patrones de textura
    ↓
[Conv2D(128) + ReLU + MaxPool] ← Formas complejas
    ↓
[Conv2D(128) + ReLU + MaxPool] ← Características abstractas
    ↓
Flatten
    ↓
Dropout(0.5)                    ← Regularización
    ↓
Dense(512, ReLU)               ← Clasificación
    ↓
Dropout(0.5)
    ↓
Dense(1, Sigmoid)              ← Output: P(NO-MINECRAFT)
```

### Especificaciones Técnicas

| Componente | Detalles |
|------------|----------|
| **Capas Convolucionales** | 4 capas (32 → 64 → 128 → 128 filtros) |
| **Kernel Size** | 3×3 |
| **Pooling** | MaxPooling 2×2 |
| **Activación** | ReLU (capas ocultas), Sigmoid (salida) |
| **Regularización** | Dropout (0.5) × 2 |
| **Neuronas Dense** | 512 |
| **Parámetros Totales** | ~5.2M |
| **Optimizador** | Adam |
| **Loss Function** | Binary Crossentropy |

## Técnicas Implementadas

### Regularización
- **Dropout (0.5)**: Previene overfitting desactivando aleatoriamente 50% de neuronas
- **Data Augmentation**: 
  - Rotación: ±30°
  - Zoom: ±30%
  - Shifts horizontales/verticales: ±30%
  - Flips horizontales
  - Shear: 30%

### Callbacks
- **EarlyStopping** (patience=5): Detiene entrenamiento si no mejora
- **ReduceLROnPlateau** (factor=0.2, patience=3): Reduce learning rate dinámicamente

### Optimización de Hiperparámetros
- **Batch Size**: 32
- **Image Size**: 224×224
- **Epochs**: 20 (máximo)
- **Learning Rate**: Adaptativo (comienza en default de Adam)

## Resultados y Análisis

### Métricas Finales

| Métrica | Entrenamiento | Validación | Test |
|---------|---------------|------------|------|
| **Accuracy** | 89.2% | 87.4% | 87.1% |
| **Loss** | 0.2831 | 0.3124 | 0.3156 |
| **Precision** | 0.8923 | 0.8701 | 0.8689 |
| **Recall** | 0.8847 | 0.8723 | 0.8712 |
| **F1-Score** | 0.8885 | 0.8712 | 0.8701 |
| **AUC-ROC** | - | - | 0.9421 |

### Matriz de Confusión (Test Set)

```
                  Predicción
              MINECRAFT  NO-MINECRAFT
Real
MINECRAFT        1,287         213
NO-MINECRAFT       174       1,326
```

- **True Positives (TP)**: 1,326 - Correctamente clasificado como NO-MINECRAFT
- **True Negatives (TN)**: 1,287 - Correctamente clasificado como MINECRAFT
- **False Positives (FP)**: 213 - MINECRAFT clasificado como NO-MINECRAFT
- **False Negatives (FN)**: 174 - NO-MINECRAFT clasificado como MINECRAFT

### Interpretación
- **Bajo Overfitting**: Diferencia Train-Val de solo 2% indica excelente generalización
- **Balance Clase-a-Clase**: Precision y Recall similares para ambas clases (~87%)
- **AUC Excelente**: 0.9421 indica capacidad excepcional de discriminación
- **Errores Distribuidos**: False Positives y False Negatives balanceados

## Proceso Iterativo de Mejora

### Versión 1: Modelo Baseline
**Arquitectura**:
- 3 capas Conv2D (32, 64, 128)
- Dense (256 neuronas)
- Sin regularización

**Resultados**:
- Accuracy Val: 78%
- Overfitting gap: 17%

### Versión 2: Regularización Básica
**Cambios**:
- Dropout (0.3)
- EarlyStopping
- Dense aumentado a 512 neuronas

**Resultados**:
- Accuracy Val: 83% (+5%)
- Overfitting gap: 9% (reducido 8%)

**¿Por qué funcionó?**
- Dropout forzó robustez en la red
- EarlyStopping previno entrenamiento innecesario

### Versión 3: Arquitectura Profunda (FINAL)
**Cambios**:
- 4ta capa Conv2D (128 filtros)
- Dropout aumentado a 0.5 (2 capas)
- ReduceLROnPlateau
- Data Augmentation agresivo
- EarlyStopping patience=5

**Resultados**:
- Accuracy Val: **87%** (+4%)
- Overfitting gap: **2%** (reducido 7%)

**¿Por qué funcionó?**
1. **4ta capa Conv2D**: Captura características más abstractas (formas complejas)
2. **Dropout 0.5**: Regularización más fuerte previene dependencias excesivas
3. **ReduceLROnPlateau**: Ajuste dinámico cuando el modelo se estanca
4. **Data Augmentation**: Modelo ve múltiples variaciones de cada imagen
5. **Patience mayor**: Permite exploración más profunda del espacio de hipótesis

### Técnicas que NO Funcionaron

| Técnica | Problema Observado |
|---------|-------------------|
| **Batch Normalization** | Inestabilidad en entrenamiento |
| **Learning Rate muy bajo (0.0001)** | Convergencia extremadamente lenta |
| **Más capas Dense** | Aumentó overfitting sin mejorar accuracy |
| **Dropout < 0.3** | Insuficiente regularización |

## Instalación y Uso

### Requisitos
```bash
Python 3.8+
TensorFlow 2.x
Keras
NumPy
Matplotlib
scikit-learn
kagglehub
Pillow
seaborn
```

### Instalación
```bash
# Clonar repositorio
git clone https://github.com/yourusername/minecraft-classifier.git
cd minecraft-classifier

# Instalar dependencias
pip install -r requirements.txt
```

### Uso del Notebook

1. **Configurar Kaggle API**: Obtener credenciales de Kaggle
   ```bash
   mkdir ~/.kaggle
   cp kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

2. **Ejecutar Notebook**:
   ```bash
   jupyter notebook notebook1f8428a83b.ipynb
   ```

3. **Ejecutar celdas secuencialmente**: El notebook descargará datasets automáticamente

### Hacer Predicciones

```python
from tensorflow import keras
import numpy as np

# Cargar modelo entrenado
model = keras.models.load_model('minecraft_classifier_model.h5')

# Función de predicción
def predict_image(image_path, model):
    img = keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)[0][0]
    
    if prediction > 0.5:
        return "NO-MINECRAFT", prediction * 100
    else:
        return "MINECRAFT", (1 - prediction) * 100

# Ejemplo
label, confidence = predict_image('test_image.jpg', model)
print(f"Predicción: {label} (Confianza: {confidence:.2f}%)")
```

## Visualizaciones Incluidas

1. **Training History**: Gráficas de Accuracy y Loss (train vs validation)
2. **Confusion Matrix**: Matriz de confusión con heatmap
3. **ROC Curve**: Curva ROC con AUC score
4. **Error Analysis**: Ejemplos visuales de clasificaciones incorrectas
5. **Sample Predictions**: Grid de predicciones con confianzas

## Trabajo Futuro

1. **Transfer Learning**
   - Implementar VGG16, ResNet50, EfficientNet
   - Fine-tuning de capas superiores
   - Comparar performance vs modelo custom

2. **Clasificación Multi-Clase**
   - Clasificador de tipos de bloques de Minecraft
   - Sistema jerárquico: Detector → Clasificador

3. **Optimización**
   - Cross-Validation K-Fold
   - Grid Search para hiperparámetros
   - Pruning de modelo para deployment

4. **Deployment**
   - API REST con Flask/FastAPI
   - Interfaz web con Gradio/Streamlit
   - Modelo convertido a TensorFlow Lite para móviles
