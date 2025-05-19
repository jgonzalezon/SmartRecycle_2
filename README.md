# SmartRecycle Mobile Application

## Descripción

SmartRecycle es una aplicación móvil diseñada para optimizar el proceso de reciclaje mediante el uso de inteligencia artificial y técnicas de aprendizaje continuo directamente en el dispositivo. Desarrollada en Kotlin e integrada con TensorFlow Lite, permite realizar inferencias y entrenamiento incremental (continual learning) sin depender de servidores externos.

### Características principales

* **Inferencia en tiempo real**: Clasifica imágenes de residuos capturadas por la cámara y muestra la probabilidad de cada categoría.
* **Entrenamiento on-device**: Permite al usuario seleccionar imágenes y refinar las etiquetas para reentrenar el modelo en el edge.
* **Estrategia de replay (archivo NPZ)**: Guarda ejemplos anteriores en un buffer NPZ para evitar el ‘catastrophic forgetting’ al incorporar datos nuevos.
* **Guardado y restauración de pesos**: Almacena checkpoints locales tras el entrenamiento y los carga para mantener el estado aprendido.
* **Restablecimiento a estado de fábrica**: Opción para eliminar checkpoints y etiquetas personalizadas, recuperando la configuración inicial.

## Instalación

1. Clona el repositorio:
2. Abre el proyecto en Android Studio.
3. Asegúrate de tener configurado el SDK de Android (mínimo API 21).
4. Compila y ejecuta en un dispositivo o emulador con cámara.

## Uso

1. **Cámara**: Pulsa el botón de cámara para acceder al preview. Toca la pantalla para capturar una foto, realizar inferencia y guardar la imagen en la galería.
2. **Entrenamiento**: Desde el menú de entrenamiento, selecciona varias imágenes. Revisa o corrige las etiquetas propuestas y pulsa “Entrenar”. Una barra de progreso indica el avance.
3. **Guardado de pesos**: Tras entrenar, usa la opción “Guardar checkpoint” en el menú. Esto crea un archivo de checkpoint en almacenamiento interno.
4. **Restauración**: Selecciona “Restaurar checkpoint” para cargar los últimos pesos guardados.
5. **Restablecer fábrica**: Elige “Restablecer fábrica” para eliminar checkpoints y etiquetas personalizadas.

## Estructura del repositorio

```
├── app/                         # Código fuente de la aplicación Android
│   ├── src/main/java/...        # Paquetes Kotlin
│   ├── src/main/res/...         # Layouts, drawables y strings
│   └── assets/                  # Modelos TFLite y buffers NPZ y labels
├── models/                      # Scripts y notebooks de entrenamiento
└── README.md                    # Este archivo
```


## Licencia

Este proyecto está bajo la licencia MIT. Consulta `LICENSE` para más detalles.
