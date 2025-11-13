# 🤖 Qwen AI Assistant

Aplicación web interactiva construida con Streamlit que utiliza el modelo **Qwen2.5-3B-Instruct** de Hugging Face para crear un asistente conversacional de inteligencia artificial.

https://github.com/user-attachments/assets/3fb3e012-d760-438e-b439-cdd9fad4a55d

## ✨ Características

- 💬 Interfaz intuitiva y moderna para interactuar con el modelo
- ⚙️ Configuración personalizable (temperatura, longitud de respuesta, nucleus sampling)
- 🚀 Optimizado con caché para cargar el modelo una sola vez
- 🎨 Diseño responsive con gradientes y estilos personalizados
- 📱 Compatible con dispositivos Apple Silicon (MPS)

## 🛠️ Requisitos

- Python 3.13+
- PyTorch
- Transformers (Hugging Face)
- Streamlit
- Dispositivo Apple Silicon (M1/M2/M3) o GPU compatible

## 📦 Instalación

1. Clona este repositorio
2. Instala las dependencias:

```bash
pip install -r requirements.txt
```

## 🚀 Uso

Ejecuta la aplicación con:

```bash
streamlit run app/app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## 🎯 Funcionalidades del Panel Lateral

- **Maximum response length**: Controla la longitud máxima del texto generado (50-512 tokens)
- **Temperature**: Ajusta la creatividad de las respuestas (0.1-1.0)
- **Top P**: Controla la diversidad mediante nucleus sampling (0.1-1.0)

## 📝 Ejemplos de Uso

- Preguntas generales sobre cualquier tema
- Generación de código
- Traducción de textos
- Escritura creativa
- Explicaciones de conceptos complejos

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Siéntete libre de abrir issues o pull requests.

## 📄 Licencia

Este proyecto utiliza el modelo Qwen2.5-3B-Instruct de Alibaba Cloud.
