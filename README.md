# 🤟 SIGNS OF PEOPLE

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?logo=python&logoColor=white)
![Flet](https://img.shields.io/badge/Flet-Framework-purple.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-ML-orange.svg?logo=google&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-en%20desarrollo-yellow.svg)

**Traductor de Lenguaje de Señas impulsado por IA**

*Rompiendo barreras de comunicación para un mundo más inclusivo* 🌍

[Demo](#) • [Documentación](#) • [Reportar Bug](https://github.com/7749P-AR/Program_analythic_finger/issues)

---

</div>

## 🎯 Sobre el Proyecto

**Signs of People** es un proyecto de impacto social que utiliza inteligencia artificial y visión por computadora para traducir lenguaje de señas en tiempo real. Diseñado como herramienta de inclusión para personas sordomudas y con discapacidades auditivas.

> 💡 Este proyecto nace de la necesidad de crear puentes de comunicación accesibles y tecnológicos para la comunidad sorda.

### ✨ Características Principales

🎥 **Traducción Visual → Texto/Audio**
- Captura lenguaje de señas mediante cámara web
- Conversión instantánea a texto escrito
- Síntesis de voz para output auditivo

✍️ **Traducción Texto/Voz → Señas**
- Input por teclado o micrófono
- Representación visual de señas en pantalla
- Animaciones fluidas y naturales

🚀 **Multiplataforma**
- Modo Desktop (aplicación nativa)
- Modo Web Browser (acceso desde navegador)
- Arquitectura adaptable para futuro mobile

## 🔮 Visión Futura

Con el apoyo de la comunidad, el proyecto expandirá sus capacidades:

- 🆘 **Sistema de emergencias** para personas con discapacidad
- 💼 **Integración laboral** con herramientas empresariales
- 🌐 **Soporte multiidioma** de lenguajes de señas
- 📱 **App móvil nativa** (iOS/Android)
- 🤝 **Modo colaborativo** para videollamadas

## 🛠️ Stack Tecnológico

- **Lenguaje**: Python 3.9+
- **Framework UI**: Flet (Flutter para Python)
- **Computer Vision**: MediaPipe (Google)
- **Machine Learning**: TensorFlow / OpenCV
- **Speech**: pyttsx3 / SpeechRecognition

## 📦 Instalación

### Prerrequisitos

- Python 3.9 o superior
- pip (gestor de paquetes)
- Cámara web funcional
- Git

### Paso 1: Clonar el repositorio

```bash
git clone https://github.com/7749P-AR/Program_analythic_finger.git
cd Program_analythic_finger
```

### Paso 2: Crear entorno virtual

```bash
python -m venv .venv
```

### Paso 3: Activar entorno virtual

**Windows (PowerShell/CMD):**
```bash
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

> ⚠️ **¿Scripts bloqueados en Windows?**
> 
> Abre PowerShell como **Administrador** y ejecuta:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> 
> O para solo la sesión actual:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
> ```

### Paso 4: Instalar dependencias

```bash
pip install -r requirements.txt
```

> 💡 **Tip**: Si VS Code no reconoce el intérprete:
> - Presiona `Ctrl+Shift+P`
> - Busca "Python: Select Interpreter"
> - Selecciona el que termine en `(.venv)`

### Paso 5: ¡Ejecutar!

```bash
python main.py
```

## 🎮 Modos de Ejecución

### Modo Web Browser (Por defecto)

```python
# main.py - línea 385
ft.app(
    target=main,
    view=ft.AppView.WEB_BROWSER,  # Modo web
)
```

### Modo Desktop

Comenta o elimina la línea `view=ft.AppView.WEB_BROWSER,`:

```python
# main.py - línea 385
ft.app(
    target=main,
    # view=ft.AppView.WEB_BROWSER,  # ← Comentado
)
```

## 🗂️ Estructura del Proyecto

```
Program_analythic_finger/
│
├── main.py                 # Punto de entrada principal
├── requirements.txt        # Dependencias del proyecto
├── README.md              # Este archivo
│
├── .venv/                 # Entorno virtual (no subir a Git)
├── assets/                # Recursos multimedia
├── models/                # Modelos ML entrenados
└── utils/                 # Funciones auxiliares
```

## 🤝 Contribuir

¡Las contribuciones son bienvenidas! Este proyecto busca mejorar vidas.

1. Fork el proyecto
2. Crea tu rama (`git checkout -b feature/MejorIncreible`)
3. Commit tus cambios (`git commit -m 'Add: nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/MejorIncreible`)
5. Abre un Pull Request

### Ideas para contribuir

- 🐛 Reportar bugs
- 💡 Sugerir nuevas características
- 📝 Mejorar documentación
- 🌍 Traducir a otros idiomas
- 🎨 Diseñar interfaces más accesibles

## 📄 Licencia

Este proyecto es de **código abierto** y libre uso bajo licencia MIT.

### Reconocimientos de Librerías

- **MediaPipe** © Google LLC - Apache License 2.0
- **Flet** - Apache License 2.0
- Todas las librerías mantienen sus licencias originales

## 👨‍💻 Autor

**Piero Abal Robles**  
*Ingeniero de Sistemas* 🚀

- GitHub: [@7749P-AR](https://github.com/7749P-AR)
- Universidad: UNHEVAL
- Proyecto: Impacto Social + Autoaprendizaje

## 🙏 Agradecimientos

Este proyecto está dedicado a:
- La comunidad sorda y sus defensores
- Personas que trabajan por la inclusión
- Desarrolladores de MediaPipe y Flet
- Comunidad open source de Python

---

<div align="center">

**¿Te gustó el proyecto?** ⭐ Dale una estrella en GitHub

**Construido con ❤️ para un mundo más inclusivo**

[⬆ Volver arriba](#-signs-of-people)

</div>

---

## 🌐 English Version

> *For international contributors and users*

### About

**Signs of People** is an AI-powered sign language translator built with Python, using MediaPipe for real-time hand tracking and gesture recognition.

### Quick Start

```bash
git clone https://github.com/7749P-AR/Program_analythic_finger.git
cd Program_analythic_finger
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
python main.py
```

### Features

- Real-time sign language to text/speech
- Text/speech to sign language animation
- Cross-platform (Desktop & Web)
- Built with accessibility in mind

---

*Last updated: October 2025*
