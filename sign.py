import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import time
import flet as ft
from difflib import get_close_matches

class SignLanguageTranslator:
    def __init__(self):
        # Inicializar MediaPipe para detección de manos
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Inicializar motor de voz
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 1.0)
        
        # Diccionario de palabras válidas en español
        self.valid_words = {
            "hola", "adios", "amor", "casa", "agua", "vida", "bien", "malo",
            "si", "no", "dia", "noche", "sol", "luna", "ave", "avion",
            "bus", "auto", "ve", "ven", "vino", "come", "fue",
            "soy", "eres", "es", "somos", "son", "ir", "ver",
            "oir", "decir", "hacer", "dar", "ser", "estar", "tener",
            "poder", "deber", "querer", "saber", "poner", "venir",
            "salir", "ir", "llegar", "pasar", "llevar", "seguir",
            "creer", "hablar", "dejar", "sentir", "quedar", "recibir",
            "vivir", "abrir", "escribir", "leer", "comer", "beber",
            "libro", "mesa", "silla", "puerta", "ventana", "telefono",
            "computadora", "cafe", "leche", "pan", "carne", "fruta",
            "verdura", "arroz", "pescado", "pollo", "huevo", "queso",
            "familia", "padre", "madre", "hijo", "hija", "hermano",
            "hermana", "abuelo", "abuela", "tio", "tia", "primo",
            "amigo", "amiga", "trabajo", "escuela", "universidad",
            "hospital", "tienda", "restaurante", "banco", "parque",
            "calle", "ciudad", "pais", "mundo", "tiempo", "hora",
            "minuto", "segundo", "semana", "mes", "ano", "hoy",
            "ayer", "manana", "ahora", "antes", "despues", "siempre",
            "nunca", "mucho", "poco", "todo", "nada", "algo",
            "grande", "pequeno", "alto", "bajo", "largo", "corto",
            "nuevo", "viejo", "bueno", "feliz", "triste", "enojado"
        }
        
        # Control de voz
        self.last_spoken_time = 0
        self.speech_delay = 2
        self.current_letter = ""
        self.last_letter = ""
        self.word = ""
        self.words_history = []
        
        # Contador de frames para estabilidad
        self.letter_counter = {}
        self.frame_threshold = 10
        
        # UI
        self.ui_letter = None
        self.ui_word = None
        self.ui_valid_word = None
        self.ui_history = None
        
    def calculate_distance(self, point1, point2):
        """Calcula la distancia euclidiana entre dos puntos"""
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def is_finger_extended(self, landmarks, finger_tips, finger_base):
        """Determina si un dedo está extendido"""
        tip = landmarks[finger_tips]
        base = landmarks[finger_base]
        return tip.y < base.y
    
    def recognize_letter(self, hand_landmarks):
        """Reconoce letras del lenguaje de señas basándose en la posición de los dedos"""
        landmarks = hand_landmarks.landmark
        
        # Puntos clave de los dedos
        thumb_tip = 4
        index_tip = 8
        middle_tip = 12
        ring_tip = 16
        pinky_tip = 20
        
        # Bases de los dedos
        thumb_base = 2
        index_base = 6
        middle_base = 10
        ring_base = 14
        pinky_base = 18
        
        # Verificar qué dedos están extendidos
        index_extended = self.is_finger_extended(landmarks, index_tip, index_base)
        middle_extended = self.is_finger_extended(landmarks, middle_tip, middle_base)
        ring_extended = self.is_finger_extended(landmarks, ring_tip, ring_base)
        pinky_extended = self.is_finger_extended(landmarks, pinky_tip, pinky_base)
        
        # Distancias para determinar configuraciones específicas
        thumb_index_dist = self.calculate_distance(landmarks[thumb_tip], landmarks[index_tip])
        thumb_middle_dist = self.calculate_distance(landmarks[thumb_tip], landmarks[middle_tip])
        
        # Reconocimiento de letras básicas
        if not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return "a"
        
        if index_extended and middle_extended and ring_extended and pinky_extended:
            if thumb_index_dist < 0.1:
                return "b"
        
        if not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            if thumb_index_dist > 0.15:
                return "c"
        
        if index_extended and not middle_extended and not ring_extended and not pinky_extended:
            if thumb_middle_dist < 0.1:
                return "d"
        
        if not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            if thumb_index_dist < 0.08:
                return "e"
        
        if not index_extended and middle_extended and ring_extended and pinky_extended:
            if thumb_index_dist < 0.1:
                return "f"
        
        if not index_extended and not middle_extended and not ring_extended and pinky_extended:
            return "i"
        
        if index_extended and not middle_extended and not ring_extended and not pinky_extended:
            if thumb_index_dist > 0.15:
                return "l"
        
        if thumb_index_dist < 0.1 and not middle_extended and not ring_extended:
            return "o"
        
        if index_extended and middle_extended and not ring_extended and not pinky_extended:
            index_middle_dist = self.calculate_distance(landmarks[index_tip], landmarks[middle_tip])
            if index_middle_dist < 0.1:
                return "u"
        
        if index_extended and middle_extended and not ring_extended and not pinky_extended:
            index_middle_dist = self.calculate_distance(landmarks[index_tip], landmarks[middle_tip])
            if index_middle_dist > 0.1:
                return "v"
        
        if index_extended and middle_extended and ring_extended and not pinky_extended:
            return "w"
        
        if not index_extended and not middle_extended and not ring_extended and pinky_extended:
            if landmarks[thumb_tip].y < landmarks[thumb_base].y:
                return "y"
        
        return None
    
    def is_valid_word(self, word):
        """Verifica si una palabra es válida o cercana a una palabra válida"""
        word_lower = word.lower()
        
        # Verificación exacta
        if word_lower in self.valid_words:
            return True, word_lower
        
        # Buscar palabras similares
        matches = get_close_matches(word_lower, self.valid_words, n=1, cutoff=0.8)
        if matches:
            return True, matches[0]
        
        return False, None
    
    def speak(self, text):
        """Pronuncia el texto en un thread separado"""
        def speak_thread():
            self.engine.say(text)
            self.engine.runAndWait()
        
        thread = threading.Thread(target=speak_thread)
        thread.daemon = True
        thread.start()
    
    def update_ui(self):
        """Actualiza la interfaz de usuario"""
        if self.ui_letter:
            self.ui_letter.value = self.current_letter.upper() if self.current_letter else "-"
            self.ui_letter.update()
        
        if self.ui_word:
            self.ui_word.value = self.word.upper() if self.word else "..."
            self.ui_word.update()
        
        # Verificar si la palabra actual es válida
        if self.word and len(self.word) >= 2:
            is_valid, corrected = self.is_valid_word(self.word)
            if is_valid and self.ui_valid_word:
                self.ui_valid_word.value = f"✓ Palabra válida: {corrected.upper()}"
                self.ui_valid_word.color = ft.Colors.GREEN
                self.ui_valid_word.update()
            elif self.ui_valid_word:
                self.ui_valid_word.value = "Formando palabra..."
                self.ui_valid_word.color = ft.Colors.ORANGE
                self.ui_valid_word.update()
        elif self.ui_valid_word:
            self.ui_valid_word.value = ""
            self.ui_valid_word.update()
    
    def process_frame(self, frame):
        """Procesa un frame de video y detecta señas"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        detected_letter = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                
                letter = self.recognize_letter(hand_landmarks)
                if letter:
                    detected_letter = letter
                    self.letter_counter[letter] = self.letter_counter.get(letter, 0) + 1
        
        if detected_letter:
            if self.letter_counter.get(detected_letter, 0) >= self.frame_threshold:
                self.current_letter = detected_letter
                
                current_time = time.time()
                if (self.current_letter != self.last_letter and 
                    current_time - self.last_spoken_time > self.speech_delay):
                    
                    self.word += self.current_letter
                    self.last_letter = self.current_letter
                    self.last_spoken_time = current_time
                    self.letter_counter = {}
                    
                    # Verificar si la palabra es válida
                    if len(self.word) >= 2:
                        is_valid, corrected = self.is_valid_word(self.word)
                        if is_valid:
                            self.speak(corrected)
                            self.words_history.append(corrected)
                            if self.ui_history:
                                self.ui_history.controls.insert(0, 
                                    ft.Text(corrected.upper(), size=16, color=ft.Colors.GREEN)
                                )
                                if len(self.ui_history.controls) > 5:
                                    self.ui_history.controls.pop()
                                self.ui_history.update()
                    
                    self.update_ui()
        
        return frame
    
    def reset_word(self):
        """Reinicia la palabra actual"""
        self.word = ""
        self.current_letter = ""
        self.last_letter = ""
        self.update_ui()

def main(page: ft.Page):
    page.title = "Traductor de Lenguaje de Señas"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 20
    page.window_width = 800
    page.window_height = 600
    
    translator = SignLanguageTranslator()
    cap = None
    running = False
    
    # Elementos de UI
    translator.ui_letter = ft.Text(
        "-",
        size=80,
        weight=ft.FontWeight.BOLD,
        color=ft.Colors.BLUE
    )
    
    translator.ui_word = ft.Text(
        "...",
        size=40,
        weight=ft.FontWeight.BOLD,
        color=ft.Colors.PURPLE
    )
    
    translator.ui_valid_word = ft.Text(
        "",
        size=20,
        italic=True
    )
    
    translator.ui_history = ft.Column(
        controls=[],
        spacing=5
    )
    
    status_text = ft.Text("Presiona INICIAR para comenzar", size=14, color=ft.Colors.GREY)
    
    def start_camera(e):
        nonlocal cap, running
        if not running:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                running = True
                status_text.value = "✓ Cámara activa - Formando señas..."
                status_text.color = ft.Colors.GREEN
                start_btn.disabled = True
                stop_btn.disabled = False
                page.update()
                process_video()
            else:
                status_text.value = "✗ Error al acceder a la cámara"
                status_text.color = ft.Colors.RED
                page.update()
    
    def stop_camera(e):
        nonlocal cap, running
        running = False
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        status_text.value = "Cámara detenida"
        status_text.color = ft.Colors.ORANGE
        start_btn.disabled = False
        stop_btn.disabled = True
        page.update()
    
    def reset_word_ui(e):
        translator.reset_word()
    
    def process_video():
        nonlocal cap, running
        while running:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame = translator.process_frame(frame)
            
            cv2.imshow('Cámara - Lenguaje de Señas', frame)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                stop_camera(None)
                break
    
    # Botones
    start_btn = ft.ElevatedButton(
        "INICIAR CÁMARA",
        on_click=start_camera,
        icon=ft.Icons.VIDEOCAM,
        bgcolor=ft.Colors.GREEN,
        color=ft.Colors.WHITE
    )
    
    stop_btn = ft.ElevatedButton(
        "DETENER",
        on_click=stop_camera,
        icon=ft.Icons.STOP,
        bgcolor=ft.Colors.RED,
        color=ft.Colors.WHITE,
        disabled=True
    )
    
    reset_btn = ft.ElevatedButton(
        "REINICIAR PALABRA",
        on_click=reset_word_ui,
        icon=ft.Icons.REFRESH,
        bgcolor=ft.Colors.ORANGE,
        color=ft.Colors.WHITE
    )
    
    # Layout
    page.add(
        ft.Container(
            content=ft.Column([
                ft.Text(
                    "🤟 Traductor de Lenguaje de Señas",
                    size=30,
                    weight=ft.FontWeight.BOLD,
                    text_align=ft.TextAlign.CENTER
                ),
                ft.Divider(),
                
                ft.Row([
                    ft.Container(
                        content=ft.Column([
                            ft.Text("Letra Actual:", size=16, weight=ft.FontWeight.BOLD),
                            translator.ui_letter,
                        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                        bgcolor=ft.Colors.BLUE_50,
                        padding=20,
                        border_radius=10,
                        expand=True
                    ),
                    ft.Container(
                        content=ft.Column([
                            ft.Text("Palabra:", size=16, weight=ft.FontWeight.BOLD),
                            translator.ui_word,
                        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                        bgcolor=ft.Colors.PURPLE_50,
                        padding=20,
                        border_radius=10,
                        expand=True
                    ),
                ], spacing=10),
                
                ft.Container(
                    content=translator.ui_valid_word,
                    padding=10,
                    alignment=ft.alignment.center
                ),
                
                ft.Row([start_btn, stop_btn, reset_btn], alignment=ft.MainAxisAlignment.CENTER, spacing=10),
                
                status_text,
                
                ft.Divider(),
                
                ft.Text("Palabras Detectadas:", size=18, weight=ft.FontWeight.BOLD),
                ft.Container(
                    content=translator.ui_history,
                    bgcolor=ft.Colors.GREEN_50,
                    padding=15,
                    border_radius=10,
                    height=150
                ),
                
                ft.Text(
                    "Letras disponibles: A, B, C, D, E, F, I, L, O, U, V, W, Y",
                    size=12,
                    color=ft.Colors.GREY,
                    italic=True
                )
            ], spacing=15),
            padding=20
        )
    )

if __name__ == "__main__":
    ft.app(target=main)