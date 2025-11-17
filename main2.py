import flet as ft
import cv2
import base64
import threading
import time
import mediapipe as mp
import numpy as np
from collections import deque
import pyttsx3

class SignLanguageRecognizer:
    """Clase para reconocer letras del lenguaje de señas"""
    
    def __init__(self):
        self.finger_tips = [4, 8, 12, 16, 20]
        self.finger_pips = [3, 6, 10, 14, 18]
        self.finger_mcps = [2, 5, 9, 13, 17]
        
    def get_finger_states(self, landmarks):
        """Determina qué dedos están extendidos"""
        states = []
        
        # Pulgar (lógica horizontal)
        thumb_extended = landmarks[4].x < landmarks[3].x if landmarks[9].x < landmarks[0].x else landmarks[4].x > landmarks[3].x
        states.append(thumb_extended)
        
        # Resto de dedos (lógica vertical)
        for tip, pip in zip(self.finger_tips[1:], self.finger_pips[1:]):
            extended = landmarks[tip].y < landmarks[pip].y
            states.append(extended)
        
        return states
    
    def get_finger_angles(self, landmarks):
        """Calcula ángulos relativos de los dedos"""
        angles = []
        for i in range(5):
            tip = self.finger_tips[i]
            mcp = self.finger_mcps[i]
            angle = np.arctan2(
                landmarks[tip].y - landmarks[mcp].y,
                landmarks[tip].x - landmarks[mcp].x
            )
            angles.append(angle)
        return angles
    
    def recognize_letter(self, landmarks):
        """Reconoce la letra basándose en los landmarks"""
        states = self.get_finger_states(landmarks)
        angles = self.get_finger_angles(landmarks)
        
        # [Pulgar, Índice, Medio, Anular, Meñique]
        thumb, index, middle, ring, pinky = states
        
        # Calcular distancias para gestos específicos
        thumb_index_dist = np.sqrt(
            (landmarks[4].x - landmarks[8].x)**2 + 
            (landmarks[4].y - landmarks[8].y)**2
        )
        
        thumb_middle_dist = np.sqrt(
            (landmarks[4].x - landmarks[12].x)**2 + 
            (landmarks[4].y - landmarks[12].y)**2
        )
        
        # A - Puño cerrado con pulgar al lado
        if not index and not middle and not ring and not pinky and thumb:
            return 'A'
        
        # B - Mano abierta, dedos juntos, pulgar cruzado
        if not thumb and index and middle and ring and pinky:
            # Verificar que los dedos estén relativamente juntos
            fingers_together = (
                abs(landmarks[8].x - landmarks[12].x) < 0.05 and
                abs(landmarks[12].x - landmarks[16].x) < 0.05 and
                abs(landmarks[16].x - landmarks[20].x) < 0.05
            )
            if fingers_together:
                return 'B'
        
        # C - Mano en forma de C
        if thumb and not index and not middle and not ring and not pinky:
            if 0.1 < thumb_index_dist < 0.3:
                return 'C'
        
        # D - Índice arriba, resto forma O
        if not thumb and index and not middle and not ring and not pinky:
            if thumb_middle_dist < 0.1:
                return 'D'
        
        # E - Todos los dedos doblados, pulgar sobre ellos
        if not thumb and not index and not middle and not ring and not pinky:
            if landmarks[4].y < landmarks[8].y:
                return 'E'
        
        # F - OK sign - índice y pulgar tocándose, resto arriba
        if thumb and not index and middle and ring and pinky:
            if thumb_index_dist < 0.08:
                return 'F'
        
        # G - Índice y pulgar horizontales apuntando
        if thumb and index and not middle and not ring and not pinky:
            horizontal = abs(landmarks[4].y - landmarks[8].y) < 0.1
            if horizontal:
                return 'G'
        
        # H - Índice y medio horizontales
        if not thumb and index and middle and not ring and not pinky:
            horizontal = abs(landmarks[8].y - landmarks[12].y) < 0.08
            fingers_together = abs(landmarks[8].x - landmarks[12].x) < 0.15
            if horizontal and fingers_together:
                return 'H'
        
        # I - Meñique arriba, resto cerrado
        if not thumb and not index and not middle and not ring and pinky:
            return 'I'
        
        # K - Índice y medio en V, pulgar en medio
        if thumb and index and middle and not ring and not pinky:
            v_shape = abs(landmarks[8].x - landmarks[12].x) > 0.1
            if v_shape and landmarks[4].y < landmarks[6].y:
                return 'K'
        
        # L - L con índice y pulgar
        if thumb and index and not middle and not ring and not pinky:
            perpendicular = abs(landmarks[4].x - landmarks[8].x) > 0.15
            if perpendicular:
                return 'L'
        
        # M - Tres dedos sobre pulgar
        if not thumb and not index and not middle and not ring and pinky:
            if landmarks[4].x > landmarks[6].x and landmarks[4].x < landmarks[10].x:
                return 'M'
        
        # N - Dos dedos sobre pulgar
        if not thumb and not index and not middle and ring and pinky:
            if landmarks[4].x > landmarks[6].x and landmarks[4].x < landmarks[10].x:
                return 'N'
        
        # O - Todos los dedos formando O
        if thumb and not index and not middle and not ring and not pinky:
            circle = thumb_index_dist < 0.1 and thumb_middle_dist < 0.15
            if circle:
                return 'O'
        
        # P - Como K pero apuntando hacia abajo
        if thumb and index and middle and not ring and not pinky:
            pointing_down = landmarks[8].y > landmarks[6].y
            if pointing_down:
                return 'P'
        
        # R - Índice y medio cruzados
        if not thumb and index and middle and not ring and not pinky:
            crossed = landmarks[8].x > landmarks[12].x if landmarks[8].y < landmarks[12].y else landmarks[8].x < landmarks[12].x
            if crossed:
                return 'R'
        
        # S - Puño con pulgar sobre dedos
        if thumb and not index and not middle and not ring and not pinky:
            if landmarks[4].y < landmarks[6].y and landmarks[4].x > landmarks[5].x and landmarks[4].x < landmarks[17].x:
                return 'S'
        
        # T - Pulgar entre índice y medio
        if thumb and not index and not middle and not ring and not pinky:
            if landmarks[4].y > landmarks[5].y and landmarks[4].y < landmarks[9].y:
                return 'T'
        
        # U - Índice y medio juntos arriba
        if not thumb and index and middle and not ring and not pinky:
            together = abs(landmarks[8].x - landmarks[12].x) < 0.05
            if together:
                return 'U'
        
        # V - Índice y medio en V separados
        if not thumb and index and middle and not ring and not pinky:
            v_shape = abs(landmarks[8].x - landmarks[12].x) > 0.08
            if v_shape:
                return 'V'
        
        # W - Tres dedos arriba separados
        if not thumb and index and middle and ring and not pinky:
            separated = (abs(landmarks[8].x - landmarks[12].x) > 0.05 and 
                        abs(landmarks[12].x - landmarks[16].x) > 0.05)
            if separated:
                return 'W'
        
        # Y - Pulgar y meñique extendidos
        if thumb and not index and not middle and not ring and pinky:
            return 'Y'
        
        # Mano abierta (todos extendidos)
        if thumb and index and middle and ring and pinky:
            fingers_spread = (abs(landmarks[8].x - landmarks[12].x) > 0.08 and
                            abs(landmarks[12].x - landmarks[16].x) > 0.08)
            if fingers_spread:
                return '5'  # o podría ser otra letra dependiendo del contexto
        
        return None


class HandDetectionApp:
    def __init__(self):
        self.cap = None
        self.camera_active = False
        self.camera_thread = None
        self.page = None
        
        # Configurar MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Reconocedor de señas
        self.recognizer = SignLanguageRecognizer()
        
        # Buffer para estabilizar detección
        self.letter_buffer = deque(maxlen=10)
        self.last_detected_letter = None
        self.letter_stable_count = 0
        self.stability_threshold = 5
        
        # Texto acumulado
        self.accumulated_text = ""
        
        # Motor de texto a voz - se inicializará bajo demanda
        self.tts_available = True
        try:
            # Probar si pyttsx3 está disponible
            test_engine = pyttsx3.init()
            test_engine.stop()
            del test_engine
        except Exception as e:
            print(f"TTS no disponible: {e}")
            self.tts_available = False
        
    def main(self, page: ft.Page):
        self.page = page
        page.title = "Detector de Lenguaje de Señas"
        page.window_width = 1100
        page.window_height = 850
        
        # Componentes de la interfaz
        self.image_display = ft.Image(
            src_base64="",
            width=640,
            height=480,
            border_radius=10
        )
        
        self.status_text = ft.Text("Cámara desactivada", size=16, weight=ft.FontWeight.BOLD)
        
        self.detected_letter = ft.Text(
            "",
            size=80,
            weight=ft.FontWeight.BOLD,
            color=ft.Colors.PURPLE_700,
            text_align=ft.TextAlign.CENTER
        )
        
        self.accumulated_display = ft.TextField(
            value="",
            text_size=24,
            text_align=ft.TextAlign.LEFT,
            multiline=True,
            min_lines=2,
            max_lines=3,
            read_only=True,
            border_color=ft.Colors.PURPLE_300,
            bgcolor=ft.Colors.WHITE
        )
        
        self.toggle_button = ft.ElevatedButton(
            text="Activar Cámara",
            icon=ft.Icons.VIDEOCAM,
            on_click=self.toggle_camera,
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.GREEN,
                shape=ft.RoundedRectangleBorder(radius=10)
            )
        )
        
        self.clear_button = ft.ElevatedButton(
            text="Limpiar Texto",
            icon=ft.Icons.CLEAR,
            on_click=self.clear_text,
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.ORANGE,
                shape=ft.RoundedRectangleBorder(radius=10)
            )
        )
        
        self.add_space_button = ft.ElevatedButton(
            text="Espacio",
            icon=ft.Icons.SPACE_BAR,
            on_click=self.add_space,
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.BLUE,
                shape=ft.RoundedRectangleBorder(radius=10)
            )
        )
        
        # Nuevo botón de texto a voz
        self.speak_button = ft.ElevatedButton(
            text="Leer en Voz Alta",
            icon=ft.Icons.VOLUME_UP,
            on_click=self.speak_text,
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.TEAL,
                shape=ft.RoundedRectangleBorder(radius=10)
            )
        )
        
        # Configurar eventos de ventana
        page.window_prevent_close = True
        page.on_window_event = self.on_window_event
        
        # Layout principal
        page.add(
            ft.Container(
                content=ft.Column([
                    # Título
                    ft.Container(
                        content=ft.Text(
                            "🤟 Detector de Lenguaje de Señas", 
                            size=28, 
                            weight=ft.FontWeight.BOLD,
                            text_align=ft.TextAlign.CENTER,
                            color=ft.Colors.PURPLE_700
                        ),
                        alignment=ft.alignment.center,
                        margin=ft.Margin(0, 0, 0, 20)
                    ),
                    
                    # Controles
                    ft.Container(
                        content=ft.Row([
                            self.toggle_button,
                            ft.Container(width=10),
                            self.clear_button,
                            ft.Container(width=10),
                            self.add_space_button,
                            ft.Container(width=10),
                            self.speak_button,
                            ft.Container(width=20),
                            self.status_text,
                        ], alignment=ft.MainAxisAlignment.CENTER),
                        margin=ft.Margin(0, 0, 0, 20)
                    ),
                    
                    # Contenido principal
                    ft.Row([
                        # Video
                        ft.Container(
                            content=self.image_display,
                            border=ft.border.all(3, ft.Colors.PURPLE_200),
                            border_radius=15,
                            padding=10,
                            bgcolor=ft.Colors.BLACK12
                        ),
                        
                        ft.Container(width=20),
                        
                        # Panel de letra detectada
                        ft.Container(
                            content=ft.Column([
                                ft.Text("Letra Detectada:", 
                                       size=18, 
                                       weight=ft.FontWeight.BOLD,
                                       color=ft.Colors.PURPLE_700),
                                ft.Container(
                                    content=self.detected_letter,
                                    alignment=ft.alignment.center,
                                    height=120,
                                    width=200,
                                    bgcolor=ft.Colors.PURPLE_50,
                                    border_radius=10,
                                    border=ft.border.all(2, ft.Colors.PURPLE_300)
                                ),
                            ], alignment=ft.MainAxisAlignment.CENTER),
                            width=250,
                            bgcolor=ft.Colors.PURPLE_50,
                            border_radius=15,
                            padding=20,
                            border=ft.border.all(2, ft.Colors.PURPLE_200)
                        )
                    ], alignment=ft.MainAxisAlignment.CENTER),
                    
                    ft.Container(height=20),
                    
                    # Área de texto acumulado
                    ft.Container(
                        content=ft.Column([
                            ft.Text("Texto Formado:", 
                                   size=18, 
                                   weight=ft.FontWeight.BOLD,
                                   color=ft.Colors.PURPLE_700),
                            self.accumulated_display,
                        ], spacing=10),
                        bgcolor=ft.Colors.PURPLE_50,
                        border_radius=15,
                        padding=20,
                        border=ft.border.all(2, ft.Colors.PURPLE_200)
                    ),
                    
                    ft.Container(height=10),
                    
                    # Instrucciones
                    ft.Container(
                        content=ft.Column([
                            ft.Text("📋 Instrucciones:", 
                                   size=16, 
                                   weight=ft.FontWeight.BOLD,
                                   color=ft.Colors.PURPLE_700),
                            ft.Text("• Activa la cámara y coloca tu mano frente a ella"),
                            ft.Text("• Forma las letras del alfabeto de señas"),
                            ft.Text("• Mantén la posición estable por 1 segundo para que se registre"),
                            ft.Text("• Las letras detectadas aparecerán automáticamente en el texto"),
                            ft.Text("• Usa el botón 'Espacio' para agregar espacios entre palabras"),
                            ft.Text("• Presiona 'Leer en Voz Alta' para escuchar el texto formado"),
                        ], spacing=5),
                        bgcolor=ft.Colors.PURPLE_50,
                        border_radius=10,
                        padding=15,
                        border=ft.border.all(1, ft.Colors.PURPLE_200)
                    )
                    
                ], alignment=ft.MainAxisAlignment.START, scroll=ft.ScrollMode.AUTO),
                padding=30
            )
        )
    
    def speak_text(self, e):
        """Lee en voz alta el texto acumulado"""
        if not self.tts_available:
            self.show_error("Motor de voz no disponible")
            return
        
        text = self.accumulated_text.strip()
        
        if not text:
            self.status_text.value = "No hay texto para leer"
            self.status_text.color = ft.Colors.ORANGE
            if self.page:
                self.page.update()
            return
        
        # Ejecutar en un hilo separado para no bloquear la UI
        def speak_thread():
            try:
                self.status_text.value = "🔊 Leyendo texto..."
                self.status_text.color = ft.Colors.TEAL
                if self.page:
                    self.page.update()
                
                # Crear una nueva instancia del motor para cada lectura
                tts_engine = pyttsx3.init()
                tts_engine.setProperty('rate', 150)
                tts_engine.setProperty('volume', 0.9)
                
                tts_engine.say(text)
                tts_engine.runAndWait()
                
                # Limpiar el motor después de usarlo
                tts_engine.stop()
                del tts_engine
                
                if self.camera_active:
                    self.status_text.value = "Cámara activada - Forma las letras..."
                    self.status_text.color = ft.Colors.GREEN
                else:
                    self.status_text.value = "Cámara desactivada"
                    self.status_text.color = ft.Colors.GREY_600
                
                if self.page:
                    self.page.update()
                    
            except Exception as ex:
                print(f"Error al leer texto: {ex}")
                self.status_text.value = "Error al leer el texto"
                self.status_text.color = ft.Colors.RED
                if self.page:
                    self.page.update()
        
        threading.Thread(target=speak_thread, daemon=True).start()
    
    def clear_text(self, e):
        """Limpia el texto acumulado"""
        self.accumulated_text = ""
        self.accumulated_display.value = ""
        if self.page:
            self.page.update()
    
    def add_space(self, e):
        """Agrega un espacio al texto"""
        self.accumulated_text += " "
        self.accumulated_display.value = self.accumulated_text
        if self.page:
            self.page.update()
    
    def frame_to_base64(self, frame):
        """Convierte frame a base64 para Flet"""
        try:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            jpg_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
            return jpg_base64
        except Exception as e:
            print(f"Error convirtiendo frame a base64: {e}")
            return ""
    
    def process_frame(self, frame):
        """Procesa el frame para detectar manos y reconocer letras"""
        try:
            # Convertir BGR a RGB para MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame = cv2.flip(rgb_frame, 1)  # Efecto espejo
            frame = cv2.flip(frame, 1)
            
            results = self.hands.process(rgb_frame)
            
            detected_letter = None
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Dibujar landmarks
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )
                    
                    # Reconocer letra
                    letter = self.recognizer.recognize_letter(hand_landmarks.landmark)
                    if letter:
                        detected_letter = letter
                        
                        # Mostrar letra en el frame
                        cv2.putText(frame, f"Letra: {letter}", (10, 50),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            # Sistema de estabilización
            if detected_letter:
                self.letter_buffer.append(detected_letter)
                
                # Contar ocurrencias de la letra más común
                if len(self.letter_buffer) >= self.stability_threshold:
                    most_common = max(set(self.letter_buffer), key=self.letter_buffer.count)
                    count = self.letter_buffer.count(most_common)
                    
                    if count >= self.stability_threshold:
                        if most_common != self.last_detected_letter:
                            # Nueva letra detectada de forma estable
                            self.accumulated_text += most_common
                            self.accumulated_display.value = self.accumulated_text
                            self.last_detected_letter = most_common
                            self.letter_stable_count = 0
                
                # Actualizar display de letra actual
                self.detected_letter.value = detected_letter
            else:
                # Solo limpiar la última letra detectada cuando no hay mano
                if len(self.letter_buffer) > 0:
                    self.letter_buffer.popleft()
                
                if len(self.letter_buffer) == 0:
                    self.last_detected_letter = None
                    self.detected_letter.value = ""
            
            return frame
            
        except Exception as e:
            print(f"Error en process_frame: {e}")
            return frame
    
    def camera_loop(self):
        """Bucle principal de la cámara"""
        while self.camera_active and self.cap:
            try:
                ret, frame = self.cap.read()
                if ret:
                    # Procesar frame
                    processed_frame = self.process_frame(frame)
                    
                    # Convertir a base64
                    base64_image = self.frame_to_base64(processed_frame)
                    if base64_image:
                        self.image_display.src_base64 = base64_image
                    
                    # Actualizar interfaz
                    if self.page:
                        self.page.update()
                        
                else:
                    print("No se pudo leer frame de la cámara")
                    break
                    
            except Exception as e:
                print(f"Error en camera_loop: {e}")
                break
                
            time.sleep(0.033)  # ~30 FPS
    
    def toggle_camera(self, e):
        """Activa/desactiva la cámara"""
        if not self.camera_active:
            try:
                self.cap = cv2.VideoCapture(0)
                
                if not self.cap.isOpened():
                    self.show_error("No se pudo acceder a la cámara")
                    return
                
                # Configurar cámara
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                self.camera_active = True
                
                # Iniciar hilo de cámara
                self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
                self.camera_thread.start()
                
                # Actualizar interfaz
                self.toggle_button.text = "Desactivar Cámara"
                self.toggle_button.icon = ft.Icons.VIDEOCAM_OFF
                self.toggle_button.style.bgcolor = ft.Colors.RED
                self.status_text.value = "Cámara activada - Forma las letras..."
                self.status_text.color = ft.Colors.GREEN
                
            except Exception as ex:
                self.show_error(f"Error: {str(ex)}")
                
        else:
            # Desactivar cámara
            self.camera_active = False
            
            if self.cap:
                self.cap.release()
                self.cap = None
            
            # Limpiar interfaz
            self.image_display.src_base64 = ""
            self.toggle_button.text = "Activar Cámara"
            self.toggle_button.icon = ft.Icons.VIDEOCAM
            self.toggle_button.style.bgcolor = ft.Colors.GREEN
            self.status_text.value = "Cámara desactivada"
            self.status_text.color = ft.Colors.GREY_600
            self.detected_letter.value = ""
        
        if self.page:
            self.page.update()
    
    def show_error(self, message):
        """Muestra mensaje de error"""
        self.status_text.value = message
        self.status_text.color = ft.Colors.RED
        if self.page:
            self.page.update()
    
    def on_window_event(self, e):
        """Maneja eventos de ventana"""
        if e.data == "close":
            self.camera_active = False
            if self.cap:
                self.cap.release()


def main(page: ft.Page):
    app = HandDetectionApp()
    app.main(page)

if __name__ == "__main__":
    ft.app(target=main)