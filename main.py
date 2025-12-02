import flet as ft
import cv2
import base64
import threading
import time
import mediapipe as mp
import numpy as np
from collections import deque

class SignLanguageRecognizer:
    """Clase para reconocer letras y números del lenguaje de señas"""
    
    def __init__(self):
        self.finger_tips = [4, 8, 12, 16, 20]
        self.finger_pips = [3, 6, 10, 14, 18]
        self.finger_mcps = [2, 5, 9, 13, 17]
        
    def get_finger_states(self, landmarks):
        """Determina qué dedos están extendidos"""
        states = []
        
        # Pulgar (lógica horizontal mejorada)
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        wrist = landmarks[0]
        
        # Determinar orientación de la mano
        hand_is_right = landmarks[17].x < landmarks[5].x
        
        if hand_is_right:
            thumb_extended = thumb_tip.x < thumb_ip.x
        else:
            thumb_extended = thumb_tip.x > thumb_ip.x
            
        states.append(thumb_extended)
        
        # Resto de dedos (lógica vertical mejorada)
        for i, (tip, pip) in enumerate(zip(self.finger_tips[1:], self.finger_pips[1:])):
            extended = landmarks[tip].y < landmarks[pip].y - 0.02
            states.append(extended)
        
        return states
    
    def get_distance(self, p1, p2):
        """Calcula distancia euclidiana entre dos puntos"""
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    def get_angle(self, p1, p2, p3):
        """Calcula el ángulo formado por tres puntos"""
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def recognize_number(self, landmarks):
        """Reconoce números del 0-17"""
        states = self.get_finger_states(landmarks)
        thumb, index, middle, ring, pinky = states
        
        # Calcular distancias importantes
        thumb_index_dist = self.get_distance(landmarks[4], landmarks[8])
        thumb_middle_dist = self.get_distance(landmarks[4], landmarks[12])
        index_middle_dist = self.get_distance(landmarks[8], landmarks[12])
        
        # 0 - Círculo con pulgar e índice (OK sign)
        if not thumb and not index and middle and ring and pinky:
            if thumb_index_dist < 0.06:
                return '0'
        
        # 1 - Solo índice extendido
        if not thumb and index and not middle and not ring and not pinky:
            return '1'
        
        # 2 - Índice y medio extendidos (forma de V)
        if not thumb and index and middle and not ring and not pinky:
            if index_middle_dist > 0.08:  # Dedos separados
                return '2'
        
        # 3 - Pulgar, índice y medio extendidos
        if thumb and index and middle and not ring and not pinky:
            if index_middle_dist > 0.08:
                return '3'
        
        # 4 - Cuatro dedos extendidos (sin pulgar)
        if not thumb and index and middle and ring and pinky:
            return '4'
        
        # 5 - Todos los dedos extendidos
        if thumb and index and middle and ring and pinky:
            return '5'
        
        # 6 - Pulgar y meñique extendidos, resto doblados
        if thumb and not index and not middle and not ring and pinky:
            return '6'
        
        # 7 - Pulgar, índice y medio extendidos, meñique doblado
        if thumb and index and middle and not ring and not pinky:
            return '7'
        
        # 8 - Pulgar, índice, medio y anular extendidos
        if thumb and index and middle and ring and not pinky:
            return '8'
        
        # 9 - Todos extendidos menos el pulgar
        if not thumb and index and middle and ring and pinky:
            fingers_together = (
                self.get_distance(landmarks[8], landmarks[12]) < 0.06 and
                self.get_distance(landmarks[12], landmarks[16]) < 0.06
            )
            if fingers_together:
                return '9'
        
        # 10 - Puño cerrado con pulgar extendido hacia arriba
        if thumb and not index and not middle and not ring and not pinky:
            if landmarks[4].y < landmarks[2].y - 0.05:
                return '10'
        
        # 11 - Índice y pulgar extendidos (forma de pistola)
        if thumb and index and not middle and not ring and not pinky:
            if thumb_index_dist > 0.15:
                return '11'
        
        # 12 - Índice, medio y pulgar formando un 3
        if thumb and index and middle and not ring and not pinky:
            if self.get_distance(landmarks[8], landmarks[12]) < 0.06:
                return '12'
        
        # 13 - Similar a 12 pero dedos más juntos
        if thumb and index and middle and ring and not pinky:
            fingers_together = (
                self.get_distance(landmarks[8], landmarks[12]) < 0.06 and
                self.get_distance(landmarks[12], landmarks[16]) < 0.06
            )
            if fingers_together:
                return '13'
        
        # 14 - Cuatro dedos juntos sin pulgar
        if not thumb and index and middle and ring and pinky:
            fingers_together = (
                self.get_distance(landmarks[8], landmarks[12]) < 0.06 and
                self.get_distance(landmarks[12], landmarks[16]) < 0.06 and
                self.get_distance(landmarks[16], landmarks[20]) < 0.06
            )
            if fingers_together:
                return '14'
        
        # 15 - Todos los dedos extendidos y juntos
        if thumb and index and middle and ring and pinky:
            fingers_together = (
                self.get_distance(landmarks[8], landmarks[12]) < 0.06 and
                self.get_distance(landmarks[12], landmarks[16]) < 0.06 and
                self.get_distance(landmarks[16], landmarks[20]) < 0.06
            )
            if fingers_together:
                return '15'
        
        return None
    
    def recognize_letter(self, landmarks):
        """Reconoce la letra basándose en los landmarks (MEJORADO)"""
        states = self.get_finger_states(landmarks)
        thumb, index, middle, ring, pinky = states
        
        # Calcular distancias importantes
        thumb_index_dist = self.get_distance(landmarks[4], landmarks[8])
        thumb_middle_dist = self.get_distance(landmarks[4], landmarks[12])
        index_middle_dist = self.get_distance(landmarks[8], landmarks[12])
        middle_ring_dist = self.get_distance(landmarks[12], landmarks[16])
        ring_pinky_dist = self.get_distance(landmarks[16], landmarks[20])
        
        # A - Puño cerrado con pulgar al lado
        if not index and not middle and not ring and not pinky and thumb:
            if landmarks[4].y > landmarks[6].y:  # Pulgar al costado
                return 'A'
        
        # B - Mano abierta, dedos juntos, pulgar cruzado (CORREGIDO)
        if not thumb and index and middle and ring and pinky:
            fingers_together = (
                index_middle_dist < 0.05 and
                middle_ring_dist < 0.05 and
                ring_pinky_dist < 0.05
            )
            # Verificar que pulgar está doblado hacia dentro
            thumb_folded = landmarks[4].x > landmarks[5].x - 0.05 and landmarks[4].x < landmarks[17].x + 0.05
            if fingers_together and thumb_folded:
                return 'B'
        
        # C - Mano en forma de C (CORREGIDO)
        if not index and not middle and not ring and not pinky:
            # Verificar forma de C: dedos curvados
            curve_check = (
                landmarks[8].x < landmarks[5].x and
                landmarks[12].x < landmarks[9].x and
                0.1 < thumb_index_dist < 0.25
            )
            if curve_check:
                return 'C'
        
        # D - Índice arriba, resto forma O (MEJOR DETECCIÓN)
        if index and not middle and not ring and not pinky:
            # Verificar que los otros dedos forman un círculo con el pulgar
            circle_formed = thumb_middle_dist < 0.08
            if circle_formed and landmarks[8].y < landmarks[6].y:
                return 'D'
        
        # E - Todos los dedos doblados (CORREGIDO)
        if not thumb and not index and not middle and not ring and not pinky:
            # Verificar que todos están realmente doblados
            all_bent = (
                landmarks[8].y > landmarks[6].y and
                landmarks[12].y > landmarks[10].y and
                landmarks[16].y > landmarks[14].y and
                landmarks[20].y > landmarks[18].y
            )
            if all_bent:
                return 'E'
        
        # F - OK sign (CORREGIDO)
        if middle and ring and pinky:
            # Verificar que pulgar e índice se tocan
            if thumb_index_dist < 0.06:
                return 'F'
        
        # G - Índice y pulgar horizontales apuntando
        if thumb and index and not middle and not ring and not pinky:
            horizontal = abs(landmarks[4].y - landmarks[8].y) < 0.08
            perpendicular = abs(landmarks[4].x - landmarks[8].x) > 0.15
            if horizontal and perpendicular:
                return 'G'
        
        # H - Índice y medio horizontales juntos
        if not thumb and index and middle and not ring and not pinky:
            horizontal = abs(landmarks[8].y - landmarks[12].y) < 0.05
            together = index_middle_dist < 0.08
            pointing_side = abs(landmarks[8].x - landmarks[6].x) > 0.1
            if horizontal and together and pointing_side:
                return 'H'
        
        # I - Meñique arriba, resto cerrado
        if not thumb and not index and not middle and not ring and pinky:
            if landmarks[20].y < landmarks[18].y:
                return 'I'
        
        # J - I con movimiento (detectamos solo la posición base)
        # Similar a I, requeriría detección de movimiento
        
        # K - Índice y medio en V, pulgar en medio (CORREGIDO)
        if thumb and index and middle and not ring and not pinky:
            v_shape = index_middle_dist > 0.1
            # Pulgar debe estar entre índice y medio
            thumb_between = (
                landmarks[4].y > landmarks[8].y and
                landmarks[4].y < landmarks[6].y
            )
            if v_shape and thumb_between:
                return 'K'
        
        # L - L con índice y pulgar
        if thumb and index and not middle and not ring and not pinky:
            # Verificar ángulo de 90 grados
            angle = self.get_angle(landmarks[4], landmarks[2], landmarks[8])
            if 70 < angle < 110:  # Aproximadamente 90 grados
                return 'L'
        
        # M - Tres dedos sobre pulgar (CORREGIDO)
        if not thumb and not index and not middle and not ring:
            # Verificar que el pulgar está debajo de los tres primeros dedos
            thumb_under = (
                landmarks[4].y > landmarks[6].y and
                landmarks[4].x > landmarks[5].x - 0.03 and
                landmarks[4].x < landmarks[9].x + 0.03
            )
            if thumb_under:
                return 'M'
        
        # N - Dos dedos sobre pulgar (CORREGIDO)
        if not thumb and not index and not middle:
            # Verificar que pulgar está debajo de índice y medio
            thumb_under = (
                landmarks[4].y > landmarks[6].y and
                landmarks[4].x > landmarks[5].x - 0.03 and
                landmarks[4].x < landmarks[9].x + 0.03
            )
            # Anular y meñique deben estar extendidos
            if thumb_under and ring and pinky:
                return 'N'
        
        # O - Todos los dedos formando O (CORREGIDO)
        if not index and not middle and not ring and not pinky:
            # Verificar forma circular
            circle = (
                thumb_index_dist < 0.08 and
                thumb_middle_dist < 0.12 and
                landmarks[8].x < landmarks[5].x
            )
            if circle:
                return 'O'
        
        # P - Como K pero apuntando hacia abajo
        if thumb and index and middle and not ring and not pinky:
            pointing_down = landmarks[8].y > landmarks[6].y
            v_shape = index_middle_dist > 0.08
            if pointing_down and v_shape:
                return 'P'
        
        # Q - Similar a G pero apuntando hacia abajo
        if thumb and index and not middle and not ring and not pinky:
            pointing_down = landmarks[8].y > landmarks[0].y
            if pointing_down and thumb_index_dist > 0.1:
                return 'Q'
        
        # R - Índice y medio cruzados
        if not thumb and index and middle and not ring and not pinky:
            # Verificar cruce
            crossed = (
                abs(landmarks[8].x - landmarks[12].x) < 0.04 and
                abs(landmarks[8].y - landmarks[12].y) < 0.04
            )
            if crossed:
                return 'R'
        
        # S - Puño con pulgar sobre dedos
        if not index and not middle and not ring and not pinky:
            thumb_on_top = (
                landmarks[4].y < landmarks[6].y and
                landmarks[4].x > landmarks[5].x and
                landmarks[4].x < landmarks[17].x
            )
            if thumb_on_top:
                return 'S'
        
        # T - Pulgar entre índice y medio
        if not index and not middle and not ring and not pinky:
            thumb_between = (
                landmarks[4].y > landmarks[5].y and
                landmarks[4].y < landmarks[9].y and
                landmarks[4].x > landmarks[6].x - 0.03 and
                landmarks[4].x < landmarks[6].x + 0.03
            )
            if thumb_between:
                return 'T'
        
        # U - Índice y medio juntos arriba
        if not thumb and index and middle and not ring and not pinky:
            together = index_middle_dist < 0.05
            pointing_up = landmarks[8].y < landmarks[6].y
            if together and pointing_up:
                return 'U'
        
        # V - Índice y medio en V separados (CORREGIDO)
        if not thumb and index and middle and not ring and not pinky:
            v_shape = index_middle_dist > 0.08
            pointing_up = (
                landmarks[8].y < landmarks[6].y and
                landmarks[12].y < landmarks[10].y
            )
            if v_shape and pointing_up:
                return 'V'
        
        # W - Tres dedos arriba separados
        if not thumb and index and middle and ring and not pinky:
            separated = (
                index_middle_dist > 0.06 and
                middle_ring_dist > 0.06
            )
            if separated:
                return 'W'
        
        # X - Índice doblado formando gancho
        if not thumb and not middle and not ring and not pinky:
            # Índice doblado en la articulación
            hooked = (
                landmarks[8].y > landmarks[7].y and
                landmarks[8].y < landmarks[6].y
            )
            if hooked:
                return 'X'
        
        # Y - Pulgar y meñique extendidos
        if thumb and not index and not middle and not ring and pinky:
            spread = self.get_distance(landmarks[4], landmarks[20]) > 0.2
            if spread:
                return 'Y'
        
        # Z - Similar a D pero con movimiento en Z (detectamos posición base)
        # Requeriría tracking de movimiento
        
        return None
    
    def recognize(self, landmarks):
        """Reconoce tanto letras como números"""
        # Primero intentar reconocer números
        number = self.recognize_number(landmarks)
        if number:
            return number
        
        # Si no es número, intentar letra
        letter = self.recognize_letter(landmarks)
        return letter


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
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Reconocedor de señas
        self.recognizer = SignLanguageRecognizer()
        
        # Buffer para estabilizar detección
        self.symbol_buffer = deque(maxlen=12)
        self.last_detected_symbol = None
        self.symbol_stable_count = 0
        self.stability_threshold = 7
        
        # Texto acumulado
        self.accumulated_text = ""
        
    def main(self, page: ft.Page):
        self.page = page
        page.title = "Detector de Lenguaje de Señas (Letras y Números)"
        page.window_width = 1150
        page.window_height = 850
        
        # Componentes de la interfaz
        self.image_display = ft.Image(
            src_base64="",
            width=640,
            height=480,
            border_radius=10
        )
        
        self.status_text = ft.Text("Cámara desactivada", size=16, weight=ft.FontWeight.BOLD)
        
        self.detected_symbol = ft.Text(
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
            min_lines=3,
            max_lines=4,
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
        
        self.delete_button = ft.ElevatedButton(
            text="Borrar Último",
            icon=ft.Icons.BACKSPACE,
            on_click=self.delete_last,
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.RED_400,
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
                            "🤟 Detector de Lenguaje de Señas - Letras y Números", 
                            size=26, 
                            weight=ft.FontWeight.BOLD,
                            text_align=ft.TextAlign.CENTER,
                            color=ft.Colors.PURPLE_700
                        ),
                        alignment=ft.alignment.center,
                        margin=ft.Margin(0, 0, 0, 15)
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
                            self.delete_button,
                            ft.Container(width=20),
                            self.status_text,
                        ], alignment=ft.MainAxisAlignment.CENTER),
                        margin=ft.Margin(0, 0, 0, 15)
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
                        
                        # Panel de símbolo detectado
                        ft.Container(
                            content=ft.Column([
                                ft.Text("Detectado:", 
                                       size=18, 
                                       weight=ft.FontWeight.BOLD,
                                       color=ft.Colors.PURPLE_700),
                                ft.Container(
                                    content=self.detected_symbol,
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
                    
                    ft.Container(height=15),
                    
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
                            ft.Text("• Ahora detecta LETRAS (A-Z) y NÚMEROS (0-17)"),
                            ft.Text("• Activa la cámara y muestra tu mano frente a ella"),
                            ft.Text("• Forma las letras o números del alfabeto de señas"),
                            ft.Text("• Mantén la posición estable por 1 segundo para registrarla"),
                            ft.Text("• Usa 'Espacio' entre palabras y 'Borrar Último' para corregir"),
                        ], spacing=5),
                        bgcolor=ft.Colors.PURPLE_50,
                        border_radius=10,
                        padding=15,
                        border=ft.border.all(1, ft.Colors.PURPLE_200)
                    )
                    
                ], alignment=ft.MainAxisAlignment.START, scroll=ft.ScrollMode.AUTO),
                padding=25
            )
        )
    
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
    
    def delete_last(self, e):
        """Borra el último carácter"""
        if self.accumulated_text:
            self.accumulated_text = self.accumulated_text[:-1]
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
        """Procesa el frame para detectar manos y reconocer símbolos"""
        try:
            # Convertir BGR a RGB para MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame = cv2.flip(rgb_frame, 1)
            frame = cv2.flip(frame, 1)
            
            results = self.hands.process(rgb_frame)
            
            detected_symbol = None
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Dibujar landmarks
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )
                    
                    # Reconocer símbolo (letra o número)
                    symbol = self.recognizer.recognize(hand_landmarks.landmark)
                    if symbol:
                        detected_symbol = symbol
                        
                        # Determinar si es letra o número
                        symbol_type = "Número" if symbol.isdigit() else "Letra"
                        
                        # Mostrar en el frame
                        cv2.putText(frame, f"{symbol_type}: {symbol}", (10, 50),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            # Sistema de estabilización mejorado
            if detected_symbol:
                self.symbol_buffer.append(detected_symbol)
                
                if len(self.symbol_buffer) >= self.stability_threshold:
                    most_common = max(set(self.symbol_buffer), key=self.symbol_buffer.count)
                    count = self.symbol_buffer.count(most_common)
                    
                    if count >= self.stability_threshold:
                        if most_common != self.last_detected_symbol:
                            # Nuevo símbolo detectado de forma estable
                            self.accumulated_text += most_common
                            self.accumulated_display.value = self.accumulated_text
                            self.last_detected_symbol = most_common
                            self.symbol_stable_count = 0
                
                self.detected_symbol.value = detected_symbol
            else:
                if len(self.symbol_buffer) > 0:
                    self.symbol_buffer.popleft()
                
                if len(self.symbol_buffer) == 0:
                    self.last_detected_symbol = None
                    self.detected_symbol.value = ""
            
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
                    processed_frame = self.process_frame(frame)
                    base64_image = self.frame_to_base64(processed_frame)
                    if base64_image:
                        self.image_display.src_base64 = base64_image
                    
                    if self.page:
                        self.page.update()
                else:
                    print("No se pudo leer frame de la cámara")
                    break
            except Exception as e:
                print(f"Error en camera_loop: {e}")
                break
            
            time.sleep(0.033)
    
    def toggle_camera(self, e):
        """Activa/desactiva la cámara"""
        if not self.camera_active:
            try:
                self.cap = cv2.VideoCapture(0)
                
                if not self.cap.isOpened():
                    self.show_error("No se pudo acceder a la cámara")
                    return
                
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                self.camera_active = True
                self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
                self.camera_thread.start()
                
                self.toggle_button.text = "Desactivar Cámara"
                self.toggle_button.icon = ft.Icons.VIDEOCAM_OFF
                self.toggle_button.style.bgcolor = ft.Colors.RED
                self.status_text.value = "Cámara activada - Forma letras o números..."
                self.status_text.color = ft.Colors.GREEN
            except Exception as ex:
                self.show_error(f"Error: {str(ex)}")
        else:
            self.camera_active = False
            if self.cap:
                self.cap.release()
                self.cap = None
            
            self.image_display.src_base64 = ""
            self.toggle_button.text = "Activar Cámara"
            self.toggle_button.icon = ft.Icons.VIDEOCAM
            self.toggle_button.style.bgcolor = ft.Colors.GREEN
            self.status_text.value = "Cámara desactivada"
            self.status_text.color = ft.Colors.GREY_600
            self.detected_symbol.value = ""
        
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