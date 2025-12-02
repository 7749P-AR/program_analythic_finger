"""
Traductor Profesional de Lenguaje de Señas
Versión mejorada con reconocimiento preciso, modo entrenamiento, historial y más
"""

import flet as ft
import cv2
import base64
import threading
import time
import mediapipe as mp
import numpy as np
from collections import deque
from datetime import datetime
import json
import os
import pyttsx3


class SignLanguageRecognizer:
    """Reconocedor mejorado de lenguaje de señas con calibración y confianza"""
    
    def __init__(self):
        self.finger_tips = [4, 8, 12, 16, 20]
        self.finger_pips = [3, 6, 10, 14, 18]
        self.finger_mcps = [2, 5, 9, 13, 17]
        
        # Umbrales calibrables
        self.thresholds = {
            'finger_together': 0.06,
            'finger_separated': 0.10,
            'thumb_touch': 0.10,
            'horizontal': 0.12,
            'vertical': 0.06,
        }
        
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
    
    def calculate_confidence(self, landmarks, detected_letter, finger_states):
        """Calcula la confianza de la detección (0-100%)"""
        confidence = 100.0
        
        # Factores que reducen confianza:
        # 1. Mano muy cerca o muy lejos
        hand_size = np.sqrt(
            (landmarks[0].x - landmarks[9].x)**2 + 
            (landmarks[0].y - landmarks[9].y)**2
        )
        if hand_size < 0.15 or hand_size > 0.35:
            confidence -= 20
        
        # 2. Mano muy inclinada
        wrist_middle = abs(landmarks[0].y - landmarks[9].y)
        if wrist_middle > 0.3:
            confidence -= 15
        
        # 3. Dedos temblorosos (se puede agregar tracking de estabilidad)
        
        return max(0, min(100, confidence))
    
    def recognize_letter(self, landmarks):
        """Reconoce la letra basándose en los landmarks - VERSIÓN CORREGIDA"""
        states = self.get_finger_states(landmarks)
        
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
        
        thumb_ring_dist = np.sqrt(
            (landmarks[4].x - landmarks[16].x)**2 + 
            (landmarks[4].y - landmarks[16].y)**2
        )
        
        index_middle_dist = np.sqrt(
            (landmarks[8].x - landmarks[12].x)**2 + 
            (landmarks[8].y - landmarks[12].y)**2
        )
        
        # ORDEN IMPORTANTE: Evaluar casos más específicos primero
        
        # B - Mano abierta, dedos juntos, pulgar cruzado
        if not thumb and index and middle and ring and pinky:
            fingers_together = (
                abs(landmarks[8].x - landmarks[12].x) < self.thresholds['finger_together'] and
                abs(landmarks[12].x - landmarks[16].x) < self.thresholds['finger_together'] and
                abs(landmarks[16].x - landmarks[20].x) < self.thresholds['finger_together']
            )
            if fingers_together:
                return 'B', self.calculate_confidence(landmarks, 'B', states)
        
        # F - OK sign - índice y pulgar tocándose, resto arriba
        if thumb and not index and middle and ring and pinky:
            if thumb_index_dist < self.thresholds['thumb_touch']:
                return 'F', self.calculate_confidence(landmarks, 'F', states)
        
        # W - Tres dedos arriba separados
        if not thumb and index and middle and ring and not pinky:
            separated = (abs(landmarks[8].x - landmarks[12].x) > self.thresholds['finger_together'] and 
                        abs(landmarks[12].x - landmarks[16].x) > self.thresholds['finger_together'])
            similar_height = (abs(landmarks[8].y - landmarks[12].y) < 0.08 and
                            abs(landmarks[12].y - landmarks[16].y) < 0.08)
            if separated and similar_height:
                return 'W', self.calculate_confidence(landmarks, 'W', states)
        
        # K - Índice y medio en V, pulgar entre ellos
        if thumb and index and middle and not ring and not pinky:
            v_shape = abs(landmarks[8].x - landmarks[12].x) > 0.12
            thumb_between = landmarks[4].y < landmarks[6].y and landmarks[4].y < landmarks[10].y
            not_pointing_down = not (landmarks[8].y > landmarks[6].y)
            if v_shape and thumb_between and not_pointing_down:
                return 'K', self.calculate_confidence(landmarks, 'K', states)
        
        # P - Como K pero apuntando hacia abajo
        if thumb and index and middle and not ring and not pinky:
            pointing_down = landmarks[8].y > landmarks[6].y and landmarks[12].y > landmarks[10].y
            v_shape = abs(landmarks[8].x - landmarks[12].x) > 0.08
            if pointing_down and v_shape:
                return 'P', self.calculate_confidence(landmarks, 'P', states)
        
        # R - Índice y medio cruzados
        if not thumb and index and middle and not ring and not pinky:
            crossed = (landmarks[8].x > landmarks[12].x if landmarks[8].y < landmarks[12].y 
                      else landmarks[8].x < landmarks[12].x)
            close = index_middle_dist < 0.08
            if crossed and close:
                return 'R', self.calculate_confidence(landmarks, 'R', states)
        
        # U - Índice y medio juntos arriba (verticales)
        if not thumb and index and middle and not ring and not pinky:
            together = abs(landmarks[8].x - landmarks[12].x) < self.thresholds['finger_together']
            vertical = abs(landmarks[8].y - landmarks[12].y) < self.thresholds['vertical']
            if together and vertical:
                return 'U', self.calculate_confidence(landmarks, 'U', states)
        
        # V - Índice y medio en V separados
        if not thumb and index and middle and not ring and not pinky:
            v_shape = abs(landmarks[8].x - landmarks[12].x) > self.thresholds['finger_separated']
            similar_height = abs(landmarks[8].y - landmarks[12].y) < 0.08
            if v_shape and similar_height:
                return 'V', self.calculate_confidence(landmarks, 'V', states)
        
        # H - Índice y medio horizontales juntos
        if not thumb and index and middle and not ring and not pinky:
            horizontal = abs(landmarks[8].y - landmarks[12].y) < self.thresholds['horizontal']
            fingers_together = abs(landmarks[8].x - landmarks[12].x) < 0.18
            if horizontal and fingers_together:
                return 'H', self.calculate_confidence(landmarks, 'H', states)
        
        # G - Índice y pulgar horizontales apuntando
        if thumb and index and not middle and not ring and not pinky:
            horizontal = abs(landmarks[4].y - landmarks[8].y) < self.thresholds['horizontal']
            pointing = landmarks[8].x > landmarks[5].x or landmarks[8].x < landmarks[5].x
            if horizontal and pointing:
                return 'G', self.calculate_confidence(landmarks, 'G', states)
        
        # L - L con índice y pulgar perpendiculares
        if thumb and index and not middle and not ring and not pinky:
            perpendicular = abs(landmarks[4].x - landmarks[8].x) > 0.18
            index_up = landmarks[8].y < landmarks[4].y
            if perpendicular and index_up:
                return 'L', self.calculate_confidence(landmarks, 'L', states)
        
        # D - Índice arriba, resto forma O con pulgar
        if not thumb and index and not middle and not ring and not pinky:
            if thumb_middle_dist < 0.12 and thumb_ring_dist < 0.12:
                return 'D', self.calculate_confidence(landmarks, 'D', states)
        
        # Y - Pulgar y meñique extendidos (shaka)
        if thumb and not index and not middle and not ring and pinky:
            thumb_pinky_dist = np.sqrt(
                (landmarks[4].x - landmarks[20].x)**2 + 
                (landmarks[4].y - landmarks[20].y)**2
            )
            if thumb_pinky_dist > 0.20:
                return 'Y', self.calculate_confidence(landmarks, 'Y', states)
        
        # I - Meñique arriba, resto cerrado
        if not thumb and not index and not middle and not ring and pinky:
            if landmarks[20].y < landmarks[18].y - 0.05:
                return 'I', self.calculate_confidence(landmarks, 'I', states)
        
        # A - Puño cerrado con pulgar al lado
        if not index and not middle and not ring and not pinky and thumb:
            if landmarks[4].y > landmarks[2].y:
                return 'A', self.calculate_confidence(landmarks, 'A', states)
        
        # Casos con todos los dedos cerrados: C, E
        if not index and not middle and not ring and not pinky:
            # E - Pulgar sobre dedos cerrados (puño completo)
            if not thumb and landmarks[4].y < landmarks[8].y and thumb_index_dist < 0.15:
                return 'E', self.calculate_confidence(landmarks, 'E', states)
            
            # C - Mano en forma de C (dedos curvados, pulgar separado)
            if not thumb and 0.15 < thumb_index_dist < 0.35:
                if landmarks[4].y > landmarks[8].y:
                    return 'C', self.calculate_confidence(landmarks, 'C', states)
        
        # Casos con pulgar extendido y resto cerrado: O, S, T
        if thumb and not index and not middle and not ring and not pinky:
            # S - Puño con pulgar sobre dedos
            thumb_on_top = (landmarks[4].y < landmarks[6].y and 
                          landmarks[4].x > landmarks[5].x and 
                          landmarks[4].x < landmarks[17].x)
            if thumb_on_top and landmarks[4].y < landmarks[2].y:
                return 'S', self.calculate_confidence(landmarks, 'S', states)
            
            # T - Pulgar sobresaliendo entre índice y medio
            thumb_between = (landmarks[4].y > landmarks[5].y and 
                           landmarks[4].y < landmarks[9].y and
                           landmarks[4].x > landmarks[6].x - 0.05 and
                           landmarks[4].x < landmarks[10].x + 0.05)
            if thumb_between:
                return 'T', self.calculate_confidence(landmarks, 'T', states)
            
            # O - Todos los dedos formando círculo
            circle = (thumb_index_dist < 0.12 and 
                     thumb_middle_dist < 0.15 and
                     thumb_ring_dist < 0.18)
            not_s = landmarks[4].y > landmarks[6].y - 0.05
            if circle and not_s:
                return 'O', self.calculate_confidence(landmarks, 'O', states)
        
        # M - Tres dedos doblados sobre pulgar
        if not thumb and not index and not middle and not ring and pinky:
            thumb_covered = (landmarks[4].x > landmarks[6].x - 0.05 and 
                           landmarks[4].x < landmarks[14].x + 0.05 and
                           landmarks[4].y > landmarks[5].y)
            if thumb_covered:
                return 'M', self.calculate_confidence(landmarks, 'M', states)
        
        # N - Dos dedos doblados sobre pulgar
        if not thumb and not index and not middle and ring and pinky:
            thumb_covered = (landmarks[4].x > landmarks[6].x - 0.05 and 
                           landmarks[4].x < landmarks[10].x + 0.05 and
                           landmarks[4].y > landmarks[5].y)
            if thumb_covered and landmarks[16].y < landmarks[14].y:
                return 'N', self.calculate_confidence(landmarks, 'N', states)
        
        # Mano abierta (todos extendidos)
        if thumb and index and middle and ring and pinky:
            fingers_spread = (abs(landmarks[8].x - landmarks[12].x) > 0.08 and
                            abs(landmarks[12].x - landmarks[16].x) > 0.08 and
                            abs(landmarks[16].x - landmarks[20].x) > 0.08)
            if fingers_spread:
                return '5', self.calculate_confidence(landmarks, '5', states)
        
        return None, 0


class TranslationEngine:
    """Motor de traducción con auto-completado y corrección"""
    
    def __init__(self):
        self.common_words = {
            'HOLA': ['H', 'O', 'L', 'A'],
            'ADIOS': ['A', 'D', 'I', 'O', 'S'],
            'GRACIAS': ['G', 'R', 'A', 'C', 'I', 'A', 'S'],
            'POR FAVOR': ['P', 'O', 'R', ' ', 'F', 'A', 'V', 'O', 'R'],
            'SI': ['S', 'I'],
            'NO': ['N', 'O'],
            'AYUDA': ['A', 'Y', 'U', 'D', 'A'],
            'BIEN': ['B', 'I', 'E', 'N'],
            'MAL': ['M', 'A', 'L'],
        }
        self.current_buffer = []
        
    def suggest_words(self, current_text):
        """Sugiere palabras basadas en el texto actual"""
        suggestions = []
        words = current_text.split()
        if words:
            last_word = words[-1].upper()
            for word in self.common_words.keys():
                if word.startswith(last_word) and word != last_word:
                    suggestions.append(word)
        return suggestions[:3]  # Top 3 sugerencias
    
    def auto_complete(self, current_text, selected_word):
        """Auto-completa con la palabra seleccionada"""
        words = current_text.split()
        if words:
            words[-1] = selected_word
            return ' '.join(words)
        return selected_word


class HistoryManager:
    """Gestor de historial y exportación"""
    
    def __init__(self):
        self.history_dir = "sign_language_history"
        if not os.path.exists(self.history_dir):
            os.makedirs(self.history_dir)
        self.current_session = {
            'start_time': datetime.now().isoformat(),
            'letters_detected': 0,
            'text': '',
            'accuracy': 0.0
        }
    
    def save_session(self, text, letters_count, avg_confidence):
        """Guarda la sesión actual"""
        self.current_session['end_time'] = datetime.now().isoformat()
        self.current_session['text'] = text
        self.current_session['letters_detected'] = letters_count
        self.current_session['accuracy'] = avg_confidence
        
        filename = f"{self.history_dir}/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.current_session, f, indent=2, ensure_ascii=False)
        
        return filename
    
    def export_text(self, text, format='txt'):
        """Exporta el texto traducido"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'txt':
            filename = f"{self.history_dir}/translation_{timestamp}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"=== Traducción de Lenguaje de Señas ===\n")
                f.write(f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n")
                f.write(f"Texto:\n{text}\n")
        
        return filename
    
    def load_history(self):
        """Carga el historial de sesiones"""
        sessions = []
        if os.path.exists(self.history_dir):
            for filename in os.listdir(self.history_dir):
                if filename.endswith('.json'):
                    with open(os.path.join(self.history_dir, filename), 'r', encoding='utf-8') as f:
                        sessions.append(json.load(f))
        return sorted(sessions, key=lambda x: x.get('start_time', ''), reverse=True)


class SignLanguageApp:
    """Aplicación principal con interfaz Flet"""
    
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
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Componentes
        self.recognizer = SignLanguageRecognizer()
        self.translator = TranslationEngine()
        self.history_manager = HistoryManager()
        
        # Buffer para estabilizar detección
        self.letter_buffer = deque(maxlen=10)
        self.confidence_buffer = deque(maxlen=10)
        self.last_detected_letter = None
        self.stability_threshold = 5
        
        # Estadísticas
        self.accumulated_text = ""
        self.letters_count = 0
        self.total_confidence = 0.0
        
        # TTS
        self.tts_available = True
        try:
            test_engine = pyttsx3.init()
            test_engine.stop()
            del test_engine
        except:
            self.tts_available = False
    
    def main(self, page: ft.Page):
        self.page = page
        page.title = "🤟 Traductor Profesional de Lenguaje de Señas"
        page.window_width = 1200
        page.window_height = 900
        page.theme_mode = ft.ThemeMode.LIGHT
        
        # Componentes UI
        self.image_display = ft.Image(
            src_base64="",
            width=640,
            height=480,
            border_radius=10,
            fit=ft.ImageFit.CONTAIN
        )
        
        self.status_text = ft.Text(
            "Cámara desactivada", 
            size=14, 
            weight=ft.FontWeight.BOLD,
            color=ft.Colors.GREY_700
        )
        
        self.detected_letter = ft.Text(
            "",
            size=100,
            weight=ft.FontWeight.BOLD,
            color=ft.Colors.PURPLE_700,
            text_align=ft.TextAlign.CENTER
        )
        
        self.confidence_text = ft.Text(
            "Confianza: --",
            size=16,
            color=ft.Colors.BLUE_700
        )
        
        self.hand_type_text = ft.Text(
            "Mano: --",
            size=14,
            color=ft.Colors.GREEN_700
        )
        
        self.suggestions_text = ft.Text(
            "",
            size=12,
            color=ft.Colors.ORANGE_700
        )
        
        self.accumulated_display = ft.TextField(
            value="",
            text_size=20,
            text_align=ft.TextAlign.LEFT,
            multiline=True,
            min_lines=3,
            max_lines=4,
            read_only=False,
            border_color=ft.Colors.PURPLE_300,
            bgcolor=ft.Colors.WHITE
        )
        
        self.stats_text = ft.Text(
            "📊 Letras: 0 | Precisión: 0%",
            size=12,
            color=ft.Colors.GREY_600
        )
        
        # Botones
        self.toggle_button = ft.ElevatedButton(
            text="Iniciar Cámara",
            icon=ft.Icons.VIDEOCAM,
            on_click=self.toggle_camera,
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.GREEN,
            )
        )
        
        self.clear_button = ft.ElevatedButton(
            text="Limpiar",
            icon=ft.Icons.CLEAR,
            on_click=self.clear_text,
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.ORANGE,
            )
        )
        
        self.space_button = ft.ElevatedButton(
            text="Espacio",
            icon=ft.Icons.SPACE_BAR,
            on_click=self.add_space,
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.BLUE,
            )
        )
        
        self.speak_button = ft.ElevatedButton(
            text="Leer",
            icon=ft.Icons.VOLUME_UP,
            on_click=self.speak_text,
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.TEAL,
            )
        )
        
        self.export_button = ft.ElevatedButton(
            text="Exportar",
            icon=ft.Icons.SAVE,
            on_click=self.export_text,
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.INDIGO,
            )
        )
        
        self.history_button = ft.ElevatedButton(
            text="Historial",
            icon=ft.Icons.HISTORY,
            on_click=self.show_history,
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.PURPLE,
            )
        )
        
        # Layout
        page.add(
            ft.Container(
                content=ft.Column([
                    # Título
                    ft.Container(
                        content=ft.Text(
                            "🤟 Traductor Profesional de Lenguaje de Señas",
                            size=24,
                            weight=ft.FontWeight.BOLD,
                            color=ft.Colors.PURPLE_700
                        ),
                        alignment=ft.alignment.center,
                        margin=ft.Margin(0, 0, 0, 10)
                    ),
                    
                    # Controles principales
                    ft.Container(
                        content=ft.Row([
                            self.toggle_button,
                            self.space_button,
                            self.clear_button,
                            self.speak_button,
                            self.export_button,
                            self.history_button,
                            ft.Container(width=10),
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
                        
                        ft.Container(width=15),
                        
                        # Panel lateral
                        ft.Container(
                            content=ft.Column([
                                ft.Text("Letra Detectada:", 
                                       size=16, 
                                       weight=ft.FontWeight.BOLD,
                                       color=ft.Colors.PURPLE_700),
                                ft.Container(
                                    content=self.detected_letter,
                                    alignment=ft.alignment.center,
                                    height=140,
                                    width=220,
                                    bgcolor=ft.Colors.PURPLE_50,
                                    border_radius=10,
                                    border=ft.border.all(2, ft.Colors.PURPLE_300)
                                ),
                                ft.Container(height=10),
                                self.confidence_text,
                                self.hand_type_text,
                                ft.Divider(height=20, color=ft.Colors.PURPLE_200),
                                ft.Text("💡 Sugerencias:", 
                                       size=14, 
                                       weight=ft.FontWeight.BOLD,
                                       color=ft.Colors.ORANGE_700),
                                self.suggestions_text,
                            ], alignment=ft.MainAxisAlignment.START),
                            width=270,
                            bgcolor=ft.Colors.PURPLE_50,
                            border_radius=15,
                            padding=15,
                            border=ft.border.all(2, ft.Colors.PURPLE_200)
                        )
                    ], alignment=ft.MainAxisAlignment.CENTER),
                    
                    ft.Container(height=15),
                    
                    # Área de texto
                    ft.Container(
                        content=ft.Column([
                            ft.Text("📝 Texto Traducido:", 
                                   size=16, 
                                   weight=ft.FontWeight.BOLD,
                                   color=ft.Colors.PURPLE_700),
                            self.accumulated_display,
                        ], spacing=8),
                        bgcolor=ft.Colors.PURPLE_50,
                        border_radius=15,
                        padding=15,
                        border=ft.border.all(2, ft.Colors.PURPLE_200)
                    ),
                    
                    ft.Container(height=10),
                    
                    # Estadísticas
                    ft.Container(
                        content=self.stats_text,
                        alignment=ft.alignment.center,
                        padding=10,
                        bgcolor=ft.Colors.GREY_100,
                        border_radius=10
                    ),
                    
                ], alignment=ft.MainAxisAlignment.START, scroll=ft.ScrollMode.AUTO),
                padding=20
            )
        )
        
        page.window_prevent_close = True
        page.on_window_event = self.on_window_event
    
    def process_frame(self, frame):
        """Procesa el frame para detectar manos y reconocer letras"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame = cv2.flip(rgb_frame, 1)
            frame = cv2.flip(frame, 1)
            
            results = self.hands.process(rgb_frame)
            
            detected_letter = None
            confidence = 0
            
            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Dibujar landmarks
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )
                    
                    # Reconocer letra
                    result = self.recognizer.recognize_letter(hand_landmarks.landmark)
                    if result and result[0]:
                        detected_letter, confidence = result
                        
                        # Determinar tipo de mano
                        hand_type = "Derecha" if results.multi_handedness[idx].classification[0].label == "Right" else "Izquierda"
                        self.hand_type_text.value = f"Mano: {hand_type}"
                        
                        # Mostrar en frame
                        cv2.putText(frame, f"{detected_letter} ({confidence:.0f}%)", (10, 50),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            # Sistema de estabilización
            if detected_letter:
                self.letter_buffer.append(detected_letter)
                self.confidence_buffer.append(confidence)
                
                if len(self.letter_buffer) >= self.stability_threshold:
                    most_common = max(set(self.letter_buffer), key=self.letter_buffer.count)
                    count = self.letter_buffer.count(most_common)
                    
                    if count >= self.stability_threshold:
                        if most_common != self.last_detected_letter:
                            # Nueva letra detectada
                            self.accumulated_text += most_common
                            self.accumulated_display.value = self.accumulated_text
                            self.last_detected_letter = most_common
                            self.letters_count += 1
                            
                            # Actualizar estadísticas
                            avg_conf = sum(self.confidence_buffer) / len(self.confidence_buffer)
                            self.total_confidence = (self.total_confidence * (self.letters_count - 1) + avg_conf) / self.letters_count
                            self.update_stats()
                            
                            # Sugerencias
                            suggestions = self.translator.suggest_words(self.accumulated_text)
                            if suggestions:
                                self.suggestions_text.value = "\n".join([f"• {s}" for s in suggestions])
                            else:
                                self.suggestions_text.value = "Sin sugerencias"
                
                # Actualizar display
                self.detected_letter.value = detected_letter
                self.confidence_text.value = f"Confianza: {confidence:.0f}%"
                
                # Color según confianza
                if confidence >= 80:
                    self.confidence_text.color = ft.Colors.GREEN_700
                elif confidence >= 60:
                    self.confidence_text.color = ft.Colors.ORANGE_700
                else:
                    self.confidence_text.color = ft.Colors.RED_700
            else:
                if len(self.letter_buffer) > 0:
                    self.letter_buffer.popleft()
                
                if len(self.letter_buffer) == 0:
                    self.last_detected_letter = None
                    self.detected_letter.value = ""
                    self.confidence_text.value = "Confianza: --"
                    self.hand_type_text.value = "Mano: --"
            
            return frame
            
        except Exception as e:
            print(f"Error en process_frame: {e}")
            return frame
    
    def update_stats(self):
        """Actualiza las estadísticas"""
        avg_accuracy = self.total_confidence if self.letters_count > 0 else 0
        self.stats_text.value = f"📊 Letras: {self.letters_count} | Precisión: {avg_accuracy:.1f}%"
    
    def frame_to_base64(self, frame):
        """Convierte frame a base64"""
        try:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return base64.b64encode(buffer.tobytes()).decode('utf-8')
        except Exception as e:
            print(f"Error convirtiendo frame: {e}")
            return ""
    
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
                
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                self.camera_active = True
                self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
                self.camera_thread.start()
                
                self.toggle_button.text = "Detener Cámara"
                self.toggle_button.icon = ft.Icons.VIDEOCAM_OFF
                self.toggle_button.style.bgcolor = ft.Colors.RED
                self.status_text.value = "✅ Cámara activa - Forma las letras..."
                self.status_text.color = ft.Colors.GREEN
                
            except Exception as ex:
                self.show_error(f"Error: {str(ex)}")
        else:
            self.camera_active = False
            
            if self.cap:
                self.cap.release()
                self.cap = None
            
            self.image_display.src_base64 = ""
            self.toggle_button.text = "Iniciar Cámara"
            self.toggle_button.icon = ft.Icons.VIDEOCAM
            self.toggle_button.style.bgcolor = ft.Colors.GREEN
            self.status_text.value = "Cámara desactivada"
            self.status_text.color = ft.Colors.GREY_700
            self.detected_letter.value = ""
        
        if self.page:
            self.page.update()
    
    def clear_text(self, e):
        """Limpia el texto"""
        self.accumulated_text = ""
        self.accumulated_display.value = ""
        self.letters_count = 0
        self.total_confidence = 0.0
        self.update_stats()
        if self.page:
            self.page.update()
    
    def add_space(self, e):
        """Agrega un espacio"""
        self.accumulated_text += " "
        self.accumulated_display.value = self.accumulated_text
        if self.page:
            self.page.update()
    
    def speak_text(self, e):
        """Lee el texto en voz alta"""
        if not self.tts_available:
            self.show_error("Motor de voz no disponible")
            return
        
        text = self.accumulated_text.strip()
        if not text:
            return
        
        def speak_thread():
            try:
                tts_engine = pyttsx3.init()
                tts_engine.setProperty('rate', 150)
                tts_engine.setProperty('volume', 0.9)
                tts_engine.say(text)
                tts_engine.runAndWait()
                tts_engine.stop()
                del tts_engine
            except Exception as ex:
                print(f"Error TTS: {ex}")
        
        threading.Thread(target=speak_thread, daemon=True).start()
    
    def export_text(self, e):
        """Exporta el texto"""
        if not self.accumulated_text:
            return
        
        try:
            filename = self.history_manager.export_text(self.accumulated_text)
            self.show_success(f"Exportado: {filename}")
        except Exception as ex:
            self.show_error(f"Error al exportar: {ex}")
    
    def show_history(self, e):
        """Muestra el historial"""
        # TODO: Implementar vista de historial
        self.show_success("Historial - Función en desarrollo")
    
    def show_error(self, message):
        """Muestra mensaje de error"""
        self.status_text.value = f"❌ {message}"
        self.status_text.color = ft.Colors.RED
        if self.page:
            self.page.update()
    
    def show_success(self, message):
        """Muestra mensaje de éxito"""
        self.status_text.value = f"✅ {message}"
        self.status_text.color = ft.Colors.GREEN
        if self.page:
            self.page.update()
    
    def on_window_event(self, e):
        """Maneja eventos de ventana"""
        if e.data == "close":
            self.camera_active = False
            if self.cap:
                self.cap.release()
            
            # Guardar sesión
            if self.accumulated_text:
                try:
                    avg_conf = self.total_confidence if self.letters_count > 0 else 0
                    self.history_manager.save_session(
                        self.accumulated_text,
                        self.letters_count,
                        avg_conf
                    )
                except:
                    pass


def main(page: ft.Page):
    app = SignLanguageApp()
    app.main(page)


if __name__ == "__main__":
    ft.app(target=main)
