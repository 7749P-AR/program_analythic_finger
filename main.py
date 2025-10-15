import flet as ft
import cv2
import base64
import threading
import time
import mediapipe as mp
import numpy as np

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
        
        # Nombres de los dedos
        self.finger_names = ["Pulgar", "Índice", "Medio", "Anular", "Meñique"]
        
        # IDs de las puntas de los dedos en MediaPipe
        self.finger_tips = [4, 8, 12, 16, 20]
        self.finger_pips = [3, 6, 10, 14, 18]  # Para determinar si está extendido
        
    def main(self, page: ft.Page):
        self.page = page
        page.title = "Detector de Posición de Dedos"
        page.window_width = 1000
        page.window_height = 700
        
        # Componentes de la interfaz
        self.image_display = ft.Image(
            src_base64="",
            width=640,
            height=480,
            border_radius=10
        )
        
        self.status_text = ft.Text("Cámara desactivada", size=16, weight=ft.FontWeight.BOLD)
        self.hands_count_text = ft.Text("Manos detectadas: 0", size=14, color=ft.Colors.BLUE)
        
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
        
        # Panel de información de dedos
        self.finger_info = ft.Column([
            ft.Text("Información de Dedos:", size=16, weight=ft.FontWeight.BOLD),
            ft.Text("Activa la cámara para comenzar", color=ft.Colors.GREY_600)
        ])
        
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
                            "🖐️ Detector de Posición de Dedos", 
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
                            ft.Container(width=20),
                            self.status_text,
                            ft.Container(width=20),
                            self.hands_count_text
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
                        
                        # Panel de información
                        ft.Container(
                            content=self.finger_info,
                            width=300,
                            bgcolor=ft.Colors.PURPLE_50,
                            border_radius=15,
                            padding=20,
                            border=ft.border.all(2, ft.Colors.PURPLE_200)
                        )
                    ], alignment=ft.MainAxisAlignment.CENTER),
                    
                    ft.Container(height=20),
                    
                    # Instrucciones
                    ft.Container(
                        content=ft.Column([
                            ft.Text("📋 Instrucciones:", 
                                   size=16, 
                                   weight=ft.FontWeight.BOLD,
                                   color=ft.Colors.PURPLE_700),
                            ft.Text("• Coloca tu mano frente a la cámara"),
                            ft.Text("• El programa detectará hasta 2 manos simultáneamente"),
                            ft.Text("• Se mostrará la posición de cada dedo y si está extendido"),
                            ft.Text("• Los puntos rojos marcan las articulaciones de los dedos"),
                        ], spacing=5),
                        bgcolor=ft.Colors.PURPLE_50,
                        border_radius=10,
                        padding=15,
                        border=ft.border.all(1, ft.Colors.PURPLE_200)
                    )
                    
                ], alignment=ft.MainAxisAlignment.START),
                padding=30
            )
        )
    
    def is_finger_extended(self, landmarks, finger_tip, finger_pip):
        """Determina si un dedo está extendido"""
        try:
            # Para el pulgar, usar lógica diferente
            if finger_tip == 4:  # Pulgar
                return landmarks[finger_tip].x > landmarks[finger_tip - 1].x
            else:  # Otros dedos
                return landmarks[finger_tip].y < landmarks[finger_pip].y
        except (IndexError, AttributeError):
            return False
    
    def get_finger_positions(self, landmarks):
        """Obtiene información detallada de cada dedo"""
        finger_data = []
        
        try:
            for i, (tip, pip) in enumerate(zip(self.finger_tips, self.finger_pips)):
                finger_name = self.finger_names[i]
                tip_landmark = landmarks[tip]
                
                # Posición en píxeles (asumiendo imagen de 640x480)
                x = int(tip_landmark.x * 640)
                y = int(tip_landmark.y * 480)
                
                # Determinar si está extendido
                is_extended = self.is_finger_extended(landmarks, tip, pip)
                
                finger_data.append({
                    'name': finger_name,
                    'x': x,
                    'y': y,
                    'extended': is_extended
                })
        except (IndexError, AttributeError) as e:
            print(f"Error obteniendo posiciones de dedos: {e}")
        
        return finger_data
    
    def update_finger_info(self, hands_data):
        """Actualiza el panel de información de dedos"""
        # Inicializar controls al inicio para evitar el error
        controls = [ft.Text("Información de Dedos:", size=16, weight=ft.FontWeight.BOLD)]
        
        try:
            if not hands_data:
                controls.append(ft.Text("No se detectaron manos", color=ft.Colors.GREY_600))
            else:
                for hand_idx, hand_data in enumerate(hands_data):
                    controls.append(
                        ft.Text(f"\n--- Mano {hand_idx + 1} ---", 
                               weight=ft.FontWeight.BOLD,
                               color=ft.Colors.PURPLE_700)
                    )
                    
                    for finger in hand_data:
                        status = "✋ Extendido" if finger['extended'] else "✊ Cerrado"
                        color = ft.Colors.GREEN if finger['extended'] else ft.Colors.RED
                        
                        controls.append(
                            ft.Text(f"{finger['name']}: {status}", color=color)
                        )
                        controls.append(
                            ft.Text(f"   Posición: ({finger['x']}, {finger['y']})", 
                                   size=12, color=ft.Colors.GREY_700)
                        )
            
            self.finger_info.controls = controls
            
        except Exception as e:
            print(f"Error actualizando información de dedos: {e}")
            # En caso de error, mostrar mensaje básico
            self.finger_info.controls = [
                ft.Text("Información de Dedos:", size=16, weight=ft.FontWeight.BOLD),
                ft.Text("Error procesando datos", color=ft.Colors.RED)
            ]
    
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
        """Procesa el frame para detectar manos"""
        try:
            # Convertir BGR a RGB para MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            hands_data = []
            hands_count = 0
            
            if results.multi_hand_landmarks:
                hands_count = len(results.multi_hand_landmarks)
                
                for hand_landmarks in results.multi_hand_landmarks:
                    # Dibujar landmarks en el frame
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )
                    
                    # Obtener información de dedos
                    finger_data = self.get_finger_positions(hand_landmarks.landmark)
                    if finger_data:  # Solo agregar si hay datos válidos
                        hands_data.append(finger_data)
                    
                    # Dibujar información en el frame
                    for finger in finger_data:
                        color = (0, 255, 0) if finger['extended'] else (0, 0, 255)
                        cv2.circle(frame, (finger['x'], finger['y']), 8, color, -1)
                        cv2.putText(frame, finger['name'][:3], 
                                  (finger['x'] - 15, finger['y'] - 15),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Actualizar contador de manos
            self.hands_count_text.value = f"Manos detectadas: {hands_count}"
            
            # Actualizar información de dedos
            self.update_finger_info(hands_data)
            
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
                    # Procesar frame para detectar manos
                    processed_frame = self.process_frame(frame)
                    
                    # Convertir a base64
                    base64_image = self.frame_to_base64(processed_frame)
                    if base64_image:  # Solo actualizar si la conversión fue exitosa
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
                self.status_text.value = "Cámara activada - Detectando manos..."
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
            self.hands_count_text.value = "Manos detectadas: 0"
            
            # Reset finger info
            self.finger_info.controls = [
                ft.Text("Información de Dedos:", size=16, weight=ft.FontWeight.BOLD),
                ft.Text("Activa la cámara para comenzar", color=ft.Colors.GREY_600)
            ]
        
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
    ft.app(target=main,
           view=ft.AppView.WEB_BROWSER,
           )