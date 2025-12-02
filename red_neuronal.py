import tkinter as tk
from tkinter import ttk
import numpy as np
import tensorflow as tf
from tensorflow import keras
import threading
import time

class NeuralNetworkVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Visualizador de Red Neuronal - Clasificador de Alfabeto")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1a1a2e')
        
        self.grid_size = 20
        self.is_running = False
        self.epoch = 0
        self.alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        self.current_letter_idx = 0
        
        # Crear modelo neuronal simple
        self.create_model()
        
        # Inicializar pesos aleatorios para visualización
        self.weights = np.random.randn(self.grid_size, self.grid_size)
        
        self.setup_ui()
        
    def create_model(self):
        """Crea un modelo simple de red neuronal para clasificación de letras"""
        self.model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(26,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(26, activation='softmax')
        ])
        self.model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        
    def setup_ui(self):
        # Frame principal
        main_frame = tk.Frame(self.root, bg='#1a1a2e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header
        header_frame = tk.Frame(main_frame, bg='#1a1a2e')
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = tk.Label(header_frame, 
                              text="🧠 Red Neuronal - Clasificador de Alfabeto",
                              font=('Arial', 24, 'bold'),
                              bg='#1a1a2e',
                              fg='#00d4ff')
        title_label.pack(side=tk.LEFT)
        
        # Botones de control
        btn_frame = tk.Frame(header_frame, bg='#1a1a2e')
        btn_frame.pack(side=tk.RIGHT)
        
        self.start_btn = tk.Button(btn_frame,
                                   text="▶ Iniciar",
                                   command=self.toggle_training,
                                   font=('Arial', 12, 'bold'),
                                   bg='#00ff88',
                                   fg='#1a1a2e',
                                   width=12,
                                   cursor='hand2')
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        reset_btn = tk.Button(btn_frame,
                             text="↻ Reset",
                             command=self.reset,
                             font=('Arial', 12, 'bold'),
                             bg='#ff6b6b',
                             fg='white',
                             width=12,
                             cursor='hand2')
        reset_btn.pack(side=tk.LEFT, padx=5)
        
        # Frame de contenido (izquierda y derecha)
        content_frame = tk.Frame(main_frame, bg='#1a1a2e')
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Panel izquierdo - Matriz de pesos
        left_frame = tk.Frame(content_frame, bg='#0f3460', relief=tk.RAISED, borderwidth=2)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        left_title = tk.Label(left_frame,
                             text="Matriz de Pesos (Capa Oculta)",
                             font=('Arial', 16, 'bold'),
                             bg='#0f3460',
                             fg='#00d4ff')
        left_title.pack(pady=10)
        
        # Canvas para la matriz de pesos
        self.canvas = tk.Canvas(left_frame, bg='#16213e', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Leyenda de colores
        legend_frame = tk.Frame(left_frame, bg='#0f3460')
        legend_frame.pack(pady=10)
        
        legends = [
            ("Positivo fuerte", "#00ff88"),
            ("Positivo débil", "#88ffaa"),
            ("Negativo débil", "#ffaa88"),
            ("Negativo fuerte", "#ff6b6b")
        ]
        
        for text, color in legends:
            lf = tk.Frame(legend_frame, bg='#0f3460')
            lf.pack(side=tk.LEFT, padx=10)
            color_box = tk.Label(lf, bg=color, width=2, height=1)
            color_box.pack(side=tk.LEFT, padx=5)
            tk.Label(lf, text=text, bg='#0f3460', fg='white', font=('Arial', 9)).pack(side=tk.LEFT)
        
        # Panel derecho - Información
        right_frame = tk.Frame(content_frame, bg='#1a1a2e')
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # Estado del modelo
        status_frame = tk.Frame(right_frame, bg='#0f3460', relief=tk.RAISED, borderwidth=2)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(status_frame,
                text="Estado del Modelo",
                font=('Arial', 14, 'bold'),
                bg='#0f3460',
                fg='#00d4ff').pack(pady=10)
        
        self.epoch_label = tk.Label(status_frame,
                                   text="Época: 0",
                                   font=('Arial', 20, 'bold'),
                                   bg='#0f3460',
                                   fg='#00ff88')
        self.epoch_label.pack(pady=5)
        
        self.status_label = tk.Label(status_frame,
                                    text="Estado: Detenido",
                                    font=('Arial', 12),
                                    bg='#0f3460',
                                    fg='#ff6b6b')
        self.status_label.pack(pady=10)
        
        # Letra actual
        input_frame = tk.Frame(right_frame, bg='#0f3460', relief=tk.RAISED, borderwidth=2)
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(input_frame,
                text="Input Actual",
                font=('Arial', 14, 'bold'),
                bg='#0f3460',
                fg='#00d4ff').pack(pady=10)
        
        letter_display = tk.Frame(input_frame, bg='#16213e', width=150, height=150)
        letter_display.pack(pady=10, padx=20)
        letter_display.pack_propagate(False)
        
        self.letter_label = tk.Label(letter_display,
                                     text="A",
                                     font=('Arial', 60, 'bold'),
                                     bg='#16213e',
                                     fg='#00d4ff')
        self.letter_label.pack(expand=True)
        
        # Predicción
        pred_frame = tk.Frame(right_frame, bg='#0f3460', relief=tk.RAISED, borderwidth=2)
        pred_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(pred_frame,
                text="Predicción",
                font=('Arial', 14, 'bold'),
                bg='#0f3460',
                fg='#00d4ff').pack(pady=10)
        
        self.pred_label = tk.Label(pred_frame,
                                   text="--",
                                   font=('Arial', 40, 'bold'),
                                   bg='#0f3460',
                                   fg='#ffeb3b')
        self.pred_label.pack(pady=10)
        
        self.conf_label = tk.Label(pred_frame,
                                   text="Confianza: --",
                                   font=('Arial', 12),
                                   bg='#0f3460',
                                   fg='white')
        self.conf_label.pack(pady=10)
        
        # Estadísticas adicionales
        stats_frame = tk.Frame(right_frame, bg='#0f3460', relief=tk.RAISED, borderwidth=2)
        stats_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(stats_frame,
                text="Estadísticas",
                font=('Arial', 14, 'bold'),
                bg='#0f3460',
                fg='#00d4ff').pack(pady=10)
        
        self.accuracy_label = tk.Label(stats_frame,
                                      text="Precisión: 0%",
                                      font=('Arial', 11),
                                      bg='#0f3460',
                                      fg='white')
        self.accuracy_label.pack(pady=5)
        
        self.loss_label = tk.Label(stats_frame,
                                  text="Loss: 0.000",
                                  font=('Arial', 11),
                                  bg='#0f3460',
                                  fg='white')
        self.loss_label.pack(pady=5)
        
        # Inicializar visualización
        self.root.after(100, self.draw_weights)
        
    def draw_weights(self):
        """Dibuja la matriz de pesos en el canvas"""
        self.canvas.delete("all")
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            self.root.after(100, self.draw_weights)
            return
        
        cell_width = canvas_width / self.grid_size
        cell_height = canvas_height / self.grid_size
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x1 = j * cell_width
                y1 = i * cell_height
                x2 = x1 + cell_width
                y2 = y1 + cell_height
                
                weight = self.weights[i, j]
                color = self.get_color(weight)
                
                self.canvas.create_rectangle(x1, y1, x2, y2,
                                            fill=color,
                                            outline='#1a1a2e',
                                            width=1)
                
                # Dibujar el valor del peso
                text_size = max(6, int(min(cell_width, cell_height) / 3))
                self.canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2,
                                       text=f"{weight:.2f}",
                                       fill='white',
                                       font=('Courier', text_size, 'bold'))
    
    def get_color(self, weight):
        """Retorna un color basado en el valor del peso"""
        if weight > 0.5:
            return '#00ff88'
        elif weight > 0:
            return '#88ffaa'
        elif weight > -0.5:
            return '#ffaa88'
        else:
            return '#ff6b6b'
    
    def toggle_training(self):
        """Inicia o detiene el entrenamiento"""
        self.is_running = not self.is_running
        
        if self.is_running:
            self.start_btn.config(text="⏸ Pausar", bg='#ff6b6b')
            self.status_label.config(text="Estado: Entrenando", fg='#00ff88')
            threading.Thread(target=self.training_loop, daemon=True).start()
        else:
            self.start_btn.config(text="▶ Iniciar", bg='#00ff88')
            self.status_label.config(text="Estado: Detenido", fg='#ff6b6b')
    
    def training_loop(self):
        """Simula el entrenamiento de la red neuronal"""
        while self.is_running:
            # Actualizar pesos (simulación de backpropagation)
            num_updates = np.random.randint(20, 50)
            for _ in range(num_updates):
                i = np.random.randint(0, self.grid_size)
                j = np.random.randint(0, self.grid_size)
                delta = np.random.randn() * 0.1
                self.weights[i, j] += delta
                self.weights[i, j] = np.clip(self.weights[i, j], -1, 1)
            
            # Actualizar letra actual
            current_letter = self.alphabet[self.current_letter_idx]
            self.current_letter_idx = (self.current_letter_idx + 1) % len(self.alphabet)
            
            # Simular predicción
            prediction = self.alphabet[np.random.randint(0, len(self.alphabet))]
            confidence = np.random.uniform(0.6, 0.99)
            accuracy = np.random.uniform(0.75, 0.95)
            loss = np.random.uniform(0.1, 0.5)
            
            # Actualizar UI desde el thread principal
            self.root.after(0, self.update_ui, current_letter, prediction, confidence, accuracy, loss)
            
            self.epoch += 1
            time.sleep(0.2)
    
    def update_ui(self, letter, prediction, confidence, accuracy, loss):
        """Actualiza la interfaz con nueva información"""
        self.letter_label.config(text=letter)
        self.pred_label.config(text=prediction)
        self.conf_label.config(text=f"Confianza: {confidence:.1%}")
        self.epoch_label.config(text=f"Época: {self.epoch}")
        self.accuracy_label.config(text=f"Precisión: {accuracy:.1%}")
        self.loss_label.config(text=f"Loss: {loss:.3f}")
        self.draw_weights()
    
    def reset(self):
        """Reinicia el visualizador"""
        self.is_running = False
        self.epoch = 0
        self.current_letter_idx = 0
        self.weights = np.random.randn(self.grid_size, self.grid_size)
        
        self.start_btn.config(text="▶ Iniciar", bg='#00ff88')
        self.status_label.config(text="Estado: Detenido", fg='#ff6b6b')
        self.epoch_label.config(text="Época: 0")
        self.letter_label.config(text="A")
        self.pred_label.config(text="--")
        self.conf_label.config(text="Confianza: --")
        self.accuracy_label.config(text="Precisión: 0%")
        self.loss_label.config(text="Loss: 0.000")
        
        self.draw_weights()

if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralNetworkVisualizer(root)
    root.mainloop()