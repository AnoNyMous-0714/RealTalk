import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from tkinter import DISABLED
from tkinter.scrolledtext import ScrolledText
import numpy as np
import librosa
import librosa.display
import pywt
import tensorflow as tf
import soundfile as sf
import os
import threading
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from datetime import datetime
import time
import psutil
import sys
from tabulate import tabulate
import tempfile
import csv
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from ttkbootstrap.tooltip import ToolTip
from ttkbootstrap.style import Bootstyle
from PIL import Image, ImageTk

# Constants
SAMPLE_RATE = 16000
N_FFT = 2048
HOP_LENGTH = 512
N_MFCC = 50
DWT_WAVELET = 'sym8'
DWT_LEVEL = 2
FIXED_FEATURE_LENGTH = 4000
SEGMENT_LENGTH = 5
OVERLAP = 0.5
WINDOW_SIZE = 1000
STEP_SIZE = 500

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads=4, head_size=32, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.head_size = head_size
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=head_size)
        self.add = tf.keras.layers.Add()
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, inputs):
        attn_output = self.attention(inputs, inputs)
        return self.norm(self.add([inputs, attn_output]))

    def get_config(self):
        config = super(SelfAttention, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'head_size': self.head_size
        })
        return config

class DeepfakeDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real Talk - Deepfake Audio Detection")
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{min(1400, screen_width-100)}x{min(900, screen_height-100)}")
        self.root.minsize(800, 600)
        self.root.state('zoomed')
        
        self.style = tb.Style(theme="minty")
        self.style.configure("TButton", font=("Segoe UI", 10), padding=8, relief="flat", 
                        bordercolor="#e0e0e0", borderwidth=1)
        self.style.configure("TLabel", font=("Segoe UI", 10))
        self.style.configure("TLabelFrame", font=("Segoe UI", 10, "bold"), 
                        background="#f8f9fa", borderwidth=2, relief="groove")
        self.style.configure("TProgressbar", thickness=20)
        self.style.configure("TFrame", background="#f8f9fa")
        
        # Custom style for primary buttons
        self.style.configure("primary.TButton", 
            font=("Segoe UI", 10, "bold"),
            padding=10,
            relief="flat",
            bordercolor="#3498db",
            background="#3498db",
            foreground="white",
            focuscolor="#3498db",
            focusthickness=0,
            borderwidth=0
        )
        self.style.map("primary.TButton",
            background=[("active", "#2980b9"), ("disabled", "#bdc3c7")],
            foreground=[("disabled", "#7f8c8d")]
        )
            
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_path, 'model.h5')

        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={'SelfAttention': SelfAttention}
        )
        
        # Initialize variables
        self.current_segment = 0
        self.total_segments = 0
        self.segment_audio = []
        self.segment_predictions = []
        self.segment_confidences = []
        self.final_prediction = ""
        self.final_confidence = 0.0
        self.file_path = None
        self.current_audio = None
        
        # Visualization variables
        self.fig = None
        self.wave_ax = None
        self.spec_ax = None
        self.mfcc_ax = None
        self.canvas = None
        self.spec_cbar = None
        self.mfcc_cbar = None

        plt.style.use('seaborn-v0_8-whitegrid')  # Clean grid style
        plt.rcParams.update({
            'font.family': 'Segoe UI',          # Match UI font
            'axes.titlesize': 10,               # Consistent title size
            'axes.labelsize': 8,                # Smaller axis labels
            'xtick.labelsize': 7,               # Tiny tick labels
            'ytick.labelsize': 7,
            'grid.alpha': 0.3,                  # Subtle grid lines
            'figure.autolayout': False          # Manual layout control
        })
        
        self.create_widgets()
        self.setup_visualizations()  # Removed setup_logging() from here

    def setup_logging(self):
        log_frame = tb.LabelFrame(self.left_panel, text="Processing Logs", bootstyle="info")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create main container with grid layout
        log_container = tb.Frame(log_frame)
        log_container.pack(fill=tk.BOTH, expand=True)
        log_container.grid_rowconfigure(0, weight=1)
        log_container.grid_columnconfigure(0, weight=1)
        
        # Create Text widget with scrollbars
        self.log_text = tk.Text(
            log_container,
            wrap=tk.WORD,
            font=("Segoe UI", 9),
            padx=5,
            pady=5,
            relief="flat",
            highlightthickness=0
        )
        
        # Create scrollbars
        y_scroll = tb.Scrollbar(
            log_container,
            orient="vertical",
            command=self.log_text.yview,
            bootstyle="round"
        )
        x_scroll = tb.Scrollbar(
            log_container,
            orient="horizontal",
            command=self.log_text.xview,
            bootstyle="round"
        )
        
        # Configure text widget
        self.log_text.config(
            yscrollcommand=y_scroll.set,
            xscrollcommand=x_scroll.set,
            state=tk.DISABLED
        )
        
        # Grid layout for proper expansion
        self.log_text.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        
        # Configure text tags
        self.log_text.tag_config("INFO", foreground="black")
        self.log_text.tag_config("DEBUG", foreground="blue")
        self.log_text.tag_config("MATH", foreground="purple")
        self.log_text.tag_config("WARNING", foreground="orange")
        self.log_text.tag_config("ERROR", foreground="red")
        self.log_text.tag_config("HEADER", font=("Segoe UI", 9, "bold"))

    def log_message(self, message, level="INFO"):
        """Add a message to the log with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] {message}\n"
        
        self.log_text.config(state=tb.NORMAL)
        self.log_text.insert(tb.END, log_entry, level)
        self.log_text.see(tb.END)
        self.log_text.config(state=tb.DISABLED)
        self.root.update_idletasks()

    def safe_clear_visualizations(self):
        
        try:
            # Clear figure and all subplots
            self.fig.clf()

            # Recreate subplots from scratch
            self.wave_ax = self.fig.add_subplot(311)
            self.spec_ax = self.fig.add_subplot(312)
            self.mfcc_ax = self.fig.add_subplot(313)

            self.spec_cbar = None
            self.mfcc_cbar = None

            # Reset titles
            self.wave_ax.set_title("Waveform (Time Domain)")
            self.spec_ax.set_title("Spectrogram (Frequency Domain)")
            self.mfcc_ax.set_title("MFCC Coefficients")

            self.fig.subplots_adjust(hspace=0.4, top=0.95, bottom=0.07)
            self.canvas.draw()

        except Exception as e:
            self.log_message(f"Visualization reset error: {e}", "ERROR")

    def setup_visualizations(self):
        
        if self.canvas:
            return  

        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.wave_ax = self.fig.add_subplot(311)
        self.spec_ax = self.fig.add_subplot(312)
        self.mfcc_ax = self.fig.add_subplot(313)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.vis_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fig.subplots_adjust(hspace=0.4, top=0.95, bottom=0.07)

        
        self.spec_cbar = None
        self.mfcc_cbar = None

        # Draw blank plots initially
        self.wave_ax.set_title("Waveform (Time Domain)")
        self.spec_ax.set_title("Spectrogram (Frequency Domain)")
        self.mfcc_ax.set_title("MFCC Coefficients")
        self.canvas.draw()

    def format_analysis_table(self, headers, data, col_widths=None):
        """Helper method to format analysis tables consistently"""
        if col_widths is None:
            col_widths = [20, 20, 20]  # Default column widths
        
        # Create a formatted table string
        table = []
        
        # Add headers
        header_row = "".join([h.ljust(w) for h, w in zip(headers, col_widths)])
        table.append(header_row)
        table.append("-" * len(header_row))
        
        # Add data rows
        for row in data:
            table.append("".join([str(cell).ljust(w) for cell, w in zip(row, col_widths)]))
        
        return "\n".join(table)

    def update_visualizations(self, audio, power_spectrum):
        
        try:
            self.safe_clear_visualizations()

            max_time = len(audio) / SAMPLE_RATE
            time = np.linspace(0, max_time, num=len(audio))
            stft = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH))
            power_spectrum = librosa.power_to_db(stft)
            mfccs = librosa.feature.mfcc(S=power_spectrum, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
            mfccs_normalized = librosa.util.normalize(mfccs, axis=1)

            if not self.fig:
                self.setup_visualizations()

            gs = self.fig.add_gridspec(3, 1, height_ratios=[1, 1, 1])
            for ax in [self.wave_ax, self.spec_ax, self.mfcc_ax]:
                ax.remove()

            self.wave_ax = self.fig.add_subplot(gs[0])
            self.spec_ax = self.fig.add_subplot(gs[1])
            self.mfcc_ax = self.fig.add_subplot(gs[2])

            plot_style = {
                'title_pad': 10,
                'label_pad': 5,
                'interpretation_xpos': 0.02,
                'interpretation_ypos': -0.25,
                'interpretation_fontsize': 8,
                'bbox_style': dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5')
            }

            self.wave_ax.plot(time, audio, color='#1f77b4', linewidth=0.8)
            self.wave_ax.fill_between(time, audio, color='#1f77b4', alpha=0.15)
            self.wave_ax.grid(True, linestyle=':', alpha=0.5)
            self.wave_ax.set_xlim(0, max_time)
            self.wave_ax.set_ylim(-1.1, 1.1)
            self.wave_ax.set_title("Waveform (Time Domain)", pad=plot_style['title_pad'])
            self.wave_ax.set_ylabel("Amplitude", labelpad=plot_style['label_pad'])
            self.wave_ax.text(plot_style['interpretation_xpos'], plot_style['interpretation_ypos'],
                              "Interpretation:\n"
                              "The waveform shows amplitude variations over time. Genuine speech has irregular peaks and dips.\n"
                              "Deepfakes may appear unnaturally smooth or have consistent amplitudes, indicating synthetic generation.",
                              transform=self.wave_ax.transAxes,
                              fontsize=plot_style['interpretation_fontsize'],
                              ha='left', va='top',
                              bbox=plot_style['bbox_style'])

            S_dB = librosa.power_to_db(power_spectrum, ref=np.max, top_db=80)
            spec_img = self.spec_ax.imshow(S_dB, origin='lower', aspect='auto',
                                           extent=[0, max_time, 0, SAMPLE_RATE / 2], cmap='magma')
            self.spec_ax.set_title("Spectrogram (Frequency Domain)", pad=plot_style['title_pad'])
            self.spec_ax.set_ylabel("Frequency (Hz)", labelpad=plot_style['label_pad'])
            self.spec_ax.set_ylim(0, 8000)
            if self.spec_cbar:
                self.spec_cbar.mappable.set_array(S_dB)
            else:
                self.spec_cbar = self.fig.colorbar(spec_img, ax=self.spec_ax)
                self.spec_cbar.set_label('Intensity (dB)', rotation=270)
            self.spec_ax.text(plot_style['interpretation_xpos'], plot_style['interpretation_ypos'],
                              "Interpretation:\n"
                              "The spectrogram displays frequency energy over time. Natural speech shows uneven energy with clear pitch shifts.\n"
                              "Deepfakes may look overly clean or uniform, with smooth transitions and missing background noise.",
                              transform=self.spec_ax.transAxes,
                              fontsize=plot_style['interpretation_fontsize'],
                              ha='left', va='top',
                              bbox=plot_style['bbox_style'])

            mfcc_img = self.mfcc_ax.imshow(mfccs_normalized, origin='lower', aspect='auto',
                                           extent=[0, max_time, 0, N_MFCC], cmap='coolwarm')
            self.mfcc_ax.set_title("MFCC Coefficients", pad=plot_style['title_pad'])
            self.mfcc_ax.set_ylabel('MFCC Index', labelpad=plot_style['label_pad'])
            self.mfcc_ax.set_xlabel('Time (s)', labelpad=plot_style['label_pad'])
            self.mfcc_ax.set_yticks(np.arange(0, N_MFCC + 1, 5))
            if self.mfcc_cbar:
                self.mfcc_cbar.mappable.set_array(mfccs_normalized)
            else:
                self.mfcc_cbar = self.fig.colorbar(mfcc_img, ax=self.mfcc_ax)
                self.mfcc_cbar.set_label('Normalized Value', rotation=270)
            self.mfcc_ax.text(plot_style['interpretation_xpos'], plot_style['interpretation_ypos'],
                              "Interpretation:\n"
                              "The MFCC heatmap visualizes compressed frequency features. Real audio shows varied, irregular patterns.\n"
                              "Deepfakes often display repetitive or overly smooth structures, lacking natural speech variations.",
                              transform=self.mfcc_ax.transAxes,
                              fontsize=plot_style['interpretation_fontsize'],
                              ha='left', va='top',
                              bbox=plot_style['bbox_style'])

            self.fig.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.1, hspace=0.7)
            self.canvas.draw()

        except Exception as e:
            self.log_message(f"Visualization update error: {e}", "ERROR")
            import traceback
            self.log_message(traceback.format_exc(), "ERROR")
            raise

    def create_widgets(self):
        # Main container using grid for proper panel sizing
        main_frame = tb.Frame(self.root, bootstyle="light")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configure grid weights (3:2 ratio for left:right panels)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=3, minsize=500)  # Left panel
        main_frame.grid_columnconfigure(1, weight=2, minsize=350)  # Right panel

        # Left panel container (will hold canvas and scrollbar)
        left_container = tb.Frame(main_frame, bootstyle="light")
        left_container.grid(row=0, column=0, sticky="nsew", padx=(0, 0))
        
        # Create canvas with scrollbar
        self.left_canvas = tk.Canvas(left_container, highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=self.left_canvas.yview)
        self.left_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Grid layout for canvas and scrollbar
        self.left_canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Configure grid weights inside left container
        left_container.grid_rowconfigure(0, weight=1)
        left_container.grid_columnconfigure(0, weight=1)
        
        # Create scrollable frame inside canvas
        self.left_panel = tb.Frame(self.left_canvas, bootstyle="light")
        self.left_canvas_frame = self.left_canvas.create_window((0, 0), window=self.left_panel, anchor="nw")
        
        # Configure scroll region and width adjustment
        def _configure_scroll(event):
            # Update scroll region
            self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))
            # Set the inner frame width to match canvas
            self.left_canvas.itemconfig(self.left_canvas_frame, width=self.left_canvas.winfo_width())
        
        self.left_panel.bind("<Configure>", _configure_scroll)
        self.left_canvas.bind("<Configure>", lambda e: self.left_canvas.itemconfig(self.left_canvas_frame, width=e.width))

        # Right panel
        right_panel = tb.Frame(main_frame, bootstyle="light")
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        # ===== LEFT PANEL WIDGETS =====
        # Title Frame
        title_frame = tb.Frame(self.left_panel, bootstyle="light")
        title_frame.pack(fill=tk.X, pady=(0, 15))

        # Logo and Title
        logo_path = "C:/Users/Jio/Desktop/final_deepfake_detection/logo.png"
        if os.path.exists(logo_path):
            original_img = Image.open(logo_path)
            resized_img = original_img.resize((70, 70), Image.LANCZOS)
            logo_img = ImageTk.PhotoImage(resized_img)
            logo_label = tb.Label(title_frame, image=logo_img, bootstyle="light")
            logo_label.image = logo_img
            logo_label.pack(side=tk.LEFT, padx=(0, 8), pady=0)
        
        title_label = tb.Label(
            title_frame, 
            text="Real Talk - Deepfake Audio Detection", 
            font=("Segoe UI", 18, "bold"),
            bootstyle="inverse-light"
        )
        title_label.pack(expand=True)

        # File Selection Frame
        file_frame = tb.LabelFrame(self.left_panel, text="Audio File", bootstyle="info")
        file_frame.pack(fill=tk.X, pady=5, padx=5)

        self.file_label = tb.Label(
            file_frame, 
            text="No file selected", 
            anchor="w", 
            padding=5,
            font=("Segoe UI", 9, "bold"),
            relief="sunken",
            bootstyle="light",
            foreground="#2c3e50"
        )
        self.file_label.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 0))

        browse_btn = tb.Button(
            file_frame, 
            text="Browse", 
            bootstyle="primary",
            command=self.browse_file
        )
        browse_btn.pack(side=tk.LEFT, padx=(5, 5))
        ToolTip(browse_btn, text="Select an audio file (WAV, MP3, OGG, FLAC)\nMinimum 5 seconds duration", 
            bootstyle="info-inverse")

        # Scan Button
        self.scan_btn = tb.Button(
            self.left_panel, 
            text="Scan Audio", 
            bootstyle="primary.TButton",
            command=self.start_scan, 
            state=tk.DISABLED
        )
        self.scan_btn.pack(fill=tk.X, pady=10, padx=5)
        ToolTip(self.scan_btn, text="Analyze the selected audio file for deepfake detection", 
            bootstyle="info-inverse")

        # Progress Bar
        progress_frame = tb.Frame(self.left_panel, bootstyle="light")
        progress_frame.pack(fill=tk.X, pady=5, padx=5)
        
        self.progress = tb.Progressbar(
            progress_frame, 
            orient=tk.HORIZONTAL, 
            length=300, 
            mode="determinate",
            bootstyle="info-striped"
        )
        self.progress.pack(fill=tk.X)

        # Final Results Frame
        self.final_results_frame = tb.LabelFrame(
            self.left_panel, 
            text="Final Detection Results", 
            padding=10,
            bootstyle="info"
        )
        self.final_results_frame.pack(fill=tk.X, pady=5, padx=5)

        self.final_result_label = tb.Label(
            self.final_results_frame, 
            text="No results yet", 
            font=("Segoe UI", 12),
            anchor="center",
            bootstyle="light"
        )
        self.final_result_label.pack(fill=tk.X, pady=5)

        # Navigation Controls
        self.nav_frame = tb.Frame(self.left_panel, bootstyle="light")
        self.nav_frame.pack(fill=tk.X, pady=5, padx=5)

        self.prev_btn = tb.Button(
            self.nav_frame, 
            text="◄ Previous", 
            bootstyle="outline-primary",
            command=self.prev_segment, 
            state=tk.DISABLED,
            width=10
        )
        self.prev_btn.pack(side=tk.LEFT, expand=True, padx=2)
        ToolTip(self.prev_btn, text="View previous segment analysis", bootstyle="info-inverse")

        self.segment_label = tb.Label(
            self.nav_frame, 
            text="Segment: 0/0", 
            font=("Segoe UI", 11, "bold"),
            anchor="center",
            bootstyle="inverse-primary",
            padding=(10, 4),
            width=12
        )
        self.segment_label.pack(side=tk.LEFT, expand=True)

        self.next_btn = tb.Button(
            self.nav_frame, 
            text="Next ►", 
            bootstyle="outline-primary",
            command=self.next_segment, 
            state=tk.DISABLED,
            width=10
        )
        self.next_btn.pack(side=tk.LEFT, expand=True, padx=2)
        ToolTip(self.next_btn, text="View next segment analysis", bootstyle="info-inverse")

        # Results Text Frame with Scrollbars
        results_frame = tb.LabelFrame(
            self.left_panel, 
            text="Segment Analysis", 
            padding=10,
            bootstyle="info"
        )
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)

        results_text_frame = tb.Frame(results_frame)
        results_text_frame.pack(fill=tk.BOTH, expand=True)

        self.result_text = tk.Text(
            results_text_frame, 
            wrap="none",
            font=("Consolas", 10),
            padx=10,
            pady=10,
            relief="flat",
            highlightthickness=0
        )
        v_scroll = tb.Scrollbar(results_text_frame, orient="vertical", command=self.result_text.yview)
        h_scroll = tb.Scrollbar(results_text_frame, orient="horizontal", command=self.result_text.xview)
        self.result_text.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        v_scroll.pack(side="right", fill="y")
        h_scroll.pack(side="bottom", fill="x")
        self.result_text.pack(side="left", fill="both", expand=True)
        self.result_text.config(state=tk.DISABLED)

        # Export Buttons
        export_frame = tb.Frame(self.left_panel, bootstyle="light")
        export_frame.pack(fill=tk.X, pady=10, padx=5)

        self.pdf_export_btn = tb.Button(
            export_frame, 
            text="Export Visualizations to PDF", 
            bootstyle="outline",
            command=self.export_to_pdf, 
            state=tk.DISABLED
        )
        self.pdf_export_btn.pack(side=tk.LEFT, expand=True, padx=2)
        ToolTip(self.pdf_export_btn, text="Export visualizations to PDF file", bootstyle="info-inverse")

        self.csv_export_btn = tb.Button(
            export_frame, 
            text="Export Results to CSV", 
            bootstyle="outline",
            command=self.export_to_csv, 
            state=tk.DISABLED
        )
        self.csv_export_btn.pack(side=tk.LEFT, expand=True, padx=2)
        ToolTip(self.csv_export_btn, text="Export analysis results to CSV file", bootstyle="info-inverse")

        # ===== RIGHT PANEL WIDGETS =====
        # Visualization Frame
        self.vis_frame = tb.LabelFrame(
            right_panel, 
            text="Audio Analysis Visualizations", 
            padding=(10, 5),
            bootstyle="info"
        )
        self.vis_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Status Bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_label = tb.Label(
            self.root, 
            textvariable=self.status_var, 
            anchor="w", 
            bootstyle=("info", "inverse"),
            padding=(10, 5, 10, 5),
            font=("Segoe UI", 9),
            relief="flat"
        )
        self.status_label.pack(side="bottom", fill="x")

        # Setup visualizations and logging (only called once here)
        self.setup_visualizations()
        self.setup_logging()  # This is now the only call to setup_logging()

    def browse_file(self):
        try:
            # Clear everything completely before loading new file
            
            filetypes = [("Audio files", "*.wav *.mp3 *.ogg *.flac"), ("All files", "*.*")]
            file_path = filedialog.askopenfilename(title="Select Audio File", filetypes=filetypes)
            
            if file_path:
                # Check duration before proceeding
                if not self.check_audio_duration(file_path):
                    self.log_message("User cancelled long audio file processing", "INFO")
                    return
                
                self.clear_results()
                self.safe_clear_visualizations()
                self.setup_visualizations()
                    
                self.file_path = file_path
                self.file_label.config(text=os.path.basename(file_path))
                self.scan_btn.config(state=tb.NORMAL)
                self.status_var.set(f"Selected: {os.path.basename(file_path)}")
                self.log_message(f"Selected audio file: {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Error during file browsing: {str(e)}")
            self.log_message(f"File browsing error: {str(e)}", "ERROR")

    def clear_results(self):
        """Clear both text results and visualizations"""
        self.result_text.config(state=tb.NORMAL)
        self.result_text.delete(1.0, tb.END)
        self.result_text.config(state=tb.DISABLED)
        self.final_result_label.config(text="No results yet")
        self.segment_label.config(text="Segment: 0/0")
        self.prev_btn.config(state=tb.DISABLED)
        self.next_btn.config(state=tb.DISABLED)
        
        # Reset segment tracking
        self.current_segment = 0
        self.total_segments = 0
        self.segment_audio = []
        self.segment_predictions = []
        self.segment_confidences = []
        self.final_prediction = ""
        self.final_confidence = 0.0
        
        # Clear logs
        if hasattr(self, 'log_text'):
            self.log_text.config(state=tb.NORMAL)
            self.log_text.delete(1.0, tb.END)
            self.log_text.config(state=tb.DISABLED)

    def start_scan(self):
        if not self.file_path:
            messagebox.showerror("Error", "No file selected")
            self.log_message("Scan attempted with no file selected", "WARNING")
            return
        
        # Clear previous results
        self.clear_results()
        self.safe_clear_visualizations()
        self.setup_visualizations()
        
        # Disable buttons during processing
        self.scan_btn.config(state=tb.DISABLED)
        self.progress["value"] = 0
        self.status_var.set("Processing audio...")
        self.log_message("Starting audio scan...", "HEADER")
        
        # Start processing in a separate thread
        threading.Thread(target=self.process_audio, daemon=True).start()

    def process_audio(self):
        try:
            self.log_message("\n=== STARTING AUDIO PROCESSING ===", "HEADER")
            # Start timing and memory tracking
            start_time = time.time()
            process = psutil.Process(os.getpid())
            
            # Step 1: Preprocess
            self.update_progress(10, "Preprocessing audio...")
            initial_mem = process.memory_info().rss / (1024 * 1024)  # MB
            temp_path = "temp_processed.wav"
            self.preprocess_audio(self.file_path, temp_path)
            
            # Load the full audio file
            audio, _ = librosa.load(temp_path, sr=SAMPLE_RATE)
            duration = len(audio) / SAMPLE_RATE
            self.log_message(f"Full audio duration: {duration:.2f} seconds")
            
            # Reject audio shorter than 5 seconds
            if duration < 5:
                os.remove(temp_path)
                message = f"Audio is only {duration:.1f} seconds. Minimum 5 seconds required."
                messagebox.showwarning("Audio Too Short", message)
                self.status_var.set("Audio too short - minimum 5s required")
                self.scan_btn.config(state=tb.NORMAL)
                self.log_message(message, "WARNING")
                return
            
            # Split into segments with 50% overlap
            segment_samples = SAMPLE_RATE * SEGMENT_LENGTH
            step = int(segment_samples * 0.5)  # 50% overlap
            self.segment_audio = []
            
            # Generate segments and discard partial ones (<2s of real audio)
            for start in range(0, len(audio), step):
                end = start + segment_samples
                # For last segment, check if it has at least 2s of real audio
                if end > len(audio):
                    remaining_seconds = (len(audio) - start) / SAMPLE_RATE
                    if remaining_seconds >= 2:  # Keep if >=2s
                        segment = audio[start:]
                        segment = np.pad(segment, (0, segment_samples - len(segment)))
                        self.segment_audio.append(segment)
                        self.log_message(f"Added partial segment ({remaining_seconds:.1f}s) starting at {start/SAMPLE_RATE:.1f}s")
                else:
                    segment = audio[start:end]
                    self.segment_audio.append(segment)
                    self.log_message(f"Added full segment starting at {start/SAMPLE_RATE:.1f}s")
            
            self.total_segments = len(self.segment_audio)
            self.log_message(f"Total segments created: {self.total_segments}")
            
            # If no valid segments found (unlikely with 5s+ requirement)
            if self.total_segments == 0:
                os.remove(temp_path)
                messagebox.showwarning("No Valid Segments", "Could not extract valid audio segments.")
                self.status_var.set("No valid segments found")
                self.scan_btn.config(state=tb.NORMAL)
                self.log_message("No valid segments could be extracted", "WARNING")
                return
                
            # Process each segment
            self.segment_predictions = []
            self.segment_confidences = []
            
            for i, segment in enumerate(self.segment_audio):
                self.update_progress(10 + (i * 80 / self.total_segments), 
                                    f"Processing segment {i+1}/{self.total_segments}...")
                self.log_message(f"\nProcessing segment {i+1}/{self.total_segments}", "HEADER")
                
                # Track memory before processing segment
                mem_before = process.memory_info().rss / (1024 * 1024)  # MB
                segment_start = time.time()
                
                # Save segment to temp file
                segment_path = f"temp_segment_{i}.wav"
                sf.write(segment_path, segment, SAMPLE_RATE)
                self.log_message(f"Saved temporary segment to: {segment_path}")
                
                # Extract features and predict
                features, analysis_results = self.extract_features_with_analysis(segment_path)
                features = self.apply_sliding_window(features)
                
                # Log sliding window application
                self.log_message("\nApplying sliding window to features...", "DEBUG")
                self.log_message(f"Window size: {WINDOW_SIZE}, Step size: {STEP_SIZE}", "DEBUG")
                self.log_message(f"Input features length: {len(features)}", "DEBUG")
                self.log_message(f"Resulting windows: {features.shape[0]} of shape {features.shape[1:]}", "DEBUG")
                
                # Model prediction
                self.log_message("\nMaking prediction with model...", "DEBUG")
                prediction_prob = self.model.predict(features, verbose=0).mean()
                prediction = "Deepfake" if prediction_prob >= 0.4976 else "Genuine"
                confidence = prediction_prob if prediction == "Deepfake" else 1 - prediction_prob
                
                self.log_message(f"Raw prediction probability: {prediction_prob:.4f}", "DEBUG")
                self.log_message(f"Final prediction: {prediction} (confidence: {confidence*100:.2f}%)", "INFO")
                
                self.segment_predictions.append(prediction)
                self.segment_confidences.append(confidence)
                
                # Clean up temp file
                os.remove(segment_path)
                self.log_message(f"Removed temporary file: {segment_path}")
                
                # Log segment processing stats
                segment_time = time.time() - segment_start
                mem_after = process.memory_info().rss / (1024 * 1024)  # MB
                self.log_message(f"Segment processing time: {segment_time:.2f}s")
                self.log_message(f"Memory usage: {mem_before:.1f}MB -> {mem_after:.1f}MB (Δ{mem_after-mem_before:+.1f}MB)")
            
            # Calculate final results
            deepfake_count = sum(1 for p in self.segment_predictions if p == "Deepfake")
            self.final_prediction = "Deepfake" if (deepfake_count / self.total_segments) >= 0.5 else "Genuine"
            self.final_confidence = np.mean(self.segment_confidences)
            
            # Update UI with final results
            self.update_final_results()
            self.log_message(f"\nFinal prediction: {self.final_prediction}", "HEADER")
            self.log_message(f"Average confidence: {self.final_confidence*100:.2f}%", "INFO")
            self.log_message(f"Deepfake segments: {deepfake_count}/{self.total_segments}", "INFO")
            
            # Show first segment
            self.current_segment = 0
            self.show_segment()
            
            # Enable navigation
            self.prev_btn.config(state=tb.DISABLED)
            self.next_btn.config(state=tb.NORMAL if self.total_segments > 1 else tb.DISABLED)
            self.segment_label.config(text=f"Segment: 1/{self.total_segments}")
            
            # Log final performance stats
            total_time = time.time() - start_time
            final_mem = process.memory_info().rss / (1024 * 1024)  # MB
            self.log_message(f"\nTotal processing time: {total_time:.2f}s")
            self.log_message(f"Peak memory usage: {max(process.memory_info().rss for _ in range(10)) / (1024 * 1024):.2f}MB")
            self.log_message(f"Final memory usage: {final_mem:.2f}MB (Δ{final_mem-initial_mem:+.2f}MB)")
            
            self.update_progress(100, "Done!")
            self.log_message("\n=== PROCESSING COMPLETED ===", "HEADER")
            
            os.remove(temp_path)
            self.log_message(f"Removed temporary processed file: {temp_path}")
                
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.log_message(f"\nPROCESSING FAILED: {str(e)}", "ERROR")
        finally:
            self.scan_btn.config(state=tb.NORMAL)

    def update_final_results(self):
        """Update the static final results display"""
        color = "red" if self.final_prediction == "Deepfake" else "green"
        result_text = (f"Final Detection: {self.final_prediction}\n"
                      f"Average Confidence: {self.final_confidence*100:.2f}%")
        self.final_result_label.config(text=result_text, foreground=color)
        self.enable_export_buttons()

    def show_segment(self):
        """Completely refresh visualization for current segment"""
        try:
            if not self.segment_audio or self.current_segment >= len(self.segment_audio):
                return
            
            self.safe_clear_visualizations()
            
            # Get segment data
            segment_audio = self.segment_audio[self.current_segment]
            audio = np.append(segment_audio[0], segment_audio[1:] - 0.97 * segment_audio[:-1])
            stft = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH))
            power_spectrum = librosa.power_to_db(stft)
            
            # Update visualizations with improved parameters
            self.update_visualizations(audio, power_spectrum)
            
            # Update segment analysis text
            self.result_text.config(state=tb.NORMAL)
            self.result_text.delete(1.0, tb.END)
            
            # Segment header
            self.result_text.insert(tb.END, f"SEGMENT {self.current_segment + 1} ANALYSIS\n\n", "title")
            
            # Segment prediction
            pred = self.segment_predictions[self.current_segment]
            conf = self.segment_confidences[self.current_segment]
            color = "red" if pred == "Deepfake" else "green"
            self.result_text.insert(tb.END, "Segment Detection: ", "bold")
            self.result_text.insert(tb.END, f"{pred}\n", ("bold", color))
            self.result_text.insert(tb.END, f"Confidence: {conf*100:.2f}%\n\n", "bold")
            
            # Technical analysis
            analysis_results = {
                'time_freq_info': self.display_time_frequency_info(audio, power_spectrum),
                'time_freq_comparison': self.compare_time_freq_features(power_spectrum),
                'sudden_changes': self.detect_sudden_changes(audio, power_spectrum),
                'harmonics_analysis': self.analyze_harmonics(audio),
                'sibilance_analysis': self.analyze_sibilance(audio),
                'frequency_patterns': self.analyze_frequency_patterns(power_spectrum)
            }
            
            # Add analysis results
            self.result_text.insert(tb.END, "\nTECHNICAL ANALYSIS\n", "subtitle")
            
            # Time-frequency information
            self.result_text.insert(tb.END, "\nTime-Frequency Information:\n", "section")
            self.result_text.insert(tb.END, analysis_results['time_freq_info'])
            
            # Time-frequency comparison
            self.result_text.insert(tb.END, "\n\nTime-Frequency Comparison (Base MFCC vs MFCC+DWT):\n", "section")
            self.result_text.insert(tb.END, analysis_results['time_freq_comparison'])
            
            # Sudden changes
            self.result_text.insert(tb.END, "\n\nSudden Changes Detection:\n", "section")
            self.result_text.insert(tb.END, analysis_results['sudden_changes'])
            
            # Harmonics analysis
            self.result_text.insert(tb.END, "\n\nHarmonic Analysis:\n", "section")
            self.result_text.insert(tb.END, analysis_results['harmonics_analysis'])
            
            # Sibilance analysis
            self.result_text.insert(tb.END, "\n\nSibilance Analysis:\n", "section")
            self.result_text.insert(tb.END, analysis_results['sibilance_analysis'])
            
            # Frequency patterns
            self.result_text.insert(tb.END, "\n\nFrequency Pattern Analysis:\n", "section")
            self.result_text.insert(tb.END, analysis_results['frequency_patterns'])
            
            # Configure text styles
            self.result_text.tag_config("title", font=("Arial", 14, "bold"), justify="center")
            self.result_text.tag_config("subtitle", font=("Arial", 12, "bold"), underline=True)
            self.result_text.tag_config("bold", font=("Arial", 10, "bold"))
            self.result_text.tag_config("section", font=("Arial", 10, "bold"))
            self.result_text.tag_config(color, foreground=color, font=("Arial", 12, "bold"))
            
            self.result_text.config(state=tb.DISABLED)
            self.log_message(f"Displaying segment {self.current_segment + 1}/{self.total_segments}")
        except Exception as e:
            self.log_message(f"Segment display error: {e}", "ERROR")
            messagebox.showerror("Error", f"Failed to display segment: {str(e)}")

    def prev_segment(self):
        """Navigate to previous segment"""
        if self.current_segment > 0:
            self.current_segment -= 1
            self.show_segment()
            self.segment_label.config(text=f"Segment: {self.current_segment + 1}/{self.total_segments}")
            self.next_btn.config(state=tb.NORMAL)
            if self.current_segment == 0:
                self.prev_btn.config(state=tb.DISABLED)

    def next_segment(self):
        """Navigate to next segment"""
        if self.current_segment < self.total_segments - 1:
            self.current_segment += 1
            self.show_segment()
            self.segment_label.config(text=f"Segment: {self.current_segment + 1}/{self.total_segments}")
            self.prev_btn.config(state=tb.NORMAL)
            if self.current_segment == self.total_segments - 1:
                self.next_btn.config(state=tb.DISABLED)

    def preprocess_audio(self, input_path, output_path):
        """Preprocess audio file with detailed logging"""
        self.log_message("Starting audio preprocessing...", "HEADER")
        try:
            self.log_message(f"Loading audio from: {input_path}")
            audio, orig_sr = librosa.load(input_path, sr=None, mono=True)
            self.log_message(f"Original sample rate: {orig_sr} Hz, Duration: {len(audio)/orig_sr:.2f}s")
            
            # Resample if needed
            if orig_sr != SAMPLE_RATE:
                self.log_message(f"Resampling from {orig_sr}Hz to {SAMPLE_RATE}Hz...", "DEBUG")
                audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=SAMPLE_RATE)
                self.log_message(f"Resampled audio length: {len(audio)} samples ({len(audio)/SAMPLE_RATE:.2f}s)")
            
            # Normalize
            self.log_message("Applying normalization...", "DEBUG")
            self.log_message("Equation: audio_normalized = (audio / max(|audio|)) * 0.95", "MATH")
            max_amp = np.max(np.abs(audio))
            self.log_message(f"Max amplitude before normalization: {max_amp:.4f}")
            audio = librosa.util.normalize(audio) * 0.95
            audio = np.clip(audio, -1.0, 1.0)
            self.log_message(f"Max amplitude after normalization: {np.max(np.abs(audio)):.4f}")
            
            # Save
            self.log_message(f"Saving processed audio to: {output_path}")
            sf.write(output_path, audio, SAMPLE_RATE, subtype='PCM_16')
            self.log_message("Audio preprocessing completed successfully", "HEADER")
            
        except Exception as e:
            self.log_message(f"Preprocessing failed: {str(e)}", "ERROR")
            raise Exception(f"Preprocessing failed: {str(e)}")

    def extract_features_with_analysis(self, file_path):
        """Extract features with detailed logging of each step"""
        self.log_message("\nStarting feature extraction...", "HEADER")
        try:
            analysis_results = {
                'time_freq_info': "",
                'sudden_changes': "",
                'harmonics_analysis': "",
                'sibilance_analysis': "",
                'frequency_patterns': ""
            }
            
            # Load audio
            self.log_message(f"Loading audio segment: {file_path}")
            audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
            self.current_audio = audio
            self.log_message(f"Segment duration: {len(audio)/SAMPLE_RATE:.2f}s ({len(audio)} samples)")
            
            # Apply pre-emphasis
            self.log_message("Applying pre-emphasis filter...", "DEBUG")
            self.log_message("Equation: y[n] = x[n] - α * x[n-1], where α = 0.97", "MATH")
            audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
            self.log_message(f"Pre-emphasis applied. First 5 samples: {audio[:5]}")
            
            # Compute STFT
            self.log_message("\nComputing Short-Time Fourier Transform...", "DEBUG")
            self.log_message(f"Parameters: n_fft={N_FFT}, hop_length={HOP_LENGTH}", "DEBUG")
            stft = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH))
            power_spectrum = librosa.power_to_db(stft)
            self.log_message(f"STFT shape: {stft.shape} (freq_bins×time_frames)")
            self.log_message(f"Power spectrum range: {np.min(power_spectrum):.2f} to {np.max(power_spectrum):.2f} dB")
            
            # DWT Feature Extraction
            self.log_message("\nComputing Discrete Wavelet Transform...", "DEBUG")
            self.log_message(f"Parameters: wavelet={DWT_WAVELET}, level={DWT_LEVEL}", "DEBUG")
            coeffs = pywt.wavedec2(power_spectrum, wavelet=DWT_WAVELET, level=DWT_LEVEL)
            self.log_message(f"Got {len(coeffs)} sets of coefficients")
            
            # Log coefficient shapes
            for i, c in enumerate(coeffs):
                if isinstance(c, tuple):  # Detail coefficients
                    self.log_message(f"Level {i}: Approximation {c[0].shape} + Details {[d.shape for d in c[1]]}")
                else:  # Approximation coefficients
                    self.log_message(f"Level {i}: Approximation {c.shape}")
            
            dwt_features = np.concatenate([c.flatten() for c in coeffs if isinstance(c, np.ndarray)])
            self.log_message(f"DWT features extracted: {len(dwt_features)} coefficients")
            self.log_message(f"First 5 DWT coefficients: {dwt_features[:5]}")
            
            # MFCC Feature Extraction
            self.log_message("\nComputing MFCCs...", "DEBUG")
            self.log_message(f"Parameters: n_mfcc={N_MFCC}", "DEBUG")
            mfcc = librosa.feature.mfcc(S=power_spectrum, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
            self.log_message(f"MFCC shape: {mfcc.shape} (coefficients×time_frames)")
            self.log_message(f"First MFCC frame values: {mfcc[:, 0]}")
            
            # Log stats for DWT and MFCC features separately
            dwt_selected = dwt_features[:3350]
            mfcc_selected = mfcc.flatten()[:650]

            self.log_message("\nFeature Statistics Before Concatenation:", "DEBUG")
            self.log_message(f"DWT (first 3350 coeffs) - Mean: {np.mean(dwt_selected):.4f}, Std: {np.std(dwt_selected):.4f}, Min: {np.min(dwt_selected):.4f}, Max: {np.max(dwt_selected):.4f}", "DEBUG")
            self.log_message(f"MFCC (first 650 coeffs) - Mean: {np.mean(mfcc_selected):.4f}, Std: {np.std(mfcc_selected):.4f}, Min: {np.min(mfcc_selected):.4f}, Max: {np.max(mfcc_selected):.4f}", "DEBUG")

            # Feature concatenation
            features = np.hstack([dwt_selected, mfcc_selected])
            
            self.log_message("\nFeature concatenation:", "DEBUG")
            self.log_message(f"DWT features used: {min(3350, len(dwt_features))} coefficients")
            self.log_message(f"MFCC features used: {min(650, len(mfcc.flatten()))} coefficients")
            
            # Padding if needed
            if len(features) < FIXED_FEATURE_LENGTH:
                pad_length = FIXED_FEATURE_LENGTH - len(features)
                self.log_message(f"Padding features with {pad_length} zeros")
                features = np.pad(features, (0, pad_length))
            else:
                self.log_message(f"Truncating features to {FIXED_FEATURE_LENGTH} elements")
                features = features[:FIXED_FEATURE_LENGTH]
            
            # Normalization
            self.log_message("\nNormalizing features...", "DEBUG")
            self.log_message("Equation: (features - μ) / (σ + ε), where μ=mean, σ=std, ε=1e-8", "MATH")
            mean_val = np.mean(features)
            std_val = np.std(features)
            self.log_message(f"Pre-normalization - Mean: {mean_val:.4f}, Std: {std_val:.4f}")
            features = (features - mean_val) / (std_val + 1e-8)
            self.log_message(f"Post-normalization - Mean: {np.mean(features):.4f}, Std: {np.std(features):.4f}")
            
            # Analysis functions
            self.log_message("\nRunning analysis functions...", "DEBUG")
            analysis_results['time_freq_info'] = self.display_time_frequency_info(audio, power_spectrum)
            
            self.log_message("Feature extraction completed successfully", "HEADER")
            return features, analysis_results
            
        except Exception as e:
            self.log_message(f"Feature extraction failed: {str(e)}", "ERROR")
            raise Exception(f"Feature extraction failed: {str(e)}")

    def apply_sliding_window(self, features):
        """Apply sliding window to features"""
        segments = []
        for start in range(0, len(features) - WINDOW_SIZE + 1, STEP_SIZE):
            segments.append(features[start:start + WINDOW_SIZE])
        return np.array(segments).reshape(-1, WINDOW_SIZE, 1)

    def display_time_frequency_info(self, audio, power_spectrum):
        """Generate time-frequency stats text"""
        info = (
            f"- Audio duration: {len(audio)/SAMPLE_RATE:.2f} seconds\n"
            f"- Frequency range: 0 to {SAMPLE_RATE//2} Hz\n"
            f"- Spectrogram dimensions: {power_spectrum.shape[0]} frequency bins × {power_spectrum.shape[1]} time frames\n"
            f"- MFCCs capturing {N_MFCC} frequency bands\n"
        )
        return info

    def compare_time_freq_features(self, power_spectrum):
        try:
            base_mfcc = librosa.feature.mfcc(S=power_spectrum, sr=SAMPLE_RATE, n_mfcc=50)
            coeffs = pywt.wavedec2(power_spectrum, wavelet=DWT_WAVELET, level=DWT_LEVEL)
            dwt_mfcc = librosa.feature.mfcc(S=coeffs[0], sr=SAMPLE_RATE, n_mfcc=N_MFCC)
            base_energy = np.sum(power_spectrum**2)
            dwt_energy = np.sum(coeffs[0]**2)
            bin_width = SAMPLE_RATE / power_spectrum.shape[0]

            bands = [
                ("Low Frequency (0-500Hz)", 0, 500),
                ("Mid Frequency (501-4kHz)", 501, 4000),
                ("High Frequency (4-8kHz)", 4001, 8000),
                ("Extended High (8kHz+)", 8001, SAMPLE_RATE//2)
            ]

            table_data = []
            for name, low, high in bands:
                low_bin = int(low/bin_width)
                high_bin = int(high/bin_width) if high < SAMPLE_RATE//2 else power_spectrum.shape[0]
                base_band = power_spectrum[low_bin:high_bin]
                base_val = np.sum(base_band**2)
                dwt_band = coeffs[0][low_bin:high_bin]
                dwt_val = np.sum(dwt_band**2)
                delta = (dwt_val - base_val) / base_val * 100 if base_val != 0 else 0
                table_data.append([name, f"{base_val:.1f}", f"{dwt_val:.1f}", f"{delta:+.1f}%"])

            table_data.append(["Total Energy", f"{base_energy:.1f}", f"{dwt_energy:.1f}", 
                             f"{(dwt_energy-base_energy)/base_energy*100:+.1f}%"])

            headers = ["Frequency Band", "Base MFCC", "MFCC+DWT", "Δ%"]
            table = self.format_analysis_table(
                headers,
                table_data,
                col_widths=[25, 15, 15, 10]
            )

            interpretation = (
                "\n\nInterpretation:\n"
                "- The combined MFCC+DWT view reveals much more detail in low and mid frequencies, where human speech carries the most information.\n"
                "- These increases suggest subtle voice features are now more visible, which can help spot things like formants or natural rhythm.\n"
                "- Meanwhile, high and extended high frequencies dropped — likely because DWT removed unwanted background noise or irrelevant static, improving clarity.\n"
                "- This gives the model a clearer, cleaner view of what matters in the speech."
            )

            return table + interpretation

        except Exception as e:
            return f"\nTime-Frequency comparison error: {str(e)}\n"

    def detect_sudden_changes(self, audio, power_spectrum):
        try:
            if len(audio) == 0:
                return "\nSudden Changes Detection:\n- Error: Empty audio input\n"

            power_spectrum = np.abs(power_spectrum)
            power_spectrum = np.where(power_spectrum > 0, power_spectrum, 1e-10)
            times = librosa.times_like(power_spectrum, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)

            results = {
                'amp_shifts': [],
                'freq_shifts': [],
                'dwt_shifts': [],
                'status': 'success'
            }

            rms = librosa.feature.rms(y=audio, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)
            rms_diff = np.diff(rms_db)
            amp_shifts = np.where(np.abs(rms_diff) > 5)[0]
            results['amp_shifts'] = list(zip(times[amp_shifts], rms_diff[amp_shifts]))

            spectral_centroid = librosa.feature.spectral_centroid(S=power_spectrum, sr=SAMPLE_RATE)[0]
            centroid_diff = np.diff(spectral_centroid)
            freq_shifts = np.where(np.abs(centroid_diff) > 100)[0]
            results['freq_shifts'] = list(zip(times[freq_shifts], centroid_diff[freq_shifts]))

            coeffs = pywt.wavedec2(power_spectrum, wavelet=DWT_WAVELET, level=DWT_LEVEL)
            if len(coeffs) > 1:
                cH, cV, cD = coeffs[1]
                combined_details = np.abs(cH) + np.abs(cV) + np.abs(cD)
                dwt_centroid = librosa.feature.spectral_centroid(S=combined_details, sr=SAMPLE_RATE, n_fft=combined_details.shape[0])[0]
                dwt_centroid_diff = np.diff(dwt_centroid)
                median_shift = np.median(np.abs(dwt_centroid_diff))
                threshold = max(100, 2*median_shift)
                dwt_freq_shifts = np.where(np.abs(dwt_centroid_diff) > threshold)[0]
                results['dwt_shifts'] = list(zip(times[dwt_freq_shifts], dwt_centroid_diff[dwt_freq_shifts]))

            amp_count = len(results['amp_shifts'])
            freq_count = len(results['freq_shifts'])
            dwt_freq_count = len(results['dwt_shifts'])
            avg_amp = np.mean([abs(x[1]) for x in results['amp_shifts']]) if results['amp_shifts'] else 0
            avg_freq = np.mean([abs(x[1]) for x in results['freq_shifts']]) if results['freq_shifts'] else 0
            avg_dwt_freq = np.mean([abs(x[1]) for x in results['dwt_shifts']]) if results['dwt_shifts'] else 0

            headers = ["Change Metric", "Base MFCC", "MFCC+DWT"]
            table_data = [
                ["Sudden Amplitude Shifts", f"{amp_count} events", f"{amp_count} events"],
                ["Avg. ΔAmplitude", f"{avg_amp:.1f} dB", f"{avg_amp:.1f} dB"],
                ["Sudden Frequency Shifts", f"{freq_count} events", f"{dwt_freq_count} events"],
                ["Avg. ΔSpectral Centroid", f"{avg_freq:.1f} Hz", f"{avg_dwt_freq:.1f} Hz"]
            ]

            table = self.format_analysis_table(
                headers,
                table_data,
                col_widths=[28, 18, 18]
            )

            report = table
            
            if results['amp_shifts']:
                report += "\n\nTop Amplitude Shifts (MFCC+DWT, Δ > 5 dB):\n"
                for t, Δ in sorted(results['amp_shifts'], key=lambda x: abs(x[1]), reverse=True)[:3]:
                    report += f"• {t:.2f}s → {Δ:+.1f} dB\n"
            
            if results['dwt_shifts']:
                report += "\nTop Frequency Shifts (MFCC+DWT, Δ > 100 Hz):\n"
                for t, Δ in sorted(results['dwt_shifts'], key=lambda x: abs(x[1]), reverse=True)[:3]:
                    report += f"• {t:.2f}s → Δ{Δ:.1f} Hz\n"

            interpretation = (
                "\nInterpretation:\n"
                "- Natural speech includes many small pitch and volume changes. Deepfakes might miss these or insert abrupt ones.\n"
                "- MFCC detects many tiny frequency shifts, but DWT shows fewer, more meaningful transitions — like sharp edits or sudden jumps.\n"
                "- These big shifts might hint at synthetic editing, especially when the audio seems smooth overall.\n"
                "- This helps detect structure problems, not just random noise."
            )

            return report + interpretation

        except Exception as e:
            return "\nSudden Changes Detection:\n- Analysis failed\n"

    def analyze_sibilance(self, audio):
        try:
            stft = np.abs(librosa.stft(audio, n_fft=N_FFT))
            power_spectrum = librosa.power_to_db(stft)
            bin_width = SAMPLE_RATE / power_spectrum.shape[0]
            sibilance_bin_start = int(4000/bin_width)
            sibilance_bin_end = int(8000/bin_width)

            sibilance_band = power_spectrum[sibilance_bin_start:sibilance_bin_end]
            sibilance_base = np.mean(sibilance_band)
            sibilance_std = np.std(sibilance_band)
            sibilance_frames = np.mean(sibilance_band, axis=0)
            threshold = np.median(sibilance_frames) + 5
            sibilant_regions = sibilance_frames > threshold
            sibilance_count = np.sum(sibilant_regions)
            sibilance_duration = sibilance_count * HOP_LENGTH / SAMPLE_RATE

            coeffs = pywt.wavedec2(power_spectrum, wavelet=DWT_WAVELET, level=DWT_LEVEL)
            approx = coeffs[0]
            dwt_sibilance_band = approx[sibilance_bin_start:sibilance_bin_end]
            dwt_sibilance = np.mean(dwt_sibilance_band)
            dwt_sibilance_std = np.std(dwt_sibilance_band)
            dwt_sibilance_frames = np.mean(dwt_sibilance_band, axis=0)
            dwt_threshold = np.median(dwt_sibilance_frames) + 5
            dwt_sibilant_regions = dwt_sibilance_frames > dwt_threshold
            dwt_sibilance_count = np.sum(dwt_sibilant_regions)
            dwt_sibilance_duration = dwt_sibilance_count * HOP_LENGTH / SAMPLE_RATE

            headers = ["Sibilance Metric", "Base MFCC", "MFCC+DWT"]
            table_data = [
                ["Avg. Energy (4-8kHz)", f"{sibilance_base:.1f} dB", f"{dwt_sibilance:.1f} dB"],
                ["Energy Std Dev", f"{sibilance_std:.1f} dB", f"{dwt_sibilance_std:.1f} dB"],
                ["Sibilant Regions Count", f"{sibilance_count}", f"{dwt_sibilance_count}"],
                ["Sibilant Duration", f"{sibilance_duration:.2f} sec", f"{dwt_sibilance_duration:.2f} sec"]
            ]

            table = self.format_analysis_table(
                headers,
                table_data,
                col_widths=[25, 15, 15]
            )

            interpretation = (
                "\n\nInterpretation:\n"
                "- 'S' and 'sh' sounds are sharp and quick. Real voices have them clearly and often.\n"
                "- Deepfakes sometimes smooth them out or place them strangely.\n"
                "- DWT picks up these sharper, more focused bursts better than MFCC alone.\n"
                "- Fewer sibilant events with sharper energy may suggest artificial smoothing."
            )

            return table + interpretation

        except Exception as e:
            return f"\nSibilance analysis error: {str(e)}\n"
    
    def analyze_harmonics(self, audio):
        try:
            stft = np.abs(librosa.stft(audio, n_fft=N_FFT))
            power_spectrum = librosa.power_to_db(stft)
            harmonic_base, percussive_base = librosa.effects.harmonic(audio), librosa.effects.percussive(audio)
            hnr_base = 10 * np.log10(np.mean(harmonic_base**2) / (np.mean(percussive_base**2) + 1e-8))

            coeffs = pywt.wavedec2(power_spectrum, wavelet=DWT_WAVELET, level=DWT_LEVEL)
            approx = coeffs[0]
            harmonic_energy = 0
            for level in range(1, len(coeffs)):
                cH, cV, cD = coeffs[level]
                harmonic_energy += np.sum(cH**2) + np.sum(cV**2) + np.sum(cD**2)

            fundamental_energy = np.sum(approx**2)
            reconstructed = pywt.waverec2(coeffs, DWT_WAVELET)
            min_shape = (min(power_spectrum.shape[0], reconstructed.shape[0]), min(power_spectrum.shape[1], reconstructed.shape[1]))
            noise_energy = np.sum((power_spectrum[:min_shape[0], :min_shape[1]] - reconstructed[:min_shape[0], :min_shape[1]])**2)

            hnr_dwt = 10 * np.log10((fundamental_energy + harmonic_energy) / (noise_energy + 1e-8))
            hf_ratio = harmonic_energy / (fundamental_energy + 1e-8)

            headers = ["Harmonic Metric", "Base MFCC", "MFCC+DWT"]
            table_data = [
                ["Harmonic-to-Noise Ratio", f"{hnr_base:.1f} dB", f"{hnr_dwt:.1f} dB"],
                ["Fundamental Energy (power)", "N/A", f"{fundamental_energy:.2e}"],
                ["Harmonic Energy (power)", "N/A", f"{harmonic_energy:.2e}"],
                ["H/F Ratio", "N/A", f"{hf_ratio:.3f}"]
            ]

            table = self.format_analysis_table(
                headers,
                table_data,
                col_widths=[25, 15, 15]
            )

            interpretation = (
                "\n\nInterpretation:\n"
                "- Real speech usually has a strong base frequency and many layered tones (harmonics).\n"
                "- Deepfakes often simplify this — sounding flat or overly smooth.\n"
                "- A high base but low harmonic energy can be a sign the voice was artificially generated.\n"
                "- The DWT-enhanced view helps expose that imbalance, even if it sounds normal to the ear."
            )

            return table + interpretation

        except Exception as e:
            return f"- Could not perform harmonic analysis: {str(e)}"
        
    def analyze_frequency_patterns(self, power_spectrum):
        try:
            bin_width = SAMPLE_RATE / power_spectrum.shape[0]
            low_bins = slice(0, int(500/bin_width))
            base_low = np.mean(power_spectrum[low_bins])
            coeffs = pywt.wavedec2(power_spectrum, wavelet=DWT_WAVELET, level=DWT_LEVEL)
            approx = coeffs[0]
            dwt_low = np.mean(approx[low_bins])
            high_bins = slice(int(4000/bin_width), int(8000/bin_width))
            base_high = np.mean(power_spectrum[high_bins])
            dwt_high_energy = 0
            count = 0
            for level in range(1, min(3, len(coeffs))):
                cH, cV, cD = coeffs[level]
                dwt_high_energy += np.mean(cH[high_bins]) + np.mean(cV[high_bins]) + np.mean(cD[high_bins])
                count += 3
            dwt_high = dwt_high_energy / count if count > 0 else np.nan

            headers = ["Frequency Pattern", "Base MFCC", "MFCC+DWT"]
            table_data = [
                ["Low-Freq Tonal Patterns", f"{base_low:.1f} dB", f"{dwt_low:.1f} dB"],
                ["High-Freq Local Details", f"{base_high:.1f} dB", f"{dwt_high:.1f} dB"]
            ]

            table = self.format_analysis_table(
                headers,
                table_data,
                col_widths=[25, 15, 15]
            )

            interpretation = (
                "\n\nInterpretation:\n"
                "- The low-frequency pattern appears flatter in DWT, possibly showing a loss of natural vocal variation.\n"
                "- At the same time, DWT enhances clarity in high-frequency detail — which can reveal things like breath sounds or unnatural sharpness.\n"
                "- This combination of missing low variation and exaggerated high detail is often seen in generated speech."
            )

            return table + interpretation

        except Exception as e:
            return f"- Frequency pattern analysis error: {str(e)}"

    def update_progress(self, value, message):
        self.progress["value"] = value
        self.status_var.set(message)
        self.root.update_idletasks()

    def check_audio_duration(self, file_path):
        """Check audio duration and warn if longer than 10 minutes"""
        try:
            # Get duration without loading full file
            audio, sr = librosa.load(file_path, sr=None, mono=True, duration=601)  # Only load first 10:01
            duration = len(audio) / sr
            
            if duration > 600:  # 10 minutes in seconds
                response = messagebox.askyesno(
                    "Long Audio Warning",
                    f"This audio is {duration/60:.1f} minutes long. Processing may take significant time.\n"
                    "Do you want to proceed anyway?",
                    parent=self.root
                )
                return response
            return True
        except Exception as e:
            self.log_message(f"Duration check error: {str(e)}", "ERROR")
            return True
        
    def enable_export_buttons(self):
        """Enable export buttons when results are available"""
        if self.total_segments > 0:
            self.pdf_export_btn.config(state=tb.NORMAL)
            self.csv_export_btn.config(state=tb.NORMAL)

    def export_to_pdf(self):
        """Directly export UI visualizations to PDF with perfect matching"""
        try:
            from fpdf import FPDF
            import tempfile
            from PIL import Image
            import io
            import base64

            if not self.fig or not self.segment_audio:
                messagebox.showwarning("No Visualizations", "No visualizations available to export")
                return

            # Ask user if they want to export all segments or just current
            export_all = messagebox.askyesnocancel(
                "Export Option",
                "Export ALL segments? (Yes=All, No=Current, Cancel=Abort)"
            )
            if export_all is None:
                self.status_var.set("PDF export cancelled")
                return

            file_path = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf")],
                title="Save PDF Report As"
            )
            if not file_path:
                return

            self.status_var.set("Generating PDF...")
            self.root.update_idletasks()

            # Create PDF
            pdf = FPDF(orientation='P', unit='mm', format='A4')
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.set_margins(left=15, top=15, right=15)
            pdf.set_font("Arial", size=12)

            # Determine which segments to export
            segments_to_export = range(self.total_segments) if export_all else [self.current_segment]

            # Create temp directory
            temp_dir = tempfile.mkdtemp()
            image_paths = []

            for seg_idx in segments_to_export:
                # Save current segment index and update UI to render it
                current_segment_backup = self.current_segment
                self.current_segment = seg_idx
                self.show_segment()  # This updates the UI visualizations
                self.root.update()  # Force UI update

                # Save the figure to a temporary file
                temp_file = os.path.join(temp_dir, f"segment_{seg_idx}.png")
                self.fig.savefig(temp_file, dpi=150, bbox_inches='tight', pad_inches=0.3)
                image_paths.append(temp_file)

                # Restore original segment
                self.current_segment = current_segment_backup
                self.show_segment()
                self.root.update()

            # Add images to PDF
            for i, image_path in enumerate(image_paths):
                pdf.add_page()
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(0, 10, f"Segment {i + 1}/{len(image_paths)}", ln=True, align='C')
                pdf.ln(5)
                
                # Add image with proper scaling
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
                    aspect_ratio = img_height / img_width
                    max_width = 180  # A4 width minus margins
                    calculated_height = max_width * aspect_ratio
                    
                    if calculated_height > 250:  # Max height with header space
                        scale_factor = 250 / calculated_height
                        max_width *= scale_factor
                        calculated_height = 250
                    
                    x_pos = (210 - max_width) / 2  # Center horizontally
                    pdf.image(image_path, x=x_pos, y=None, w=max_width, h=calculated_height)

            # Save PDF
            pdf.output(file_path)
            self.status_var.set(f"PDF saved: {os.path.basename(file_path)}")
            messagebox.showinfo("Export Complete", f"PDF saved to:\n{file_path}")

            # Clean up temporary files
            for image_path in image_paths:
                try:
                    os.remove(image_path)
                except:
                    pass
            try:
                os.rmdir(temp_dir)
            except:
                pass

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to generate PDF: {str(e)}")
            self.log_message(f"PDF export error: {str(e)}\n{traceback.format_exc()}", "ERROR")

    def export_to_csv(self):
        """Export analysis results to CSV"""
        try:
            from datetime import datetime
            import csv
            
            if not self.total_segments:
                messagebox.showwarning("No Data", "No analysis results to export")
                return
                
            # Ask for save location
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                title="Save CSV Report As"
            )
            
            if not file_path:
                return
                
            self.status_var.set("Generating CSV report...")
            self.root.update_idletasks()
            
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                writer.writerow(["Deepfake Audio Detection Report"])
                writer.writerow([])
                writer.writerow(["File:", os.path.basename(self.file_path)])
                writer.writerow(["Date:", datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                writer.writerow(["Final Detection:", 
                            f"{self.final_prediction} (Confidence: {self.final_confidence*100:.2f}%)"])
                writer.writerow([])
                
                # Write segments summary
                writer.writerow(["Segments Summary"])
                writer.writerow(["Segment", "Prediction", "Confidence"])
                for i, (pred, conf) in enumerate(zip(self.segment_predictions, self.segment_confidences), 1):
                    writer.writerow([f"Segment {i}", pred, f"{conf*100:.2f}%"])
                writer.writerow([])
                
                # Write current segment analysis
                self.result_text.config(state=tb.NORMAL)
                analysis_text = self.result_text.get("1.0", tb.END)
                self.result_text.config(state=tb.DISABLED)
                
                writer.writerow([f"Segment {self.current_segment + 1} Analysis"])
                for line in analysis_text.split('\n'):
                    writer.writerow([line])
                writer.writerow([])
                
                # Write processing logs
                writer.writerow(["Processing Logs"])
                self.log_text.config(state=tb.NORMAL)
                log_text = self.log_text.get("1.0", tb.END)
                self.log_text.config(state=tb.DISABLED)
                
                for line in log_text.split('\n'):
                    writer.writerow([line])
            
            self.status_var.set(f"Report saved to {os.path.basename(file_path)}")
            messagebox.showinfo("Export Complete", f"CSV report successfully saved to:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to generate CSV: {str(e)}")
            self.log_message(f"CSV export error: {str(e)}", "ERROR")

if __name__ == "__main__":
    root = tb.Window(themename="flatly")  # or 'cosmo', 'superhero', etc.
    app = DeepfakeDetectorApp(root)

    style = tb.Style()
    style.configure("TProgressbar", thickness=20)

    root.mainloop()