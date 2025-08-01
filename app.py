import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import librosa
from scipy import signal
from scipy.fft import rfft, rfftfreq 
import pyaudio
import os

# --- Utility Functions ---
def load_audio_file(filepath):
    print(f"Attempting to load audio file: {filepath}")
    try:
        audio_data, sample_rate = librosa.load(filepath, sr=None, mono=True)
        print(f"Successfully loaded {os.path.basename(filepath)}, Sample Rate: {sample_rate}, Data Shape: {audio_data.shape}")
        return audio_data, sample_rate
    except FileNotFoundError:
        messagebox.showerror("Error", f"File not found: {filepath}")
        return None, None
    except Exception as e:
        messagebox.showerror("Loading Error", f"Could not load audio file {os.path.basename(filepath)}: {e}\n\nEnsure FFmpeg is installed and in your system's PATH for MP3/MOV support.")
        return None, None

def record_audio_pyaudio(duration_seconds=5, sample_rate=44100, chunk_size=1024):
    audio_format = pyaudio.paInt16
    channels = 1
    p = pyaudio.PyAudio()
    stream = None # Initialize stream to None for finally block
    print(f"Preparing to record for {duration_seconds} seconds at {sample_rate} Hz...")
    try:
        stream = p.open(format=audio_format, channels=channels, rate=sample_rate,
                        input=True, frames_per_buffer=chunk_size)
        frames = []
        print(f"Recording...")
        for _ in range(0, int(sample_rate / chunk_size * duration_seconds)):
            data = stream.read(chunk_size, exception_on_overflow=False)
            frames.append(data)
        print("Recording finished.")

    except IOError as e:
        messagebox.showerror("Recording Error", f"PyAudio IOError: {e}\nIs a microphone connected and enabled?")
        return None, None
    except Exception as e:
        messagebox.showerror("Recording Error", f"An unexpected error occurred during recording: {e}")
        return None, None
    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        p.terminate()
        print("PyAudio terminated.")

    if not frames:
        print("No frames recorded.")
        return None, None

    audio_data_raw = b''.join(frames)
    audio_data_int16 = np.frombuffer(audio_data_raw, dtype=np.int16)
    audio_data_float = audio_data_int16.astype(np.float32) / 32768.0 # Normalize to -1.0 to 1.0
    print(f"Recording processed. Sample Rate: {sample_rate}, Data Shape: {audio_data_float.shape}")
    return audio_data_float, sample_rate


def generate_spectrogram_data(audio_data, sample_rate, fft_window_size_samples=1024, overlap_samples=None):
    print(f"Generating spectrogram for audio of shape {audio_data.shape if audio_data is not None else 'None'} with SR {sample_rate}")
    if audio_data is None or len(audio_data) == 0:
        print("No audio data provided to generate_spectrogram_data.")
        return np.array([]), np.array([]), np.array([]) # Return empty arrays with consistent ndim for unpacking

    if len(audio_data) < fft_window_size_samples :
        print(f"Not enough audio data (len: {len(audio_data)}) for FFT window size ({fft_window_size_samples}).")
        return np.array([]), np.array([]), np.array([])

    if overlap_samples is None:
        overlap_samples = fft_window_size_samples // 2
    
    print(f"Using FFT window: {fft_window_size_samples}, Overlap: {overlap_samples}")
    try:
        frequencies, times, Sxx = signal.spectrogram(
            audio_data, fs=sample_rate, window='hann',
            nperseg=fft_window_size_samples, noverlap=overlap_samples,
            nfft=fft_window_size_samples, scaling='density' # 'spectrum' for V_rms, 'density' for V_rms^2/Hz
        )
        # Add a small epsilon to avoid log(0)
        Sxx_db = 10 * np.log10(Sxx + 1e-9)
        print(f"Spectrogram generated: Freqs shape {frequencies.shape}, Times shape {times.shape}, Sxx_db shape {Sxx_db.shape}")
        return frequencies, times, Sxx_db
    except ValueError as e:
        messagebox.showwarning("Spectrogram Error", f"Could not generate spectrogram: {e}")
        print(f"ValueError in signal.spectrogram: {e}")
        return np.array([]), np.array([]), np.array([])


# --- Main Application Class ---
class AudioAnalyzerApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Audio Signal Processor")
        self.root.geometry("1000x700")

        self.full_audio_data = None
        self.full_sample_rate = None
        
        self.start_time_sec = 0.0
        self.end_time_sec = 0.0
        self.min_freq_hz = 0.0
        self.max_freq_hz = 0.0
        self.colorbar = None # Initialize colorbar attribute

        self.control_frame = ttk.LabelFrame(self.root, text="Controls")
        self.control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5, ipady=5)

        self.plot_frame = ttk.LabelFrame(self.root, text="Spectrogram / Plot")
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        self._create_control_widgets()
        self._create_plot_canvas()

    def _create_control_widgets(self):
        load_file_button = ttk.Button(self.control_frame, text="Load Audio File", command=self._load_audio_file_dialog)
        load_file_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.loaded_file_label = ttk.Label(self.control_frame, text="No file loaded.", width=50, anchor="w")
        self.loaded_file_label.grid(row=0, column=1, columnspan=3, padx=5, pady=5, sticky="ew")

        record_button = ttk.Button(self.control_frame, text="Record from Mic (5s)", command=self._record_audio)
        record_button.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        ttk.Label(self.control_frame, text="Time Range (s or M:S.s):").grid(row=2, column=0, padx=5, pady=2, sticky="e")
        self.time_start_entry = ttk.Entry(self.control_frame, width=10)
        self.time_start_entry.grid(row=2, column=1, padx=(0,2), pady=2, sticky="w")
        self.time_start_entry.insert(0, "0.00")
        ttk.Label(self.control_frame, text="to").grid(row=2, column=2, padx=2, pady=2, sticky="ew")
        self.time_end_entry = ttk.Entry(self.control_frame, width=10)
        self.time_end_entry.grid(row=2, column=3, padx=(0,5), pady=2, sticky="w")
        
        ttk.Label(self.control_frame, text="Freq Range (Hz):").grid(row=3, column=0, padx=5, pady=2, sticky="e")
        self.freq_min_entry = ttk.Entry(self.control_frame, width=10)
        self.freq_min_entry.grid(row=3, column=1, padx=(0,2), pady=2, sticky="w")
        self.freq_min_entry.insert(0, "0")
        ttk.Label(self.control_frame, text="to").grid(row=3, column=2, padx=2, pady=2, sticky="ew")
        self.freq_max_entry = ttk.Entry(self.control_frame, width=10)
        self.freq_max_entry.grid(row=3, column=3, padx=(0,5), pady=2, sticky="w")

        apply_button = ttk.Button(self.control_frame, text="Apply Filter & Update Plot", command=self._update_plot_with_filters)
        apply_button.grid(row=4, column=0, columnspan=4, padx=5, pady=10, sticky="ew")

        self.control_frame.grid_columnconfigure(0, weight=1)
        self.control_frame.grid_columnconfigure(1, weight=2) # Give more space to entry
        self.control_frame.grid_columnconfigure(2, weight=0)
        self.control_frame.grid_columnconfigure(3, weight=2) # Give more space to entry


    def _create_plot_canvas(self):
        self.fig, self.ax = plt.subplots()
        # self.fig.set_tight_layout(True) # Call this after plotting and colorbar

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame, pack_toolbar=False)
        self.toolbar.update()
        # Pack toolbar first so it's at the bottom
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        # Then pack the canvas widget to fill the rest
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self._initialize_plot() # Draw an empty plot initially
        self.canvas.draw()

    def _initialize_plot(self):
        self.ax.clear()
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Frequency (Hz)")
        self.ax.set_title("Spectrogram - Load Audio or Record")
        if self.colorbar: # If a colorbar exists from a previous plot
            try:
                self.colorbar.remove()
            except Exception as e:
                print(f"Error removing colorbar: {e}") # Should not happen often
            self.colorbar = None
        self.canvas.draw_idle() # Use draw_idle for potentially better performance

    def _load_audio_file_dialog(self):
        filepath = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=(("Audio Files", "*.wav *.mp3 *.m4a *.flac *.ogg"),
                       ("Video Files (for audio)", "*.mov *.mp4 *.avi"),
                       ("All files", "*.*"))
        )
        if filepath:
            self.full_audio_data, self.full_sample_rate = load_audio_file(filepath)
            if self.full_audio_data is not None and self.full_sample_rate is not None:
                self.loaded_file_label.config(text=os.path.basename(filepath))
                duration = len(self.full_audio_data) / self.full_sample_rate
                self.time_start_entry.delete(0, tk.END)
                self.time_start_entry.insert(0, "0.00")
                self.time_end_entry.delete(0, tk.END)
                self.time_end_entry.insert(0, f"{duration:.2f}")
                
                self.freq_min_entry.delete(0, tk.END)
                self.freq_min_entry.insert(0, "0")
                self.freq_max_entry.delete(0, tk.END)
                self.freq_max_entry.insert(0, f"{self.full_sample_rate / 2:.0f}")
                
                self._update_plot_with_filters() # Plot full spectrogram
            else:
                self.loaded_file_label.config(text="Failed to load file.")
                self._initialize_plot() # Clear plot if loading failed

    def _record_audio(self):
        self.loaded_file_label.config(text="Recording...")
        self.root.update_idletasks()

        self.full_audio_data, self.full_sample_rate = record_audio_pyaudio()
        
        if self.full_audio_data is not None and self.full_sample_rate is not None:
            self.loaded_file_label.config(text="Microphone input recorded.")
            duration = len(self.full_audio_data) / self.full_sample_rate
            self.time_start_entry.delete(0, tk.END)
            self.time_start_entry.insert(0, "0.00")
            self.time_end_entry.delete(0, tk.END)
            self.time_end_entry.insert(0, f"{duration:.2f}")
            
            self.freq_min_entry.delete(0, tk.END)
            self.freq_min_entry.insert(0, "0")
            self.freq_max_entry.delete(0, tk.END)
            self.freq_max_entry.insert(0, f"{self.full_sample_rate / 2:.0f}")
            
            self._update_plot_with_filters()
        else:
            self.loaded_file_label.config(text="Recording failed or no audio captured.")
            self._initialize_plot()

    def _parse_time_string(self, time_str):
        try:
            if ':' in time_str:
                parts = time_str.split(':')
                if len(parts) == 2: # M:S.s or M:S
                    minutes = float(parts[0])
                    seconds = float(parts[1])
                    return minutes * 60 + seconds
                elif len(parts) == 3: # H:M:S.s or H:M:S
                    hours = float(parts[0])
                    minutes = float(parts[1])
                    seconds = float(parts[2])
                    return hours * 3600 + minutes * 60 + seconds
                else:
                    messagebox.showwarning("Time Parse Error", "Invalid time format. Use S.s, M:S.s, or H:M:S.s")
                    return None
            else: # Assume seconds
                return float(time_str)
        except ValueError:
            messagebox.showwarning("Time Parse Error", "Time components must be numeric.")
            return None

    def _validate_inputs(self):
        if self.full_audio_data is None or self.full_sample_rate is None:
            messagebox.showerror("Error", "No audio loaded or recorded.")
            return False
        
        total_duration = len(self.full_audio_data) / self.full_sample_rate
        nyquist_freq = self.full_sample_rate / 2

        start_time_str = self.time_start_entry.get()
        self.start_time_sec = self._parse_time_string(start_time_str)
        if self.start_time_sec is None or not (0 <= self.start_time_sec <= total_duration):
            messagebox.showerror("Input Error", f"Invalid start time. Must be between 0.00 and {total_duration:.2f}s.")
            return False

        end_time_str = self.time_end_entry.get()
        self.end_time_sec = self._parse_time_string(end_time_str)
        if self.end_time_sec is None or not (self.start_time_sec < self.end_time_sec <= total_duration + 0.001): # Add small epsilon for float compare
            messagebox.showerror("Input Error", f"Invalid end time. Must be > start time ({self.start_time_sec:.2f}s) and <= total duration ({total_duration:.2f}s).")
            return False
        
        try:
            self.min_freq_hz = float(self.freq_min_entry.get())
            if not (0 <= self.min_freq_hz <= nyquist_freq):
                raise ValueError("Min frequency out of range")
        except ValueError:
            messagebox.showerror("Input Error", f"Invalid min frequency. Must be numeric and between 0 and {nyquist_freq:.0f} Hz.")
            return False

        try:
            self.max_freq_hz = float(self.freq_max_entry.get())
            if not (self.min_freq_hz < self.max_freq_hz <= nyquist_freq + 0.1): # Add small epsilon for nyquist comparison
                 messagebox.showerror("Input Error", f"Invalid max frequency. Must be > min frequency ({self.min_freq_hz:.0f}Hz) and <= Nyquist ({nyquist_freq:.0f} Hz).")
                 return False
        except ValueError:
            messagebox.showerror("Input Error", "Max frequency must be numeric.")
            return False
            
        return True

    def _update_plot_with_filters(self):
        print("Updating plot with filters...")
        if not self._validate_inputs():
            print("Input validation failed.")
            return

        print(f"Validated inputs: Time {self.start_time_sec:.2f}-{self.end_time_sec:.2f}s, Freq {self.min_freq_hz:.0f}-{self.max_freq_hz:.0f}Hz")

        start_sample = int(self.start_time_sec * self.full_sample_rate)
        end_sample = int(self.end_time_sec * self.full_sample_rate)
        
        # Ensure slicing is within bounds
        start_sample = max(0, start_sample)
        end_sample = min(len(self.full_audio_data), end_sample)

        if start_sample >= end_sample:
            messagebox.showwarning("Plot Info", "Start sample is after or at end sample. No audio segment to plot.")
            self._plot_spectrogram(np.array([]), np.array([]), np.array([]), title_suffix=" - Invalid Time Segment")
            return

        current_audio_segment = self.full_audio_data[start_sample:end_sample]
        print(f"Current audio segment shape: {current_audio_segment.shape}")
        
        if len(current_audio_segment) == 0:
            messagebox.showwarning("Plot Info", "Selected audio segment is empty.")
            self._plot_spectrogram(np.array([]), np.array([]), np.array([]), title_suffix=" - Empty Segment")
            return

        fft_win_size = 1024 # Can make this configurable later
        overlap = fft_win_size // 2
        
        spec_freqs, spec_times_relative, current_Sxx_db = generate_spectrogram_data(
            current_audio_segment, self.full_sample_rate,
            fft_window_size_samples=fft_win_size, overlap_samples=overlap
        )

        plot_title_suffix = f"(Time: {self.start_time_sec:.2f}-{self.end_time_sec:.2f}s, Freq: {self.min_freq_hz:.0f}-{self.max_freq_hz:.0f}Hz)"

        if current_Sxx_db.size == 0 or spec_freqs.size == 0 or spec_times_relative.size == 0:
            print("Spectrogram data is empty after generation.")
            self._plot_spectrogram(np.array([]), np.array([]), np.array([]), title_suffix=plot_title_suffix + " - No Spectrogram Data")
            return
        
        # Filter by frequency
        # np.where returns a tuple of arrays; for 1D freqs, we need the first element
        freq_indices_to_plot_tuple = np.where((spec_freqs >= self.min_freq_hz) & (spec_freqs <= self.max_freq_hz))
        freq_indices_to_plot = freq_indices_to_plot_tuple[0]


        if freq_indices_to_plot.size == 0:
            print("No frequencies fall within the selected range.")
            filtered_Sxx_db_for_plot = np.array([]) # Empty 2D array
            filtered_frequencies_for_plot = np.array([])
        else:
            filtered_Sxx_db_for_plot = current_Sxx_db[freq_indices_to_plot, :]
            filtered_frequencies_for_plot = spec_freqs[freq_indices_to_plot]
        
        print(f"Filtered Sxx_db shape: {filtered_Sxx_db_for_plot.shape}, Filtered freqs shape: {filtered_frequencies_for_plot.shape}")
        
        # Adjust times to be absolute from the start of the original audio file
        absolute_spec_times = spec_times_relative + self.start_time_sec
        
        self._plot_spectrogram(filtered_frequencies_for_plot, absolute_spec_times, filtered_Sxx_db_for_plot, title_suffix=plot_title_suffix)

    def _plot_spectrogram(self, frequencies, times, Sxx_db, title_suffix=""):
        print(f"Plotting spectrogram. Freqs shape: {frequencies.shape}, Times shape: {times.shape}, Sxx_db shape: {Sxx_db.shape}")
        self.ax.clear()
        if self.colorbar:
            try:
                self.colorbar.remove()
            except Exception as e:
                 print(f"Minor error removing colorbar (can be ignored): {e}")
            self.colorbar = None

        # Check if there's valid data to plot
        if frequencies.size == 0 or times.size == 0 or Sxx_db.ndim != 2 or Sxx_db.shape[0] == 0 or Sxx_db.shape[1] == 0:
            self.ax.set_title(f"Spectrogram - No data to display {title_suffix}")
            print(f"No data to display. Freq size: {frequencies.size}, Time size: {times.size}, Sxx_db shape: {Sxx_db.shape}")
        # Check for shape consistency for pcolormesh
        elif Sxx_db.shape[0] == frequencies.shape[0] and Sxx_db.shape[1] == times.shape[0]:
            try:
                # Robust vmin/vmax to handle potential all-zero or very flat data after filtering
                vmin_val = np.percentile(Sxx_db, 1) if Sxx_db.size > 0 else -80
                vmax_val = np.percentile(Sxx_db, 99) if Sxx_db.size > 0 else -20
                if vmin_val >= vmax_val : # Ensure vmin < vmax
                    vmax_val = vmin_val + 10 

                pcm = self.ax.pcolormesh(times, frequencies, Sxx_db, cmap='viridis', shading='gouraud',
                                         vmin=vmin_val, vmax=vmax_val)
                self.colorbar = self.fig.colorbar(pcm, ax=self.ax, format='%+2.0f dB', pad=0.02)
                self.colorbar.set_label('Power/Frequency (dB/Hz)')
            except Exception as e:
                self.ax.set_title(f"Error during plotting: {e} {title_suffix}")
                print(f"Exception during pcolormesh or colorbar: {e}")
        else:
            self.ax.set_title(f"Spectrogram - Data shape mismatch {title_suffix}")
            print(f"Shape mismatch for pcolormesh: Sxx_db: {Sxx_db.shape}, freqs: {frequencies.shape}, times: {times.shape}")

        self.ax.set_ylabel('Frequency (Hz)')
        self.ax.set_xlabel('Time (s)')
        if not self.ax.get_title(): # Set title if not already set by an error message
            self.ax.set_title(f"Spectrogram {title_suffix}")
        
        try:
            self.fig.tight_layout(pad=1.0) # Smaller pad, adjust if needed
        except Exception as e:
            print(f"Error during tight_layout: {e}") # tight_layout can sometimes fail with complex plots

        self.canvas.draw_idle()
        print("Plotting complete.")

if __name__ == '__main__':
    # Forcing TkAgg backend might be necessary on some systems
    # import matplotlib
    # try:
    #     matplotlib.use('TkAgg')
    # except Exception as e:
    #     print(f"Could not set Matplotlib backend to TkAgg: {e}")

    root = tk.Tk()
    app = AudioAnalyzerApp(root)
    root.mainloop()
