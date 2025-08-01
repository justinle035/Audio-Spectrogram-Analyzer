# **Audio Signal Analyzer \- Python Code Documentation**

## **1\. Overview**

This document provides a detailed explanation of the Python code for the "Audio Signal Analyzer" application. The application is a desktop GUI tool built with Tkinter that allows users to:

* Load audio files in various formats (WAV, MP3, etc.).  
* Record audio directly from a microphone.  
* Visualize the audio's frequency content over time as a spectrogram.  
* Interactively select time and frequency ranges to zoom into specific parts of the spectrogram.

The core functionalities rely on libraries such as Librosa for audio loading, SciPy for signal processing (spectrogram generation), PyAudio for microphone input, NumPy for numerical operations, and Matplotlib for plotting, all integrated within a Tkinter GUI.

## **2\. Prerequisites**

To run this application, you need:

* **Python 3.8+**  
* **Libraries**:  
  * tkinter (usually included with Python)  
  * matplotlib  
  * numpy  
  * librosa  
  * scipy  
  * pyaudio  
* **FFmpeg**: An external dependency required by Librosa for handling compressed audio formats (like MP3) and audio from video files. FFmpeg must be installed and its bin directory added to your system's PATH environment variable.

You can install the Python libraries using pip:  
```python 
pip install matplotlib numpy librosa scipy pyaudio
```
## **3\. Code Structure**

The Python script is organized into the following main sections:

1. **Imports**: Importing all necessary libraries.  
2. **Utility Functions**: Standalone functions for specific tasks:  
   * load\_audio\_file(): Handles loading audio from a file.  
   * record\_audio\_pyaudio(): Handles recording audio from the microphone.  
   * generate\_spectrogram\_data(): Computes the spectrogram from audio data.  
3. **Main Application Class (AudioAnalyzerApp)**: Encapsulates the entire GUI, its logic, and state.  
   * Constructor (\_\_init\_\_)  
   * GUI widget creation methods (\_create\_control\_widgets, \_create\_plot\_canvas)  
   * Callback methods for user interactions (e.g., \_load\_audio\_file\_dialog, \_record\_audio, \_update\_plot\_with\_filters)  
   * Input parsing and validation methods (\_parse\_time\_string, \_validate\_inputs)  
   * Plotting methods (\_initialize\_plot, \_plot\_spectrogram)  
4. **Main Execution Block**: Code to instantiate and run the application when the script is executed.

## **4\. Detailed Code Explanation**

### **4.1. Imports**
```python
import tkinter as tk  
from tkinter import filedialog, messagebox, ttk  
import matplotlib.pyplot as plt  
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  
import numpy as np  
import librosa  
from scipy import signal  
import pyaudio  
import os 
```
* **tkinter (as tk)**: The primary library for creating the graphical user interface.  
  * **filedialog**: Provides pre-built dialogs for opening and saving files.  
  * **messagebox**: Used for displaying standard dialog boxes (errors, warnings, info).  
  * **ttk**: Themed Tkinter widgets, offering a more modern look than classic Tk widgets.  
* **matplotlib.pyplot (as plt)**: The main plotting interface for Matplotlib.  
* **matplotlib.backends.backend\_tkagg**: Contains classes to embed Matplotlib plots in Tkinter applications.  
  * **FigureCanvasTkAgg**: Creates a Tkinter-compatible canvas for a Matplotlib figure.  
  * **NavigationToolbar2Tk**: Provides the standard Matplotlib navigation toolbar (zoom, pan, save) for the Tkinter canvas.  
* **numpy (as np)**: Fundamental package for numerical computation in Python, used here for handling audio data as arrays.  
* **librosa**: A powerful library for audio analysis. Used for loading audio files with robust format support and resampling capabilities.  
* **scipy.signal**: Part of the SciPy library, used for advanced signal processing tasks, specifically signal.spectrogram for generating spectrograms.  
* **pyaudio**: Provides Python bindings for PortAudio, enabling audio recording and playback.  
* **os**: Standard library for interacting with the operating system, used here for path manipulations like os.path.basename().

### **4.2. Utility Functions**

These functions perform specific, reusable tasks related to audio processing.

#### **4.2.1. load\_audio\_file(filepath)**
```python
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

```
* **Purpose**: Loads an audio file from the specified filepath.  
* **Parameters**:  
  * filepath (str): The path to the audio file.  
* **Logic**:  
  * Uses librosa.load(filepath, sr=None, mono=True):  
    * sr=None: Loads the audio at its original sampling rate.  
    * mono=True: Converts the audio to a single (mono) channel, simplifying subsequent processing. Librosa returns audio data as a NumPy array of floats, typically normalized between \-1.0 and 1.0.  
  * **Error Handling**:  
    * Catches FileNotFoundError if the file does not exist.  
    * Catches generic Exception for other loading issues (e.g., unsupported format, FFmpeg missing for MP3/MOV files) and displays an informative error message.  
* **Returns**:  
  * A tuple (audio\_data, sample\_rate) on success:  
    * audio\_data (NumPy array): The audio samples.  
    * sample\_rate (int): The sampling rate of the audio in Hz.  
  * (None, None) if loading fails.

#### **4.2.2. record\_audio\_pyaudio(duration\_seconds=5, sample\_rate=44100, chunk\_size=1024)**
```python
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
```
* **Purpose**: Records audio from the default microphone.  
* **Parameters**:  
  * duration\_seconds (int): The duration of the recording in seconds (default: 5).  
  * sample\_rate (int): The desired sampling rate in Hz (default: 44100).  
  * chunk\_size (int): The number of audio frames to read from the buffer at a time (default: 1024).  
* **Logic**:  
  1. **Initialization**:  
     * audio\_format \= pyaudio.paInt16: Sets the sample format to 16-bit signed integers.  
     * channels \= 1: Specifies mono recording.  
     * p \= pyaudio.PyAudio(): Creates a PyAudio instance.  
  2. **Stream Opening**:  
     * p.open(...): Opens an input audio stream from the microphone with the specified parameters. frames\_per\_buffer=chunk\_size defines how many samples are read in each stream.read() call.  
  3. **Recording Loop**:  
     * Reads audio data in chunks using stream.read(chunk\_size, exception\_on\_overflow=False).  
       * exception\_on\_overflow=False: If Python is too slow to read the buffer, an overflow might occur. This setting prevents the program from crashing by discarding overflowing frames (some audio data might be lost).  
     * Appends each chunk (raw bytes) to the frames list.  
  4. **Resource Cleanup (finally block)**: This block ensures that the PyAudio stream is stopped and closed, and the PyAudio instance is terminated, even if errors occur during recording. This is crucial for releasing microphone access.  
  5. **Data Conversion**:  
     * b''.join(frames): Concatenates all recorded byte chunks.  
     * np.frombuffer(...): Converts the raw byte string into a NumPy array of np.int16 samples.  
     * .astype(np.float32) / 32768.0: Converts the 16-bit integer samples to 32-bit floating-point samples and normalizes them to the range \[-1.0, 1.0\]. 32768.0 is 2^15, the maximum absolute value for a 16-bit signed integer.  
* **Returns**:  
  * A tuple (audio\_data\_float, sample\_rate) on success.  
  * (None, None) if recording fails or no frames are captured.

#### **4.2.3. generate\_spectrogram\_data(audio\_data, sample\_rate, fft\_window\_size\_samples=1024, overlap\_samples=None)**
```python
def generate_spectrogram_data(audio_data, sample_rate, fft_window_size_samples=1024, overlap_samples=None):
    print(f"Generating spectrogram for audio of shape {audio_data.shape if audio_data is not None else 'None'} with SR {sample_rate}")
    if audio_data is None or len(audio_data) == 0:
        print("No audio data provided to generate_spectrogram_data.")
        return np.array([]), np.array([]), np.array([[]]) 

    if len(audio_data) < fft_window_size_samples :
        print(f"Not enough audio data (len: {len(audio_data)}) for FFT window size ({fft_window_size_samples}).")
        return np.array([]), np.array([]), np.array([[]])

    if overlap_samples is None:
        overlap_samples = fft_window_size_samples // 2
    
    print(f"Using FFT window: {fft_window_size_samples}, Overlap: {overlap_samples}")
    try:
        frequencies, times, Sxx = signal.spectrogram(
            audio_data, fs=sample_rate, window='hann',
            nperseg=fft_window_size_samples, noverlap=overlap_samples,
            nfft=fft_window_size_samples, scaling='density' 
        )
        Sxx_db = 10 * np.log10(Sxx + 1e-9) # Add epsilon to avoid log(0)
        print(f"Spectrogram generated: Freqs shape {frequencies.shape}, Times shape {times.shape}, Sxx_db shape {Sxx_db.shape}")
        return frequencies, times, Sxx_db
    except ValueError as e:
        messagebox.showwarning("Spectrogram Error", f"Could not generate spectrogram: {e}")
        print(f"ValueError in signal.spectrogram: {e}")
        return np.array([]), np.array([]), np.array([[]])
```
* **Purpose**: Computes the Short-Time Fourier Transform (STFT) of the audio data to produce a spectrogram.  
* **Parameters**:  
  * audio\_data (NumPy array): The 1D array of audio samples.  
  * sample\_rate (int): The sampling rate of the audio.  
  * fft\_window\_size\_samples (int): The number of samples per segment for the FFT (default: 1024). This is nperseg in scipy.signal.spectrogram.  
  * overlap\_samples (int, optional): The number of samples to overlap between segments. Defaults to half the fft\_window\_size\_samples. This is noverlap.  
* **Logic**:  
  1. **Input Validation**: Checks if audio\_data is valid and has enough samples for at least one FFT window.  
  2. **scipy.signal.spectrogram()**:  
     * window='hann': Applies a Hann window to each segment before FFT to reduce spectral leakage.  
     * nperseg: Length of each segment (window size).  
     * noverlap: Number of overlapping samples between segments.  
     * nfft: Length of the FFT. Setting it to nperseg is common.  
     * scaling='density': Computes the power spectral density (PSD), typically units of V²/Hz. 'spectrum' would yield power spectrum (V²).  
  3. **Conversion to Decibels (dB)**:  
     * Sxx\_db \= 10 \* np.log10(Sxx \+ 1e-9): Converts the power values (Sxx) to a logarithmic scale (dB). The \+ 1e-9 (epsilon) prevents log10(0) errors for silent portions.  
* **Returns**:  
  * A tuple (frequencies, times, Sxx\_db) on success:  
    * frequencies (1D NumPy array): Array of sample frequencies.  
    * times (1D NumPy array): Array of segment times.  
    * Sxx\_db (2D NumPy array): The spectrogram data in dB, where rows are frequencies and columns are time segments.  
  * (np.array(\[\]), np.array(\[\]), np.array(\[\[\]\])) (empty arrays with appropriate dimensions) if generation fails.

### **4.3. Main Application Class: AudioAnalyzerApp**

This class defines the structure and behavior of the GUI application.

#### **4.3.1. \_\_init\_\_(self, root\_window) \- Constructor**
```python
class AudioAnalyzerApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Audio Signal Analyzer")
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
```
* **Purpose**: Initializes the main application window and its core components.  
* **Parameters**:  
  * root\_window (tk.Tk): The main Tkinter window instance.  
* **Logic**:  
  * Sets the window title and initial dimensions (self.root.geometry(...)).  
  * **Initializes State Variables**:  
    * self.full\_audio\_data, self.full\_sample\_rate: Store the currently loaded/recorded audio. Initialized to None.  
    * self.start\_time\_sec, self.end\_time\_sec, self.min\_freq\_hz, self.max\_freq\_hz: Store user-defined ranges for filtering the spectrogram view.  
    * self.colorbar: Stores a reference to the Matplotlib colorbar object, so it can be removed when the plot is updated.  
  * **Creates GUI Frames**:  
    * self.control\_frame (ttk.LabelFrame): A container for user input widgets (buttons, entry fields). It's packed at the top and fills horizontally.  
    * self.plot\_frame (ttk.LabelFrame): A container for the Matplotlib spectrogram plot. It's packed below the control frame and expands to fill the remaining window space.  
  * Calls internal methods \_create\_control\_widgets() and \_create\_plot\_canvas() to populate these frames with their respective UI elements.

#### **4.3.2. \_create\_control\_widgets(self)**
```python
    def _create_control_widgets(self):
        # Row 0: File Loading
        load_file_button = ttk.Button(self.control_frame, text="Load Audio File", command=self._load_audio_file_dialog)
        load_file_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.loaded_file_label = ttk.Label(self.control_frame, text="No file loaded.", width=50, anchor="w")
        self.loaded_file_label.grid(row=0, column=1, columnspan=3, padx=5, pady=5, sticky="ew")

        # Row 1: Microphone Recording
        record_button = ttk.Button(self.control_frame, text="Record from Mic (5s)", command=self._record_audio)
        record_button.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        # Row 2: Time Range Inputs
        ttk.Label(self.control_frame, text="Time Range (s or M:S.s):").grid(row=2, column=0, padx=5, pady=2, sticky="e")
        self.time_start_entry = ttk.Entry(self.control_frame, width=10)
        self.time_start_entry.grid(row=2, column=1, padx=(0,2), pady=2, sticky="w")
        self.time_start_entry.insert(0, "0.00")
        ttk.Label(self.control_frame, text="to").grid(row=2, column=2, padx=2, pady=2, sticky="ew")
        self.time_end_entry = ttk.Entry(self.control_frame, width=10)
        self.time_end_entry.grid(row=2, column=3, padx=(0,5), pady=2, sticky="w")
        
        # Row 3: Frequency Range Inputs
        ttk.Label(self.control_frame, text="Freq Range (Hz):").grid(row=3, column=0, padx=5, pady=2, sticky="e")
        self.freq_min_entry = ttk.Entry(self.control_frame, width=10)
        self.freq_min_entry.grid(row=3, column=1, padx=(0,2), pady=2, sticky="w")
        self.freq_min_entry.insert(0, "0")
        ttk.Label(self.control_frame, text="to").grid(row=3, column=2, padx=2, pady=2, sticky="ew")
        self.freq_max_entry = ttk.Entry(self.control_frame, width=10)
        self.freq_max_entry.grid(row=3, column=3, padx=(0,5), pady=2, sticky="w")

        # Row 4: Apply Filter Button
        apply_button = ttk.Button(self.control_frame, text="Apply Filter & Update Plot", command=self._update_plot_with_filters)
        apply_button.grid(row=4, column=0, columnspan=4, padx=5, pady=10, sticky="ew")

        # Configure column weights for better resizing behavior
        self.control_frame.grid_columnconfigure(0, weight=1)
        self.control_frame.grid_columnconfigure(1, weight=2) 
        self.control_frame.grid_columnconfigure(2, weight=0) 
        self.control_frame.grid_columnconfigure(3, weight=2)
```
* **Purpose**: Creates and arranges all interactive UI elements (widgets) within the self.control\_frame.  
* **Layout**: Uses the .grid() layout manager for precise placement of widgets in rows and columns.  
* **Widgets Created**:  
  * **Load File Button**: ttk.Button that calls self.\_load\_audio\_file\_dialog when clicked.  
  * **Loaded File Label**: ttk.Label (self.loaded\_file\_label) to display the name of the currently loaded file or status messages. anchor="w" aligns text to the west (left).  
  * **Record Button**: ttk.Button that calls self.\_record\_audio.  
  * **Time Range Inputs**:  
    * Labels ("Time Range:", "to").  
    * ttk.Entry fields (self.time\_start\_entry, self.time\_end\_entry) for users to input start and end times. Default values are inserted.  
  * **Frequency Range Inputs**:  
    * Labels ("Freq Range (Hz):", "to").  
    * ttk.Entry fields (self.freq\_min\_entry, self.freq\_max\_entry) for frequency limits. Default values are inserted.  
  * **Apply Button**: ttk.Button that calls self.\_update\_plot\_with\_filters to process the current time/frequency inputs and refresh the spectrogram.  
* **sticky="ew" / sticky="e" / sticky="w"**: Controls how widgets expand or align within their grid cells if the cell is larger than the widget. "ew" means stretch horizontally (east-west).  
* **grid\_columnconfigure(..., weight=...)**: Configures how columns in the grid expand when the window is resized. Columns with higher weights get more of the extra space.

#### **4.3.3. \_create\_plot\_canvas(self)**
```python
    def _create_plot_canvas(self):
        self.fig, self.ax = plt.subplots()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame, pack_toolbar=False) 
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X) 
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self._initialize_plot() 
        self.canvas.draw()
```
* **Purpose**: Sets up the Matplotlib plotting area within the self.plot\_frame.  
* **Logic**:  
  1. self.fig, self.ax \= plt.subplots(): Creates a Matplotlib figure (self.fig) and a single subplot/axes object (self.ax) on that figure. All plotting will happen on self.ax.  
  2. self.canvas \= FigureCanvasTkAgg(self.fig, master=self.plot\_frame): Embeds the Matplotlib figure (self.fig) into a Tkinter-compatible canvas, making self.plot\_frame its parent.  
  3. self.canvas\_widget \= self.canvas.get\_tk\_widget(): Retrieves the underlying Tkinter widget from the FigureCanvasTkAgg object. This widget is what will be packed into the Tkinter layout.  
  4. self.toolbar \= NavigationToolbar2Tk(...): Creates the standard Matplotlib navigation toolbar (for zoom, pan, save plot, etc.) associated with the self.canvas. pack\_toolbar=False allows for manual placement using Tkinter's layout managers.  
  5. self.toolbar.update(): An essential step to initialize the toolbar.  
  6. self.toolbar.pack(...) and self.canvas\_widget.pack(...): Uses Tkinter's pack layout manager. The toolbar is packed at the bottom of self.plot\_frame, and the canvas widget is packed above it, set to fill all available space (fill=tk.BOTH, expand=True).  
  7. self.\_initialize\_plot(): Calls a helper method to draw an initial empty state for the plot.  
  8. self.canvas.draw(): Redraws the canvas to display the initial plot.

#### **4.3.4. \_initialize\_plot(self)**
```python
    def _initialize_plot(self):
        self.ax.clear()
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Frequency (Hz)")
        self.ax.set_title("Spectrogram - Load Audio or Record")
        if self.colorbar: 
            try:
                self.colorbar.remove()
            except Exception as e:
                print(f"Error removing colorbar during initialization (can often be ignored): {e}")
            self.colorbar = None 
        self.canvas.draw_idle() 
```
* **Purpose**: Clears the plot area and sets it to a default empty state.  
* **Logic**:  
  * self.ax.clear(): Removes all artists (lines, images, text, etc.) from the current axes (self.ax).  
  * Sets default X-axis label, Y-axis label, and title for the plot.  
  * **Colorbar Removal**: If self.colorbar (a reference to a previously drawn colorbar) exists, it calls self.colorbar.remove() to remove it from the figure and then sets self.colorbar \= None. This is crucial to prevent multiple colorbars from appearing when the plot is updated.  
  * self.canvas.draw\_idle(): Schedules a redraw of the canvas. draw\_idle is generally preferred over draw() in callbacks as it can be more efficient by coalescing multiple pending draw requests.

#### **4.3.5. \_load\_audio\_file\_dialog(self)**
```python
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
                
                self._update_plot_with_filters()
            else:
                self.loaded_file_label.config(text="Failed to load file.")
                self._initialize_plot() 
```
* **Purpose**: Handles the action when the "Load Audio File" button is clicked.  
* **Logic**:  
  1. filedialog.askopenfilename(...): Opens a system file dialog, allowing the user to browse and select a file. The filetypes argument provides filters for common audio and video formats.  
  2. If a filepath is selected (i.e., the user didn't cancel the dialog):  
     * Calls the utility function load\_audio\_file(filepath) to load the audio data.  
     * If loading is successful (self.full\_audio\_data is not None):  
       * Updates self.loaded\_file\_label to display the name of the loaded file.  
       * Calculates the duration of the audio.  
       * Clears and sets the default values in the time and frequency entry fields to span the entire loaded audio (time from 0 to duration, frequency from 0 to Nyquist frequency, which is sample\_rate / 2).  
       * Calls self.\_update\_plot\_with\_filters() to immediately display the spectrogram of the newly loaded audio.  
     * If loading fails, updates self.loaded\_file\_label with an error message and calls self.\_initialize\_plot() to clear any existing plot.

#### **4.3.6. \_record\_audio(self)**
```python
    def _record_audio(self):
        self.loaded_file_label.config(text="Recording...")
        self.root.update_idletasks() # Force Tkinter to update the GUI now

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
```
* **Purpose**: Handles the action when the "Record from Mic" button is clicked.  
* **Logic**:  
  1. Updates self.loaded\_file\_label to "Recording...".  
  2. self.root.update\_idletasks(): Forces Tkinter to process pending GUI events. This ensures the "Recording..." label is displayed *before* the potentially blocking record\_audio\_pyaudio() call starts.  
  3. Calls the utility function record\_audio\_pyaudio() to perform the recording.  
  4. Similar to file loading, if recording is successful:  
     * Updates self.loaded\_file\_label.  
     * Sets default time and frequency ranges in the entry fields based on the recorded audio's duration and sample rate.  
     * Calls self.\_update\_plot\_with\_filters() to display the spectrogram of the recorded audio.  
  5. If recording fails, updates the label and resets the plot.

#### **4.3.7. \_parse\_time\_string(self, time\_str)**
```python
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
```
* **Purpose**: Converts user-entered time strings (from the time range entry fields) into a total number of seconds (float).  
* **Parameters**:  
  * time\_str (str): The time string to parse.  
* **Logic**:  
  * Checks if the string contains a colon (:).  
  * If colons are present, splits the string and attempts to parse it as MM:SS.ss or HH:MM:SS.ss.  
  * If no colons, attempts to parse it directly as seconds.  
  * Uses float() to convert parts to numbers, allowing for fractional seconds.  
  * Catches ValueError if conversion to float fails (e.g., non-numeric input).  
  * Displays a warning message via messagebox.showwarning for invalid formats or non-numeric components.  
* **Returns**:  
  * The total time in seconds (float) on successful parsing.  
  * None if parsing fails.

#### **4.3.8. \_validate\_inputs(self)**
```python
    def _validate_inputs(self):
        if self.full_audio_data is None or self.full_sample_rate is None:
            messagebox.showerror("Error", "No audio loaded or recorded.")
            return False 
        
        total_duration = len(self.full_audio_data) / self.full_sample_rate
        nyquist_freq = self.full_sample_rate / 2

        start_time_str = self.time_start_entry.get()
        self.start_time_sec = self._parse_time_string(start_time_str)
        if self.start_time_sec is None or not (0 <= self.start_time_sec <= total_duration):
            messagebox.showerror("Input Error", f"Invalid start time. Must be a number between 0.00 and {total_duration:.2f}s.")
            return False

        end_time_str = self.time_end_entry.get()
        self.end_time_sec = self._parse_time_string(end_time_str)
        if self.end_time_sec is None or not (self.start_time_sec < self.end_time_sec <= total_duration + 0.001): 
            messagebox.showerror("Input Error",
                                 f"Invalid end time. Must be > start time ({self.start_time_sec:.2f}s) "
                                 f"and <= total duration ({total_duration:.2f}s).")
            return False
        
        try:
            self.min_freq_hz = float(self.freq_min_entry.get())
            if not (0 <= self.min_freq_hz <= nyquist_freq):
                raise ValueError("Min frequency out of valid range [0, Nyquist]")
        except ValueError:
            messagebox.showerror("Input Error",
                                 f"Invalid min frequency. Must be numeric and between 0 and {nyquist_freq:.0f} Hz.")
            return False

        try:
            self.max_freq_hz = float(self.freq_max_entry.get())
            if not (self.min_freq_hz < self.max_freq_hz <= nyquist_freq + 0.1): 
                 messagebox.showerror("Input Error",
                                      f"Invalid max frequency. Must be > min frequency ({self.min_freq_hz:.0f}Hz) "
                                      f"and <= Nyquist frequency ({nyquist_freq:.0f} Hz).")
                 return False
        except ValueError: 
            messagebox.showerror("Input Error", "Max frequency must be numeric.")
            return False
            
        return True 
```
* **Purpose**: Validates the time and frequency range values entered by the user in the GUI's entry fields.  
* **Logic**:  
  1. Checks if audio data (self.full\_audio\_data) has been loaded or recorded. If not, shows an error and returns False.  
  2. Calculates total\_duration of the audio and nyquist\_freq (sample\_rate / 2).  
  3. **Start Time Validation**:  
     * Gets the string from self.time\_start\_entry.  
     * Parses it using self.\_parse\_time\_string().  
     * Checks if the parsed time is valid (numeric, non-negative, and not exceeding total\_duration). Updates self.start\_time\_sec.  
  4. **End Time Validation**:  
     * Similar parsing and validation for self.time\_end\_entry.  
     * Ensures end time is greater than start time and within total\_duration. A small epsilon (0.001) is added to total\_duration for robust floating-point comparison. Updates self.end\_time\_sec.  
  5. **Min Frequency Validation**:  
     * Gets string from self.freq\_min\_entry, converts to float.  
     * Checks if it's numeric, non-negative, and not exceeding nyquist\_freq. Updates self.min\_freq\_hz.  
  6. **Max Frequency Validation**:  
     * Similar conversion for self.freq\_max\_entry.  
     * Ensures it's numeric, greater than self.min\_freq\_hz, and not exceeding nyquist\_freq. An epsilon (0.1) is used for Nyquist comparison. Updates self.max\_freq\_hz.  
  * Uses messagebox.showerror() to display specific error messages to the user if validation fails.  
* **Returns**:  
  * True if all inputs are valid.  
  * False if any input is invalid.

#### **4.3.9. \_update\_plot\_with\_filters(self)**
```python
    def _update_plot_with_filters(self):
        print("Updating plot with filters...")
        if not self._validate_inputs():
            print("Input validation failed.")
            return 

        print(f"Validated inputs: Time {self.start_time_sec:.2f}-{self.end_time_sec:.2f}s, Freq {self.min_freq_hz:.0f}-{self.max_freq_hz:.0f}Hz")

        start_sample = int(self.start_time_sec * self.full_sample_rate)
        end_sample = int(self.end_time_sec * self.full_sample_rate)
        
        start_sample = max(0, start_sample)
        end_sample = min(len(self.full_audio_data), end_sample) # Corrected variable name

        if start_sample >= end_sample: 
            messagebox.showwarning("Plot Info", "Start time is after or at end time. No audio segment to plot.")
            self._plot_spectrogram(np.array([]), np.array([]), np.array([[]]), title_suffix=" - Invalid Time Segment")
            return

        current_audio_segment = self.full_audio_data[start_sample:end_sample]
        print(f"Current audio segment shape: {current_audio_segment.shape}")
        
        if len(current_audio_segment) == 0: 
            messagebox.showwarning("Plot Info", "Selected audio segment is empty.")
            self._plot_spectrogram(np.array([]), np.array([]), np.array([[]]), title_suffix=" - Empty Segment")
            return

        fft_win_size = 1024  
        overlap = fft_win_size // 2 
        
        spec_freqs, spec_times_relative, current_Sxx_db = generate_spectrogram_data(
            current_audio_segment, self.full_sample_rate,
            fft_window_size_samples=fft_win_size, overlap_samples=overlap
        )

        plot_title_suffix = (f"(Time: {self.start_time_sec:.2f}-{self.end_time_sec:.2f}s, "
                             f"Freq: {self.min_freq_hz:.0f}-{self.max_freq_hz:.0f}Hz)")

        if current_Sxx_db.size == 0 or spec_freqs.size == 0 or spec_times_relative.size == 0:
            print("Spectrogram data is empty after generation for the current segment.")
            self._plot_spectrogram(np.array([]), np.array([]), np.array([[]]), title_suffix=plot_title_suffix + " - No Spectrogram Data")
            return
        
        freq_indices_to_plot_tuple = np.where((spec_freqs >= self.min_freq_hz) & (spec_freqs <= self.max_freq_hz))
        freq_indices_to_plot = freq_indices_to_plot_tuple[0] 

        if freq_indices_to_plot.size == 0: 
            print("No frequencies fall within the selected display range.")
            filtered_Sxx_db_for_plot = np.array([[]]) 
            filtered_frequencies_for_plot = np.array([])
        else:
            filtered_Sxx_db_for_plot = current_Sxx_db[freq_indices_to_plot, :]
            filtered_frequencies_for_plot = spec_freqs[freq_indices_to_plot]
        
        print(f"Filtered Sxx_db shape: {filtered_Sxx_db_for_plot.shape}, Filtered freqs shape: {filtered_frequencies_for_plot.shape}")
        
        absolute_spec_times = spec_times_relative + self.start_time_sec
        
        self._plot_spectrogram(filtered_frequencies_for_plot, absolute_spec_times, filtered_Sxx_db_for_plot, title_suffix=plot_title_suffix)
```
*Correction Note*: In the original code provided in the prompt for \_update\_plot\_with\_filters, there was a typo: min(len(self.full\_audio\_\_data), end\_sample). It should be min(len(self.full\_audio\_data), end\_sample). This version of the documentation assumes the corrected variable name.

* **Purpose**: This is the main callback method triggered when the "Apply Filter & Update Plot" button is clicked or after new audio is loaded/recorded. It processes the audio based on the user's time/frequency selections and updates the spectrogram display.  
* **Logic**:  
  1. **Validate Inputs**: Calls self.\_validate\_inputs(). If validation fails, it stops.  
  2. **Time Slicing**:  
     * Converts the validated self.start\_time\_sec and self.end\_time\_sec (which were updated by \_validate\_inputs) into audio sample indices (start\_sample, end\_sample).  
     * Slices the self.full\_audio\_data to get current\_audio\_segment corresponding to the selected time range. Boundary checks (max(0, ...), min(len(...), ...)) ensure indices are valid.  
     * Checks if the resulting segment is valid (e.g., not zero-length).  
  3. **Generate Spectrogram**: Calls generate\_spectrogram\_data() with the current\_audio\_segment to compute its spectrogram. FFT parameters (fft\_win\_size, overlap) are currently hardcoded.  
  4. **Handle Empty Spectrogram**: If generate\_spectrogram\_data returns empty data (e.g., segment too short), it calls \_plot\_spectrogram with empty arrays to clear the plot and show a "No Data" message.  
  5. **Frequency Filtering (on Spectrogram Data)**:  
     * np.where(...): Finds the indices of the frequency bins (spec\_freqs from the spectrogram) that fall within the user-specified self.min\_freq\_hz and self.max\_freq\_hz.  
     * Uses these freq\_indices\_to\_plot to select the relevant rows from current\_Sxx\_db (the 2D spectrogram power data) and the corresponding spec\_freqs. This effectively "filters" the spectrogram data for display.  
     * If no frequencies match the range, empty arrays are prepared for plotting.  
  6. **Adjust Time Axis**: The spec\_times\_relative from generate\_spectrogram\_data are relative to the start of current\_audio\_segment. This line, absolute\_spec\_times \= spec\_times\_relative \+ self.start\_time\_sec, converts these to absolute time values (relative to the beginning of the original full audio). This makes the X-axis of the plot display the correct time segment from the original audio.  
  7. **Plot**: Calls self.\_plot\_spectrogram() with the (potentially filtered) filtered\_frequencies\_for\_plot, absolute\_spec\_times, and filtered\_Sxx\_db\_for\_plot, along with a plot\_title\_suffix indicating the current filter ranges.

#### **4.3.10. \_plot\_spectrogram(self, frequencies, times, Sxx\_db, title\_suffix="")**
```python
    def _plot_spectrogram(self, frequencies, times, Sxx_db, title_suffix=""):
        print(f"Plotting spectrogram. Freqs shape: {frequencies.shape}, Times shape: {times.shape}, Sxx_db shape: {Sxx_db.shape}")
        
        self.ax.clear() 
        if self.colorbar: 
            try:
                self.colorbar.remove()
            except Exception as e:
                 print(f"Minor error removing colorbar (can be ignored): {e}")
            self.colorbar = None 

        valid_data_for_plot = (
            frequencies.ndim == 1 and frequencies.size > 0 and
            times.ndim == 1 and times.size > 0 and
            Sxx_db.ndim == 2 and
            Sxx_db.shape[0] == frequencies.shape[0] and
            Sxx_db.shape[1] == times.shape[0]
        )

        if not valid_data_for_plot:
            self.ax.set_title(f"Spectrogram - No valid data to display {title_suffix}")
            print(f"No valid data or shape mismatch for pcolormesh. "
                  f"Freq size: {frequencies.size}, Time size: {times.size}, Sxx_db shape: {Sxx_db.shape} {title_suffix}")
        else:
            try:
                vmin_val = np.percentile(Sxx_db, 1) 
                vmax_val = np.percentile(Sxx_db, 99)
                if vmin_val >= vmax_val : 
                    vmax_val = vmin_val + 10.0 
                    if vmin_val == vmax_val == 0.0: 
                         vmin_val = -10.0 

                pcm = self.ax.pcolormesh(times, frequencies, Sxx_db, cmap='viridis', shading='gouraud',
                                         vmin=vmin_val, vmax=vmax_val)
                
                self.colorbar = self.fig.colorbar(pcm, ax=self.ax, format='%+2.0f dB', pad=0.02)
                self.colorbar.set_label('Power/Frequency (dB/Hz)')
                
            except Exception as e: 
                self.ax.set_title(f"Error during plotting: {e} {title_suffix}")
                print(f"Exception during pcolormesh or colorbar: {e}")

        self.ax.set_ylabel('Frequency (Hz)')
        self.ax.set_xlabel('Time (s)')
        if not self.ax.get_title(): 
            self.ax.set_title(f"Spectrogram {title_suffix}")
        
        try:
            self.fig.tight_layout(pad=1.0) 
        except Exception as e:
            print(f"Error during fig.tight_layout: {e} (Plot might not be perfectly aligned)")

        self.canvas.draw_idle() 
        print("Plotting complete.")
```
* **Purpose**: Responsible for actually drawing the spectrogram on the Matplotlib canvas embedded in the Tkinter GUI.  
* **Parameters**:  
  * frequencies (1D NumPy array): The frequency values for the Y-axis.  
  * times (1D NumPy array): The time values for the X-axis.  
  * Sxx\_db (2D NumPy array): The spectrogram power data (in dB) to plot. Shape should be (len(frequencies), len(times)).  
  * title\_suffix (str, optional): Text to append to the plot title, usually indicating filter ranges.  
* **Logic**:  
  1. self.ax.clear(): Clears the axes of any previous plot.  
  2. **Colorbar Removal**: Removes the existing self.colorbar if present.  
  3. **Data Validation for pcolormesh**:  
     * Performs a rigorous check to ensure frequencies, times, and Sxx\_db are not empty and have compatible dimensions for self.ax.pcolormesh(). Sxx\_db must be 2D, frequencies and times 1D, and Sxx\_db.shape must be (len(frequencies), len(times)).  
  4. If data is invalid or shapes mismatch, sets a title indicating "No valid data" or "Shape mismatch."  
  5. If data is valid:  
     * **Color Scaling (vmin\_val, vmax\_val)**: Calculates vmin and vmax using np.percentile(Sxx\_db, 1\) and np.percentile(Sxx\_db, 99). This makes the color mapping robust by ignoring the extreme 1% of data points at both ends, which helps in visualizing the main dynamic range of the spectrogram, especially if there are strong outliers or very flat data. It also ensures vmin \< vmax.  
     * self.ax.pcolormesh(times, frequencies, Sxx\_db, ...): Draws the spectrogram.  
       * cmap='viridis': A perceptually uniform and commonly used colormap.  
       * shading='gouraud': Interpolates colors between grid cells for a smoother appearance. 'flat' or 'auto' are alternatives.  
     * self.colorbar \= self.fig.colorbar(...): Adds a colorbar to the plot, linked to the pcm (PseudoColorMesh) object, and formats its labels to show dB values. The pad argument controls spacing.  
  6. Sets the Y-axis label, X-axis label, and the main plot title (if not already set by an error message).  
  7. self.fig.tight\_layout(pad=1.0): Attempts to adjust subplot parameters to provide a tight layout, preventing labels or titles from overlapping. Wrapped in try-except as it can occasionally fail.  
  8. self.canvas.draw\_idle(): Schedules a redraw of the Tkinter canvas to display the new spectrogram.

### **4.4. Main Execution Block**
```python
if __name__ == '__main__':
    # Forcing TkAgg backend might be necessary on some systems
    # import matplotlib
    # try:
    #     matplotlib.use('TkAgg')
    # except Exception as e:
    #     print(f"Could not set Matplotlib backend to TkAgg: {e}")

    root = tk.Tk()  # Create the main Tkinter window
    app = AudioAnalyzerApp(root)  # Instantiate our application class
    root.mainloop()  # Start the Tkinter event loop
```
* **Purpose**: This is the standard entry point for a Python script.  
* **Logic**:  
  * if \_\_name\_\_ \== '\_\_main\_\_':: Ensures this code block only runs when the script is executed directly (not when imported as a module).  
  * \# matplotlib.use('TkAgg'): This commented-out section shows how one might explicitly set the Matplotlib backend to TkAgg. This is sometimes necessary if Matplotlib defaults to a different backend that's incompatible with Tkinter, though FigureCanvasTkAgg usually handles this.  
  * root \= tk.Tk(): Creates the main, top-level Tkinter window.  
  * app \= AudioAnalyzerApp(root): Creates an instance of our AudioAnalyzerApp class, passing the root window to its constructor. This initializes the entire GUI.  
  * root.mainloop(): Starts the Tkinter event loop. This is a blocking call that keeps the GUI window open, listening for and responding to user events (mouse clicks, key presses, window resizing, etc.) until the window is closed by the user.

This detailed documentation should provide a thorough understanding of the Audio Signal Analyzer's codebase.
