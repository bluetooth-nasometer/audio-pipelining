import sounddevice as sd
import scipy.io.wavfile as wavfile 
import numpy as np
import threading
import time

def list_devices():
    """List all available audio input devices"""
    print("Available Audio Devices:")
    print(sd.query_devices(), '\n\n')
    # for i, d in enumerate(sd.query_devices()):
    #     print(f"{i}: {d}")

def record_from_single_mic(mic_idx, duration, sample_rate, recordings):
    """Record from a single microphone"""
    try:
        with sd.InputStream(device=mic_idx, channels=1, samplerate=sample_rate) as stream:
            frames = []
            for _ in range(int(duration * sample_rate / 1024)):
                data, _ = stream.read(1024)
                frames.append(data)
        recordings[mic_idx] = np.concatenate(frames)
    except Exception as e:
        print(f"Error recording from mic {mic_idx}: {str(e)}")

def record_from_mics(duration=5, sample_rate=44100):
    """Record from multiple microphones simultaneously"""
    list_devices()
    devices = sd.query_devices()
    recordings = {}
    
    threads = []
    for mic_idx, _ in enumerate(devices):
        thread = threading.Thread(
            target=record_from_single_mic,
            args=(mic_idx, duration, sample_rate, recordings)
        )
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    for mic_idx, recording in recordings.items():
        filename = f"recording_mic_{mic_idx}.wav"
        wavfile.write(filename, sample_rate, recording)
        print(f"Saved recording to {filename}")

if __name__ == "__main__":
    record_from_mics(duration=3)