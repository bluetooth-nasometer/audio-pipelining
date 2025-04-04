import librosa
import numpy as np
import matplotlib.pyplot as plt
import time

from feed import AudioRecorder

def analyze_audio_spl(audio_files):
    fig, axes = plt.subplots(len(audio_files), 2, figsize=(12, 8 * len(audio_files)))
    
    if len(audio_files) == 1:
        axes = [axes]  # Ensure axes is iterable for a single file
    
    results = []
    
    for i, audio_path in enumerate(audio_files):
        y, sr = librosa.load(audio_path)
        
        rms = librosa.feature.rms(y=y, frame_length=512)[0]  # Get RMS energy
        rsp = 20e-6                                          # Reference Sound Pressure
        spl_db = 20 * np.log10(rms / rsp)
        
        mean_spl = np.mean(spl_db)
        max_spl = np.max(spl_db)
        min_spl = np.min(spl_db)
        
        # Plot 1: Raw Audio
        times_raw = np.arange(len(y)) / sr
        axes[i][0].plot(times_raw, y)
        axes[i][0].set_xlabel('Time (s)')
        axes[i][0].set_ylabel('Amplitude')
        axes[i][0].set_title(f'Raw Audio Waveform - {audio_path}')
        axes[i][0].grid(True)
        
        # Plot 2: SPL over time
        times = librosa.times_like(spl_db, sr=sr)
        axes[i][1].plot(times, spl_db)
        axes[i][1].axhline(y=mean_spl, color='r', linestyle='--', label=f'Mean: {mean_spl:.1f} dB')
        axes[i][1].set_xlabel('Time (s)')
        axes[i][1].set_ylabel('SPL (dB)')
        axes[i][1].set_title(f'Sound Pressure Level Over Time - {audio_path}')
        axes[i][1].grid(True)
        axes[i][1].legend()
        
        results.append({
            'audio_file': audio_path,
            'mean_spl': mean_spl,
            'max_spl': max_spl,
            'min_spl': min_spl
        })
    
    plt.tight_layout()
    return results


if __name__ == "__main__":
    recorder = AudioRecorder(sample_rate=44100)
    recorder.list_devices()

    a = input("Pick mic1 - ")
    b = input("Pick mic2 - ")
    
    recorder.start_recording(mic_indices=[a, b])
    time.sleep(5)
    recorder.stop_recording()
    recorder.save_recordings()
    recorder.clear_recordings()


    audio_files = [f"recording_mic_{a}.wav", f"recording_mic_{b}.wav"]
    stats = analyze_audio_spl(audio_files)
    
    for stat in stats:
        print(f"File: {stat['audio_file']}")
        print(f"  Mean SPL: {stat['mean_spl']:.1f} dB")
        print(f"  Max SPL: {stat['max_spl']:.1f} dB")
        print(f"  Min SPL: {stat['min_spl']:.1f} dB")
    
    plt.show()