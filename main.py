import librosa
import numpy as np
import matplotlib.pyplot as plt

def analyze_audio_spl(audio_path):
    y, sr = librosa.load(audio_path)
    
    rms = librosa.feature.rms(y=y, frame_length=512)[0] # Get RMS energy
    rsp = 20e-6                                         # Reference Sound Pressure
    spl_db = 20 * np.log10(rms / rsp)
    
    mean_spl = np.mean(spl_db)
    max_spl = np.max(spl_db)
    min_spl = np.min(spl_db)
    
    # # FFT
    # n = len(y)
    # yf = np.fft.fft(y)
    # xf = np.fft.fftfreq(n, 1/sr)
    # yf = 2.0/n * np.abs(yf[0:n//2])
    # xf = xf[0:n//2]

    # Create 4 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
    
    # Plot 1: Raw Audio (existing)
    times_raw = np.arange(len(y)) / sr
    ax1.plot(times_raw, y)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Raw Audio Waveform')
    ax1.grid(True)
    
    # Plot 2: SPL over time (existing)
    times = librosa.times_like(spl_db, sr=sr)
    ax2.plot(times, spl_db)
    ax2.axhline(y=mean_spl, color='r', linestyle='--', label=f'Mean: {mean_spl:.1f} dB')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('SPL (dB)')
    ax2.set_title('Sound Pressure Level Over Time')
    ax2.grid(True)
    ax2.legend()

    # # Plot 3: Spectrogram (existing)
    # D = librosa.stft(y, n_fft=2048, hop_length=512)
    # S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    # img = librosa.display.specshow(S_db, x_axis='time', y_axis='hz', ax=ax3, sr=sr)
    # ax3.set_title('Spectrogram')
    # fig.colorbar(img, ax=ax3, format='%+2.0f dB')
    
    # # Plot 4: Frequency Spectrum (new)
    # ax4.plot(xf, yf)
    # ax4.set_xlabel('Frequency (Hz)')
    # ax4.set_ylabel('Magnitude')
    # ax4.set_title('Frequency Spectrum')
    # ax4.set_xlim(0, sr//2)
    # ax4.grid(True)
    
    plt.tight_layout()
    
    return {
        'mean_spl': mean_spl,
        'max_spl': max_spl,
        'min_spl': min_spl
    }


if __name__ == "__main__":
    audio_file = "recording_mic_2.wav"
    stats = analyze_audio_spl(audio_file)
    print(f"Mean SPL: {stats['mean_spl']:.1f} dB")
    print(f"Max SPL: {stats['max_spl']:.1f} dB")
    print(f"Min SPL: {stats['min_spl']:.1f} dB")
    plt.show()