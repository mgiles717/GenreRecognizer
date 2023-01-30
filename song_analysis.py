import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from typing import Any
class SongAnalysis:
    def __init__(self, song: str) -> None:
        self.name = song.split("/")[-1].split(".")[0]
        with open(song, 'rb') as f:
            self.y, self.sr = librosa.load(f, sr=16000)
        self.bpm = librosa.beat.tempo(y=self.y, sr=self.sr)
        self.tonnetz_features = self.tonnetz_features()
        self.spectrogram = self.compute_spectrogram()

    def tonnetz_features(self) -> Any:
        features = librosa.feature.tonnetz(y=self.y, sr=self.sr)
        tonnetz_features = np.ravel(features)
        return tonnetz_features

    def plot_waveform(self) -> None:
        plt.figure(figsize=(14, 5))
        librosa.display.waveshow(y=self.y, sr=self.sr)
        plt.show()

    def compute_spectrogram(self) -> Any:
        spectrogram = np.abs(librosa.stft(self.y))
        # plt.figure()
        # librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), y_axis='log', x_axis='time')
        # plt.show()
        return spectrogram

if __name__ == '__main__':
    x = SongAnalysis("data/train_data/sabotage.wav")
    print(x.compute_spectrogram())
