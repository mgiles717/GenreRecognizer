import os
import sys
import song_analysis as sa
from sklearn.naive_bayes import GaussianNB

def main():
    songs = [sa.SongAnalysis(f"data/train_data/{i}") for i in os.listdir("data/train_data")
             if i.endswith(".wav")]
    for song in songs:
        print(song.name, song.bpm)

if __name__ == '__main__':
    main()