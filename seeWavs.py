import SPECtogram
import os


wavfiles = ["data/up/" + wavfile for wavfile in os.listdir("data/up/")]

for wavfile in wavfiles:
    SPECtogram.gimmeDaSPECtogram(wavfile)