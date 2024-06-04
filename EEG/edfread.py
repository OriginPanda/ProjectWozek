import mne
import matplotlib.pyplot as plt
from mne import concatenate_raws
from mne.channels import get_builtin_montages
from mne.io import Raw, read_raw_edf

# Load your EDF+ file
# file_path = "S001R03.edf"  # replace with your file path
# raws = mne.io.read_raw_edf(file_path, preload=True)
# Print the parameters

# raw = concatenate_raws(raws,preload=True)
path = ['EEG/data/S002R12.edf']#tablica sciezek do plikow

raws = [read_raw_edf(file, preload=True) for file in path]#czytanie do tablicy raw edf
raw = concatenate_raws(raws, preload=True)
print(raw.info)
print(raw.annotations.onset)
print(raw.annotations.duration)
print(raw.annotations.description)


#print(get_builtin_montages())
