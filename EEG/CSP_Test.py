import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline

from mne import Epochs, events_from_annotations, pick_types
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.io import concatenate_raws, read_raw_edf

# print(__doc__)

# #############################################################################
# # Set parameters and read data

# avoid classification of evoked responses by using epochs that start 1s after
# cue onset.
tmin, tmax = -1.0, 4.0
event_id = dict( rest=2, hands=3, feet=4)

subject = 1  # Numer badania
runs = [5, 6, 10, 13, 14]  # motor imagery: hands vs feet


# czytanie edf


# path = ['S001R03.edf', 'S001R05.edf', 'S001R02.edf']  # tablica sciezek do plikow
# raws = [read_raw_edf(file, preload=True) for file in path]  # czytanie do tablicy raw edf #https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw
# raw = concatenate_raws(raws, preload=True)  # https://mne.tools/stable/generated/mne.concatenate_raws.html#mne-concatenate-raws

raw_fnames = eegbci.load_data(subject, runs)

raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
eegbci.standardize(raw)  # set channel names

#montage sa odpowiedzialne za informacje o umiesczeniu elektrod na głowie, TODO mozna dodac swoj wlasny
montage = make_standard_montage("standard_1005")  #https://mne.tools/stable/generated/mne.channels.make_standard_montage.html#mne.channels.make_standard_montage
raw.set_montage(montage)

# filtr pasmowo przepustowy
raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge") #filtr pomija oznaczenie o koncu

events, _ = events_from_annotations(raw, event_id=dict(T0=2, T1=3, T2=4))

picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

# Read epochs (train will be done only between 1 and 2s)
# Testing will be done with a running classifier

# mozna wykluczyc zbyt silne sygnaly
epochs = Epochs(  # https://mne.tools/stable/generated/mne.Epochs.html#mne.Epochs
    raw,
    events,  # lista kategori ktore sa w plikach i ich czasy
    event_id,  # kategorie ktore chcemy sprawdzic
    tmin,  # czasy epoki epoki startu i konca dookola eventu
    tmax,
    proj=True,  #usuwanie szumow
    picks=picks, #kanaly uczestniczace
    baseline=None,#ewentualna korekcja sygnalu u stala warotsc
    preload=True,
)
epochs_train = epochs.copy().crop(tmin=1.0, tmax=2.0)  # kopia i ustwienie czasu epok
labels = epochs.events[:, -1] # przypisanie tabeli klasyfikacji czyli ostatniej kolumny z kazdego rzedu

# Define a monte-carlo cross-validation generator (reduce variance):
scores = []
epochs_data = epochs.get_data(copy=False)
epochs_data_train = epochs_train.get_data(copy=False)
cv = ShuffleSplit(20, test_size=0.3, random_state=30)  #mozna cos pozmieniac by zyskac lepsze efekty
cv_split = cv.split(epochs_data_train)

# Assemble a classifier
lda = LinearDiscriminantAnalysis()
csp = CSP(n_components=6, reg=None, log=True, norm_trace=False)

# Use scikit-learn Pipeline with cross_val_score function
clf = Pipeline([("CSP", csp), ("LDA", lda)])
scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=None)

# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1.0 - class_balance)
print(
    "Classification accuracy: %f / Chance level: %f" % (np.mean(scores), class_balance)
)

# plot CSP patterns estimated on full data for visualization
csp.fit_transform(epochs_data, labels)

csp.plot_patterns(epochs.info, ch_type="eeg", units="Patterns (AU)", size=2)
sfreq = raw.info["sfreq"]
w_length = int(sfreq * 0.5)  # running classifier: window length
w_step = int(sfreq * 0.1)  # running classifier: window step size
w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)

scores_windows = []

for train_idx, test_idx in cv_split:
    y_train, y_test = labels[train_idx], labels[test_idx]

    X_train = csp.fit_transform(epochs_data_train[train_idx], y_train)
    X_test = csp.transform(epochs_data_train[test_idx])

    # fit classifier
    lda.fit(X_train, y_train)

    # running classifier: test classifier on sliding window
    score_this_window = []
    for n in w_start:
        X_test = csp.transform(epochs_data[test_idx][:, :, n: (n + w_length)])
        score_this_window.append(lda.score(X_test, y_test))
    scores_windows.append(score_this_window)

# Plot scores over time
w_times = (w_start + w_length / 2.0) / sfreq + epochs.tmin

plt.figure()
plt.plot(w_times, np.mean(scores_windows, 0), label="Score")
plt.axvline(0, linestyle="--", color="k", label="Onset")
plt.axhline(0.5, linestyle="-", color="k", label="Chance")
plt.xlabel("time (s)")
plt.ylabel("classification accuracy")
plt.title("Classification score over time")
plt.legend(loc="lower right")
plt.show()

# testowanie
new_raw = read_raw_edf('S001R09.edf', preload=True)
new_raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")
new_events, _ = events_from_annotations(new_raw, event_id=dict(T0=2, T1=3, T2=4))
new_picks = pick_types(new_raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

# sfreq = new_raw.info['sfreq']  # sampling frequency
# t_epoch = 4
# n_samples = int(t_epoch * sfreq)  # number of samples in each epoch

# new_epochs_data = []
# for i in range(0, len(new_raw.times), n_samples):
#     epoch = new_raw[:, i:i+n_samples][0]
#     new_epochs_data.append(epoch)
# new_epochs_data = np.array(new_epochs_data)

new_epochs = Epochs(new_raw, new_events, event_id, tmin, tmax, proj=True, picks=new_picks, baseline=None, preload=True)
new_epochs_data = new_epochs.get_data(copy=True)

new_X = csp.transform(new_epochs_data)
new_labels_pred = lda.predict(new_X)
for i, label in enumerate(new_labels_pred):
    print(f'Epoch {i+1}: {label}')