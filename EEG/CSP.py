import mne
from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import joblib
import numpy as np
import re
import os
# Function to load and preprocess EDF files
tmin, tmax = -1, 5
filter_param = [8, 25]
def load_and_preprocess_edf_both(edf_files):
    X = []
    y = []

    for file in edf_files:
        raw = mne.io.read_raw_edf(file, preload=True)
        events, event_id = mne.events_from_annotations(raw)
        raw = raw.filter(filter_param[0], filter_param[1],fir_design="firwin", skip_by_annotation="edge")
        epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin, tmax=tmax, preload=True)
        print(events)
        print(epochs.drop_log)
        X.append(epochs.get_data(copy=False))
        y.append(epochs.events[:, -1])
        print(y)
    X = np.concatenate(X)
    y = np.concatenate(y)
    return X, y
def load_and_preprocess_edf_hands(edf_files):
    X = []
    y = []

    for file in edf_files:
        raw = mne.io.read_raw_edf(file, preload=True)
        events, event_id = mne.events_from_annotations(raw)
        raw = raw.filter(filter_param[0], filter_param[1])
        epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin, tmax=tmax, preload=True)
        # Modify the event codes
        epochs.events[:, -1][epochs.events[:, -1] == 2] = 4
        epochs.events[:, -1][epochs.events[:, -1] == 3] = 5
        X.append(epochs.get_data(copy=False))
        y.append(epochs.events[:, -1])
        print(epochs.events)
        print(y)
    X = np.concatenate(X)
    y = np.concatenate(y)
    return X, y
def crop_edf(new_edf_file, start_time, end_time):
    # Load the raw EDF file
    raw_new = mne.io.read_raw_edf(new_edf_file, preload=True)

    # Crop the raw data to the specified time range
    raw_new.crop(tmin=start_time, tmax=end_time)

    return raw_new
def predict_from_new_edf(new_edf_file, start_time, end_time):
    clf_loaded = joblib.load('classifier_model.pkl')

    # Crop the raw EDF file
    raw_new = crop_edf(new_edf_file, start_time, end_time)

    # Preprocess the cropped EDF file
    raw_new = raw_new.filter(filter_param[0], filter_param[1])

    events_new, event_id_new = mne.events_from_annotations(raw_new)
    epochs_new = mne.Epochs(raw_new, events=events_new, event_id=event_id_new, tmin=tmin, tmax=tmax, preload=True)
    X_new = epochs_new.get_data(copy=False)

    # Predict labels
    y_pred_new = clf_loaded.predict(X_new)
    y_test_new = epochs_new.events[:, -1]
    accuracy = accuracy_score(y_test_new, y_pred_new)
    print("Accuracy on test set:", accuracy)

    return y_pred_new

def predict_from_new_edf_new(new_edf_file,start_time, end_time):
    clf_loaded = joblib.load('classifier_model.pkl')
    # Load and preprocess the new EDF file
    raw_new = mne.io.read_raw_edf(new_edf_file, preload=True)
    raw_new.crop(tmin=start_time, tmax=end_time)
    raw_new = raw_new.filter(filter_param[0], filter_param[1], skip_by_annotation ="edge")
    # Create fixed length epochs
    epochs_new = mne.make_fixed_length_epochs(raw_new, duration=5.0, preload=True)
    X_new = epochs_new.get_data(copy=False)
    # Predict labels
    y_pred_new = clf_loaded.predict(X_new)
    print("Predicted labels:", y_pred_new)

    return y_pred_new

def find_edf_files(root_dir, subjects, sessions):
    edf_files = []
    pattern = re.compile(r'S(\d{3})R(\d{2})\.edf')

    for subject in subjects:
        subject_dir = os.path.join(root_dir, f'S{subject:03d}')
        if os.path.isdir(subject_dir):
            for file in os.listdir(subject_dir):
                if file.endswith('.edf') and not file.endswith('.edf.event'):
                    match = pattern.match(file)
                    if match:
                        _, session = match.groups()
                        if int(session) in sessions:
                            edf_files.append(os.path.join(subject_dir, file))

    return edf_files

# 3,4,7,8,11,12 fists
# 5,6,9,10,13,14 both
if __name__ == "__main__":
    
    
    root_dir = 'EEG/data'  # Adjust this to your root directory

    # Define specific subjects and sessions to process
    specific_subjects = [2]  # 24 , 2 , 35
    specific_sessions_fists = [3,4,7,8,11,12]

    specific_sessions_both = [5,6,9,10,13,14]
    
    
    edf_files1 = find_edf_files(root_dir, specific_subjects, specific_sessions_fists)
    edf_files2 = find_edf_files(root_dir, specific_subjects, specific_sessions_both)
    
    edf_files_set1 = [
        'EEG/data/S002R03.edf', 'EEG/data/S002R04.edf', 'EEG/data/S002R07.edf', 'EEG/data/S002R08.edf', 'EEG/data/S002R11.edf', 'EEG/data/S002R12.edf'
    ]
    # edf_files_set1 = [
    #     'EEG/data/S002R03.edf', 'EEG/data/S002R07.edf','EEG/data/S002R11.edf',
    # ]
    edf_files_set2 = [
        'EEG/data/S002R09.edf', 'EEG/data/S002R05.edf', 'EEG/data/S002R06.edf', 'EEG/data/S002R10.edf', 'EEG/data/S002R13.edf', 'EEG/data/S002R14.edf'
    ]
    
    # Load and preprocess the first set of EDF files
    # X1, y1 = load_and_preprocess_edf_hands(edf_files1)

    # Load and preprocess the second set of EDF files
    X, y = load_and_preprocess_edf_both(edf_files2)
    
    # # Combine data from both sets
    # X = np.concatenate([X1, X2])
    # y = np.concatenate([y1, y2])
    # print(y)
    # Define CSP parameters
    n_components = 5  # Number of components to keep
    # Apply CSP filtering
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)

    # Define the classifier
    clf = make_pipeline(csp, StandardScaler(), SVC(kernel='linear'))  # linear, rbf, poly

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

    # Train the classifier
    clf.fit(X_train, y_train)
    # Evaluate accuracy on the test set
    accuracy = accuracy_score(y_test, clf.predict(X_test))
    print("Accuracy on test set:", accuracy)


    # Save the trained classifier
    joblib.dump(clf, 'classifier_model.pkl')

    # Load the trained classifier
    clf_loaded = joblib.load('classifier_model.pkl')

    # Use the loaded classifier for prediction
    y_pred = clf_loaded.predict(X_test)

    print(y_pred)

    # new_edf_file = 'EEG/data/S002R09.edf'
    # predicted_labels = predict_from_new_edf(new_edf_file,0,122)
    # raw = mne.io.read_raw_edf(new_edf_file, preload=True)
    # events, event_id = mne.events_from_annotations(raw)
    # # print(events)
    # print("Predicted labels from the new EDF file:", predicted_labels)