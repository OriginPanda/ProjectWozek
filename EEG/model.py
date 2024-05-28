import mne
from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import joblib
import numpy as np


# Function to load and preprocess EDF files
tmin, tmax = -1, 5.1
filter_param = [13, 18]
def load_and_preprocess_edf(edf_files):
    X = []
    y = []
    for file in edf_files:
        raw = mne.io.read_raw_edf(file, preload=True)
        events, event_id = mne.events_from_annotations(raw)
        raw = raw.filter(filter_param[0], filter_param[1])
        epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin, tmax=tmax, preload=True)
        print(events)
        print(epochs.drop_log)
        X.append(epochs.get_data(copy=False))
        y.append(epochs.events[:, -1])
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

def predict_from_new_edf_new(new_edf_file):
    clf_loaded = joblib.load('classifier_model.pkl')
    # Load and preprocess the new EDF file
    raw_new = mne.io.read_raw_edf(new_edf_file, preload=True)
    raw_new = raw_new.filter(filter_param[0], filter_param[1])

    # Create fixed length epochs
    epochs_new = mne.make_fixed_length_epochs(raw_new, duration=1.0, preload=True)
    X_new = epochs_new.get_data(copy=False)

    # Predict labels
    y_pred_new = clf_loaded.predict(X_new)
    print("Predicted labels:", y_pred_new)

    return y_pred_new




if __name__ == "__main__":
    edf_files = [
        #'S001R09.edf', 'S001R05.edf', 'S001R06.edf', 'S001R10.edf', 'S001R13.edf', 'S001R14.edf',]
     'S002R09.edf', 'S002R05.edf', 'S002R06.edf', 'S002R10.edf', 'S002R13.edf', 'S002R14.edf',]
    # 'S003R09.edf', 'S003R05.edf', 'S003R06.edf', 'S003R10.edf', 'S003R13.edf', 'S003R14.edf',]
    # 'S004R09.edf', 'S004R05.edf', 'S004R06.edf', 'S004R10.edf', 'S004R13.edf', 'S004R14.edf']
    # ]  # Add more files as needed
    results=[]
    # for i in range(2, 10, 2):
    #     for j in range(5, 28, i):

    X, y = load_and_preprocess_edf(edf_files)

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

    # # Perform cross-validation on the entire dataset
    # cv_scores = cross_val_score(clf, X, y, cv=5)
    # print("Cross-validation scores:", cv_scores)
    # print("Mean cross-validation accuracy:", np.mean(cv_scores))
    # results.append(np.mean(cv_scores))



    # Save the trained classifier
    joblib.dump(clf, 'classifier_model.pkl')

    # Load the trained classifier
    clf_loaded = joblib.load('classifier_model.pkl')

    # Use the loaded classifier for prediction
    y_pred = clf_loaded.predict(X_test)

    print(y_pred)
    print(results)
    # Example usage
    new_edf_file = 'S002R09.edf'
    predicted_labels = predict_from_new_edf(new_edf_file)
    print("Predicted labels from the new EDF file:", predicted_labels)



