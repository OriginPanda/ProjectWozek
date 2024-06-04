from . import CSP
root_dir = 'EEG/data' 

if __name__ == "__main__":
    specific_subjects = [72] 
    specific_sessions = [5]
    new_edf_file = CSP.find_edf_files(root_dir, specific_subjects, specific_sessions)[0]
    print(new_edf_file)
    predicted_labels = CSP.predict_from_new_edf_new(new_edf_file,0,122)
    print("Predicted labels from the new EDF file:", predicted_labels)
    
    
def load_file():
    specific_subjects = [72] 
    specific_sessions = [5]
    
    new_edf_file = CSP.find_edf_files(root_dir, specific_subjects, specific_sessions)[0]
    
    return new_edf_file

def prediction(tmin,tmax):
    
    specific_subjects = [2] 
    specific_sessions = [5]
    new_edf_file = CSP.find_edf_files(root_dir, specific_subjects, specific_sessions)[0]
    print(new_edf_file)
    predicted_labels = CSP.predict_from_new_edf_new(new_edf_file,tmin,tmax)
    print("Predicted labels from the new EDF file:", predicted_labels)
    return predicted_labels
