import CSP


root_dir = 'EEG/data' 

specific_subjects = [72] 
specific_sessions = [5]
new_edf_file = CSP.find_edf_files(root_dir, specific_subjects, specific_sessions)[0]
predicted_labels = CSP.predict_from_new_edf(new_edf_file, 0, 122)
print("Predicted labels from the new EDF file:", predicted_labels)
