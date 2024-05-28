import model



new_edf_file = 'S002R13.edf'
predicted_labels = model.predict_from_new_edf(new_edf_file, 0, 22)
print("Predicted labels from the new EDF file:", predicted_labels)