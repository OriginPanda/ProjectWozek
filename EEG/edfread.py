import pyedflib
import numpy as np
import matplotlib.pyplot as plt

f = pyedflib.EdfReader("C:\\Users\\titro\\Downloads\\S001R01.edf")
n = f.signals_in_file
signal_labels = f.getSignalLabels()
sigbufs = np.zeros((n, f.getNSamples()[0]))
f.file_info()
print(n,f.file_duration,f.datarecords_in_file)
fig = plt.figure()
ax = plt.axes()
for i in np.arange(n):
    sigbufs[i, :] = f.readSignal(i)
    ax.plot(f.readSignal(i))
    print(f.readSignal(i))
    plt.show()