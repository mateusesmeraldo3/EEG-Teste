import numpy as np
import mne
import sklearn

sample_data_raw_file = ("Exemplo paciente normal - FA0273WD.edf")
raw = mne.io.read_raw_edf(sample_data_raw_file)

#print(raw)
#print(raw.info)

#raw.plot_psd(fmax=50)
#raw.plot(duration=1, n_channels=32, block = True)

# set up and fit the ICA
ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
ica.fit(raw)
ica.exclude = [1, 2]  # details on how we picked these are omitted here
ica.plot_properties(raw, picks=ica.exclude)