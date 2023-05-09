import numpy as np
import mne
import matplotlib.pyplot as plt

sample_data_raw_file = ('V1.edf')

raw = mne.io.read_raw_edf(sample_data_raw_file)

print(raw)
print(raw.info)
ch_names1 = raw.info.ch_names
print(ch_names1)
print(len(ch_names1))

montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)

# raw.plot_psd(fmax=50)
# NOTE: plot_psd() is a legacy function. New code should use .compute_psd().plot().
raw.compute_psd(fmax=50).plot()  # psd power spectral density

plt.pause(-1)  # Pause the script execution, keeping the plot window open

# set up and fit the ICA

ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
ica.fit(raw)
ica.exclude = [1, 2]  # details on how we picked these are omitted here
ica.plot_properties(raw, picks=ica.exclude)


