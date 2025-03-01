import mne
from sleep_scoring.gui import Sleep

path = r"F:\EGI cleaning\1.mff"
raw = mne.io.read_raw(path)
# raw = raw.crop(tmin=909*60)
# raw = raw.resample(100)
Sleep(data=raw.get_data(), sf=raw.info['sfreq'],  channels=raw.ch_names).show()
# Sleep().show()
print(0)