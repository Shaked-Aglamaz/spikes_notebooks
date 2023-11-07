import numpy as np
import pandas as pd
import mne
import scipy.stats as sp_stats
import scipy.signal as sp_sig
import antropy as ant
from scipy.integrate import simps
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import robust_scale

sr = 1000
depth_edf_path = 'C:\\Lilach\\%s_for_tag_filtered_fix_tag.edf'


# yasa function for power features in each band
def bandpower_from_psd_ndarray(psd, freqs, bands, relative=True):
    # Type checks
    assert isinstance(bands, list), 'bands must be a list of tuple(s)'
    assert isinstance(relative, bool), 'relative must be a boolean'

    # Safety checks
    freqs = np.asarray(freqs)
    psd = np.asarray(psd)
    assert freqs.ndim == 1, 'freqs must be a 1-D array of shape (n_freqs,)'
    assert psd.shape[-1] == freqs.shape[-1], 'n_freqs must be last axis of psd'

    # Extract frequencies of interest
    all_freqs = np.hstack([[b[0], b[1]] for b in bands])
    fmin, fmax = min(all_freqs), max(all_freqs)
    idx_good_freq = np.logical_and(freqs >= fmin, freqs <= fmax)
    freqs = freqs[idx_good_freq]
    res = freqs[1] - freqs[0]

    # Trim PSD to frequencies of interest
    psd = psd[..., idx_good_freq]

    # Check if there are negative values in PSD
    if (psd < 0).any():
        msg = (
            "There are negative values in PSD. This will result in incorrect "
            "bandpower values. We highly recommend working with an "
            "all-positive PSD. For more details, please refer to: "
            "https://github.com/raphaelvallat/yasa/issues/29")
        print(msg)

    # Calculate total power
    total_power = simps(psd, dx=res, axis=-1)
    total_power = total_power[np.newaxis, ...]

    # Initialize empty array
    bp = np.zeros((len(bands), *psd.shape[:-1]), dtype=np.float)

    # Enumerate over the frequency bands
    labels = []
    for i, band in enumerate(bands):
        b0, b1, la = band
        labels.append(la)
        idx_band = np.logical_and(freqs >= b0, freqs <= b1)
        bp[i] = simps(psd[..., idx_band], dx=res, axis=-1)

    if relative:
        bp /= total_power
    return bp


def get_metrics(cm):
    if len(cm) == 0:
        cm = np.zeros((2,2), dtype=int)
    if np.squeeze(cm).ndim < 2:
        new_cm = np.zeros((2,2), dtype=int)
        new_cm[1, 1] = int(cm[0][0])
        cm = new_cm
    numerator = cm[0, 0] + cm[1, 1]
    denominator = cm[0, 0] + cm[1, 1] + cm[0, 1] + cm[1, 0]
    accuracy = numerator / denominator
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
    recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    numerator = precision * recall
    denominator = (0.25 * precision) + recall
    f_score = 1.25 * numerator / denominator
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f_score': f_score}


# read edf of tags and return epochs of 250ms and y array represent the occurrence of spikes
def format_raw(edf, channel, norm='raw'):
    epochs, epochs_zscore, epochs_random, epochs_index = [], [], [], []
    window_size = int(sr / 4)
    raw = mne.io.read_raw_edf(edf)
    spikes = raw.annotations
    spikes_df = pd.DataFrame(spikes)
    # END annotation represent the complete time segment that the doctor tag (second round)
    end_onset = list(spikes_df[spikes_df['description'].str.contains('END')]['onset'])[0]
    spikes_df = spikes_df[spikes_df['description'].str.contains(channel[0] + 't')]
    spikes_df = spikes_df.drop_duplicates(subset=['onset'])
    complete_spikes_df = spikes_df[spikes_df['onset'] < end_onset]

    # from first tags round- only some spikes here and there
    random_spikes_df = spikes_df[spikes_df['onset'] > end_onset]
    raw_data = raw.copy().pick_channels([channel]).resample(sr).get_data()[0]
    if norm == 'raw':
        raw_data = (raw_data - raw_data.mean()) / raw_data.std()
    for onset in random_spikes_df['onset']:
        epochs_random.append(raw_data[int(onset * sr - window_size / 2) : int (onset * sr + window_size / 2)])
        epochs_index.append(int(onset * sr / window_size))
    y_windows_random = np.full(len(epochs_random), 1)

    # from second tags round- complete data
    raw_data = raw.crop(tmax=end_onset).pick_channels([channel]).resample(sr).get_data()[0]
    if norm == 'raw':
        raw_data = (raw_data - raw_data.mean()) / raw_data.std()
    for i in range(0, len(raw_data), window_size):
        curr_block = raw_data[i: i + window_size]
        if i + window_size < len(raw_data):
            epochs.append(curr_block)

    y_windows = np.zeros(len(epochs))
    for onset in complete_spikes_df['onset']:
        y_windows[int(onset * 1000 / window_size)] = 1

    final_index = np.concatenate((np.arange(len(epochs)), epochs_index))
    epochs = np.concatenate((epochs, epochs_random))

    # normalize epochs
    if norm == 'epochs':
        epochs = (epochs - epochs.mean()) / epochs.std()
    y_windows = np.concatenate((y_windows, y_windows_random))

    return np.array(epochs), np.array(epochs), y_windows, final_index


def calc_features_before_split(epochs, subj):
    # Bandpass filter
    freq_broad = (0.1, 500)
    # FFT & bandpower parameters
    bands = [
        (0.1, 4, 'delta'), (4, 8, 'theta'),
        (8, 12, 'alpha'), (12, 16, 'sigma'), (16, 30, 'beta'),
        (30, 100, 'gamma'), (100, 300, 'fast')
    ]

    # Calculate standard descriptive statistics
    hmob, hcomp = ant.hjorth_params(epochs, axis=1)

    feat = {
        'subj': np.full(len(epochs), subj),
        'epoch_id': np.arange(len(epochs)),
        'std': np.std(epochs, ddof=1, axis=1),
        'iqr': sp_stats.iqr(epochs, axis=1),
        'skew': sp_stats.skew(epochs, axis=1),
        'kurt': sp_stats.kurtosis(epochs, axis=1),
        'nzc': ant.num_zerocross(epochs, axis=1),
        'hmob': hmob,
        'hcomp': hcomp
    }

    # Calculate spectral power features (for EEG + EOG)
    freqs, psd = sp_sig.welch(epochs, sr)
    bp = bandpower_from_psd_ndarray(psd, freqs, bands=bands)
    for j, (_, _, b) in enumerate(bands):
        feat[b] = bp[j]

    # Add power ratios for EEG
    feat['at'] = feat['alpha'] / feat['theta']
    feat['ag'] = feat['gamma'] / feat['alpha']
    feat['sf'] = feat['sigma'] / feat['fast']
    feat['bf'] = feat['beta'] / feat['fast']
    feat['gf'] = feat['gamma'] / feat['fast']
    # need those?
    feat['gt'] = feat['gamma'] / feat['theta']
    feat['ft'] = feat['fast'] / feat['theta']
    feat['af'] = feat['fast'] / feat['alpha']

    # Add total power
    idx_broad = np.logical_and(
        freqs >= freq_broad[0], freqs <= freq_broad[1])
    dx = freqs[1] - freqs[0]
    feat['abspow'] = np.trapz(psd[:, idx_broad], dx=dx)

    # Calculate entropy and fractal dimension features
    feat['perm'] = np.apply_along_axis(
        ant.perm_entropy, axis=1, arr=epochs, normalize=True)
    feat['higuchi'] = np.apply_along_axis(
        ant.higuchi_fd, axis=1, arr=epochs)
    feat['petrosian'] = ant.petrosian_fd(epochs, axis=1)

    # Convert to dataframe
    feat = pd.DataFrame(feat)

    return feat


def calc_features_after_split(feat):
    # SMOOTHING & NORMALIZATION
    roll1 = feat.rolling(window=1, center=True, min_periods=1, win_type='triang').mean()
    roll1[roll1.columns] = robust_scale(roll1, quantile_range=(5, 95))
    roll1 = roll1.iloc[:, 2:].add_suffix('_cmin_norm')

    # Add to current set of features
    # feat = feat.join(roll1)
    # pd.concat([feat, roll1])

    return pd.concat([feat, roll1], axis=1)


def run_specific_chan_with_bi(chan, bi=True, subjects=['38', '396', '398', '402', '406', '415', '416']):
    x_all = np.empty((0, 250))
    y_all = np.empty(0)
    feat_all = pd.DataFrame()
    y_all_train = np.empty(0)
    y_all_test = np.empty(0)
    feat_all_train = pd.DataFrame()
    feat_all_test = pd.DataFrame()
    for subj in subjects:
        for channel in [f'R{chan}1', f'L{chan}1']:
            if not ((subj == '396' and channel == 'RAH1') or (subj == '38' and channel == 'LAH1')):
                x, x_zscore, y, x_index = format_raw(depth_edf_path % subj, channel)
                features = calc_features_before_split(x, subj)
                x_all = np.concatenate((x_all, x))
                y_all = np.concatenate((y_all, y))
                if bi:
                    x, x_zscore, y, x_index = format_raw(depth_edf_path % subj, f'{channel}-{channel[:-1]}2')
                    features_bi = calc_features_before_split(x, subj).add_prefix('bi_')
                    features = pd.concat([features, features_bi.iloc[:, 2:]], axis=1)
                feat_all = pd.concat([feat_all, features], axis=0)

                X_train, X_test, y_train, y_test = train_test_split(features, y, stratify=y, random_state=20)
                # Add separated norm
                X_train = calc_features_after_split(X_train)
                X_test = calc_features_after_split(X_test)

                feat_all_train = pd.concat([feat_all_train, X_train], axis=0)
                feat_all_test = pd.concat([feat_all_test, X_test], axis=0)
                y_all_train = np.concatenate((y_all_train, y_train))
                y_all_test = np.concatenate((y_all_test, y_test))

    feat_all_train = feat_all_train.reset_index(drop=True)
    feat_all_train.index.name = 'epoch'
    feat_all_test = feat_all_test.reset_index(drop=True)
    feat_all_test.index.name = 'epoch'
    return feat_all_train, feat_all_test, y_all_train, y_all_test


def run_specific_chan(channels, subjects=['38', '396', '398', '402', '406', '415', '416']):
    x_all = np.empty((0, 250))
    y_all = np.empty(0)
    feat_all = pd.DataFrame()
    y_all_train = np.empty(0)
    y_all_test = np.empty(0)
    feat_all_train = pd.DataFrame()
    feat_all_test = pd.DataFrame()
    for subj in subjects:
        for channel in channels:
            if not ((subj == '396' and 'RAH1' in channel) or (subj == '38' and 'LAH1' in channel)):
                x, x_zscore, y, x_index = format_raw(depth_edf_path % subj, channel)
                features = calc_features_before_split(x, subj)
                x_all = np.concatenate((x_all, x))
                y_all = np.concatenate((y_all, y))
                feat_all = pd.concat([feat_all, features], axis=0)

                X_train, X_test, y_train, y_test = train_test_split(features, y, stratify=y, random_state=20)
                # Add separated norm
                X_train = calc_features_after_split(X_train)
                X_test = calc_features_after_split(X_test)

                feat_all_train = pd.concat([feat_all_train, X_train], axis=0)
                feat_all_test = pd.concat([feat_all_test, X_test], axis=0)
                y_all_train = np.concatenate((y_all_train, y_train))
                y_all_test = np.concatenate((y_all_test, y_test))

    feat_all_train = feat_all_train.reset_index(drop=True)
    feat_all_train.index.name = 'epoch'
    feat_all_test = feat_all_test.reset_index(drop=True)
    feat_all_test.index.name = 'epoch'
    return feat_all_train, feat_all_test, y_all_train, y_all_test

def run_all(subjects=['396', '398', '402', '406', '415', '416']):
    neighbors = {'R': ['RAH1-RAH2', 'RA1'], 'L': ['LAH1-LAH2', 'LA1']}
    x_AH = np.empty((0, 250))
    x_A = np.empty((0, 250))
    x_bi = np.empty((0, 250))
    y_all_train = np.empty(0)
    y_all_test = np.empty(0)
    feat_all_train = pd.DataFrame()
    feat_all_test = pd.DataFrame()
    for subj in subjects:
        for channel in ['RAH1', 'LAH1']:
            if not (subj == '396' and channel == 'RAH1'):
                x, x_zscore, y, x_index = format_raw(depth_edf_path % subj, channel)
                x_AH = np.concatenate((x_AH, x))
                # y_all = np.concatenate((y_all, y))
                features = calc_features_before_split(x, subj)
                for neighbor in neighbors[channel[0]]:
                    x, x_zscore, y, x_index = format_raw(depth_edf_path % subj, neighbor)
                    if '-' in neighbor:
                        x_bi = np.concatenate((x_bi, x))
                    else:
                        x_A = np.concatenate((x_A, x))
                    prefix = neighbor.replace(channel[0], '')
                    features_neighbor = calc_features_before_split(x, subj).add_prefix(f'{prefix}_')
                    features = pd.concat([features, features_neighbor.iloc[:, 2:]], axis=1)

                X_train, X_test, y_train, y_test = train_test_split(features, y, stratify=y, random_state=20)
                # Add separated norm
                X_train = calc_features_after_split(X_train)
                X_test = calc_features_after_split(X_test)

                feat_all_train = pd.concat([feat_all_train, X_train], axis=0)
                feat_all_test = pd.concat([feat_all_test, X_test], axis=0)
                y_all_train = np.concatenate((y_all_train, y_train))
                y_all_test = np.concatenate((y_all_test, y_test))

    feat_all_train = feat_all_train.reset_index(drop=True)
    feat_all_train.index.name = 'epoch'
    feat_all_test = feat_all_test.reset_index(drop=True)
    feat_all_test.index.name = 'epoch'

    return feat_all_train, feat_all_test, y_all_train, y_all_test


def channel_feat(edf, channel):
    raw_data = mne.io.read_raw(edf).pick_channels([channel]).resample(sr).get_data()[0]
    feat = {
        'median': np.median(raw_data),
        'ptp': np.ptp(raw_data),
        # 'iqr': sp_stats.iqr(chan),
        # 'skew': sp_stats.skew(chan),
        # 'kurt': sp_stats.kurtosis(chan),
        # bf, gf
    }

    # feat = pd.DataFrame(feat, index=[0])

    return feat

# def run_specific_chan_no_split(channels, subjects=['38', '396', '398', '402', '406', '415', '416']):
#     x_all = np.empty((0, 250))
#     y_all = np.empty(0)
#     feat_all = pd.DataFrame()
#     y_all_train = np.empty(0)
#     y_all_test = np.empty(0)
#     feat_all_train = pd.DataFrame()
#     feat_all_test = pd.DataFrame()
#     for subj in subjects:
#         for channel in channels:
#             if not ((subj == '396' and 'RAH1' in channel) or (subj == '38' and 'LAH1' in channel)):
#                 x, x_zscore, y, x_index = format_raw(depth_edf_path % subj, channel)
#                 features = calc_features_before_split(x, subj)
#                 x_all = np.concatenate((x_all, x))
#                 y_all = np.concatenate((y_all, y))
#                 feat_all = pd.concat([feat_all, features], axis=0)
#
#                 X_train, X_test, y_train, y_test = train_test_split(features, y, stratify=y, random_state=20)
#                 # Add separated norm
#                 X_train = calc_features_after_split(X_train)
#                 X_test = calc_features_after_split(X_test)
#
#                 feat_all_train = pd.concat([feat_all_train, X_train], axis=0)
#                 feat_all_test = pd.concat([feat_all_test, X_test], axis=0)
#                 y_all_train = np.concatenate((y_all_train, y_train))
#                 y_all_test = np.concatenate((y_all_test, y_test))
#
#     feat_all_train = feat_all_train.reset_index(drop=True)
#     feat_all_train.index.name = 'epoch'
#     feat_all_test = feat_all_test.reset_index(drop=True)
#     feat_all_test.index.name = 'epoch'
#     return feat_all_train, feat_all_test, y_all_train, y_all_test


def run_specific_chan_no_split(subjects=['38', '396', '398', '402', '406', '415', '416'], bi=True):
    channels = ['RAH1-RAH2', 'LAH1-LAH2'] if bi else ['RAH1', 'LAH1']
    y_all = np.empty(0)
    feat_all = pd.DataFrame()
    for subj in subjects:
        for channel in channels:
            if not ((subj == '396' and 'RAH1' in channel) or (subj == '38' and 'LAH1' in channel)):
                x, x_zscore, y, x_index = format_raw(f'C:\\Lilach\\{subj}_for_tag_filtered_fix_tag.edf', channel)
                y_all = np.concatenate((y_all, y))
                chan_feat = channel_feat(f'C:\\Lilach\\{subj}_for_tag_filtered_fix_tag.edf', channel)
                features = calc_features_before_split(x, subj)
                for feat in chan_feat.keys():
                    features[feat] = chan_feat[feat]

                feat_all = pd.concat([feat_all, features], axis=0)

    # feat_all = feat_all.drop('epoch', axis=1)
    feat_all = feat_all.reset_index(drop=True)
    feat_all.index.name = 'epoch'

    return feat_all, y_all

def run_all_chans_no_split(subjects=['38', '396', '398', '402', '406', '415', '416'], include_bi=False):
    y_all = np.empty(0)
    feat_all = pd.DataFrame()
    for subj in subjects:
        ch_names = mne.io.read_raw_edf(f'C:\\Lilach\\{subj}_for_tag_filtered_fix_tag.edf').ch_names
        if not include_bi:
            ch_names = [x for x in ch_names if '-' not in x and '3' not in x]
        for channel in ch_names:
            if not ((subj == '396' and channel[0] == 'R') or (subj == '38' and channel[0] == 'L')):
                x, x_zscore, y, x_index = format_raw(f'C:\\Lilach\\{subj}_for_tag_filtered_fix_tag.edf', channel)
                y_all = np.concatenate((y_all, y))
                chan_feat = channel_feat(f'C:\\Lilach\\{subj}_for_tag_filtered_fix_tag.edf', channel)
                features = calc_features_before_split(x, subj)
                for feat in chan_feat.keys():
                    features[feat] = chan_feat[feat]

                feat_all = pd.concat([feat_all, features], axis=0)

    # feat_all = feat_all.drop('epoch', axis=1)
    feat_all = feat_all.reset_index(drop=True)
    feat_all.index.name = 'epoch'

    return feat_all, y_all

# run_all_chans_no_split()