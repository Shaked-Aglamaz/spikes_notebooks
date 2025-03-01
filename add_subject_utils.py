import os
from pathlib import Path

import antropy as ant
import mne
import mne_features
import numpy as np
import pandas as pd
import pyedflib
import scipy.stats as sp_stats
from mne_features.univariate import (compute_pow_freq_bands,
                                     get_univariate_funcs)

depth = sorted(['RAH1', 'LAH1', 'RA1', 'LA1', 'LEC1', 'REC1', 'RPHG1', 'LPHG1', 'RMH1', 'LMH1', 'RAH2', 'LAH2', 'RA2', 'LA2', 'LEC2', 'REC2', 'RPHG2', 'LPHG2', 'RMH2', 'LMH2'])
scalp = ['C3', 'C4', 'Cz', 'Pz', 'EOG1', 'EOG2']
frontal = scalp + ['F3', 'F4']


def write_edf_nan(fname, raw):
    # same function as mne but support nan values (const physical values)
    """Export raw to EDF/BDF file (requires pyEDFlib)."""

    suffixes = Path(fname).suffixes
    ext = "".join(suffixes[-1:])
    if ext == ".edf":
        filetype = pyedflib.FILETYPE_EDFPLUS
        dmin, dmax = -32768, 32767
    elif ext == ".bdf":
        filetype = pyedflib.FILETYPE_BDFPLUS
        dmin, dmax = -8388608, 8388607
    data = raw.get_data() * 1e6  # convert to micro-volts
    fs = raw.info["sfreq"]
    nchan = raw.info["nchan"]
    ch_names = raw.info["ch_names"]
    if raw.info["meas_date"] is not None:
        meas_date = raw.info["meas_date"]
    else:
        meas_date = None
    prefilter = (f"{raw.info['highpass']}Hz - "
                 f"{raw.info['lowpass']}")
    f = pyedflib.EdfWriter(fname, nchan, filetype)
    channel_info = []
    data_list = []
    for i in range(nchan):
        channel_info.append(dict(label=ch_names[i],
                                 dimension="uV",
                                 sample_rate=fs,
                                 physical_min=-5000,
                                 physical_max=5000,
                                 digital_min=dmin,
                                 digital_max=dmax,
                                 transducer="",
                                 prefilter=prefilter))
        data_list.append(data[i])
    f.setTechnician("Exported by MNELAB")
    f.setSignalHeaders(channel_info)
    if raw.info["meas_date"] is not None:
        f.setStartdatetime(meas_date)
    
    # note that currently, only blocks of whole seconds can be written
    f.writeSamples(data_list)
    for annot in raw.annotations:
        f.writeAnnotation(annot["onset"], annot["duration"], annot["description"])


def extract_epochs_features(epochs, subj):
    # extract features from the epochs of a single channel
    mobility, complexity = ant.hjorth_params(epochs, axis=1)
    feat = {
        'subj': np.full(len(epochs), subj),
        'epoch_id': np.arange(len(epochs)),
        'kurtosis': sp_stats.kurtosis(epochs, axis=1),
        'hjorth_mobility': mobility,
        'hjorth_complexity': complexity,
        'ptp_amp': np.ptp(epochs, axis=1),
        'samp_entropy': np.apply_along_axis(ant.sample_entropy, axis=1, arr=epochs)
    }
    kaiser = mne_features.univariate.compute_teager_kaiser_energy(np.array(epochs))
    reshaped_list = np.array(kaiser).reshape(-1, 12)
    X_new = pd.DataFrame(reshaped_list)
    X_new.columns = [
        f'teager_kaiser_energy_{i}_mean' if j % 2 == 0 else f'teager_kaiser_energy_{i}_std'
        for i in range(6) for j in range(2)
    ]

    feat = pd.DataFrame(feat)
    feat = pd.concat([feat, X_new], axis=1)
    return feat


def get_subj_data(raw, chan, subj, depth=True):
    # get features and labels of a single subject (all channels)
    window_size = 250  # ms
    epochs = []
    chan_raw = raw.copy().pick([chan]).get_data().flatten()

    # normalize channel
    chan_norm = (chan_raw - np.nanmean(chan_raw)) / np.nanstd(chan_raw)
    
    # run on all 250ms epochs excluding the last 1s
    for i in range(0, len(chan_norm) - 4 * window_size, window_size):
        if not np.isnan(chan_norm[i: i + window_size]).any():
            epochs.append(chan_norm[i: i + window_size])

    # add epoch-level features
    if depth:
        curr_feat = extract_epochs_features(epochs, subj)
    else: # zeeg
        curr_feat = get_subj_features_zeeg_fast(epochs, subj, raw.info['sfreq'])
    
    # add channel-level features
    chan_feat = {
        'chan_name': chan,
        'chan_ptp': np.ptp(chan_norm),
        'chan_kurt': sp_stats.kurtosis(chan_norm),
    }
    
    for feat in chan_feat.keys():
        curr_feat[feat] = chan_feat[feat]

    # save the epochs as column for debugging/visualization
    curr_feat['epoch'] = epochs
    return curr_feat


def extract_all_epochs_features(epochs, subj, sr):
    feat = {
        'subj': np.full(len(epochs), subj),
        'epoch_id': np.arange(len(epochs)),
    }

    selected_funcs = get_univariate_funcs(sr)
    selected_funcs.pop('spect_edge_freq', None)
    bands_dict = {'theta': (4, 8), 'alpha': (8, 12), 'sigma': (12, 16), 'beta': (16, 30), 'gamma': (30, 100), 'fast': (100, 300)}
    params = {'pow_freq_bands__freq_bands': bands_dict, 'pow_freq_bands__ratios': 'all', 'pow_freq_bands__psd_method': 'multitaper',
              'energy_freq_bands__freq_bands': bands_dict}
    X_new = None # Should change this: extract_features(np.array(epochs)[:, np.newaxis, :], sr, selected_funcs, funcs_params=params, return_as_df=True)
    X_new['abspow'] = compute_pow_freq_bands(sr, np.array(epochs), {'total': (0.1, 500)}, False, psd_method='multitaper')
    
    # rename columns
    names = []
    for name in X_new.columns:
        if type(name) is tuple:
            if name[1] == 'ch0':
                names.append(name[0])
            else:
                names.append(name[0] + '_' + name[1].replace('ch0_', ''))
        else:
            names.append(name)

    X_new.columns = names

    # add ratios between bands
    X_new['energy_freq_bands_ab'] = X_new['energy_freq_bands_alpha'] / X_new['energy_freq_bands_beta']
    X_new['energy_freq_bands_ag'] = X_new['energy_freq_bands_alpha'] / X_new['energy_freq_bands_gamma']
    X_new['energy_freq_bands_as'] = X_new['energy_freq_bands_alpha'] / X_new['energy_freq_bands_sigma']
    X_new['energy_freq_bands_af'] = X_new['energy_freq_bands_alpha'] / X_new['energy_freq_bands_fast']
    X_new['energy_freq_bands_at'] = X_new['energy_freq_bands_alpha'] / X_new['energy_freq_bands_theta']
    X_new['energy_freq_bands_bt'] = X_new['energy_freq_bands_beta'] / X_new['energy_freq_bands_theta']
    X_new['energy_freq_bands_bs'] = X_new['energy_freq_bands_beta'] / X_new['energy_freq_bands_sigma']
    X_new['energy_freq_bands_bg'] = X_new['energy_freq_bands_beta'] / X_new['energy_freq_bands_gamma']
    X_new['energy_freq_bands_bf'] = X_new['energy_freq_bands_beta'] / X_new['energy_freq_bands_fast']
    X_new['energy_freq_bands_st'] = X_new['energy_freq_bands_sigma'] / X_new['energy_freq_bands_theta']
    X_new['energy_freq_bands_sg'] = X_new['energy_freq_bands_sigma'] / X_new['energy_freq_bands_gamma']
    X_new['energy_freq_bands_sf'] = X_new['energy_freq_bands_sigma'] / X_new['energy_freq_bands_fast']
    X_new['energy_freq_bands_gt'] = X_new['energy_freq_bands_gamma'] / X_new['energy_freq_bands_theta']
    X_new['energy_freq_bands_gf'] = X_new['energy_freq_bands_gamma'] / X_new['energy_freq_bands_fast']
    X_new['energy_freq_bands_ft'] = X_new['energy_freq_bands_fast'] / X_new['energy_freq_bands_theta']

    # Convert to dataframe
    feat = pd.DataFrame(feat)
    feat = pd.concat([feat, X_new], axis=1)
    return feat


def get_subj_features_zeeg_fast(epochs, subj, sr):
    data = np.array(epochs)

    ptp_amp = mne_features.univariate.compute_ptp_amp(data)
    hjorth_mobility = mne_features.univariate.compute_hjorth_mobility(data)
    hjorth_complexity = mne_features.univariate.compute_hjorth_complexity(data)
    samp_entropy = mne_features.univariate.compute_samp_entropy(data, emb=2, metric='chebyshev')
    compute_spect_slope = mne_features.univariate.compute_spect_slope(
        sr, data, fmin=0.1, fmax=50, with_intercept=True, psd_method='welch', psd_params=None
    )
    # Compute power in specified frequency bands with normalization enabled and no ratios
    bands_dict = {'theta': (4, 8), 'alpha': (8, 12), 'sigma': (12, 16), 'beta': (16, 30), 'gamma': (30, 100), 'fast': (100, 300)}
    energy_freq_bands = mne_features.univariate.compute_energy_freq_bands(sr, data, freq_bands=bands_dict)

    app_entropy = mne_features.univariate.compute_app_entropy(data, emb=2, metric='chebyshev')
    decorr_time = mne_features.univariate.compute_decorr_time(sr, data)
    higuchi_fd = mne_features.univariate.compute_higuchi_fd(data, kmax=10)
    hjorth_complexity_spect = mne_features.univariate.compute_hjorth_complexity_spect(sr, data, normalize=False, psd_method='welch', psd_params=None)
    hjorth_mobility_spect = mne_features.univariate.compute_hjorth_mobility_spect(sr, data, normalize=False, psd_method='welch', psd_params=None)
    hurst_exp = mne_features.univariate.compute_hurst_exp(data)
    katz_fd = mne_features.univariate.compute_katz_fd(data)
    kurtosis = mne_features.univariate.compute_kurtosis(data)
    line_length = mne_features.univariate.compute_line_length(data)
    mean = mne_features.univariate.compute_mean(data)
    quantile = mne_features.univariate.compute_quantile(data, q=0.75)
    rms = mne_features.univariate.compute_rms(data)
    skewness = mne_features.univariate.compute_skewness(data)
    spect_entropy = mne_features.univariate.compute_spect_entropy(sr, data, psd_method='welch', psd_params=None)
    std = mne_features.univariate.compute_std(data)
    svd_entropy = mne_features.univariate.compute_svd_entropy(data, tau=2, emb=10)
    svd_fisher_info = mne_features.univariate.compute_svd_fisher_info(data, tau=2, emb=10)
    variance = mne_features.univariate.compute_variance(data)
    zero_crossings = mne_features.univariate.compute_zero_crossings(data, threshold=2.220446049250313e-16)
    abspow = mne_features.univariate.compute_pow_freq_bands(sr, data, {'total': (0.1, 500)}, False,
                                                            psd_method='multitaper')
    # Stack the computed feature arrays into columns for easier DataFrame creation
    stacked_arrays = np.column_stack(
        (hjorth_mobility, hjorth_complexity, samp_entropy, ptp_amp, app_entropy, decorr_time, higuchi_fd,
         hjorth_complexity_spect, hjorth_mobility_spect, hurst_exp, katz_fd, kurtosis, line_length, mean,
         quantile, rms, skewness, spect_entropy, std, svd_entropy, svd_fisher_info,
         variance, zero_crossings, abspow))

    df = pd.DataFrame(stacked_arrays)
    df.columns = ['hjorth_mobility', 'hjorth_complexity', 'samp_entropy', 'ptp_amp', 'app_entropy', 'decorr_time',
                  'higuchi_fd', 'hjorth_complexity_spect', 'hjorth_mobility_spect', 'hurst_exp', 'katz_fd', 'kurtosis',
                  'line_length', 'mean', 'quantile', 'rms', 'skewness', 'spect_entropy',
                  'std', 'svd_entropy', 'svd_fisher_info', 'variance', 'zero_crossings', 'abspow_']

    # Add metadata columns for subject, channel name, epoch label, and epoch ID
    df['subj'] = np.full(df.shape[0], subj)
    df['epoch_id'] = list(range(0, len(epochs)))

    kaiser = mne_features.univariate.compute_teager_kaiser_energy(data)
    kaiser_df = pd.DataFrame(
        np.array(kaiser).reshape(-1, 12),
        columns=['teager_kaiser_energy_0_mean', 'teager_kaiser_energy_0_std',
                 'teager_kaiser_energy_1_mean', 'teager_kaiser_energy_1_std',
                 'teager_kaiser_energy_2_mean', 'teager_kaiser_energy_2_std',
                 'teager_kaiser_energy_3_mean', 'teager_kaiser_energy_3_std',
                 'teager_kaiser_energy_4_mean', 'teager_kaiser_energy_4_std',
                 'teager_kaiser_energy_5_mean', 'teager_kaiser_energy_5_std']
    )

    pow_freq_bands = mne_features.univariate.compute_pow_freq_bands(data=epochs, sfreq=sr, freq_bands=bands_dict,
                                                                    normalize=True, ratios=None, ratios_triu=False,
                                                                    psd_method='multitaper', log=False, psd_params=None)
    pow_freq_bands_df = pd.DataFrame(np.array(pow_freq_bands).reshape(-1, 6),
                                     columns=['pow_freq_bands_theta', 'pow_freq_bands_alpha', 'pow_freq_bands_sigma',
                                              'pow_freq_bands_beta', 'pow_freq_bands_gamma', 'pow_freq_bands_fast'])

    for band1 in ['theta', 'alpha', 'sigma', 'beta', 'gamma', 'fast']:
        for band2 in ['theta', 'alpha', 'sigma', 'beta', 'gamma', 'fast']:
            if band1 == band2:
                continue
            else:
                pow_freq_bands_df[f'pow_freq_bands_{band1}/{band2}'] = pow_freq_bands_df[f'pow_freq_bands_{band1}'] / \
                                                                       pow_freq_bands_df[f'pow_freq_bands_{band2}']

    # Spectral slope components
    spect_slope = pd.DataFrame(
        np.array(compute_spect_slope).reshape(-1, 4),
        columns=['spect_slope_intercept', 'spect_slope_slope', 'spect_slope_MSE', 'spect_slope_R2']
    )

    # Energy frequency bands with each band as a separate column
    energy_freq_bands = pd.DataFrame(
        np.array(energy_freq_bands).reshape(-1, 6),
        columns=['energy_freq_bands_theta', 'energy_freq_bands_alpha', 'energy_freq_bands_sigma',
                 'energy_freq_bands_beta', 'energy_freq_bands_gamma', 'energy_freq_bands_fast']
    )
    # Add ratios between bands
    energy_freq_bands['energy_freq_bands_gf'] = energy_freq_bands['energy_freq_bands_gamma'] / energy_freq_bands[
        'energy_freq_bands_fast']
    energy_freq_bands['energy_freq_bands_bg'] = energy_freq_bands['energy_freq_bands_beta'] / energy_freq_bands[
        'energy_freq_bands_gamma']

    wavelet_coef_energy = mne_features.univariate.compute_wavelet_coef_energy(data, wavelet_name='db4')
    wavelet_coef_energy_df = pd.DataFrame(
        np.array(wavelet_coef_energy).reshape(-1, 5),
        columns=['wavelet_coef_energy_0', 'wavelet_coef_energy_1', 'wavelet_coef_energy_2', 'wavelet_coef_energy_3',
                 'wavelet_coef_energy_4'])

    # Concatenate all feature DataFrames side by side
    subj_features = pd.concat(
        [df, spect_slope, energy_freq_bands, kaiser_df, pow_freq_bands_df, wavelet_coef_energy_df], axis=1)

    # add ratios between bands
    for band1 in ['theta', 'alpha', 'sigma', 'beta', 'gamma', 'fast']:
        for band2 in ['theta', 'alpha', 'sigma', 'beta', 'gamma', 'fast']:
            if band1 == band2:
                continue
            else:
                subj_features[f'energy_freq_bands_{band1[0]}{band2[0]}'] = subj_features[f'energy_freq_bands_{band1}'] / \
                                                                           subj_features[f'energy_freq_bands_{band2}']
    subj_features = subj_features[
        ['subj', 'epoch_id', 'app_entropy', 'decorr_time', 'energy_freq_bands_theta', 'energy_freq_bands_alpha',
         'energy_freq_bands_sigma', 'energy_freq_bands_beta', 'energy_freq_bands_gamma', 'energy_freq_bands_fast',
         'higuchi_fd', 'hjorth_complexity', 'hjorth_complexity_spect', 'hjorth_mobility', 'hjorth_mobility_spect',
         'hurst_exp', 'katz_fd', 'kurtosis', 'line_length', 'mean', 'pow_freq_bands_theta', 'pow_freq_bands_alpha',
         'pow_freq_bands_sigma', 'pow_freq_bands_beta', 'pow_freq_bands_gamma', 'pow_freq_bands_fast',
         'pow_freq_bands_theta/alpha', 'pow_freq_bands_theta/sigma', 'pow_freq_bands_theta/beta',
         'pow_freq_bands_theta/gamma', 'pow_freq_bands_theta/fast', 'pow_freq_bands_alpha/theta',
         'pow_freq_bands_alpha/sigma', 'pow_freq_bands_alpha/beta', 'pow_freq_bands_alpha/gamma',
         'pow_freq_bands_alpha/fast', 'pow_freq_bands_sigma/theta', 'pow_freq_bands_sigma/alpha',
         'pow_freq_bands_sigma/beta', 'pow_freq_bands_sigma/gamma', 'pow_freq_bands_sigma/fast',
         'pow_freq_bands_beta/theta', 'pow_freq_bands_beta/alpha', 'pow_freq_bands_beta/sigma',
         'pow_freq_bands_beta/gamma', 'pow_freq_bands_beta/fast', 'pow_freq_bands_gamma/theta',
         'pow_freq_bands_gamma/alpha', 'pow_freq_bands_gamma/sigma', 'pow_freq_bands_gamma/beta',
         'pow_freq_bands_gamma/fast', 'pow_freq_bands_fast/theta', 'pow_freq_bands_fast/alpha',
         'pow_freq_bands_fast/sigma', 'pow_freq_bands_fast/beta', 'pow_freq_bands_fast/gamma', 'ptp_amp', 'quantile',
         'rms', 'samp_entropy', 'skewness', 'spect_entropy', 'spect_slope_intercept', 'spect_slope_slope',
         'spect_slope_MSE', 'spect_slope_R2', 'std', 'svd_entropy', 'svd_fisher_info', 'teager_kaiser_energy_0_mean',
         'teager_kaiser_energy_0_std', 'teager_kaiser_energy_1_mean', 'teager_kaiser_energy_1_std',
         'teager_kaiser_energy_2_mean', 'teager_kaiser_energy_2_std', 'teager_kaiser_energy_3_mean',
         'teager_kaiser_energy_3_std', 'teager_kaiser_energy_4_mean', 'teager_kaiser_energy_4_std',
         'teager_kaiser_energy_5_mean', 'teager_kaiser_energy_5_std', 'variance', 'wavelet_coef_energy_0',
         'wavelet_coef_energy_1', 'wavelet_coef_energy_2', 'wavelet_coef_energy_3', 'wavelet_coef_energy_4',
         'zero_crossings', 'abspow_', 'energy_freq_bands_ab', 'energy_freq_bands_ag', 'energy_freq_bands_as',
         'energy_freq_bands_af', 'energy_freq_bands_at', 'energy_freq_bands_bt', 'energy_freq_bands_bs',
         'energy_freq_bands_bg', 'energy_freq_bands_bf', 'energy_freq_bands_st', 'energy_freq_bands_sg',
         'energy_freq_bands_sf', 'energy_freq_bands_gt', 'energy_freq_bands_gf', 'energy_freq_bands_ft']]
    return subj_features


def map_nan_index(edf):
    # visualize the models
    sr = 1000
    window_size = 250
    raw = mne.io.read_raw(edf)
    raw_data = raw.pick([raw.ch_names[0]])
    if raw_data.info['sfreq'] != sr:
        raw_data.resample(sr)
    raw_data = raw_data.get_data(reject_by_annotation='NaN')[0]
    map = []
    for j, i in enumerate(range(0, len(raw_data), window_size)):
        curr_block = raw_data[i: i + window_size]
        if i + window_size < len(raw_data):
            if not np.isnan(curr_block).any():
                map.append(j)
    return map


def get_scalp_channels_info(excel_path, sheets={"TLV": 0, "Bonn": 753590609}):
    channels_per_sub = {}
    channels_count = {}
    for sheet in sheets.values():
        info = pd.read_csv(f"{excel_path}{sheet}", index_col=0)
        for sub, channels in info.loc["scalp"].items():
            channels_list = [x for x in channels.split(", ") if x.strip()]
            channels_list = [channel if channel != "X" else "None" for channel in channels_list]
            channels_per_sub[str(sub)] = channels_list
            for channel in channels_list:
                channels_count[channel] = channels_count.get(channel, 0) + 1
    return channels_per_sub, channels_count


def get_files_from_dir(dir_path, exclude_prefix=800):
    file_names = [f.name for f in dir_path.iterdir() if f.is_file()]
    first_files = [x for x in file_names if x.split(".")[0][-2:] not in ("-1", "-2")]
    if exclude_prefix:
        subjects = list(set([x.split("_")[0] for x in first_files]))
        relevant_subjects = [x for x in subjects if int(x[1:]) < exclude_prefix or int(x[1:]) > exclude_prefix + 99]
        first_files = [x for x in first_files if x.split("_")[0] in relevant_subjects]
    return first_files


def get_all_channels_info(dir_path):
    files = get_files_from_dir(dir_path)
    channels_per_sub = {}
    channels_count = {}
    for file in files:
        subject = file.split("_")[0]
        clean_raw = mne.io.read_raw(rf"{dir_path}\{file}")
        channels_per_sub[subject] = clean_raw.ch_names
        for channel in clean_raw.ch_names:
            channels_count[channel] = channels_count.get(channel, 0) + 1
    return channels_per_sub, channels_count

    
def rename_bonn_depth_channels(raw):
    new_names = {}
    for name in raw.info.ch_names:
        new_name = name
        new_name = new_name.replace('PHC', 'PHG')
        if new_name not in scalp + ['Cb1', 'Cb2', 'EKG+', 'EMG+']:
            new_names[name] = new_name[-2] + new_name[:-2] + new_name[-1]
        else:
            new_names[name] = new_name
    raw.rename_channels(new_names)


def compare_and_remove_identical_files(old_dir, new_dir):
    deleted = []
    files = sorted(os.listdir(old_dir))
    for old_file in files:
        psub = old_file.split("_")[0]
        old_path = os.path.join(old_dir, old_file)
        new_path = fr"{new_dir}\{psub}_scalp_filtered.fif"
        old_raw = mne.io.read_raw_fif(old_path, preload=True)
        new_raw = mne.io.read_raw_fif(new_path, preload=True)
        if np.array_equal(old_raw.get_data(), new_raw.get_data()) and str(old_raw.info) == str(new_raw.info):
            os.remove(old_path)
            deleted.append(old_path)
    return deleted