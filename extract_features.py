import joblib
import mne
import os
from add_subject_utils import depth, get_subj_data
import pandas as pd
import numpy as np
import pickle
import gc

mne.set_log_level("error")
base_dir = r"I:\Shaked\clean_with_scalp"
features_dir = r"I:\Shaked\features_data"

# Define running parameters
model_name = "lgbm_full_f15_s25_b_V5.pkl"
use_all_channels = True
confidence = 0.8
num_channels_agreement = 2
out_dir = r"I:\Shaked\test1"
depth_model, features = joblib.load(rf"I:\Shaked\{model_name}").values()


def save_params_info(info_path=None):
    info_path = rf"{out_dir}\info.txt" if not info_path else info_path
    if not os.path.exists(info_path):
        with open(info_path, "w") as f:
            f.write(f"model_name: {model_name}\n")
            f.write(f"use_all_channels: {use_all_channels}\n")
            f.write(f"confidence: {confidence}\n")
            f.write(f"num_channels_agreement: {num_channels_agreement}\n")


def extract_depth_vector(sub, raw):
    if not os.path.exists(fr"{out_dir}\y_depth_{sub}.npy"):
        print(f"Extracting depth: {sub}")
        total_y = None
        curr_channels = [chan for chan in raw.ch_names if chan in depth]
        if not use_all_channels:
            # get only one deepest channel from each brain region
            min_indexes = {}
            for ch_name in curr_channels:
                prefix = ch_name[:-1]
                index = int(ch_name[-1])
                if prefix not in min_indexes or index < int(min_indexes[prefix][-1]):
                    min_indexes[prefix] = ch_name
            curr_channels = list(min_indexes.values())
        
        for chan in curr_channels:
            curr_data = get_subj_data(raw, chan, sub, depth=True)
            predictions = depth_model.predict_proba(curr_data[features])
            curr_y = (predictions[:, 1] >= confidence).astype(int)
            if total_y is None:
                total_y = curr_y
            else:
                # summing answers from all the channels
                total_y += curr_y

        # at least X channels should be above threshold
        total_y[total_y < num_channels_agreement] = 0
        total_y[total_y >= num_channels_agreement] = 1
        print(f"Number of spikes: {sum(total_y)}")
        print(f"Spikes ratio: {total_y.sum() / len(total_y)}")
        np.save(fr"{out_dir}\y_depth_{sub}.npy", total_y)


def extract_scalp_features(sub, raw):
    scalp_channels = [["C4", "F3"], ["C3", "F4"], ["EOG1", "EOG2"]]
    for channel_couple in scalp_channels:
        curr_path = rf'{features_dir}\scalp_features_{"_".join(channel_couple)}_{sub}.pkl'
        if all(ch in raw.ch_names for ch in channel_couple) and not os.path.exists(curr_path):
            print(f"Extracting features: {sub}")
            scalp1 = get_subj_data(raw, channel_couple[0], sub, depth=False)
            scalp2 = get_subj_data(raw, channel_couple[1], sub, depth=False)
            # combine and rename columns
            subj_feat = pd.concat([scalp1, scalp2], axis=1, ignore_index=True) 
            subj_feat.columns = [f'{channel_couple[0]}_{col}' for col in scalp1.columns] + [f'{channel_couple[1]}_{col}' for col in scalp2.columns]
            subj_feat.to_pickle(curr_path)
        elif not os.path.exists(curr_path):
            print(f"Subject {sub} is missing: {[x for x in channel_couple if x not in raw.ch_names]}")


def main():
    save_params_info()
    errors = {}
    for file_path in sorted(os.listdir(base_dir)):
        try:
            if file_path.split(".")[0][-2:] == "-1":
                continue

            sub = file_path.split("_")[0][1:]
            clean_raw = mne.io.read_raw_fif(f"{base_dir}/{file_path}")
            nan_clean = clean_raw.get_data(reject_by_annotation='NaN')
            nan_raw = mne.io.RawArray(nan_clean, clean_raw.info)
            extract_depth_vector(sub, nan_raw)
            extract_scalp_features(sub, nan_raw)
            
            del clean_raw
            del nan_raw
            gc.collect()
        
        except Exception as e:
            errors[sub] = e
        
    print(errors)
    with open(f"{out_dir}/errors.pkl", "wb") as f:
        pickle.dump(errors, f)


if __name__ == "__main__":
    main()
