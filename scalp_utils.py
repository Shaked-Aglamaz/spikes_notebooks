import numpy as np
import pandas as pd
import mne
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from lightgbm import LGBMClassifier
from depth_utils import get_metrics, calc_features_before_split, calc_features_after_split, channel_feat
import joblib
from IPython.display import clear_output
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE

# General params
sr = 1000
# scalp_edf_path = 'C:\\Users\\user\\PycharmProjects\\pythonProject\\%s_clean.edf'
# scalp_edf_path = 'C:\\UCLA\\%s_clean_eog.edf'
scalp_edf_path = 'C:\\UCLA\\%s_clean_eog.edf'
mtl_path = 'C:\\UCLA\\%s_clean_mtl_annot.fif'
only_right = ['017', '018', '025', '38', '422']
only_left = ['44', '46', '396']
nrem_sample = {'38': 80, '394': 20, '396': 65, '398': 50, '400': 10, '402': 10, '404': 100, '405': 53, '406': 90, '414': 75, '415': 55,
               '416': 190, '417': 86, '423': 20, '426': 185, '429': 7}

# Thesis model
# model = joblib.load('lgbm_fast.pkl')
# features_names = pd.read_csv('features_fast.csv').columns.tolist()
# only AH
model_AH = joblib.load('results/lgbm_AH_only_7.pkl')
features_names_AH = pd.read_csv('results/features_AH_only_7.csv').columns.tolist()

# no leak
model = joblib.load('results/lgbm_full_no_leak.pkl')
features_names = pd.read_csv('results/features_lgbm_no_leak.csv').columns.tolist()

all_subject = ['394', '396', '398', '400', '402', '404', '405', '406', '414', '415',
               '416', '417', '422', '423', '426', '429', '489'] + \
              ['013', '017', '018', '025', '38', '39', '44', '46', '47', '48', '51']
# rf biased
subj_eog1 = ['404', '414', '415', '422', '423', '429']
subj_eog2 = ['398', '402', '405', '406', '416', '417', '426']
subj_eog12 = ['38', '394', '396', '400']
# all rf are ~0
subj_bad = ['394', '396', '400', '405', '417', '422', '423', '426', '429']
# good = ['38', '398', '402', '404', '406', '414', '415', '416']
good_subj = [x for x in all_subject if x not in subj_bad]


def format_raw_night_without_nan(edf, channel, norm='raw'):
    epochs = []
    window_size = int(sr / 4)
    if '-' in channel and 'REF' not in channel:
        chans = channel.split('-')
        raw = mne.io.read_raw_edf(edf).pick_channels(chans).resample(sr)
        # raw = remove_rem(raw, edf)
        mne.set_bipolar_reference(raw, chans[0], chans[1], ch_name=channel)
        raw_data = raw.get_data()[0]
    else:
        raw_data = mne.io.read_raw_edf(edf).pick_channels([channel])
        if raw_data.info['sfreq'] != 1000:
            raw_data.resample(sr)
        raw_data = raw_data.get_data()[0]

    if norm == 'raw':
        raw_data = (raw_data - np.nanmean(raw_data)) / np.nanstd(raw_data)
    for i in range(0, len(raw_data), window_size):
        curr_block = raw_data[i: i + window_size]
        if i + window_size < len(raw_data):
            epochs.append(curr_block)

    # Normalization
    epochs = np.array(epochs)
    if norm == 'epochs':
        epochs = (epochs - epochs.mean()) / epochs.std()
    return epochs

def format_raw_night(edf, channel):
    epochs = []
    window_size = int(sr / 4)
    raw_data = mne.io.read_raw(edf).pick_channels([channel])
    if raw_data.info['sfreq'] != sr:
        raw_data.resample(sr)
    raw_data = raw_data.get_data(reject_by_annotation='NaN')[0]

    # normalization
    raw_data = (raw_data - np.nanmean(raw_data)) / np.nanstd(raw_data)
    for i in range(0, len(raw_data), window_size):
        curr_block = raw_data[i: i + window_size]
        if i + window_size < len(raw_data) and not np.isnan(curr_block).any():
            epochs.append(curr_block)

    epochs = np.array(epochs)
    return epochs


def map_nan_index(edf):
    window_size = int(sr / 4)
    raw = mne.io.read_raw(edf)
    raw_data = raw.pick_channels([raw.ch_names[0]])
    if raw_data.info['sfreq'] != sr:
        raw_data.resample(sr)
    raw_data = raw_data.get_data(reject_by_annotation='NaN')[0]
    map = []

    for j, i in enumerate(range(0, len(raw_data), window_size)):
        curr_block = raw_data[i: i + window_size]
        if i + window_size < len(raw_data):
            if not np.isnan(curr_block).any():
                map.append(j)
            else:
                print('nan')
    return map


def get_all_y_multi_channel(subjects=['38', '396', '398', '400', '402', '406', '415', '416', '423']):
    side1_y = None
    channels = ['RAH1', 'LAH1']
    y_all = np.empty(0)
    for subj in subjects:
        if subj == '404':
            neighbors = {'R': ['RAH1-RAH2', 'RAH3'], 'L': ['LAH1-LAH2', 'LPHG1']}
        elif subj == '426':
            neighbors = {'R': ['RA1-RA2', 'REC1'], 'L': ['LA1-LA2', 'LEC1']}
            channels = ['RA1', 'LA1']
        elif subj == '422':
            neighbors = {'R': ['RAH1-RAH2', 'RPHG1']}
        else:
            neighbors = {'R': ['RAH1-RAH2', 'RA1'], 'L': ['LAH1-LAH2', 'LA1']}
        for channel in channels:
            if not ((subj in only_left and channel == 'RAH1') or (subj in only_right and channel == 'LAH1')):
                print(channel)
                x = format_raw_night(scalp_edf_path % subj, channel)
                features = calc_features_after_split(calc_features_before_split(x, subj))
                for neighbor in neighbors[channel[0]]:
                    x_neighbor = format_raw_night(scalp_edf_path % subj, neighbor)
                    if subj == '426':
                        prefix = 'AH1-AH2' if '-' in neighbor else 'A1'
                    else:
                        prefix = neighbor.replace(channel[0], '') if '-' in neighbor else 'A1'
                    features_neighbor = calc_features_after_split(calc_features_before_split(x_neighbor, subj)).add_prefix(f'{prefix}_')
                    features = pd.concat([features, features_neighbor.iloc[:, 2:]], axis=1)

                # Here I have all features for one side
                if side1_y is None:
                    side1_y = model.predict(features[features_names[1:]])

        if subj in only_right + only_left:
            y_all = np.concatenate((y_all, side1_y))
        else:
            side2_y = model.predict(features[features_names[1:]])
            y_bilateral = side1_y + side2_y
            y_bilateral[y_bilateral == 2] = 1
            y_all = np.concatenate((y_all, y_bilateral))

        side1_y = None

    return y_all


def get_all_y_AH(subjects=['38', '396', '398', '400', '402', '406', '415', '416', '423'], bi=False,
                 path=scalp_edf_path, model_path=None):
    if model_path:
        model_AH = joblib.load(model_path)
        features_names_AH = model_AH.get_booster().feature_names
    else:
        if bi:
            model_AH = joblib.load("results/lgbm_bi_only_7.pkl")
        else:
           model_AH = joblib.load('results/lgbm_AH_only_no_p.pkl')
            # model_AH = joblib.load('results/rf_AH_only_no_p.pkl')
        features_names_AH = model_AH.feature_name_

    channels = ['RAH1-RAH2', 'LAH1-LAH2'] if bi else ['RAH1', 'LAH1']
    side1_y = None
    y_all = np.empty(0)
    for subj in subjects:
        for channel in channels:
            if not ((subj in only_left and 'RAH1' in channel) or (subj in only_right and 'LAH1' in channel)):
                if subj == '426':
                    channel = channel.replace('H', '')
                x = format_raw_night(path % subj, channel)
                features = calc_features_after_split(calc_features_before_split(x, subj))

                # Here I have all features for one side
                if side1_y is None:
                    side1_y = model_AH.predict(features[features_names_AH])

        if subj in only_right + only_left:
            y_all = np.concatenate((y_all, side1_y))
        else:
            side2_y = model_AH.predict(features[features_names_AH])
            y_bilateral = side1_y + side2_y
            y_bilateral[y_bilateral == 2] = 1
            y_all = np.concatenate((y_all, y_bilateral))

        side1_y = None

    return y_all


def get_all_y_AH_bi(subjects=['38', '396', '398', '400', '402', '406', '415', '416', '423']):
    model_bi = joblib.load('results/lgbm_full_AH+bi_7.pkl')
    feature_names = model_bi.feature_name_
    side1_y = None
    y_all = np.empty(0)
    for subj in subjects:
        for channel in ['RAH1', 'LAH1']:
            if not ((subj in only_left and channel == 'RAH1') or (subj in only_right and channel == 'LAH1')):
                if subj == '426':
                    channel = channel.replace('H', '')
                x = format_raw_night(scalp_edf_path % subj, channel)
                features = calc_features_after_split(calc_features_before_split(x, subj))
                x_bi = format_raw_night(scalp_edf_path % subj, f'{channel}-{channel[:-1]}2')
                features_bi = calc_features_after_split(calc_features_before_split(x_bi, subj)).add_prefix('bi_')
                features = pd.concat([features, features_bi.iloc[:, 2:]], axis=1)
                # Here I have all features for one side
                if side1_y is None:
                    side1_y = model_bi.predict(features[feature_names])

        if subj in only_right + only_left:
            y_all = np.concatenate((y_all, side1_y))
        else:
            side2_y = model_bi.predict(features[feature_names])
            y_bilateral = side1_y + side2_y
            y_bilateral[y_bilateral == 2] = 1
            y_all = np.concatenate((y_all, y_bilateral))

        side1_y = None

    return y_all


def format_combine_channel(subj, chans=['PZ','C3','C4'], norm='raw'):
    epochs = []
    window_size = int(sr / 4)

    if subj in ['422', '429']:
        chans = ['C3', 'C4']
        raw = mne.io.read_raw_edf(scalp_edf_path % subj).pick_channels(chans).resample(sr)
        raw = mne.channels.combine_channels(raw, {'avg': [0, 1]}, method='mean')
    else:
        raw = mne.io.read_raw_edf(scalp_edf_path % subj).pick_channels(chans).resample(sr)
        raw = mne.channels.combine_channels(raw, {'avg': [0, 1, 2]}, method='mean')
    raw_data = raw.get_data()[0]

    if norm == 'raw':
        raw_data = (raw_data - raw_data.mean()) / raw_data.std()

    for i in range(0, len(raw_data), window_size):
        curr_block = raw_data[i: i + window_size]
        if i + window_size < len(raw_data):
            epochs.append(curr_block)

    # Normalization
    epochs = np.array(epochs)
    if norm == 'epochs':
        epochs = (epochs - epochs.mean()) / epochs.std()
    return epochs


def get_all_feat_eog(eog_num, subjects=['38', '396', '398', '402', '406', '415', '416']):
    feat_all = pd.DataFrame()
    for subj in subjects:
        x = format_raw_night(scalp_edf_path % subj, 'EOG' + eog_num)
        features = calc_features_before_split(x, subj)
        feat_all = pd.concat([feat_all, features], axis=0)

    return feat_all



def get_all_feat_eog_with_chan_feat(eog_num, subjects=['38', '396', '398', '402', '406', '415', '416'], path=mtl_path):
    feat_all = pd.DataFrame()
    for subj in subjects:
        x = format_raw_night(path % subj, 'EOG' + eog_num)
        chan_feat = channel_feat(path % subj, 'EOG' + eog_num)
        features = calc_features_before_split(x, subj)
        for feat in chan_feat.keys():
            features[feat] = chan_feat[feat]
        feat_all = pd.concat([feat_all, features], axis=0)

    return feat_all


def get_all_feat_chan_with_chan_feat(chan, subjects=['38', '396', '398', '402', '406', '415', '416'], path=scalp_edf_path):
    feat_all = pd.DataFrame()
    for subj in subjects:
        x = format_raw_night(path % subj, chan)
        chan_feat = channel_feat(path % subj, chan)
        features = calc_features_before_split(x, subj)
        for feat in chan_feat.keys():
            features[feat] = chan_feat[feat]
        feat_all = pd.concat([feat_all, features], axis=0)

    return feat_all


def get_all_feat_avg(subjects=['38', '396', '398', '402', '406', '415', '416']):
    feat_all = pd.DataFrame()
    for subj in subjects:
        x = format_combine_channel(subj)
        features = calc_features_before_split(x, subj)
        feat_all = pd.concat([feat_all, features], axis=0)

    return feat_all


def run_all(subjects=['396', '398', '402', '406', '415', '416']):
    y_all_bi = get_all_y_multi_channel(subjects=subjects)
    feat_avg = get_all_feat_avg(subjects=subjects)
    feat_all_eog2 = get_all_feat_eog('2', subjects=subjects)
    clear_output()
    feat_avg.reset_index(inplace=True, drop=True)
    feat_all_eog2.reset_index(inplace=True, drop=True)
    feat_eog2_avg = pd.concat([feat_avg, feat_all_eog2.iloc[:, 2:].add_suffix('_2')], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(feat_eog2_avg, y_all_bi, stratify=y_all_bi, random_state=20)

    # Add separated norm values
    X_train = calc_features_after_split(X_train)
    X_test = calc_features_after_split(X_test)

    return X_train, X_test, y_train, y_test


def plt_cls_old(feat, y, stratify=False, over=False):
    if stratify:
        strat = pd.DataFrame()
        strat['y'] = list(map(str, y))
        strat['subj'] = list(feat['subj'].astype(str))
        strat['stratify'] = strat[['y', 'subj']].apply(lambda x: ''.join(x), axis=1)
        X_train, X_test, y_train, y_test = train_test_split(feat, y, stratify=strat['stratify'], random_state=20)
    else:
        X_train, X_test, y_train, y_test = train_test_split(feat, y, stratify=y, random_state=20)

    if over:
        oversample = SMOTE()
        X_train, y_train = oversample.fit_resample(X_train, y_train)
    X_train_data = X_train.iloc[:, 2:]
    selector = VarianceThreshold(.1)
    selector.fit_transform(X_train_data)
    X_train_data = X_train_data[X_train_data.columns[selector.get_support(indices=True)]]
    X_test_data = X_test[X_train_data.columns]

    classifiers_2_all = {
        "Random Forest": RandomForestClassifier(),
        "LGBM": LGBMClassifier(),
        # "XGB": xgb.XGBClassifier(objective="multi:softmax", num_class=2, use_label_encoder=False),
    }

    pred_details = {}
    f, axes = plt.subplots(1, len(classifiers_2_all), figsize=(10, 5), sharey='row')

    for i, (key, classifier) in enumerate(classifiers_2_all.items()):
        y_pred = classifier.fit(X_train_data, y_train).predict(X_test_data)
        cf_matrix = confusion_matrix(y_test, y_pred)
        metrics = get_metrics(cf_matrix)
        disp = ConfusionMatrixDisplay(cf_matrix)
        disp.plot(ax=axes[i], xticks_rotation=45)
        precision = '{0:.2f}'.format(metrics['precision'])
        recall = '{0:.2f}'.format(metrics['recall'])
        text = """precision: {0}
                  recall: {1}""".format(precision, recall)
        axes[i].annotate(text, xy=(1, 0), xycoords='axes fraction', fontsize=16,
                         xytext=(-60, -40), textcoords='offset points',
                         ha='right', va='top')
        disp.ax_.set_title(key)
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel('')
        if i != 0:
            disp.ax_.set_ylabel('')
        curr_pred_details = pd.DataFrame(data=X_test, copy=True)
        curr_pred_details['pred'] = y_pred
        pred_details[key] = curr_pred_details

    f.text(0.45, 0.1, 'Predicted label', ha='left')
    plt.subplots_adjust(wspace=0.40, hspace=0.1)
    plt.show()
    return classifiers_2_all, pred_details


def plot_results(x_train, x_test, y_train, y_test, prob=None):
    x_train_data = x_train.iloc[:, 2:]
    selector = VarianceThreshold(.1)
    selector.fit_transform(x_train_data)
    x_train_data = x_train_data[x_train_data.columns[selector.get_support(indices=True)]]
    x_test_data = x_test[x_train_data.columns]
    classifiers = {
        "Random Forest": RandomForestClassifier(),
        "LGBM": LGBMClassifier(),
    }

    pred_details = {}
    f, axes = plt.subplots(1, len(classifiers), figsize=(10, 5), sharey='row')

    for i, (key, classifier) in enumerate(classifiers.items()):
        if prob is not None:
            y_pred = classifier.predict_proba(x_test_data).T
            cf_matrix = confusion_matrix(y_test, [p > prob for p in y_pred[1]])
        else:
            y_pred = classifier.fit(x_train_data, y_train).predict(x_test_data)
            cf_matrix = confusion_matrix(y_test, y_pred)
        metrics = get_metrics(cf_matrix)
        disp = ConfusionMatrixDisplay(cf_matrix)
        disp.plot(ax=axes[i], xticks_rotation=45)
        precision = '{0:.2f}'.format(metrics['precision'])
        recall = '{0:.2f}'.format(metrics['recall'])
        text = """precision: {0}
                      recall: {1}""".format(precision, recall)
        axes[i].annotate(text, xy=(1, 0), xycoords='axes fraction', fontsize=16,
                         xytext=(-60, -40), textcoords='offset points',
                         ha='right', va='top')
        disp.ax_.set_title(key)
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel('')
        if i != 0:
            disp.ax_.set_ylabel('')
        curr_pred_details = pd.DataFrame(data=x_test, copy=True)
        curr_pred_details['pred'] = y_pred
        pred_details[key] = curr_pred_details

    f.text(0.45, 0.1, 'Predicted label', ha='left')
    plt.subplots_adjust(wspace=0.40, hspace=0.1)
    plt.show()
    return classifiers, pred_details


def plot_prob(cls, x_train, x_test, y_test, prob=0.8):
    x_train_data = x_train.iloc[:, 2:]
    selector = VarianceThreshold(.1)
    selector.fit_transform(x_train_data)
    x_train_data = x_train_data[x_train_data.columns[selector.get_support(indices=True)]]
    x_test_data = x_test[x_train_data.columns]
    pred_details = {}
    f, axes = plt.subplots(1, len(cls), figsize=(10, 5), sharey='row')

    for i, (key, classifier) in enumerate(cls.items()):
        y_pred = classifier.predict_proba(x_test_data).T
        cf_matrix = confusion_matrix(y_test, [p > prob for p in y_pred[1]])
        metrics = get_metrics(cf_matrix)
        disp = ConfusionMatrixDisplay(cf_matrix)
        disp.plot(ax=axes[i], xticks_rotation=45)
        precision = '{0:.2f}'.format(metrics['precision'])
        recall = '{0:.2f}'.format(metrics['recall'])
        text = """precision: {0}
                      recall: {1}""".format(precision, recall)
        axes[i].annotate(text, xy=(1, 0), xycoords='axes fraction', fontsize=16,
                         xytext=(-60, -40), textcoords='offset points',
                         ha='right', va='top')
        disp.ax_.set_title(key)
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel('')
        if i != 0:
            disp.ax_.set_ylabel('')
        curr_pred_details = pd.DataFrame(data=x_test, copy=True)
        curr_pred_details['pred'] = y_pred
        pred_details[key] = curr_pred_details

    f.text(0.45, 0.1, 'Predicted label', ha='left')
    plt.subplots_adjust(wspace=0.40, hspace=0.1)
    plt.show()

    return pred_details


def save_depth_spikes(subjects=['396', '398', '402', '406', '415', '416']):
    for subj in subjects:
        y_bi = get_all_y_multi_channel([subj])
        spike_index = np.transpose((y_bi == 1).nonzero())
        pd.DataFrame(spike_index).to_csv(f"filter_depth_lgbm_{subj}.csv", header=False, index=False)


def save_scalp_predictions(details, file_name):
    details[(details['pred'] == 1)].sort_values(by='subj')[['subj', 'epoch_id']].to_csv(file_name)


def get_importance(cls, feature_names=None, top_num=15):
    if feature_names is None:
        feature_names = pd.DataFrame(cls.feature_name_)
    feature_names['imp'] = cls.feature_importances_
    feature_imp = feature_names.sort_values(by=['imp'], ascending=False)
    feature_imp.head(top_num)


# maybe plot both side by side
def plot_detection_distribution_hist(details, confidence=0.8):
    # details[(details['pred'] >= confidence)].sort_values(by='subj')['subj'].hist()
    data = details[(details['pred'] >= confidence)].sort_values(by='subj')['subj'].value_counts()
    fig, ax = plt.subplots()
    ax.bar(data.index, data.values)
    unique, counts = np.unique(details[(details['pred'] >= confidence)].sort_values(by='subj')['subj'].to_numpy(), return_counts=True)
    return dict(zip(unique, counts))


def plot_detection_distribution_pie(details, confidence=0.8):
    unique, counts = np.unique(details[(details['pred'] >= confidence)].sort_values(by='subj')['subj'].to_numpy(), return_counts=True)
    patches, labels, pct_texts = plt.pie(counts, labels=unique,  rotatelabels=True, startangle=10,
                                         # colors=['tab:green', 'tab:red', 'tab:blue', 'tab:orange', 'tab:purple'],
                                         autopct=lambda p: '{:.1f}% ({:.0f})'.format(p, (p/100)*counts.sum()))

    for label, pct_text in zip(labels, pct_texts):
        pct_text.set_rotation(label.get_rotation())


def save_features_dicts(subjects=['38', '394', '396', '398', '400', '402', '404', '405', '406', '414', '415', '416', '417', '423', '426', '429'], detection_func='AH', file_name='all'):
    # get everyone feat and y
    eog1_dict = {}
    eog2_dict = {}
    avg_dict = {}
    y_dict = {}
    for subj in subjects:
        if detection_func == 'multi':
            y = get_all_y_multi_channel(subjects=[subj])
        elif detection_func == 'AH':
            y = get_all_y_AH(subjects=[subj])
        elif detection_func == 'AH+bi':
            y = get_all_y_AH_bi(subjects=[subj])
        clear_output()
        y_dict[subj] = y
        # feat_avg_396_fast = get_all_feat_avg(subjects=[subj])
        # clear_output()
        # avg_dict[subj] = feat_avg_396_fast
        feat_eog1 = get_all_feat_eog('1', subjects=[subj])
        clear_output()
        eog1_dict[subj] = feat_eog1
        feat_eog2 = get_all_feat_eog('2', subjects=[subj])
        clear_output()
        eog2_dict[subj] = feat_eog2

    joblib.dump(y_dict, f'y_dict_{file_name}_{detection_func}.pkl')
    joblib.dump(eog1_dict, f'eog1_dict_{file_name}.pkl')
    joblib.dump(eog2_dict, f'eog2_dict_{file_name}.pkl')
    # joblib.dump(avg_dict, f'avg_dict_{file_name}.pkl')

def save_y(subjects=['38', '394', '396', '398', '400', '402', '404', '405', '406', '414', '415', '416', '417', '423', '426', '429'], detection_func='AH', file_name='all'):
    y_dict = {}
    for subj in subjects:
        if detection_func == 'multi':
            y = get_all_y_multi_channel(subjects=[subj])
        elif detection_func == 'AH':
            y = get_all_y_AH(subjects=[subj], model_path='validation_models/xgb_AH_no_split.pkl')
        elif detection_func == 'AH+bi':
            y = get_all_y_AH_bi(subjects=[subj])
        clear_output()
        y_dict[subj] = y

    joblib.dump(y_dict, f'y_dict_{file_name}_{detection_func}.pkl')

# save_y(subjects=['38', '394', '396', '398', '400', '402', '404', '405', '406', '414', '415', '416', '417', '423', '426', '429'] + ['013', '017', '018', '025', '39', '47', '48', '51'], file_name='xgb')

def get_all_chans_y(subjects=['38', '396', '398', '400', '402', '406', '415', '416', '423'], model_path='xgb_all_chan_no_split.pkl', path=mtl_path):
    model = joblib.load(model_path)
    features_names_AH = model.get_booster().feature_names if 'xgb' in model_path else model.feature_name_
    channels = ['RAH1', 'LAH1', 'RA1', 'LA1', 'LEC1', 'REC1', 'RPHG1', 'LPHG1', 'RMH1', 'LMH1', 'LH1', 'RH1',
                'RAH2', 'LAH2', 'RA2', 'LA2', 'LEC2', 'REC2', 'RPHG2', 'LPHG2', 'RMH2', 'LMH2', 'LH2', 'RH2']
    y_all = np.empty(0)
    for subj in subjects:
        if subj in ['018', '025']:
            channels = channels + ['RA3']
        y_curr = None
        subj_chans = [x for x in mne.io.read_raw(path % subj).ch_names if x in channels]
        for channel in subj_chans:
            if not ((subj in only_left and channel[0] == 'R') or (subj in only_right and channel[0] == 'L')):
                x = format_raw_night(path % subj, channel)
                features = calc_features_before_split(x, subj)
                chan_feat = channel_feat(path % subj, channel)
                for feat in chan_feat.keys():
                    features[feat] = chan_feat[feat]

                if y_curr is None:
                    y_curr = model.predict(features[features_names_AH])
                else:
                    y_temp = model.predict(features[features_names_AH])
                    y_curr += y_temp

        y_curr[y_curr > 1] = 1
        y_all = np.concatenate((y_all, y_curr))

    return y_all
# get_all_chans_y(['49'], path='C:\\UCLA\\P%s_overnightData.edf', model_path='lgbm_all_chan_no_split.pkl')

def save_dicts_with_chan_feat(subjects=['38', '394', '396', '398', '400', '402', '404', '405', '406', '414', '415', '416', '417', '423', '426', '429'], file_name='all'):
    # get everyone feat and y
    eog1_dict = {}
    eog2_dict = {}
    y_dict = {}
    n = len(subjects)
    for subj in subjects:
        y = get_all_chans_y(subjects=[subj], model_path='validation_models/lgbm_mtl_depth.pkl')
        clear_output()
        y_dict[subj] = y
        feat_eog1 = get_all_feat_eog_with_chan_feat('1', subjects=[subj])
        clear_output()
        eog1_dict[subj] = feat_eog1
        if subj == '025':
            feat_eog2 = get_all_feat_eog_with_chan_feat('1', subjects=[subj])
        else:
            feat_eog2 = get_all_feat_eog_with_chan_feat('2', subjects=[subj])
        clear_output()
        eog2_dict[subj] = feat_eog2

    joblib.dump(y_dict, f'y_dict_{file_name}_{n}.pkl')
    joblib.dump(eog1_dict, f'eog1_dict_{file_name}_{n}.pkl')
    joblib.dump(eog2_dict, f'eog2_dict_{file_name}_{n}.pkl')

# save_dicts_with_chan_feat(subjects=all_subject, file_name='multi_ldepth_mtl')



def save_features_dicts_new(subjects=['38', '394', '396', '398', '400', '402', '404', '405', '406', '414', '415', '416', '417', '423', '426', '429'],
                            detection_func='AH', file_name='all'):
    # get everyone feat and y
    eog1_dict = {}
    eog2_dict = {}
    avg_dict = {}
    y_dict = {}
    for subj in subjects:
        if detection_func == 'multi':
            y = get_all_y_multi_channel(subjects=[subj])
        elif detection_func == 'AH':
            y = get_all_y_AH(subjects=[subj])
        elif detection_func == 'AH+bi':
            y = get_all_y_AH_bi(subjects=[subj])
        clear_output()
        y_dict[subj] = y
        # feat_avg_396_fast = get_all_feat_avg(subjects=[subj])
        # clear_output()
        # avg_dict[subj] = feat_avg_396_fast
        feat_eog1 = get_all_feat_eog('1', subjects=[subj])
        clear_output()
        eog1_dict[subj] = feat_eog1
        feat_eog2 = get_all_feat_eog('2', subjects=[subj])
        clear_output()
        eog2_dict[subj] = feat_eog2

    joblib.dump(y_dict, f'y_dict_{file_name}_{detection_func}.pkl')
    joblib.dump(eog1_dict, f'eog1_dict_{file_name}.pkl')
    joblib.dump(eog2_dict, f'eog2_dict_{file_name}.pkl')
    # joblib.dump(avg_dict, f'avg_dict_{file_name}.pkl')
# save_features_dicts(file_name='clean_eog')
# save_dicts_with_chan_feat(file_name='with_chan')
