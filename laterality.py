from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from zeeg_utils import *
import joblib

eog1 = pd.read_csv(f'eog1_mne.csv', dtype={'subj': str}).drop(columns='Unnamed: 0')
eog2 = pd.read_csv(f'eog2_mne.csv', dtype={'subj': str}).drop(columns='Unnamed: 0')
# y_l, y_r = joblib.load('lateral_y.pkl')
y = joblib.load('lateral_y.pkl')
meta_data = ['subj', 'epoch_id', 'chan_name', 'epoch']
all_results = {}
subj_data = {}
for subj in all_subjects:
    eog1_subj = eog1[eog1['subj'] == subj]
    eog2_subj = eog2[eog2['subj'] == subj]
    subj_data[subj] = {'eog1': eog1_subj, 'eog2': eog2_subj, 'y_l': y[subj].get('L', None), 'y_r': y[subj].get('R', None)}

for subj in all_subjects:
    print(f'Processing {subj}')
    for side in ['l', 'r']:
        for num in range(1, 3):
            x = subj_data[subj][f'eog{num}']
            y = subj_data[subj][f'y_{side}']

            if y is None or y.shape[0] == 0 or x.shape[0] == 0:
                continue

            # undersample
            rus = RandomUnderSampler(random_state=8)
            x_balanced, y_balanced = rus.fit_resample(x, y)

            # remove meta data
            x_feat = x_balanced[x_balanced.columns[~x_balanced.columns.isin(meta_data)]]

            metrics = {'accuracy': [], 'precision': [], 'sensitivity': [], 'specificity': [], 'f1': [], 'ROCAUC': [],
                       'PRAUC': []}
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=8)
            for train_index, test_index in kf.split(x_feat, y_balanced):
                model = xgb.XGBClassifier()
                x_train_fold, x_test_fold = x_feat.iloc[train_index], x_feat.iloc[test_index]
                y_train_fold, y_test_fold = y_balanced[train_index], y_balanced[test_index]
                model.fit(x_train_fold, y_train_fold)
                y_pred = model.predict(x_test_fold)
                y_true = y_test_fold
                # save scores in dict
                metrics['accuracy'].append(accuracy_score(y_true, y_pred))
                metrics['precision'].append(precision_score(y_true, y_pred))
                metrics['sensitivity'].append(recall_score(y_true, y_pred))
                metrics['f1'].append(f1_score(y_true, y_pred))
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                metrics['specificity'].append(tn / (tn + fp))
                metrics['ROCAUC'].append(roc_auc_score(y_true, y_pred))
                metrics['PRAUC'].append(average_precision_score(y_true, y_pred))

            results = pd.DataFrame(metrics)
            # add mean row
            results.loc['mean'] = results.mean()
            all_results[f'{subj}_{side}{num}'] = results
#             joblib.dump(all_results, 'laterality_results.pkl')
