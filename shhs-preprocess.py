import os
import numpy as np
import torch

import argparse
import glob
import math
import ntpath

import shutil
import urllib
# import urllib2
import re 
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
import pyedflib
import collections
import pandas as pd
from mne.io import concatenate_raws, read_raw_edf
import dhedfreader
import xml.etree.ElementTree as ET
EPOCH_SEC_SIZE=30
EPOCH_SEC_HOUR=60*60
EPOCH_SEC_15MINUTE=60*15
EPOCH_SEC_5MINUTE=60*5
sampling_rate1=125
sampling_rate2=10
sampling_rate3=1
from scipy.signal import butter, filtfilt
def standardize_ecg_with_highpass(signal, fs, cutoff=0.5, order=1, clip_val=None, return_stats=False):
    """
    Apply high-pass filtering followed by global z-score normalization to an ECG signal.

    Parameters:
    - signal: Raw ECG signal as a 1D numpy array.
    - fs: Sampling rate (in Hz).
    - cutoff: Cutoff frequency for high-pass filter (default: 0.5 Hz).
    - order: Order of the Butterworth filter (default: 3).
    - clip_val: If specified, clips the normalized signal within [-clip_val, +clip_val].
    - return_stats: If True, also returns the mean and standard deviation (for inverse normalization).

    Returns:
    - The normalized ECG signal (numpy array).
    - If return_stats=True, also returns the mean and standard deviation.
    """
    b, a = butter(order, cutoff / (0.5 * fs), btype='high')
    filtered = filtfilt(b, a, signal)

    # Global z-score normalization
    mean = np.mean(filtered)
    std = np.std(filtered) + 1e-12
    print(mean,std)
    z = (filtered - mean) / std

    # 可选 clip
    if clip_val is not None:
        z = np.clip(z, -clip_val, clip_val)

    if return_stats:
        return z, mean, std
    else:
        return z
def extract_label(filename):
    labels = []
    t = ET.parse(filename)
    r = t.getroot()
    faulty_File = 0
    for i in range(len(r[4])):
        lbl = int(r[4][i].text)
        if lbl == 4:  # make stages N3, N4 same as N3
            labels.append(3)
        elif lbl == 5:  # Assign label 4 for REM stage
            labels.append(4)
        else:
            labels.append(lbl)
        if lbl > 5:  # some files may contain labels > 5 BUT not the selected ones.
            faulty_File = 1
    
    if faulty_File == 1:
        print( "============================== Faulty file ==================")
        
    labels = np.asarray(labels)
    
    return labels, faulty_File


def extract_data(filename):
    f = pyedflib.EdfReader(filename)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    dict_data = collections.defaultdict(list)
    print(signal_labels)
    # Normalize the labels for consistency
    normalized_labels = {
        'NEW AIR': 'NewAir',
        'NEWAIR': 'NewAir',
        'AIRFLOW': 'Airflow',
        'airflow': 'Airflow',
        'new A/F': 'NewAir',
        'New Air': 'NewAir',
        'AUX': 'NewAir',
        'THOR RES': 'THORRES',
        'ABDO RES': 'ABDORES',
    }
    
    for i in range(n):
        label = signal_labels[i].strip()  # Remove extra spaces
        label = normalized_labels.get(label.upper(), re.sub("[^A-Za-z0-9]+", "", signal_labels[i]))  # Normalize
        #print(label)
        if label in ['THORRES', 'ABDORES', 'ECG', 'Airflow', 'SaO2', 'EEG', 'EEGsec']:
            sig_bufs = np.zeros((f.getNSamples()[i], 1), dtype=np.float32)  # Read raw signal
            sig_bufs[:, 0] = f.readSignal(i)
            dict_data[label].append(sig_bufs[:, 0])

    # Handle NewAir and Airflow selection
    if 'NewAir' not in dict_data and 'Airflow' in dict_data:
        dict_data['NewAir'] = dict_data['Airflow']  # Use Airflow as fallback if NewAir is missing
    return dict_data
def extract_respiratory_events_to_labels(xml_file, signal_duration):
    """
    Converts respiratory events in an XML annotation to a 1 Hz label array.
    
    Args:
        xml_file (str): Path to the XML annotation file.
        signal_duration (int): Total duration of the signal in seconds.
        
    Returns:
        np.ndarray: A 1 Hz label array of the same duration as the signal.
    """
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Initialize the label array with zeros (no event)
    labels = np.zeros((signal_duration, 2), dtype=int)
    # By default, set the third column (normal breathing) to 1
    labels[:, 1] = 1
    
    # Define event types and their corresponding labels
    event_label_map = {
        "Central Apnea": 1,
        "Obstructive Apnea": 1,
        "Mixed Apnea": 1,
        "Hypopnea": 2,
        "Obstructive Hypopnea": 2,
        "Central Hypopnea": 2,
        "Mixed Hypopnea": 2
    }
    all_events = set()
    
    # Process each ScoredEvent
    for event in root.findall(".//ScoredEvent"):
        name = event.find("Name").text
        if name in event_label_map:
            start = float(event.find("Start").text)
            duration = float(event.find("Duration").text)
            label = event_label_map[name]
            
            start_idx = int(np.floor(start))  # Convert start time to index
            end_idx = int(np.ceil(start + duration))  # Convert end time to index
            
            if label == 2:
                # Ensure indices are within signal bounds
                start_idx = max(0, start_idx)
                end_idx = min(signal_duration, end_idx)
                
                # Assign the label to the appropriate indices
                labels[start_idx:end_idx, label - 1] = 1
                # Set the normal breathing label to 0 during event periods
                labels[start_idx:end_idx, 1] = 0
                
    return labels

def zscore_global(x):
    mean = np.mean(x)
    std = np.std(x) + 1e-8
    return (x - mean) / std
import numpy as np
import matplotlib.pyplot as plt
import os


def generate_dataset(edf_filename, ann_filename, save_path, features, mode):
    """
    Generate dataset in PyTorch (.pt) format from EDF and annotation files.
    """
    try:
        # Get ID
        current_id = int(edf_filename.split('/')[-1].split('-')[1].split('.')[0])
        save_filename = os.path.join(save_path, f'shhs-{current_id}.pt')  # 改为.pt格式
        if os.path.isfile(save_filename):
            return 0

        # Read EDF file
        data_dict = extract_data(edf_filename)
        sleep_labels=extract_label(ann_filename)[0]
        w_edge_mins=30
        nw_idx=np.where(sleep_labels!=0)[0]
        start_idx=nw_idx[0]-w_edge_mins*2
        end_idx=nw_idx[-1]+w_edge_mins*2
        if start_idx<0:
            start_idx=0
        if end_idx>len(sleep_labels):
            end_idx=len(sleep_labels)-1
        if data_dict is None:
            return 0
            
        # Get ECG data
        if 'ECG' not in data_dict:
            print(f"No ECG data found in {edf_filename}")
            return 0
            
        raw_ch = data_dict['ECG'][0]
        raw_ch1 = data_dict['EEG'][0]
        raw_ch2 = data_dict['EEGsec'][0]
        raw_ch3 = data_dict['THORRES'][0]
        raw_ch4 = data_dict['ABDORES'][0]
        if raw_ch.shape[0] % (EPOCH_SEC_SIZE * sampling_rate1) != 0:
            print(f"Signal length not divisible by epoch size in {edf_filename}")
            return 0
        x = np.array(raw_ch)
        x1 = np.array(raw_ch1)
        x2 = np.array(raw_ch2)
        x3 = np.array(raw_ch3)
        x4 = np.array(raw_ch4)
        # Replace the original standardization method
        x = standardize_ecg_with_highpass(x, sampling_rate1,cutoff=0.5)
        x1 = standardize_ecg_with_highpass(x1, sampling_rate1,cutoff=0.3)
        x2 = standardize_ecg_with_highpass(x2, sampling_rate1,cutoff=0.3)
        x3 = x3 + x4
        x3 = standardize_ecg_with_highpass(x3, sampling_rate2,cutoff=0.05)
        # 计算每小时的采样点数
        points_per_epoch = EPOCH_SEC_SIZE * sampling_rate1
        points_per_epoch2 = EPOCH_SEC_SIZE * sampling_rate2
        # 计算完整的小时数
        n_complete_epoch0 = x.shape[0] // points_per_epoch
        n_complete_epoch1 = x1.shape[0] // points_per_epoch
        n_complete_epoch2 = x2.shape[0] // points_per_epoch
        n_complete_epoch3 = x3.shape[0] // points_per_epoch2
        # n_complete_hours4 = x4.shape[0] // points_per_hour2
        print(n_complete_epoch0,n_complete_epoch1,n_complete_epoch2,n_complete_epoch3)
        # Keep complete hours data
        x = x[:n_complete_epoch0 * points_per_epoch]
        x1 = x1[:n_complete_epoch1 * points_per_epoch]
        x2=x2[:n_complete_epoch2 * points_per_epoch]
        x3=x3[:n_complete_epoch3 * points_per_epoch2]
        print(f"Original length: {len(raw_ch)}")
        print(f"Adjusted length: {x.shape[0]}")
        print(f"Number of complete 30 seconds: {n_complete_epoch0}")
        x=x.reshape(n_complete_epoch0,EPOCH_SEC_SIZE*sampling_rate1)
        x1=x1.reshape(n_complete_epoch1,EPOCH_SEC_SIZE*sampling_rate1)
        x2=x2.reshape(n_complete_epoch2,EPOCH_SEC_SIZE*sampling_rate1)
        x3=x3.reshape(n_complete_epoch3,EPOCH_SEC_SIZE*sampling_rate2)
        if start_idx>=0 and end_idx<=x.shape[0]:
            x=x[start_idx:end_idx+1,:]
            x1=x1[start_idx:end_idx+1,:]
            x2=x2[start_idx:end_idx+1,:]
            x3=x3[start_idx:end_idx+1,:]
            sleep_labels=sleep_labels[start_idx:end_idx+1]
        else:
            print('invalid indices')
            exit()


        if mode=='train':
            y1 = features[0]  # prevalent_mi
            y2 = features[1]  # rbbb
            y3 = features[2]  # congestive_heart_failure
            y4 = features[3]  # cvd_death
            y5 = features[4]  # hypertension
            y6 = features[5]  # hypertension_current
            afib = features[6]  # afib
            incident_afib = features[7]  # incident_afib
            age=features[8]
            sex=features[9]
            race=features[10]
            bmi=features[11]
            chol=features[12]
            hdl=features[13]
            systbp=features[14]
            parrptdiab=features[15]
            smokstat_s1=features[16]
            htnmed1=features[17]
            
            # Convert to PyTorch tensor
            x_tensor = torch.FloatTensor(x)
            x1_tensor = torch.FloatTensor(x1)
            x2_tensor = torch.FloatTensor(x2)
            x3_tensor = torch.FloatTensor(x3)
            y1_tensor = torch.FloatTensor([y1])
            y2_tensor = torch.FloatTensor([y2])
            y3_tensor = torch.FloatTensor([y3])
            y4_tensor = torch.FloatTensor([y4])
            y5_tensor = torch.FloatTensor([y5])
            y6_tensor = torch.FloatTensor([y6])
            afib_tensor = torch.FloatTensor([afib])
            incident_afib_tensor = torch.FloatTensor([incident_afib])
            age_tensor = torch.FloatTensor([age])
            sex_tensor = torch.FloatTensor([sex])
            race_tensor = torch.FloatTensor([race])
            bmi_tensor = torch.FloatTensor([bmi])
            chol_tensor = torch.FloatTensor([chol])
            hdl_tensor = torch.FloatTensor([hdl])
            systbp_tensor = torch.FloatTensor([systbp])
            parrptdiab_tensor = torch.FloatTensor([parrptdiab])
            smokstat_s1_tensor = torch.FloatTensor([smokstat_s1])
            htnmed1_tensor = torch.FloatTensor([htnmed1])
            # labels_tensor = torch.FloatTensor(labels)
            labels_tensor = torch.FloatTensor(sleep_labels)
            if(x_tensor.shape[0]!=labels_tensor.shape[0]):
                print(x_tensor.shape)
                print(labels_tensor.shape)
                exit()
            save_dict = {
                "x": x_tensor,
                "x1": x1_tensor,
                "x2": x2_tensor,
                "x3": x3_tensor,
                "y1": y1_tensor,
                "y2": y2_tensor,
                "y3": y3_tensor,
                "y4": y4_tensor,
                "y5": y5_tensor,
                "y6": y6_tensor,
                "afib": afib_tensor,
                "incident_afib": incident_afib_tensor,
                "age": age_tensor,
                "sex": sex_tensor,
                "race": race_tensor,
                "bmi": bmi_tensor,
                "chol": chol_tensor,
                "hdl": hdl_tensor,
                "systbp": systbp_tensor,
                "parrptdiab": parrptdiab_tensor,
                "smokstat_s1": smokstat_s1_tensor,
                "htnmed1": htnmed1_tensor,
                "labels": labels_tensor,
                "fs": sampling_rate1,
                "fs1": sampling_rate2
            }
            
            # 使用torch.save保存
            torch.save(save_dict, save_filename)
            print(f"Successfully saved {save_filename}")
            return 1
        else:
            y1 = features[0]  # prevalent_mi
            y2 = features[1]  # rbbb
            y3 = features[2]  # congestive_heart_failure
            y4 = features[3]  # cvd_death
            y5 = features[4]  # hypertension
            y6 = features[5]  # hypertension_current
            afib = features[6]
            incident_afib = features[7]
            age=features[8]
            sex=features[9]
            race=features[10]
            bmi=features[11]
            chol=features[12]
            hdl=features[13]
            systbp=features[14]
            parrptdiab=features[15]
            smokstat_s1=features[16]
            htnmed1=features[17]
            
            # Convert to PyTorch tensor
            x_tensor = torch.FloatTensor(x)
            x1_tensor = torch.FloatTensor(x1)
            x2_tensor = torch.FloatTensor(x2)
            x3_tensor = torch.FloatTensor(x3)
            y1_tensor = torch.FloatTensor([y1])
            y2_tensor = torch.FloatTensor([y2])
            y3_tensor = torch.FloatTensor([y3])
            y4_tensor = torch.FloatTensor([y4])
            y5_tensor = torch.FloatTensor([y5])
            y6_tensor = torch.FloatTensor([y6])
            afib_tensor = torch.FloatTensor([afib])
            incident_afib_tensor = torch.FloatTensor([incident_afib])
            age_tensor = torch.FloatTensor([age])
            sex_tensor = torch.FloatTensor([sex])
            race_tensor = torch.FloatTensor([race])
            bmi_tensor = torch.FloatTensor([bmi])
            chol_tensor = torch.FloatTensor([chol])
            hdl_tensor = torch.FloatTensor([hdl])
            systbp_tensor = torch.FloatTensor([systbp])
            parrptdiab_tensor = torch.FloatTensor([parrptdiab])
            smokstat_s1_tensor = torch.FloatTensor([smokstat_s1])
            htnmed1_tensor = torch.FloatTensor([htnmed1])
            labels_tensor = torch.FloatTensor(sleep_labels)
            if(x_tensor.shape[0]!=labels_tensor.shape[0]):
                print(x_tensor.shape)
                print(labels_tensor.shape)
                exit()
            save_dict = {
                "x": x_tensor,
                "x1": x1_tensor,
                "x2": x2_tensor,
                "x3": x3_tensor,
                "y1": y1_tensor,
                "y2": y2_tensor,
                "y3": y3_tensor,
                "y4": y4_tensor,
                "y5": y5_tensor,
                "y6": y6_tensor,
                "afib": afib_tensor,
                "incident_afib": incident_afib_tensor,
                "age": age_tensor,
                "sex": sex_tensor,
                "race": race_tensor,
                "bmi": bmi_tensor,
                "chol": chol_tensor,
                "hdl": hdl_tensor,
                "systbp": systbp_tensor,
                "parrptdiab": parrptdiab_tensor,
                "smokstat_s1": smokstat_s1_tensor,
                "htnmed1": htnmed1_tensor,
                "labels": labels_tensor,
                "fs": sampling_rate1,
                "fs1": sampling_rate2
            }
            
            # Use torch.save to save
            torch.save(save_dict, save_filename)
            print(f"Successfully saved {save_filename}")
            return 1
    except Exception as e:
        import traceback
        print(f"\n=== ERROR PROCESSING FILE: {edf_filename} ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        print(f"=== END OF ERROR REPORT ===\n")
        return 0
def main():
    print('hi')
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/mnt/d/sleep/shhs/polysomnography/edfs/shhs1/",
                        help="File path to the PSG files.")
    parser.add_argument("--ann_dir", type=str, default="/mnt/d/sleep/shhs/polysomnography/annotations-events-profusion/shhs1/",
                        help="File path to the annotation files.")
    parser.add_argument("--output_dir", type=str, default="/mnt/d/sleep/shhs/output_npz/shhs-all_signals-30s",
                        help="Directory where to save numpy files outputs.")
    parser.add_argument("--select_ch", type=str, default="ECG",
                        help="The selected channel")
    args = parser.parse_args()
    path_pain_csv=r'/mnt/d/sleep/shhs/datasets/shhs1-dataset-0.21.0.csv' 
    outcome_csv=r'/mnt/d/sleep/shhs/datasets/shhs-cvd-summary-dataset-0.21.0.csv'
    interim_csv=r'/mnt/d/sleep/shhs/datasets/shhs-interim-followup-dataset-0.21.0.csv'
    # Use os.path.join to create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    outcome_csv = pd.read_csv(outcome_csv)
    df=pd.read_csv(path_pain_csv)
    df =df[df['overall_shhs1']>=4]
    print(len(df))
    encodings = ['utf-8', 'ISO-8859-1', 'cp1252']
    for encoding in encodings:
        try:
            df2 = pd.read_csv(interim_csv, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    high_quality_merge = pd.merge(df, outcome_csv, on='nsrrid', how='outer')
    high_quality_merge = pd.merge(high_quality_merge, df2, on='nsrrid', how='outer')
    age_columns = [col for col in high_quality_merge.columns if 'age_s1' in col]
    if 'htnderv_s1' in high_quality_merge.columns:
        high_quality_merge = high_quality_merge.dropna(subset=['htnderv_s1'])
    else:
        print("Column 'htnderv_s1' not found in the dataset.")
    high_quality_merged=high_quality_merge#Note
    if 'age_s1_x' in high_quality_merged.columns:
        high_quality_merged = high_quality_merged.dropna(subset=['age_s1_x'])
    else:
        print("Column 'age_s1_x' not found in the dataset.")
    print(len(high_quality_merged))
    if 'gender_x' in high_quality_merged.columns:
        high_quality_merged = high_quality_merged.dropna(subset=['gender_x'])
    else:
        print("Column 'gender_x' not found in the dataset.")
    print(len(high_quality_merged))
    if 'race_x' in high_quality_merged.columns:
        high_quality_merged = high_quality_merged.dropna(subset=['race_x'])
    else:
        print("Column 'race_x' not found in the dataset.")
    print(len(high_quality_merged))
    if 'bmi_s1' in high_quality_merged.columns:
        high_quality_merged = high_quality_merged.dropna(subset=['bmi_s1'])
    else:
        print("Column 'bmi_s1' not found in the dataset.")
    print(len(high_quality_merged))
    if 'chol' in high_quality_merged.columns:
        high_quality_merged = high_quality_merged.dropna(subset=['chol'])
    else:
        print("Column 'chol' not found in the dataset.")
    print(len(high_quality_merged))
    if 'hdl' in high_quality_merged.columns:
        high_quality_merged = high_quality_merged.dropna(subset=['hdl'])
    else:
        print("Column 'hdl' not found in the dataset.")
    print(len(high_quality_merged))
    if 'systbp' in high_quality_merged.columns:
        high_quality_merged = high_quality_merged.dropna(subset=['systbp'])
    else:
        print("Column 'systbp' not found in the dataset.")
    print(len(high_quality_merged))
    if 'parrptdiab' in high_quality_merged.columns:
        high_quality_merged = high_quality_merged.dropna(subset=['parrptdiab'])
    else:
        print("Column 'parrptdiab' not found in the dataset.")
    print(len(high_quality_merged))
    if 'smokstat_s1' in high_quality_merged.columns:
        high_quality_merged = high_quality_merged.dropna(subset=['smokstat_s1'])
    else:
        print("Column 'nsrr_current_smoker' not found in the dataset.")
    print(len(high_quality_merged))
    if 'htnmed1' in high_quality_merged.columns:
        high_quality_merged = high_quality_merged.dropna(subset=['htnmed1'])
    else:
        print("Column 'htnmed1' not found in the dataset.")
    high_quality_merged['status_id1'] = 0  # prevalent_mi
    high_quality_merged['status_id2'] = 0  # rbbb
    high_quality_merged['status_id3'] = 0  # congestive_heart_failure
    high_quality_merged['cvd_death'] = 0  # cvd_death
    high_quality_merged['hypertension'] = 0  # hypertension
    high_quality_merged['hypertension_current'] = 0  # hypertension_current
    high_quality_merged['afib'] = 0  # afib
    high_quality_merged['incident_afib'] = 0  # incident_afib
    hypertension_mask = (
    (((high_quality_merged['dias220'] + high_quality_merged['dias320']) / 2 >= 80) |
     ((high_quality_merged['syst220'] + high_quality_merged['syst320']) / 2 >= 130)) |
    (high_quality_merged['htnmed1'] == 1)
)

    high_quality_merged.loc[hypertension_mask, 'hypertension_current'] = 1
    print(hypertension_mask.value_counts())
    cvd_death_mask = high_quality_merged['cvd_dthdt'].notna() & (high_quality_merged['cvd_dthdt'] >=1 )& (high_quality_merged['cvd_dthdt'] <= 5*365)
    print(cvd_death_mask.value_counts())
    high_quality_merged.loc[cvd_death_mask, 'cvd_death'] = high_quality_merged.loc[cvd_death_mask, 'cvd_dthdt']
    future_hypertension_mask = (high_quality_merged['hypertension_current']==0) & (high_quality_merged['intrevdt']<=3*365) & ((high_quality_merged['avgsys']>=130) | (high_quality_merged['avgdias']>=80))
    high_quality_merged.loc[future_hypertension_mask, 'hypertension'] = 1
    future_hypertension_mask1 =high_quality_merged['hypertension_current']==1
    high_quality_merged.loc[future_hypertension_mask1, 'hypertension'] = 2
    mi_mask = high_quality_merged['prev_mi'].notna() & (high_quality_merged['prev_mi'] >=1 )
    high_quality_merged.loc[mi_mask, 'status_id1'] = 1
    afib_mask = (high_quality_merged['rbbb']==1)
    high_quality_merged.loc[afib_mask, 'status_id2'] = 1
     
    congestive_heart_failure_mask = (high_quality_merged['prev_chf'].notna() & (high_quality_merged['prev_chf']>=1)) 
    high_quality_merged.loc[congestive_heart_failure_mask, 'status_id3'] = 1
    afib_mask = (high_quality_merged['afibprevalent']==1) & (high_quality_merged['afibprevalent'].notna())
    high_quality_merged.loc[afib_mask, 'afib'] = 1
    incident_afib_mask = (high_quality_merged['afibincident'].notna() & (high_quality_merged['afibincident']==1))
    high_quality_merged.loc[incident_afib_mask, 'incident_afib'] = 1
    high_quality_merged.loc[afib_mask, 'incident_afib'] = 2
    health_status_1=high_quality_merged[high_quality_merged['status_id1']==1]
    health_status_2=high_quality_merged[high_quality_merged['status_id2']==1]
    health_status_3=high_quality_merged[high_quality_merged['status_id3']==1]
    health_status_4=high_quality_merged[high_quality_merged['hypertension_current']==1]
    health_status_5=high_quality_merged[high_quality_merged['hypertension']==1]
    health_status_6=high_quality_merged[(high_quality_merged['cvd_death']!=0)]
    health_status_7=high_quality_merged[(high_quality_merged['afib']==1)]
    health_status_8=high_quality_merged[(high_quality_merged['incident_afib']==1)]
    positive_nsrrids = set(health_status_1['nsrrid']) | set(health_status_2['nsrrid']) | set(health_status_3['nsrrid']) | set(health_status_4['nsrrid']) | set(health_status_5['nsrrid']) | set(health_status_6['nsrrid'])|set(health_status_7['nsrrid'])|set(health_status_8['nsrrid'])
    positive_samples = high_quality_merged[high_quality_merged['nsrrid'].isin(positive_nsrrids)]
    negative_samples=high_quality_merged[~high_quality_merged['nsrrid'].isin(positive_nsrrids)]
    train_positive=positive_samples.sample(n=int(len(positive_samples)*0.8), random_state=2020)
    rest_positive=positive_samples.drop(train_positive.index)
    test_positive=rest_positive.sample(n=int(len(rest_positive)), random_state=2020)
    train_negative=negative_samples.sample(n=int(len(negative_samples)*0.8), random_state=2020)
    rest_negative=negative_samples.drop(train_negative.index)
    test_negative=rest_negative.sample(n=int(len(rest_negative)), random_state=2020)
    
    train_records_df = pd.concat([train_positive, train_negative])
    test_records_df = pd.concat([test_positive, test_negative])
    ids=[]
    ids_pain=[]
    for i in glob.glob(args.data_dir+"*.edf"):
        ids_pain.append(i.split('/')[-1].split('-')[1].split('.')[0])
        ids.append(i.split('/')[-1].split('.')[0])
    edf_fnames = [os.path.join(args.data_dir.strip(), i.strip() + ".edf") for i in ids]
    ann_fnames = [os.path.join(args.ann_dir.strip(), i.strip() + "-profusion.xml") for i in ids]

    edf_fnames.sort()
    ann_fnames.sort()
    edf_fnames = np.asarray(edf_fnames)
    ann_fnames = np.asarray(ann_fnames)
    ids_pain=[int(i) for i in ids_pain]
    pain_dict = dict()

    filtered_ids_train = train_records_df['nsrrid'].tolist()
    filtered_ids_test = test_records_df['nsrrid'].tolist()
    
    print("Train IDs:", len(filtered_ids_train))
    print("Test IDs:", len(filtered_ids_test))

    for index, row in train_records_df.iterrows():
        features = [
           row['status_id1'],
           row['status_id2'],
           row['status_id3'],
           row['hypertension_current'],
           row['hypertension'],
           row['cvd_death'],
           row['afib'],
           row['incident_afib'],
            row['age_s1_x'],
            row['gender_x'],
            row['race_x'],
            row['bmi_s1'],
            row['chol'],
            row['hdl'],
            row['systbp'],
            row['parrptdiab'],
            row['smokstat_s1'],
            row['htnmed1']
        ]
        pain_dict[int(row['nsrrid'])] = features
    for index, row in test_records_df.iterrows():
        features = [
           row['status_id1'],
           row['status_id2'],
           row['status_id3'],
           row['hypertension_current'],
           row['hypertension'],
           row['cvd_death'],
           row['afib'],
           row['incident_afib'],
            row['age_s1_x'],
            row['gender_x'],
            row['race_x'],
            row['bmi_s1'],
            row['chol'],
            row['hdl'],
            row['systbp'],
            row['parrptdiab'],
            row['smokstat_s1'],
            row['htnmed1']
        ]
        pain_dict[int(row['nsrrid'])] = features
    
    print("Number of valid entries:", len(pain_dict))
    print("Sample values:", list(pain_dict.items())[:5])
    
    pain_dict = dict(sorted(pain_dict.items(), key=lambda x: x[0]))
    
    for file_id in range(len(edf_fnames)):
        current_id = int(edf_fnames[file_id].split('/')[-1].split('-')[1].split('.')[0])
        
        # Check which dataset the current ID belongs to
        # if current_id not in filtered_ids_train and current_id not in filtered_ids_val and current_id not in filtered_ids_test:
        if current_id not in filtered_ids_train and current_id not in filtered_ids_test:
            continue
            
        # Get the label for the current file
        y = pain_dict[current_id]
        
        # Determine the save path and mode
        if current_id in filtered_ids_train:
            save_dir = os.path.join(args.output_dir, 'train')
            mode = 'train'
        else:
            save_dir = os.path.join(args.output_dir, 'test')
            mode = 'test'
            
        os.makedirs(save_dir, exist_ok=True)
        print(edf_fnames[file_id], ann_fnames[file_id])
        
        # Generate dataset
        success = generate_dataset(
            edf_fnames[file_id],
            ann_fnames[file_id],
            save_dir,
            y,
            mode,
        )
        
        if success:
            print(f"Processed file {file_id+1}/{len(edf_fnames)}")

if __name__ == "__main__":
    main()