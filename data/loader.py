import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import skew
import statistics as stats
import json
from torch.utils.data import DataLoader, TensorDataset
from prepare_data import prepare_features, get_disease_specific_data, f_get_Normalization
import torch
from sklearn.model_selection import StratifiedKFold

def y_per_disease(data, num_diseases, disease_ids):
    # Structured array for y:
    ys = [None]*num_diseases
    for i, disease in enumerate(disease_ids):
        data[f'event_{disease}'] = data[f'event_{disease}']
        dtype = np.dtype([('event', bool), ('time', np.float64)])
        y = np.array(list(zip(data[f'event_{disease}'], data[f'event_time_{disease}'])), dtype=dtype)
        ys[i] = y
    return ys

def drop_nans(data, labels, column = None):
    data.reset_index(drop=True, inplace=True)
    labels.reset_index(drop=True, inplace=True)
    if column is not None:
        na_rows = data[column].isna()
    else:
        na_rows = data.isna().any(axis=1)
    dropped_indices = np.where(na_rows)[0]
    try:
        data = data.drop(dropped_indices, axis=0).reset_index(drop=True)
    except:
        import pdb
        pdb.set_trace()
    labels = labels.drop(dropped_indices, axis=0).reset_index(drop=True)
    return data, labels

def categorical_encoding(data, y):
    
    data = data.map(lambda x: x.lower() if type(x) == str else x)
    str_cols = []
    for column in data.columns:
        if not np.issubdtype(data[column].dtype, np.number):
            data, y = drop_nans(data, y, column)
            unique_values = (data[column].unique())
            if (set(unique_values) == {'yes', 'no'}):
                data[column] = data[column].replace({'yes': 1, 'no': 0})
            elif all(isinstance(value, str) for value in unique_values):
           
                if column == 'alchohol_status_0' or column == 'alchohol_status_2_0':
                    mask = data[column] != 'prefer not to answer'
                    data = data[mask]
                    y = y[mask]
                    data[column] = data[column].replace({'never': 0, 'previous': 1, 'current': 2})
                else:
                    str_cols.append(column)
    if str_cols:
        data = pd.get_dummies(data, columns=str_cols, drop_first=True, dtype=int)

    return data, y
    
def fill_nans_and_drop(ref_data, fill_data, labels, fill_nan_cols):
    
    if fill_nan_cols != None:
        for col in fill_nan_cols:
            assert ((ref_data[col].dtype == float) or (ref_data[col].dtype == int)), "Fill NaNs can only be done in columns of dtype float or int."
            subset = ref_data[col].dropna()
            skewness = skew(subset)
            if (skewness>0.5) or (skewness<-0.5):
                fill_data[col] = fill_data[col].fillna(stats.median(subset))
            else:
                fill_data[col] = fill_data[col].fillna(stats.mean(subset))
        
    data, labels = drop_nans(fill_data, labels)

    return data, labels   

def clean_df(data):
    
    # Remove rows with nan values
    data.replace('NA', np.nan, inplace=True)

    # Remove rows with negative times
    time_cols = data.filter(like='time').columns
    mask = (data[time_cols] < 0).any(axis=1)
    data = data[~mask].reset_index(drop=True)
    return data

def filter_columns(data, features, disease_ids):
    event_columns = [col for col in data.columns if col.startswith('event')]
    y_columns = [col for col in event_columns if any(disease_id in col for disease_id in disease_ids)]

    keep_cols = features + y_columns
    data = data[keep_cols]
    return data, y_columns

def preprocess_data(data, args, disease_ids):
    
    data, event_columns = filter_columns(data, args.features, disease_ids)
    # Remove rows with Na values (Cox and Random forest cannot handle them)
    
    data = clean_df(data.copy())
    
    # Split data into X and y
    X = data[args.features]
    y = data[event_columns]

    # Categorical encoding
    X, y = categorical_encoding(X, y)
    
    return X, y

def split_data(X, y, test_split, seed):
    tr_data, te_data, tr_label, te_label = train_test_split(
                X, y, train_size=test_split, random_state=seed
            )
    
    train_data = (tr_data, tr_label)
    test_data = (te_data, te_label)

    return train_data, test_data

def split_data_deep(data, labels, train_split, test_split, seed):
  
    (tr_data_temp,te_data, tr_label_temp,te_label)  = train_test_split(data, labels, train_size=test_split, random_state=seed) 
    (tr_data,va_data, tr_label,va_label)  = train_test_split(tr_data_temp, tr_label_temp, train_size=train_split, random_state=seed) 

    tr_data.reset_index(drop=True, inplace=True)
    va_data.reset_index(drop=True, inplace=True)
    te_data.reset_index(drop=True, inplace=True)
    tr_label.reset_index(drop=True, inplace=True)
    va_label.reset_index(drop=True, inplace=True)
    te_label.reset_index(drop=True, inplace=True)
    
    train_data = (tr_data, tr_label)
    val_data = (va_data, va_label)
    test_data = (te_data, te_label)

    return train_data, val_data, test_data

def get_disease_names(code_disease_dict, ids, save=False):
    subset_dict = {id: code_disease_dict[id] for id in ids if id in code_disease_dict}
    if save == True:
        with open('disease_subset_mapping.json', 'w') as file:
            json.dump(subset_dict, file, indent=4)
    return subset_dict

def get_diseases(data, map_path, num_diseases=0, disease_list=None):
    
    code_to_disease_dict = {}
    disease_to_code_dict = {}
    with open(map_path, 'r') as file:
        for line in file:
            # Split each line at the tab character to separate the code and the disease
            code, disease = line.strip().split('\t')
            
            # Add the code and disease to the dictionary
            code_to_disease_dict[code] = disease
            disease_to_code_dict[disease] = code

    def get_code_or_raise(disease):
        if disease in disease_to_code_dict:
            return disease_to_code_dict[disease]
        else:
            raise ValueError(f"{disease} is not a valid key in the disease to code dictionary.")

    # Filter diseases to be used according to provided list, or use all:
    if disease_list is not None:
        keys = [get_code_or_raise(disease) for disease in disease_list]
    else:
        keys = code_to_disease_dict.keys()
    
    # Filter number of diseases if specified:
    if num_diseases is not None:
        assert num_diseases <= len(disease_list), "num_diseases must be <= than length of disease_list provided!"

        num_events = pd.DataFrame(index = keys, columns=['event_count'])
        for code in num_events.index:
            column_name = f'event_{code}'
            if column_name in data.columns:
                num_events.loc[code, 'event_count'] = data[column_name].sum()
            else:
                num_events.loc[code, 'event_count'] = pd.NA

        diseases = num_events.sort_values(by='event_count', ascending=False).head(num_diseases)
        disease_ids = diseases.index

    else:
        disease_ids = keys

    return code_to_disease_dict, disease_to_code_dict, disease_ids

def prepare_input(X, y, args, disease_ids):
    
    num_feat, data = prepare_features(X, args.norm_mode)
    labels, times, masks1, masks2 = [], [], [], []
    for disease_id in disease_ids:
        (label, time), (mask1,mask2), time_span = get_disease_specific_data(y, disease_id)
        
        labels.append(label)
        times.append(time)
        masks1.append(mask1)
        masks2.append(mask2)
    labels_all = np.stack(labels, axis=0)
    labels_all = np.transpose(labels_all, axes=(1, 0)) # Each subject has a label for each disease (0 or 1)
    times_all = np.stack(times, axis=0)
    times_all = np.transpose(times_all, axes=(1, 0))
    # Put masks together, output size: [batch_size, num_diseases, num_Category]
    masks1_all = np.stack(masks1, axis=0) 
    masks1_all = np.transpose(masks1_all, axes=(1, 0, 2))
    masks2_all = np.stack(masks2, axis=0) 
    masks2_all = np.transpose(masks2_all, axes=(1, 0, 2))
    # Put masks together, output size: [batch_size,num_diseases*num_Category]
    return num_feat, data, labels_all, times_all, masks1_all, masks2_all, time_span

def load_data(args):
  
    data = pd.read_csv(args.data_path)
    map_path = args.map_path
   
    _, _, disease_ids = get_diseases(data, map_path, args.num_diseases, args.disease_list)
   
    X, y = preprocess_data(data, args, disease_ids)
    
    
    if args.model_type != 'deep' and args.model_type != 'deep_time' and args.model_type != 'deep_time_prob':
        
        (tr_data, tr_label), (te_data, te_label) = split_data(X, y, args.test_split, args.seed)
       
        if args.fill_nan_cols is not None:
            tr_data_clean, tr_label = fill_nans_and_drop(tr_data, tr_data, tr_label, args.fill_nan_cols)

            te_data_clean, te_label = fill_nans_and_drop(tr_data, te_data, te_label, args.fill_nan_cols)
            
        tr_labels = y_per_disease(tr_label, args.num_diseases, disease_ids)
        te_labels = y_per_disease(te_label, args.num_diseases, disease_ids)
        train_data = (tr_data_clean, tr_labels)
        test_data = (te_data_clean, te_labels)
        

        return train_data, test_data, disease_ids
    
    elif args.model_type == 'deep' or args.model_type == 'deep_time' or args.model_type == 'deep_time_prob':

        (tr_data, tr_labels), (va_data, va_labels), (te_data, te_labels) = split_data_deep(X, y, args.train_split, args.test_split, args.seed)
        
        tr_data_clean, tr_labels = fill_nans_and_drop(tr_data, tr_data, tr_labels, args.fill_nan_cols)
        va_data_clean, va_labels = fill_nans_and_drop(tr_data, va_data, va_labels, args.fill_nan_cols)
        te_data_clean, te_labels = fill_nans_and_drop(tr_data, te_data, te_labels, args.fill_nan_cols)

        num_feat, tr_data, tr_label, tr_time, tr_mask1, tr_mask2, time_span = prepare_input(tr_data_clean, tr_labels, args, disease_ids)
        _, va_data, va_label, va_time, va_mask1, va_mask2, _ = prepare_input(va_data_clean, va_labels, args, disease_ids)
        _, te_data, te_label, te_time, te_mask1, te_mask2, _ = prepare_input(te_data_clean, te_labels, args, disease_ids)
        
        train_data = (tr_data, tr_label, tr_time, tr_mask1, tr_mask2)
        val_data = (va_data, va_label, va_time, va_mask1, va_mask2)
        test_data = (te_data, te_label, te_time, te_mask1, te_mask2)
       
        return train_data, val_data, test_data, disease_ids, num_feat, time_span

def get_dataloaders(args, train_data, val_data, test_data):
    
    train_data = [torch.tensor(arr).float() for arr in train_data]
    val_data = [torch.tensor(arr).float() for arr in val_data]
    test_data = [torch.tensor(arr).float() for arr in test_data]

    train_dataset = TensorDataset(*train_data)
    val_dataset = TensorDataset(*val_data)
    test_dataset = TensorDataset(*test_data)

    # Create dataloaders:
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle=False, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, drop_last=True)

    return train_dataloader, val_dataloader, test_dataloader