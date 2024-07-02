import numpy as np

# Normalize covariates for DL:
def f_get_Normalization(X, norm_mode):#, normalize_binary=True):
    _, num_Feature = np.shape(X)
    # if normalize_binary:
    if norm_mode == 'standard': #zero mean unit variance
        for j in range(num_Feature):
            if np.std(X[:,j]) != 0:
                X[:,j] = (X[:,j] - np.mean(X[:, j]))/np.std(X[:,j])
            else:
                X[:,j] = (X[:,j] - np.mean(X[:, j]))
    elif norm_mode == 'normal': #min-max normalization
        for j in range(num_Feature):
            X[:,j] = (X[:,j] - np.min(X[:,j]))/(np.max(X[:,j]) - np.min(X[:,j]))
    else:
        print("Normalization MODE ERROR!")

    return X

### MASK FUNCTIONS
'''
    fc_mask1      : To calculate LOSS_1 (log-likelihood loss)
    fc_mask2      : To calculate LOSS_2 (ranking loss)
'''
def f_get_fc_mask1(time, label, num_Event, num_Category):
    '''
        mask4 is required to get the log-likelihood loss
        mask4 size is [N, num_Event, num_Category]
            if not censored : one element = 1 (0 elsewhere)
            if censored     : fill elements with 1 after the censoring time (for all events)
    '''
    mask = np.zeros([np.shape(time)[0], num_Category]) # for the first loss function
    for i in range(np.shape(time)[0]):
        if label[i] != 0:  #not censored
            mask[i,int(time[i])] = 1
        else: #label[i,2]==0: censored
            mask[i,int(time[i]+1):] =  1 #fill 1 from censoring time +1 until end (to get 1 - \sum F)
    return mask

def prepare_features(X, norm_mode):
    
    data            = np.asarray(X)
    data            = f_get_Normalization(data, norm_mode)

    x_dim           = np.shape(data)[1]

    DIM             = (x_dim)
    DATA            = (data)
    
    return DIM, DATA

def f_get_fc_mask2(time, num_Category):
    '''
        mask2 is required calculate the ranking loss (for pair-wise comparision)
        mask2 size is [N, num_Category], each row has 1's from start to the event time(inclusive)
    '''
    
    mask = np.zeros([np.shape(time)[0], num_Category]) # for the first loss function
    for i in range(np.shape(time)[0]):
        t = int(time[i]) # censoring/event time
        mask[i,:(t+1)] = 1  #this excludes the last measurement time and includes the event time
    return mask

def get_disease_specific_data(y, disease_id):
    label           = np.asarray(y[f'event_{disease_id}'].astype(int))
    time            = np.asarray(y[[f'event_time_{disease_id}']]).flatten()
   
    num_Category    = int(np.max(time) * 1.2)  #to have enough time-horizon
    num_Event       = int(len(np.unique(label)) - 1) #only count the number of events (do not count censoring as an event)
  
    mask1           = f_get_fc_mask1(time, label, num_Event, num_Category)
    mask2           = f_get_fc_mask2(time, num_Category)
    LABELS          = (label, time)
    TIME_SPAN       = (num_Category)
    MASK            = (mask1, mask2) # mask1 size = (N, num_Event, num_Category)

    return LABELS, MASK, TIME_SPAN 