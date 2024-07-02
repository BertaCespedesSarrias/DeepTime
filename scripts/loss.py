import torch

def log_likelihood(outputs, targets, mask1):

    # Compute loss
    epsilon = 1e-10 
    uncensored_subjects = targets*torch.log(torch.sum(mask1*outputs, dim=2)+epsilon)
    censored_subjects = (1-targets)*torch.log(torch.sum(mask1*outputs,dim=2)+epsilon)
    
    loss = torch.mean(uncensored_subjects+censored_subjects)

    return - loss

def ranking_loss(outputs, targets, mask2, time):

    _sigma = 0.1
    

    eta = []
    for event in range(outputs.shape[1]):
        time_event = time[:,event].unsqueeze(1)
        mask2_event = mask2[:,event,:]
        ones_vec = torch.ones_like(time_event, dtype=torch.float32)
        # Take binary event labels for each event
        indicator = targets[:,event]
        # Put binary labels in array diagonal
        
        indicator = torch.diag(targets[:,event]).to(dtype=torch.float32)
        #Even specific joint probability: (joint?)
        event_prob = outputs[:,event,:] 
        
        R = torch.matmul(event_prob,torch.transpose(mask2_event,0,1))
        # Each element r_{ij} = risk assessment for the i-th patient based on the j-th time condition

        diag_R = torch.diag(R).view(-1, 1)
        
        # R = torch.matmul(ones_vec, diag_R.T) - R --> IN PAPER
        
        R = torch.matmul(ones_vec, diag_R.T) - R # R_{ij} = r_{j}(T_{j}) - r_{i}(T_{j})
        R = R.T # Now, R_{ij} (i-th row j-th column) = r_{i}(T_{i}) - r_{j}(T_{i})
        
        T = torch.nn.functional.relu(torch.sign(torch.matmul(ones_vec, time_event.T) - torch.matmul(time_event, ones_vec.T))) 
        # T_{ij}=1 if t_i < t_j  and T_{ij}=0 if t_i >= t_j
       
        T = torch.matmul(indicator, T) # only remains T_{ij}=1 when event occured for subject i
     
        tmp_eta = torch.mean(T*torch.exp(-R/_sigma), dim=1, keepdim=True)

        eta.append(tmp_eta)


    eta = torch.stack(eta, dim=1)
    eta = torch.mean(eta.view(-1,outputs.shape[1]), dim=1, keepdim=True)
    loss = torch.mean(eta)
    return loss

def custom_loss(outputs, labels, mask1, mask2, time):

    loss1 = log_likelihood(outputs, labels, mask1)
    loss2 = ranking_loss(outputs, labels, mask2, time)
    loss = loss1 + loss2 
    return loss


def custom_loss_time(T_preds, T_true, targets):

    
    subt = T_preds - T_true
    mask = T_preds<T_true
    censored = mask*((1-targets)*subt)
    loss_censored = torch.norm(censored, p=2,dim=0).mean()
    loss_uncensored = torch.norm(targets*subt, p=2, dim=0).mean()
    loss = loss_censored+loss_uncensored
    
    return loss