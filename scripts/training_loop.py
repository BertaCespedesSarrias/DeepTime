import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import contextlib
import torch.optim as optim
import torch.nn.functional as F
from lifelines.utils import concordance_index
from eval import weighted_c_index 

class Trainer:
    def __init__(self, model, model_type, device, seed=0):
        
        self.device = device
        self.model = model.to(self.device)
        self.model_type = model_type
        self.seed = seed

        # Set random seed:
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def train(self, epochs, train_loader, val_loader, loss_fn, out_dir, run, batch_count, disease_ids, learning_rate=0.001, fold=None, log=True, batch_size=128, early_stop=True):
        """
        Train model:

        Inputs:
        1. train_loader:
        2. val_loader:
        3. test_loader:
        4. epochs: Number of epochs for the training session.
        5. out_dir: 
        6. learning_rate: Learning rate for optimizer.
        7. log: if True, tensorboard logging is done

        Outputs:
        train_loss = List of loss at every epoch

        """
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.out_dir = out_dir
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.log = log
        self.max_c_train = -99
        self.max_c_val = -99
        self.batch_size = batch_size
        self.disease_ids = disease_ids
        # self.max_brier = -99
        self.stop_flag = 0

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate
        )

        metrics = {
            "loss_train": [],
            "loss_val": [],
            "best_ci_train": [],
            "best_ci_val": [],
        }

        
        out_dir_ES = f"{out_dir}/best_model_{self.model_type}_run{run}.pth"

        # Instantiate Early Stopping:
        
        early_stopping = EarlyStopping(
            out_dir=out_dir_ES,
            patience=7,
            min_delta=0.01,
        )

        tr_time = train_loader.dataset[:][2]
        tr_time_unique = torch.unique(tr_time.flatten())
        eval_time = [int(np.percentile(tr_time_unique, 25)), int(np.percentile(tr_time_unique, 50)), int(np.percentile(tr_time_unique, 75))]
        

        for epoch in range(epochs):
            print(f"Epoch {epoch}")
            (loss_train, preds_train, labels_train, times_train) = self.step(
                train_loader,
                is_train=True,
                log=log
            )
            

            # c_index_train = self.evaluation_train(train_loader, train_loader, preds_train, eval_time, "train", )
            c_index_train, ci_tr_ind = self.evaluation(labels_train, labels_train, times_train, times_train, preds_train, eval_time, "train", batch_size=self.batch_size, batch_count=batch_count, model_type = self.model_type)
            
            # metrics["loss_train"] = loss_train
            # metrics["c_index_train"] = c_index_train

            (loss_val, preds_val, labels_val, times_val) = self.step(
                val_loader,
                is_train=False,
                log=log
            )
            
            # c_index_val_test = self.evaluation_train(labels_train, labels_val, times_train, times_val, preds_val, eval_time, "val", batch_size=self.batch_size)
            c_index_val, ci_va_ind = self.evaluation(labels_train, labels_val, times_train, times_val, preds_val, eval_time, "val", batch_size=self.batch_size, batch_count=None, model_type = self.model_type)


            print(f"Epoch: {epoch}")
            print(f"Training:       Loss: {loss_train:.3f}, Weighted C-index: {c_index_train}")
            print(f"Validation:     Loss: {loss_val:.3f}, Weighted C-index: {c_index_val}")

            if early_stop:
                early_stopping(loss_val, loss_train, self.model, epoch, c_index_train, ci_tr_ind, c_index_val, ci_va_ind, disease_ids)
                if early_stopping.early_stop:
                    metrics["best_ci_train"] = early_stopping.best_model_info['ci_tr']
                    metrics["best_ci_val"] = early_stopping.best_model_info['ci_va']
                    metrics["loss_val"] = early_stopping.best_model_info['loss_va']
                    metrics["loss_train"] = early_stopping.best_model_info['loss_tr']
                    if len(disease_ids) > 1:
                        for i, id in enumerate(disease_ids):
                            metrics[f"ci_train_{id}"] = early_stopping.best_model_info[f"ci_train_{id}"] 
                            metrics[f"ci_val_{id}"] = early_stopping.best_model_info[f"ci_val_{id}"]
                    print("Early stopping")
                    break
            elif epoch == epochs-1:
                early_stopping.save_checkpoint(self.model, epoch, c_index_train, ci_tr_ind, c_index_val, ci_va_ind, loss_val, loss_train, disease_ids)
                metrics["best_ci_train"] = early_stopping.best_model_info['ci_tr']
                metrics["best_ci_val"] = early_stopping.best_model_info['ci_va']
                metrics["loss_val"] = early_stopping.best_model_info['loss_va']
                metrics["loss_train"] = early_stopping.best_model_info['loss_tr']
                if len(disease_ids) > 1:
                    for i, id in enumerate(disease_ids):
                        metrics[f"ci_train_{id}"] = early_stopping.best_model_info[f"ci_train_{id}"] 
                        metrics[f"ci_val_{id}"] = early_stopping.best_model_info[f"ci_val_{id}"]
        
        return metrics, labels_train, times_train
    
    def step(self, dataloader, is_train, log):
        if is_train:
            self.model.train()
            context = contextlib.nullcontext()
        else:
            self.model.eval()
            context = torch.no_grad()
        
        with context:
            batch_loss = 0
            predictions, labels_all, times = torch.tensor([]), torch.tensor([]), torch.tensor([])
            
            for batch, data in enumerate(dataloader):
                x = data[0].to(self.device)
                labels = data[1].to(self.device).long()
                time = data[2].to(self.device)
                mask1 = data[3].to(self.device)
                mask2 = data[4].to(self.device)
            
                if is_train:
                    # Clear gradient:
                    self.optimizer.zero_grad()
            
                if self.model_type == 'deep':
                    # Compute prediction with forward pass
                  
                    logits = self.model(x).to(self.device)
                   
                    outputs = F.softmax(logits, dim=1) # shape [batch_size,num_diseases*num_Category] E.g. [128,36]

                    outputs = outputs.reshape(-1, mask1.shape[1], mask1.shape[2]) # shape [128,num_diseases,num_Category]

                    # Compute loss:
                    loss_val = self.loss_fn(outputs, labels, mask1, mask2, time)
                
                elif self.model_type == 'deep_time':
               
                    time_pred = self.model(x)
                    time_pred = time_pred.to(self.device)
                    outputs = time_pred

                    # Compute loss:
                    
                    loss_val = self.loss_fn(time_pred, time, labels)

                if is_train:
                    # Backward pass:
                    loss_val.backward()

                    # Update parameters
                    self.optimizer.step()

                # Compute performance evaluation

                predictions = torch.cat([predictions, outputs.cpu()])
                labels_all = torch.cat([labels_all, labels.cpu()])
                times = torch.cat([times, time.cpu()])
                # Track loss over batch
                batch_loss += loss_val.item()
            
            loss = batch_loss / (batch+1)
            
            return loss, predictions, labels_all, times
        
    def evaluation(self, tr_label, va_label, tr_time, va_time, preds, eval_time, mode, batch_size, batch_count, model_type):
        num_diseases = preds.shape[1]
        
        if model_type == 'deep':
            results_ci = np.zeros([num_diseases, len(eval_time)])
            
            if mode == 'train':

                # We separate data into batches to compute concordance index to decrease memory allocation. We shuffle so that all batches are representative:
                indices = np.arange(len(preds))
                np.random.shuffle(indices)
                times_val = np.array(va_time)[indices]
                labels_val = np.array(va_label)[indices]
                preds = np.array(preds.detach().numpy())[indices]
                
                times_val_batches = [times_val[i:i + (batch_size*batch_count)] for i in range(0, len(times_val), (batch_size*batch_count))]
                labels_val_batches = [labels_val[i:i + (batch_size*batch_count)] for i in range(0, len(labels_val), (batch_size*batch_count))]
                
                for t, t_time in enumerate(eval_time):
                    eval_horizon = int(t_time)
                    
                    results_ci_batch_sum = np.zeros([num_diseases])

                    num_batches = 0
                    
                    risk_batches = [np.sum(preds[i:i + (batch_size*batch_count), :, :(eval_horizon+1)], axis=2) for i in range(0, len(preds), (batch_size*batch_count))]
                
                    for batch, (va_time, va_label, risk) in enumerate(zip(times_val_batches, labels_val_batches, risk_batches)):
                        
                        for k in range(num_diseases): # loop through diseases
                            results_ci_batch_sum[k] += weighted_c_index(
                                tr_time[:,k].numpy(), # time
                                tr_label[:,k].numpy(), # labels
                                risk[:,k], 
                                va_time[:,k], 
                                va_label[:,k], eval_horizon
                                )
                        num_batches += 1

                    
                    results_ci[:,t] = results_ci_batch_sum / num_batches
               
                ci_ind = np.mean(results_ci, axis=1)
                ci = np.mean(results_ci)
                if ci > self.max_c_train:
                    self.stop_flag = 0
                    self.max_c_train = ci
                
            elif mode == 'val' or mode == 'test':
                
                for t, t_time in enumerate(eval_time):
                    eval_horizon = int(t_time)
                    risk = preds[:,:,:(eval_horizon+1)].sum(dim=2) # risk score until eval_time
                    for k in range(preds.shape[1]): # loop through diseases
                        results_ci[k,t] = weighted_c_index(tr_time[:,k].numpy(), (tr_label[:,k]).to(dtype=torch.int32).numpy(), risk[:,k], va_time[:,k].numpy(), (va_label[:,k]).to(dtype=torch.int32).numpy(), eval_horizon)
                
                ci_ind = np.mean(results_ci, axis=1)
                ci = np.mean(results_ci)
                if mode == 'val':
                    if ci > self.max_c_val:
                        self.stop_flag = 0
                        self.max_c_val = ci
                    else:
                        self.stop_flag +=1
            return ci, ci_ind
        
        elif model_type == 'deep_time' or model_type == 'deep_time_prob':
            
            ci_ind = np.zeros(num_diseases)
            for k in range(preds.shape[1]):
                ci_ind[k] = concordance_index(va_time[:,k], preds[:,k].detach(), va_label[:,k])
            ci = ci_ind.mean()
            return ci, ci_ind

class EarlyStopping:
    def __init__(self, out_dir, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta  # Amount validation metric has to vary to be considered an improvement.
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.out_dir = out_dir
        self.best_model_info = {}

    def __call__(self, val_loss, tr_loss, model, epoch, metric_tr, metric_tr_ind, metric_va, metric_va_ind, disease_ids):
        score = val_loss
     
        if self.best_score is None:  # first evaluation
            self.best_score = score
            self.save_checkpoint(model, epoch, metric_tr, metric_tr_ind, metric_va, metric_va_ind, val_loss, tr_loss, disease_ids)
        elif (
            score < self.best_score - self.min_delta
        ):  # If no improvement, increment counter.
            self.best_score = score
            self.save_checkpoint(model, epoch, metric_tr, metric_tr_ind, metric_va, metric_va_ind, val_loss, tr_loss, disease_ids)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                

    def save_checkpoint(self, model, epoch, metric_tr, metric_tr_ind, metric_va, metric_va_ind, val_loss, tr_loss, disease_ids):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.out_dir)
    
        self.best_model_info = {
            'epoch': epoch,
            'loss_tr': tr_loss,
            'loss_va': val_loss, 
            'ci_tr': metric_tr,
            'ci_va': metric_va
        }

        if len(metric_tr_ind) >1:
            for i, id in enumerate(disease_ids):
                self.best_model_info[f"ci_train_{id}"] = metric_tr_ind[i]
                self.best_model_info[f"ci_val_{id}"] = metric_va_ind[i]
