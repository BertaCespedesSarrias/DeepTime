import torch
import torch.nn.functional as F
from training_loop import Trainer
import numpy as np

def test_loop(model, test_loader, metrics, labels_train, times_train, args, device, disease_ids):

    tr_time_unique = torch.unique(times_train.flatten())
    eval_time = [int(np.percentile(tr_time_unique, 25)), int(np.percentile(tr_time_unique, 50)), int(np.percentile(tr_time_unique, 75))]

    with torch.no_grad():
        model.eval()
        predictions, labels_all, times = torch.tensor([]), torch.tensor([]), torch.tensor([])
        for _, data in enumerate(test_loader):
            x = data[0].to(device)
            labels = data[1].to(device).long()
            time = data[2].to(device)
            mask1 = data[3].to(device)

            if args.model_type == 'deep':
                logits = model(x).to(device)
                outputs = F.softmax(logits, dim=1)
                outputs = outputs.reshape(-1, mask1.shape[1], mask1.shape[2])
            elif args.model_type == 'deep_time':
                time_pred = model(x)
                time_pred = time_pred.to(device)
                outputs = time_pred
            predictions = torch.cat([predictions, outputs.cpu()])
            labels_all = torch.cat([labels_all, labels.cpu()])
            times = torch.cat([times, time.cpu()])
    
    trainer = Trainer(model, args.model_type, device)

    ci, ci_ind = trainer.evaluation(labels_train, labels_all, times_train, times, predictions, eval_time, mode='test', batch_size=args.batch_size, batch_count = None, model_type = args.model_type)
    
    metrics['ci_test'] = ci
    if len(disease_ids) > 1:
        for i, id in enumerate(disease_ids):
            metrics[f"ci_test_{id}"] = ci_ind[i]
    return metrics
    