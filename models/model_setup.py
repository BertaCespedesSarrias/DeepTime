from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
import torch.nn as nn
from training_loop import Trainer

class BaseModel: 
    def __init__(self, args):
        self.args = args
        self.model = None
    
    def train(self, X_train, y_train):
        pass

    def evaluate(self, X_test, y_test):
        pass

class CoxModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.model = CoxPHSurvivalAnalysis()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        score = self.model.score(X_test, y_test)
        return score
    
    def predict(self, X):
        pred = self.model.predict(X)
        return pred

class RandomForestModel(BaseModel):   
    def __init__(self, args):
        super().__init__(args)
        self.model = RandomSurvivalForest(random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        pred = self.model.predict(X)
        return pred

def choose_model(args, device=None, num_feat=None, num_diseases=None, time_span=None, fold=None):
    if args.model_type == 'cox':
        model = CoxModel(args)
    elif args.model_type == 'random_forest':
        model = RandomForestModel(args)
    elif args.model_type == 'deep':
        fcnnsurv_model = FCNNSurv(args, num_feat, num_diseases,time_span)
        model = DeepModel(args, device, fcnnsurv_model, fold)
    elif args.model_type == 'deep_time':
        fcnnsurv_time = FCNNSurv_time(num_feat, num_diseases)
        model = DeepModel(args, device, fcnnsurv_time, fold)
    return model


class DeepModel(BaseModel):
    def __init__(self, args, device, torch_model, fold):
        super().__init__(args)
        self.model = torch_model
        self.trainer = Trainer(self.model, args.model_type, device)
        self.epochs = args.epochs
        self.out_dir = args.out_dir
        self.learning_rate = args.learning_rate
        self.log = args.log # Tensorboard log flag
        self.run = args.run
        self.fold = fold

    def train(self, train_dataloader, val_dataloader, loss_fn, out_dir_save, batch_size, batch_count, early_stop, disease_ids):
        metrics = self.trainer.train(self.epochs, train_dataloader, val_dataloader, loss_fn, out_dir = out_dir_save, run=self.run, batch_count=batch_count, disease_ids = disease_ids, learning_rate = self.learning_rate, fold=self.fold, log = self.log, batch_size= batch_size, early_stop=early_stop)
        return metrics
    
    def predict(self, X_test):
        return self.model(X_test)
    

class FCNNSurv(nn.Module):
    def __init__(self, args, num_feat, num_diseases, time_span):
        super(FCNNSurv, self).__init__()
        self.layer1 = nn.Linear(num_feat, num_feat*3)
        self.dropout1 = nn.Dropout(p=0.4)
        self.layer2 = nn.Linear(num_feat*3, num_feat*5)
        self.dropout2 = nn.Dropout(p=0.4)
        self.layer3 = nn.Linear(num_feat*5, num_feat*3)
        self.dropout3 = nn.Dropout(p=0.4)
        self.layer4 = nn.Linear(num_feat*3, num_diseases*time_span)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout1(self.relu(x))
        x = self.layer2(x)
        x = self.dropout2(self.relu(x))
        x = self.layer3(x)
        x = self.dropout3(self.relu(x))
        x = self.layer4(x)
        return x

class FCNNSurv_time(nn.Module):
    def __init__(self, num_feat, num_diseases):
        super(FCNNSurv_time, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(num_feat, num_feat*3),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(num_feat*3, num_feat*5),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(num_feat*5, num_feat*3),
            nn.ReLU(),
            nn.Dropout(p=0.4)
        )

        self.time_head = nn.Linear(num_feat*3, num_diseases)

    def forward(self, x):
        features = self.shared_layers(x)
        time = self.time_head(features)
        return time