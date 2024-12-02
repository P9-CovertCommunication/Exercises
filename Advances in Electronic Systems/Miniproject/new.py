from __future__ import absolute_import, division, print_function
import random
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.io import savemat
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import wandb

torch.set_printoptions(precision=4)
np.set_printoptions(precision=4)


# ------ DNN ------
class Net(nn.Module):
    def __init__(self, num_sub, num_chan, hidden_dim, num_l, temperature, neg_slope = 0.1):
        super(Net, self).__init__()
        self.num_sub = num_sub
        self.num_chan = num_chan
        self.hidden_dim = hidden_dim
        self.num_l = num_l
        self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32), requires_grad=True)
        self.neg_slope = neg_slope
        ## List of linear layers
        self.linear1 = nn.Linear(num_sub * num_chan, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim)
        
        self.layers = [self.linear1, self.linear2, self.linear3, self.linear4]
        ## output for softmax and sigmoid
        self.out_RA = nn.Linear(hidden_dim, num_sub * num_chan)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor

        for layer in self.layers:
            x = F.relu(layer(x))
        # output layer
        out_RA = self.out_RA(x)
        out_RA = F.softmax(out_RA, dim=1)
        return out_RA

 
def build_dataset(batch_size):


    return train_loader, val_loader, len(y_val), X_val_uuid_total

def train(config=None):
    # Intialize wandb
    with wandb.init(config=config):
        # Initialize the model
        config = wandb.config
        #print(config)

        train_loader, val_loader, y_val_len, uuids_to_val = build_dataset(config.batch_size)
        
        model = BinaryClassifier(input_size=768, dropout_p=config.dropout_p)
        model.to(device)
        # Define the optimizer
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
        criterion = nn.BCELoss()

        # Logging metrics during training
        for epoch in range(config.epochs):
            # Training loop
            model.train()  # Set the model to training mode
            for inputs, labels in train_loader:
                optimizer.zero_grad()  # Zero the gradients
                #print("Input size: ", inputs.size())
                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Compute the loss
                loss.backward()  # Backward pass
                optimizer.step()  # Update the weights
                
                # Log metrics to wandb
            wandb.log({'train_loss': loss.item(), 'epoch': epoch})

            # Validation loop
            model.eval()  # Set the model to evaluation mode
            val_loss = 0.0
            correct = 0
            total = 0
            predicted_list = np.array([])
            labels_list = np.array([])
        
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels)
                    predicted = torch.round(outputs)
                    total += labels.size(0)
                    predicted_list = np.append(predicted_list, predicted.cpu().numpy())
                    labels_list = np.append(labels_list, labels.cpu().numpy())

                    for predictions, label in zip(predicted, labels):
                        if predictions == label:
                            correct += 1
                # Compute validation metrics
                #print(correct, total)
                accuracy = correct / total
                val_loss = val_loss/val_loader.__len__()
                
                # Log metrics to wandb
                wandb.log({'val_loss': val_loss, 'val_accuracy': accuracy, 'epoch': epoch, "user ": "Anders"})
                # wandb.log({'val_loss': val_loss, 'val_accuracy': accuracy, 'epoch': epoch, "user ": "David"})
                # wandb.log({'val_loss': val_loss, 'val_accuracy': accuracy, 'epoch': epoch, "user ": "Mads"})
                # wandb.log({'val_loss': val_loss, 'val_accuracy': accuracy, 'epoch': epoch, "user ": "Nicolai"})

        if CONFMATRIX == True:
            print(predicted_list.shape)
            cf_matrix = confusion_matrix(labels_list, predicted_list)
            print(len(labels_list))
            # Build confusion matrix
            classes = ('COVID-19', 'Healthy')
            df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None],
                                index = [i for i in classes],
                                columns = [i for i in classes])
            plt.figure(figsize = (12,7))
            ax = sn.heatmap(df_cm, annot=True,)
            ax.collections[0].set_clim(0,1)
            plt.title('Confusion matrix 4 layers')
            plt.savefig('119_4_layer+1.png')

        five_random_task = True
        if five_random_task == True:
            false_positives = []
            false_negatives = []
            correct = []
            for idx, pred in enumerate(predicted_list):

                # True positive
                if pred == 1 and labels_list[idx] == 1:
                    print(f"True positive: {uuids_to_val[idx]}")
                    correct.append(uuids_to_val[idx])
                # False positive
                if pred == 1 and labels_list[idx] == 0:
                    print(f"False positive: {uuids_to_val[idx]}")
                    false_positives.append(uuids_to_val[idx])

                # False negative
                if pred == 0 and labels_list[idx] == 1:
                    print(f"False negative: {uuids_to_val[idx]}")
                    false_negatives.append(uuids_to_val[idx])

            dic = {'False positives': false_positives, 'False negatives': false_negatives, 'Correct': correct}
            import pickle
            with open('false_pos_neg.pkl', 'wb') as f:
                pickle.dump(dic, f)

        with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels)
                    predicted = torch.round(outputs)
                    total += labels.size(0)
                    predicted_list = np.append(predicted_list, predicted.cpu().numpy())
                    labels_list = np.append(labels_list, labels.cpu().numpy())

                    for predictions, label in zip(predicted, labels):
                        if predictions == label:
                            correct += 1
                # Compute validation metrics
                #print(correct, total)
                accuracy = correct / total
                val_loss = val_loss/val_loader.__len__()
                
                # Log metrics to wandb
                wandb.log({'val_loss': val_loss, 'val_accuracy': accuracy, 'epoch': epoch, "user ": "Anders"})


# WandDB stads for logging
wandb.login()
# Initialize wandb with your project name and optionally specify other configurations

# Define sweep configuration
sweep_configuration = {
    "name": "4 layers 119 for plot",
    "method": "grid",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters": {
        "learning_rate": {'values':[0.0003875] },#list(np.linspace(0.0005,0.005,5))},
        "batch_size": {"values": [32]},#[16, 32, 64]},
        "epochs": {"values": [150]},#[20, 50, 100, 150]},
        "hidden_layers":{'value': 4},
        "dropout_p":{'values': [0.5]},#[0, 0.2, 0.5]},
        "optimizer": {"values": ["adam"]},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project='825-miniproject-DL')
#sweep_id = '825-miniproject-DL/825-miniproject-DL/wkcqgpfu'
#print("sweepid: ", sweep_id)
wandb.agent(sweep_id, function=train)