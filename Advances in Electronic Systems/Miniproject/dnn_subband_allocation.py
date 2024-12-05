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
from tqdm import tqdm

torch.set_printoptions(precision=4)
np.set_printoptions(precision=4)

logging = False  # Set this flag to True to enable logging

class Net(nn.Module):
    def __init__(self, num_sub, num_chan, hidden_dim, num_l, temperature, neg_slope = 0.001):
        super(Net, self).__init__()
        self.num_sub = num_sub
        self.num_chan = num_chan
        self.hidden_dim = hidden_dim
        self.num_l = num_l
        self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32), requires_grad=True)
        self.neg_slope = neg_slope
        ## List of linear layers
        self.linear1 = nn.Linear(num_sub**2, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim)
        
        self.layers = [self.linear1, self.linear2, self.linear3, self.linear4]
        ## output for softmax and sigmoid
        self.out_RA = nn.Linear(hidden_dim, num_sub * num_chan)

    def forward(self, x):
        if logging:
            print(f"x shape in forward: {x.shape}")
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = F.relu(layer(x))
        out_RA = self.out_RA(x)
        
        return out_RA

def Loss(self, subn_channel_index, channel, noise, chan_mean, chan_std, rate_thr, power, device):
    if logging:
        print(f"subn_channel_index shape in Loss: {subn_channel_index.shape}")

    loss = 0
    loss_func = nn.LeakyReLU(self.neg_slope).to(device)
    #loss_func = nn.ReLU()
    _, _, _, _, _, cap_tot_mean = capacity(subn_channel_index, channel, noise, chan_mean, chan_std, rate_thr, power, device)  
    rate_thr = torch.tensor(rate_thr).to(device)
    diff = (torch.subtract(rate_thr, cap_tot_mean))
    # print(diff < 0)
    loss = torch.div(loss_func(diff),rate_thr)   # evaluating the LReLU
    loss = torch.sum(loss)

    return loss

def capacity(subn_channel_index, chan, noise, chan_mean, chan_std, rate_thr, power, device):
    if logging:
        print(f"subn_channel_index shape in capacity: {subn_channel_index.shape}")

    N = chan.size(1)
    N_low = int(3 * N / 4)

    cap_tot = torch.zeros((chan.size(0), N,), dtype=torch.float32).to(device)
    channel = torch.exp(chan * chan_std + chan_mean)

    tr_power = power * torch.ones(channel.size(0), channel.size(1), channel.size(1))
    tr_power = torch.transpose(tr_power, 1, 2)
    cap_val_matrix = torch.zeros((chan.size(0), N,subn_channel_index.shape[-1]), dtype=torch.float32).to(device)
    
    for k in range(subn_channel_index.shape[-1]):
        mask = subn_channel_index[:, :, k].unsqueeze(-1).expand(channel.size(0), channel.size(1),
                                                                channel.size(1)) * torch.transpose(
            subn_channel_index[:, :, k].unsqueeze(-1).expand(channel.size(0), channel.size(1), channel.size(1)), 1, 2)
        tot_ch = tr_power.to(device) * torch.mul(channel.to(device), mask.to(device)).to(device)
        sig_ch = torch.diagonal(tot_ch, dim1=1, dim2=2).to(device)
        inter_ch = tot_ch - torch.diag_embed(sig_ch).to(device)
        inter_vec = torch.sum(inter_ch, -1).to(device)
        SINR_val = torch.div(sig_ch, (inter_vec + noise)).to(device)   # Signal to Interference and Noise
        cap_val = torch.log2(1.0 + SINR_val)                # spectral efficiency
        cap_tot += cap_val.to(device)
        cap_val_matrix[:, :, k] = cap_val
    rate_thr = torch.tensor(rate_thr).expand(cap_tot.size(0), cap_tot.size(1)).float().to(device)

    score_low_DNN = torch.sum((cap_tot[:, 0:N_low] > rate_thr[:, 0:N_low]), 1).float()  # Not Used
    score_high_DNN = torch.sum((cap_tot[:, N_low:] > rate_thr[:, N_low:]), 1).float()   # Not Used
    score_DNN = score_low_DNN.data + score_high_DNN.data  # .float()
    max_sub_chan = torch.argmax(cap_val_matrix, -1).to(device)
    max_rates, max_subchannels = torch.max(cap_val_matrix, dim=-1)
    max_rates_mean = torch.mean(max_rates,0).to(device)

    cap = torch.mean(torch.sum(cap_tot, 1))
    
   
    return cap, cap_tot, score_DNN, score_low_DNN, score_high_DNN, max_rates_mean

def generate_cdf(values, bins_, types):
    values = values.cpu().detach().numpy()
    if types == 's':
        values = np.sum(values, 1)
    data = (values.reshape((-1, 1)))  #
    count, bins_count = np.histogram(data, bins=bins_)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    return bins_count[1:], cdf

class ChanDataset(Dataset):
    def __init__(self, loc_val):
        super(ChanDataset, self).__init__()
        self.data = torch.Tensor(loc_val)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx],device='cuda')
        return x  # Return the input tensor as both input and output

def DNN_model(loc_val_tr, loc_val_te, config, target_rate, max_power):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = Net(config.num_of_subnetworks, config.n_subchannel, 1024, 4, 0.01).to(device)

    loc_val_tr_db = torch.log(loc_val_tr).to(device) #We use natural log as this was implemented in the capacity method
    loc_val_te_db = torch.log(loc_val_te).to(device)
    
    train_mean = torch.mean(loc_val_tr_db).to(device)
    train_std = torch.std(loc_val_tr_db).to(device)
    
    loc_val_tr_norm = torch.div((loc_val_tr_db - train_mean), train_std).to(device)
    loc_val_te_norm = torch.div((loc_val_te_db - train_mean), train_std).to(device)
    
    #Create dataset and dataloader
    train_dataset = ChanDataset(loc_val_tr_norm)
    test_dataset = ChanDataset(loc_val_te_norm)
    
    bs = 1024
    epochs = 200
    learningRate = 1e-5
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
    
    optimizer = optim.Adam(net.parameters(), lr=learningRate)
    softmax = nn.Softmax(dim=1).to(device)

    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        if logging:
            print(f"Training")
        net.train()
        
        for batch in train_loader:
            if logging:
                print(f"Batch shape: {batch.shape}")

            batch = batch.to(device)
            optimizer.zero_grad()
            output = net(batch)
            if logging:
                print(f"output shape: {output.shape} before loss")
            output_flat = output.view(-1, config.num_of_subnetworks, config.n_subchannel)
            if logging:
                print(f"output_flat shape: {output_flat.shape} before loss")
            output = softmax(output_flat)
            
            loss = Loss(net, output, batch, config.noise_power, train_mean, train_std, torch.tensor(target_rate), max_power, device)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Epoch {epoch}, Loss: {running_loss}")
                        
        net.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                if logging:
                    print(f" Eval Batch shape: {batch.shape}")
                batch = batch.to(device)
                output = net(batch)
                output_flat = output.view(-1, config.num_of_subnetworks, config.n_subchannel)
                if logging:
                    print(f" Eval output shape: {output.shape} before loss")
                loss = Loss(net, output_flat, batch, config.noise_power, train_mean, train_std, target_rate, max_power, device)
                test_loss += loss.item()
    
        torch.save(net.state_dict(), f'model.pth')
    return net

def evaluate_model_on_new_data(model_path, hidden_dim, new_data, config, train_mean, train_std, target_rate):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = Net(config.num_of_subnetworks, config.n_subchannel, hidden_dim, 4, 0.01).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    new_data_db = torch.log(new_data)
    new_data_norm = (new_data_db - train_mean) / train_std

    new_data_dataset = ChanDataset(new_data_norm)
    new_data_loader = DataLoader(new_data_dataset, batch_size=32, shuffle=False)

    net.eval()
    results = []
    with torch.no_grad():
        for batch in new_data_loader:
            batch = batch.to(device)
            output = net(batch)
            output_flat = output.view(-1, config.num_of_subnetworks, config.n_subchannel)
            output_decision = torch.argmax(output_flat, dim=-1)  # Get the sub-band allocation decisions
            results.append(output_decision)

    results = torch.concatenate(results, axis=0)
    tensor_results = torch.tensor(results)

    #power = config.max_power * torch.ones(new_data.shape[0], config.num_of_subnetworks).to(device)
    power = config.max_power
    dl_sel_dec = F.one_hot(tensor_results, num_classes=config.n_subchannel).float().to(device)
    _, cap_dl_new, score_new, score_low_new, score_high_new, cap_tot_new_mean = capacity(dl_sel_dec, torch.Tensor(new_data_norm).to(device),
                                                                        config.noise_power, train_mean, train_std,
                                                                        torch.tensor(target_rate).to(device), power, device
    )
   

    return {
        "predictions": results,
        "capacity": cap_dl_new.mean().item(),
        "score": score_new.mean().item(),
        "low_load_score": score_low_new.mean().item(),
        "high_load_score": score_high_new.mean().item()
    }
