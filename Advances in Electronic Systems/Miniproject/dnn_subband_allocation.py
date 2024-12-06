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
import wandb
import static_subnetwork_generator

torch.set_printoptions(precision=4)
np.set_printoptions(precision=4)

logging = False  # Set this flag to True to enable logging

sweep_configuration = {
    "name": "Advance in electronic system",
    "method": "grid",
    "metric": {"goal": "minimize", "name": "test_loss"},
    "parameters": {
        "batch_size": {"values": [1024/2, 1024, 1024*2]},#[16, 32, 64]},
        "epochs": {"values": [100]},#[20, 50, 100, 150]},
        "dropout_p":{'values': [0, 0.1, 0.3, 0.4]},#[0, 0.2, 0.5]},
        "learning_rate": {'values' : [1e-7, 1e-6, 1e-5]},
        "loss_func" : {'values': [0,1,2,3,4]},
        "normalize" : {'values': [True, False]},
        "optimizer": {"values": ["adam"]},
    },
}

class init_parameters:
    def __init__(self,rng, num_of_subn, target_rate):
        self.num_of_subnetworks = num_of_subn
        self.target_rate = target_rate
        self.n_subchannel = 4
        self.deploy_length = 20                                 # the length and breadth of the factory area (m)
        self.subnet_radius = 1                                  # the radius of the subnetwork cell (m)
        self.minD = 0.8                                         # minimum distance from device to controller(access point) (m)
        self.minDistance = 2 * self.subnet_radius               # minimum controller to controller distance (m)
        self.rng_value = np.random.RandomState(rng)
        self.bandwidth = 100e6                                    # bandwidth (Hz)
        self.ch_bandwidth = self.bandwidth / self.n_subchannel
        self.fc = 1e10                                          # Carrier frequency (Hz)
        self.lambdA = 3e8/self.fc
        # self.plExponent = 3                                   # path loss exponent
        self.clutType = 'dense'                                 # Type of clutter (sparse or dense)
        self.clutSize = 2.0                                     # Clutter element size [m]
        self.clutDens = 0.6                                     # Clutter density [%]
        self.shadStd = 7.2                                      # Shadowing std (NLoS)
        self.max_power = 1
        self.no_dbm = -174
        self.noise_figure_db = 5
        self.noise_power = 10 ** ((self.no_dbm + self.noise_figure_db + 10 * np.log10(self.ch_bandwidth)) / 10)
        self.mapXPoints = np.linspace(0, self.deploy_length, num=401, endpoint=True)
        self.mapYPoints = np.linspace(0, self.deploy_length, num=401, endpoint=True)
        self.correlationDistance = 5


class Net(nn.Module):
    def __init__(self, num_sub, num_chan, hidden_dim, num_l, temperature, neg_slope = 0.001, dropout = 0.1):
        super(Net, self).__init__()
        self.num_sub = num_sub
        self.num_chan = num_chan
        self.hidden_dim = hidden_dim
        self.num_l = num_l
        self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32), requires_grad=True)
        self.neg_slope = neg_slope
        ## List of linear layers
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(num_sub**2, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim)
        
        self.layers = [self.linear1, self.linear2, self.linear3, self.linear4]
        ## output for softmax and sigmoid
        self.out_RA = nn.Linear(hidden_dim, num_sub * num_chan)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        out_RA = self.out_RA(x)
        
        return out_RA

def Loss(self, subn_channel_index, channel, noise, chan_mean, chan_std, rate_thr, power, device, N_low, func_idx):
    loss = 0
    loss_functions = [nn.ReLU(), nn.LeakyReLU(self.neg_slope), nn.SiLU()]

    loss_func = loss_functions[func_idx].to(device)
    
    _, _, _, score_low, score_high, max_rates_sum, max_rate = capacity(subn_channel_index, channel, noise, chan_mean, chan_std, rate_thr, power, device, N_low)
    #diff = (torch.subtract(rate_thr, max_rate_mean))
    diff = (torch.subtract(rate_thr, max_rate))
    # print(diff < 0)
    #loss = torch.div(loss_func(diff),rate_thr)   # evaluating the LReLU
    loss = loss_func(diff)   # evaluating the LReLU
    loss = torch.sum(torch.mean(loss,0))

    return loss, score_low, score_high

def capacity(subn_channel_index, chan, noise, chan_mean, chan_std, rate_thr, power, device, N_low):

    N = chan.size(1)
    
    cap_tot = torch.zeros((chan.size(0), N,), dtype=torch.float32).to(device)
    channel = torch.exp(chan * chan_std + chan_mean)

    tr_power = power * torch.ones(channel.size(0), channel.size(1), channel.size(1))
    tr_power = torch.transpose(tr_power, 1, 2)
    cap_val_matrix = torch.zeros((chan.size(0), N,subn_channel_index.shape[-1]), dtype=torch.float32).to(device)
    
    for k in range(subn_channel_index.shape[-1]):
        mask =  subn_channel_index[:, :, k].unsqueeze(-1).expand(channel.size(0), channel.size(1),
                                                                channel.size(1)) * torch.transpose(
                subn_channel_index[:, :, k].unsqueeze(-1).expand(channel.size(0), channel.size(1), channel.size(1)), 1, 2)
        tot_ch = tr_power.to(device) * torch.mul(channel.to(device), mask.to(device)).to(device)
        sig_ch = torch.diagonal(tot_ch, dim1=1, dim2=2).to(device)
        inter_ch = tot_ch - torch.diag_embed(sig_ch).to(device)
        inter_vec = torch.sum(inter_ch, -1).to(device)
        SINR_val = torch.div(sig_ch, (inter_vec + noise)).to(device)   # Signal to Interference and Noise
        
        cap_val = torch.log2(1.0 + SINR_val)                # spectral efficiency
        cap_val_matrix[:, :, k] = cap_val

    rate_thr = torch.tensor(rate_thr).expand(cap_tot.size(0), cap_tot.size(1)).float().to(device)
    max_rates, max_subchannels = torch.max(cap_val_matrix, dim=-1)
    score_low_DNN = torch.sum((max_rates[:, 0:N_low] > rate_thr[:, 0:N_low]), 1).float()  # Not Used
    score_high_DNN = torch.sum((max_rates[:, N_low:] > rate_thr[:, N_low:]), 1).float()   # Not Used
    score_DNN = score_low_DNN.data + score_high_DNN.data  # .float()

    max_rates_mean = torch.mean(max_rates,0).to(device)
    max_rates_sum = torch.sum(max_rates_mean).to(device)

    cap = torch.mean(torch.sum(cap_tot, 1))
    
   
    return cap, cap_tot, score_DNN, score_low_DNN, score_high_DNN, max_rates_sum, max_rates

def generate_cdf(values, bins_, types=None):
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
        x = torch.tensor(self.data[idx],device='cpu')
        return x  # Return the input tensor as both input and output

def DNN_model(wandb_config = None):
    with wandb.init(config=wandb_config):
        conf = wandb.config
        net = Net(config.num_of_subnetworks, config.n_subchannel, 1024, 4, 0.01, 0.001, conf.dropout_p).to(device)

        loc_val_tr_db = torch.log(loc_val_tr).to(device) #We use natural log as this was implemented in the capacity method
        loc_val_te_db = torch.log(loc_val_te).to(device)
        
        train_mean = torch.mean(loc_val_tr_db).to(device)
        train_std = torch.std(loc_val_tr_db).to(device)
        
        loc_val_tr_norm = torch.div((loc_val_tr_db - train_mean), train_std).to(device)
        loc_val_te_norm = torch.div((loc_val_te_db - train_mean), train_std).to(device)
        
        #Create dataset and dataloader
        train_dataset = ChanDataset(loc_val_tr_norm)
        test_dataset = ChanDataset(loc_val_te_norm)
        
        bs = conf.batch_size
        epochs = conf.epochs
        learningRate = conf.learning_rate
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
        
        optimizer = optim.Adam(net.parameters(), lr=learningRate)
        softmax = nn.Softmax(dim=-1).to(device)
        sigmoid = nn.Sigmoid()
        temperatures = torch.linspace(1 , 0.01, epochs, device=device)

        for epoch in tqdm(range(epochs)):
            running_loss = 0.0
            net.train()
            for batch in train_loader:

                batch = batch.to(device)
                optimizer.zero_grad()
                output = net(batch)
                output_flat = output.view(-1, config.num_of_subnetworks, config.n_subchannel)
                #delta_sharpness =  1-nn.functional.sigmoid(torch.tensor(-3+(torch.mul(5,torch.div(epoch,epochs)))))
                output = softmax(torch.div(output_flat,temperatures[epoch]))
                
                loss, _, _= Loss(net, output, batch, config.noise_power, train_mean, train_std, target_rate, config.max_power, device, N_low)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            wandb.log({'train_loss': running_loss, 'epoch': epoch})
            # print(f"Epoch {epoch}, Loss: {running_loss}")
                            
            net.eval()
            test_loss = 0.0
            score_low_list = []
            score_high_list = []
            
            with torch.no_grad():
                for batch in test_loader:

                    batch = batch.to(device)
                    output = net(batch)
                    output_flat = output.view(-1, config.num_of_subnetworks, config.n_subchannel)
                    loss, score_low, score_high = Loss(net, output_flat, batch, config.noise_power, train_mean, train_std, target_rate, config.max_power, device, N_low, conf.loss_func)
                    score_low_list.append(score_low)
                    score_high_list.append(score_high)
                    test_loss += loss.item()
            
            wandb.log({'test_loss': test_loss,
                        'epoch': epoch,
                        'score_low' : torch.mean(torch.cat(score_low_list,dim=0)),
                        'score_high' : torch.mean(torch.cat(score_high_list,dim=0))})

            torch.save(net.state_dict(), f'model.pth')


def evaluate_model_on_new_data(model_path, hidden_dim, new_data, config, train_mean, train_std, target_rate, N_low,device):
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
  
    _, cap_dl_new, score_new, score_low_new, score_high_new, max_rates_sum, max_rates = capacity(
        dl_sel_dec, torch.Tensor(new_data_norm).to(device), config.noise_power,
        train_mean, train_std, torch.tensor(target_rate).to(device), power, device, N_low
    )

    bins_low, low_cdf = generate_cdf(score_low_new, N_low)
    bins_high, high_cdf = generate_cdf(score_high_new, N_low)
    return {
        "predictions": results,
        "capacity": max_rates_sum,
        "score": score_new.mean().item(),
        "low_load_score": score_low_new.mean().item(),
        "low mean subnet capacities": torch.mean(max_rates,0)[:N_low].to(device),
        "low_load mean capacity" : torch.mean(torch.mean(max_rates,0)[:N_low]).to(device),
        "high_load_score": score_high_new.mean().item(),
        "High mean subnet capacities": torch.mean(max_rates,0)[N_low:].to(device),
        "high_load mean capacity" : torch.mean(torch.mean(max_rates,0)[N_low:]).to(device),
        "low_cdf" : low_cdf,
        "high_cdf": high_cdf,
        "low_bins": bins_low,
        "high_bins" : bins_high
    }

if __name__ == '__main__':
    limited_CSI = 0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = 'cpu'
    num_subn = 20
    N_low = int(num_subn*1/2)
    N_high = int(num_subn*1/2)
    target_rate = torch.tensor(np.squeeze(np.concatenate((0.4*np.ones((1,N_low)), 8*np.ones((1,N_high))), axis=1)),device=device)

    config = init_parameters(0, num_subn, target_rate)

    # print('#### Generating subnetwork ####')

    #ch_coef = static_subnetwork_generator.generate_static_samples(config, 10)

    ch_coef = np.load('Channel_matrix_gain.npy')
    ch_coef = ch_coef[:,0,:,:] # reduce the dimension of the channel matrix, only using one subchannel gain maritrix
    ch_coef = torch.from_numpy(ch_coef).float().to(device)
    


    tot_sample_faktor = 0.8
    test_size = 10000
    snapshots = 10000
    tot_sample_tr = int(ch_coef.shape[0]-snapshots-test_size)
    loc_val_tr = ch_coef[0:tot_sample_tr,:,:]
    loc_val_te = ch_coef[tot_sample_tr:tot_sample_tr+snapshots,:,:]
    loc_val_test = ch_coef[tot_sample_tr+snapshots:]

    # Train model
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="Advance in electronic system")
    wandb.agent(sweep_id, DNN_model)
    # DNN_model(loc_val_tr, loc_val_te, config, target_rate, config.max_power, N_low,device)



    # ## Evaluation set
    # new_snapshots = 10000
    # # new_ch_gain = torch.tensor(static_subnetwork_generator.generate_static_samples(config, new_snapshots),device=device).float()
    # train_mean = torch.mean(torch.log(loc_val_tr))
    # train_std = torch.std(torch.log(loc_val_tr))

    # # Load the trained model
    # #model_path = 'model.pth'
    # model_path = 'model_learningRate_e-6.pth'

    # # Evaluate model on new data
    # results = evaluate_model_on_new_data(model_path, 1024, loc_val_test, config, train_mean, train_std, target_rate, N_low,device)


    # # Print the results
    # print("Predictions for new data:", results["predictions"])
    # print(f"New data capacity: {results['capacity']}")
    # print(f"New data DNN Score: {results['score']}")
    # print("---------------------------------------------------------------------------")
    # print(f"Low-load Score: {results['low_load_score']}")
    # print(f"Low-load mean subnet capacities: {results['low mean subnet capacities']}")
    # print(f"low_load mean capacity : {results['low_load mean capacity']}")
    # print("---------------------------------------------------------------------------")
    # print(f"High-load Score: {results['high_load_score']}")
    # print(f"High-load mean subnet capacities: {results['High mean subnet capacities']}")
    # print(f"high_load mean capacity : {results['high_load mean capacity']}")

    # plt.figure("High rate cdf")
    # plt.plot(results['high_bins'], results["high_cdf"])
    # plt.grid(True,zorder=2)

    # plt.figure("Low rate cdf")
    # plt.plot(results['low_bins'], results["low_cdf"])
    # plt.grid(True,zorder=2)

    # plt.show()

