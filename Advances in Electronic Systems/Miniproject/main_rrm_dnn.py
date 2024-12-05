import numpy as np

import matplotlib.pyplot as plt
# import seaborn as sns
import static_subnetwork_generator
# import DNN_subbandd
# from resourcemanager_2d import ResourceAllocator
from tqdm import tqdm
from dnn_subband_allocation import DNN_model, evaluate_model_on_new_data
import torch

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
        
plt.close('all')


limited_CSI = 0

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_subn = 20
N_low = int(num_subn*1/2)
N_high = int(num_subn*1/2)
target_rate = torch.tensor(np.squeeze(np.concatenate((0.4*np.ones((1,N_low)), 8*np.ones((1,N_high))), axis=1)))

config = init_parameters(0, num_subn, target_rate)

# print('#### Generating subnetwork ####')

#ch_coef = static_subnetwork_generator.generate_static_samples(config, 10)

ch_coef = np.load('Channel_matrix_gain.npy')
ch_coef = ch_coef[:,0,:,:] # reduce the dimension of the channel matrix, only using one subchannel gain maritrix
ch_coef = torch.from_numpy(ch_coef).float().to(device)
tot_sample_faktor = 0.8
snapshots = int(ch_coef.shape[0]*(1-tot_sample_faktor))
tot_sample_tr = int(ch_coef.shape[0]-snapshots)
loc_val_tr = ch_coef[0:tot_sample_tr,:,:]
loc_val_te = ch_coef[tot_sample_tr:tot_sample_tr+snapshots,:,:]

# Train model

DNN_model(loc_val_tr, loc_val_te, config, target_rate, config.max_power)



## Evaluation set
new_snapshots = 64
new_ch_gain = torch.tensor(static_subnetwork_generator.generate_static_samples(config, new_snapshots),device=device).float()
train_mean = torch.mean(torch.log(loc_val_tr))
train_std = torch.std(torch.log(loc_val_tr))

# Load the trained model
model_path = 'model.pth'

# Evaluate model on new data
results = evaluate_model_on_new_data(model_path, 1024,new_ch_gain, config, train_mean, train_std, target_rate)

# Print the results
print("Predictions for new data:", results["predictions"])
print(f"New data capacity: {results['capacity']}")
print(f"New data DNN Score: {results['score']}")
print(f"Low-load Score: {results['low_load_score']}")
print(f"High-load Score: {results['high_load_score']}")

