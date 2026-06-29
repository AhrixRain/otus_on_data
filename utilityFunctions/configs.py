import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

float_type = 'float32'

data_dir = '../../data/'

data_dims = {
    'ppzee': {'z_dim': 8, 'x_dim': 12}, # x_dim = 12 includes MET, this is removed in experiments
    'ppttbar': {'z_dim': 24, 'x_dim': 24},
}

TRAIN_NUM_SLICES = 1000
EVAL_NUM_SLICES = 1000  
