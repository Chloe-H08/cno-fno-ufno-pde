import copy
import json
import os
import sys

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


from Problems.CNOBenchmarks import Darcy, Airfoil, DiscContTranslation, ContTranslation, AllenCahn, SinFrequency, WaveEquation, ShearLayer, BurgersCNO, NavierStokes

class TeeLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

os.makedirs("logs", exist_ok=True)
sys.stdout = TeeLogger("logs/navier_stokes_cno_train_log.txt")
sys.stderr = sys.stdout 

if len(sys.argv) <= 2:
    
    training_properties = {
        "learning_rate": 0.001, 
        "weight_decay": 1e-6,
        "scheduler_step": 10,
        "scheduler_gamma": 0.98,
        "epochs":500,
        "batch_size": 16,
        "exp": 2,                # Do we use L1 or L2 errors? Default: L1
        "training_samples": 800  # How many training samples?
    }
    model_architecture_ = {
        
        #Parameters to be chosen with model selection:
        "N_layers": 3,            # Number of (D) & (U) blocks 
        "channel_multiplier": 32, # Parameter d_e (how the number of channels changes)
        "N_res": 4,               # Number of (R) blocks in the middle networs.
        "N_res_neck" : 6,         # Number of (R) blocks in the BN
        
        #Other parameters:
        "in_size": 128,            # Resolution of the computational grid
        "retrain": 4,             # Random seed
        "kernel_size": 3,         # Kernel size.
        "FourierF": 0,            # Number of Fourier Features in the input channels. Default is 0.
        "activation": 'cno_lrelu',# cno_lrelu or cno_lrelu_torch or lrelu or 
        
        #Filter properties:
        "cutoff_den": 2.0001,     # Cutoff parameter.
        "lrelu_upsampling": 2,    # Coefficient N_{\sigma}. Default is 2.
        "half_width_mult": 0.8,   # Coefficient c_h. Default is 1
        "filter_size": 6,         # 2xfilter_size is the number of taps N_{tap}. Default is 6.
        "radial_filter": 0,       # Is the filter radially symmetric? Default is 0 - NO.
    }
    
    #   "which_example" can be 
    
    #   poisson             : Poisson equation 
    #   wave_0_5            : Wave equation
    #   cont_tran           : Smooth Transport
    #   disc_tran           : Discontinuous Transport
    #   allen               : Allen-Cahn equation
    #   shear_layer         : Navier-Stokes equations
    #   airfoil             : Compressible Euler equations
    #   darcy               : Darcy Flow

    # which_example = sys.argv[1]
    which_example = "ns"

    # Save the models here:
    folder = "TrainedModels/"+"CNO_"+which_example+"_1"
        
else:
    
    # Do we use a script to run the code (for cluster):
    folder = sys.argv[1]
    training_properties = json.loads(sys.argv[2].replace("\'", "\""))
    model_architecture_ = json.loads(sys.argv[3].replace("\'", "\""))
    which_example = sys.argv[4]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(log_dir=folder) #usage of TensorBoard

learning_rate = training_properties["learning_rate"]
epochs = training_properties["epochs"]
batch_size = training_properties["batch_size"]
weight_decay = training_properties["weight_decay"]
scheduler_step = training_properties["scheduler_step"]
scheduler_gamma = training_properties["scheduler_gamma"]
training_samples = training_properties["training_samples"]
p = training_properties["exp"]

if not os.path.isdir(folder):
    print("Generated new folder")
    os.mkdir(folder)

df = pd.DataFrame.from_dict([training_properties]).T
df.to_csv(folder + '/training_properties.txt', header=False, index=True, mode='w')
df = pd.DataFrame.from_dict([model_architecture_]).T
df.to_csv(folder + '/net_architecture.txt', header=False, index=True, mode='w')

if which_example == "shear_layer":
    example = ShearLayer(model_architecture_, device, batch_size, training_samples, size = 64)
elif which_example == "poisson":
    example = SinFrequency(model_architecture_, device, batch_size, training_samples)
elif which_example == "wave_0_5":
    example = WaveEquation(model_architecture_, device, batch_size, training_samples)
elif which_example == "allen":
    example = AllenCahn(model_architecture_, device, batch_size, training_samples)
elif which_example == "cont_tran":
    example = ContTranslation(model_architecture_, device, batch_size, training_samples)
elif which_example == "disc_tran":
    example = DiscContTranslation(model_architecture_, device, batch_size, training_samples)
elif which_example == "airfoil":
    model_architecture_["in_size"] = 128
    example = Airfoil(model_architecture_, device, batch_size, training_samples)
elif which_example == "darcy":
    example = Darcy(model_architecture_, device, batch_size, training_samples)
elif which_example == "burgers":
    example = BurgersCNO(model_architecture_, device, batch_size, training_samples)
elif which_example == "ns":
    example = NavierStokes(model_architecture_, device, batch_size, training_samples)
else:
    raise ValueError()
    
#-----------------------------------Train--------------------------------------
model = example.model
n_params = model.print_size()
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model has {n_params:,} trainable parameters")
train_loader = example.train_loader #TRAIN LOADER
test_loader = example.test_loader #Test LOADER

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
freq_print = 1

if p == 1:
    loss = torch.nn.L1Loss()
elif p == 2:
    loss = torch.nn.MSELoss()
    
for epoch in range(epochs):
    with tqdm(unit="batch", disable=False) as tepoch:
        model.train()
        tepoch.set_description(f"Epoch {epoch}")
        train_mse = 0.0
        for step, (input_batch, output_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            input_batch = input_batch.to(device)
            output_batch = output_batch.to(device)
            output_pred_batch = model(input_batch)

            if which_example == "airfoil":
                output_pred_batch[input_batch == 1] = 1
                output_batch[input_batch == 1] = 1

            loss_f = loss(output_pred_batch, output_batch) / loss(torch.zeros_like(output_batch).to(device), output_batch)
            loss_f.backward()
            optimizer.step()
            train_mse = train_mse * step / (step + 1) + loss_f.item() / (step + 1)
            tepoch.set_postfix({'Train Loss': train_mse})

        writer.add_scalar("train_loss/train_loss", train_mse, epoch)

        with torch.no_grad():
            model.eval()
            total_l2 = 0.0
            total_h1 = 0.0
            for step, (input_batch, output_batch) in enumerate(test_loader):
                input_batch = input_batch.to(device)
                output_batch = output_batch.to(device)
                output_pred_batch = model(input_batch)

                if which_example == "airfoil":
                    output_pred_batch[input_batch == 1] = 1
                    output_batch[input_batch == 1] = 1

                l2_error = torch.norm(output_pred_batch - output_batch) / torch.norm(output_batch)
                grad_output_pred = torch.gradient(output_pred_batch, dim=(-2, -1))
                grad_output_true = torch.gradient(output_batch, dim=(-2, -1))
                h1_error = sum(torch.norm(gp - gt) for gp, gt in zip(grad_output_pred, grad_output_true))
                total_l2 += l2_error.item()
                total_h1 += h1_error.item()

            avg_l2 = total_l2 / len(test_loader)
            avg_h1 = total_h1 / len(test_loader)
            writer.add_scalar("test/L2_error", avg_l2, epoch)
            writer.add_scalar("test/H1_error", avg_h1, epoch)
            print(f"\nEpoch {epoch}: Test L2 Error = {avg_l2:.6f}, H1 Error = {avg_h1:.6f}")

        tepoch.close()
        scheduler.step()

# Save model at last epoch
final_model_path = folder + ".pth"
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved to {final_model_path}")
