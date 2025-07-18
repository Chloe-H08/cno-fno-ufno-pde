import sys
import os
import torch
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

from neuralop import H1Loss, LpLoss, BurgersEqnLoss, ICLoss, WeightedSumLoss, Trainer, get_model
from neuralop.data.datasets import load_mini_burgers_1dtime
from neuralop.data.transforms.data_processors import MGPatchingDataProcessor
from neuralop.training import setup, AdamW
from neuralop.utils import get_wandb_api_key, count_model_params, get_project_root

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

# Save to logs/burgers_train_log.txt
os.makedirs("logs", exist_ok=True)
sys.stdout = TeeLogger("logs/burgers_train_log.txt")


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Read the configuration
config_name = "default"
# Read the configuration
from zencfg import cfg_from_commandline, cfg_from_nested_dict
import sys 
sys.path.insert(0, '../')
from config.burgers_config import Default


config = cfg_from_commandline(Default)
config = config.to_dict()

# Set-up distributed communication, if using
device, is_logger = setup(config)

# Set up WandB logging
if config.wandb.log and is_logger:
    wandb.login(key=get_wandb_api_key())
    if config.wandb.name:
        wandb_name = config.wandb.name
    else:
        wandb_name = "_".join(
            f"{var}"
            for var in [
                config_name,
                config.model.model_arch,
                config.model.n_layers,
                config.model.n_modes,
                config.model.hidden_channels,
            ]
        )
    wandb_init_args = dict(
        config=config,
        name=wandb_name,
        group=config.wandb.group,
        project=config.wandb.project,
        entity=config.wandb.entity,
    )
    if config.wandb.sweep:
        for key in wandb.config.keys():
            config.params[key] = wandb.config[key]
    wandb.init(**wandb_init_args)
else: 
    wandb_init_args = None
# Make sure we only print information when needed
config.verbose = config.verbose and is_logger

# Print config to screen
if config.verbose:
    print("##### CONFIG ######")
    print(config)
    sys.stdout.flush()

data_path = get_project_root() / config.data.folder
print ("path",data_path)
# Load the Burgers dataset
train_loader, test_loaders, data_processor = load_mini_burgers_1dtime(data_path=data_path,
        n_train=config.data.n_train, batch_size=config.data.batch_size, 
        n_test=config.data.n_tests[0], test_batch_size=config.data.test_batch_sizes[0],
        temporal_subsample=config.data.get("temporal_subsample", 1),
        spatial_subsample=config.data.get("spatial_subsample", 1),
        )

model = get_model(config)

# Use distributed data parallel
if config.distributed.use_distributed:
    model = DDP(
        model, device_ids=[device.index], output_device=device.index, static_graph=True
    )

# Create the optimizer
optimizer = AdamW(
    model.parameters(),
    lr=config.opt.learning_rate,
    weight_decay=config.opt.weight_decay,
)

if config.opt.scheduler == "ReduceLROnPlateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config.opt.gamma,
        patience=config.opt.scheduler_patience,
        mode="min",
    )
elif config.opt.scheduler == "CosineAnnealingLR":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.opt.scheduler_T_max
    )
elif config.opt.scheduler == "StepLR":
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.opt.step_size, gamma=config.opt.gamma
    )
else:
    raise ValueError(f"Got scheduler={config.opt.scheduler}")


# Creating the losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)
ic_loss = ICLoss()
equation_loss = BurgersEqnLoss(method=config.opt.get('pino_method', 'fdm'), 
                               visc=0.01, loss=F.mse_loss)

training_loss = config.opt.training_loss
if not isinstance(training_loss, (tuple, list)):
    training_loss = [training_loss]

losses = []
weights = []
for loss in training_loss:
    # Append loss
    if loss == 'l2':
        losses.append(l2loss)
    elif loss == 'h1':
        losses.append(h1loss)
    elif loss == 'equation':
        losses.append(equation_loss)
    elif loss == 'ic':
        losses.append(ic_loss)
    else:
        raise ValueError(f'Training_loss={loss} is not supported.')

    # Append loss weight
    if "loss_weights" in config.opt:
        weights.append(config.opt.loss_weights.get(loss, 1.))
    else:
        weights.append(1.)

train_loss = WeightedSumLoss(losses=losses, weights=weights)
eval_losses = {"h1": h1loss, "l2": l2loss}

if config.verbose:
    print("\n### MODEL ###\n", model)
    print("\n### OPTIMIZER ###\n", optimizer)
    print("\n### SCHEDULER ###\n", scheduler)
    print("\n### LOSSES ###")
    print(f"\n * Train: {train_loss}")
    print(f"\n * Test: {eval_losses}")
    print(f"\n### Beginning Training...\n")
    sys.stdout.flush()

# only perform MG patching if config patching levels > 0
if config.patching.levels > 0:
    data_processor = MGPatchingDataProcessor(model=model,
                                        levels=config.patching.levels,
                                        padding_fraction=config.patching.padding,
                                        stitching=config.patching.stitching,
                                        device=device,
                                        in_normalizer=data_processor.in_normalizer,
                                        out_normalizer=data_processor.out_normalizer)

trainer = Trainer(
    model=model,
    n_epochs=config.opt.n_epochs,
    data_processor=data_processor,
    device=device,
    mixed_precision=config.opt.mixed_precision,
    eval_interval=config.opt.eval_interval,
    log_output=config.wandb.log_output,
    use_distributed=config.distributed.use_distributed,
    verbose=config.verbose,
    wandb_log = config.wandb.log
)

# Log parameter count
if is_logger:
    n_params = count_model_params(model)

    if config.verbose:
        print(f"\nn_params: {n_params}")
        sys.stdout.flush()

    if config.wandb.log:
        to_log = {"n_params": n_params}
        if config.n_params_baseline is not None:
            to_log["n_params_baseline"] = (config.n_params_baseline,)
            to_log["compression_ratio"] = (config.n_params_baseline / n_params,)
            to_log["space_savings"] = 1 - (n_params / config.n_params_baseline)
        wandb.log(to_log, commit=False)
        wandb.watch(model)

trainer.train(
    train_loader,
    test_loaders,
    optimizer,
    scheduler,
    regularizer=False,
    training_loss=train_loss,
    eval_losses=eval_losses,
)

if is_logger:
    model_path = os.path.join("TrainedModels", "burger_final_model.pth")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # If using DDP, get the wrapped model
    model_to_save = model.module if isinstance(model, DDP) else model

    torch.save(model_to_save.state_dict(), model_path)

    if config.verbose:
        print(f"\nModel saved to {model_path}\n")

if config.wandb.log and is_logger:
    wandb.finish()
