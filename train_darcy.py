from pathlib import Path
import sys
import os

import torch

from torch.utils.data import DataLoader, DistributedSampler
import wandb

from neuralop import H1Loss, LpLoss, Trainer, get_model
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.data.transforms.data_processors import MGPatchingDataProcessor
from neuralop.training import setup, AdamW
from neuralop.mpu.comm import get_local_rank
from neuralop.utils import get_wandb_api_key, count_model_params
from torch.nn.parallel import DistributedDataParallel as DDP

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

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
    sys.stdout = TeeLogger("logs/darcy_train_log.txt")

    
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    # Read the configuration
    from zencfg import ConfigBase, cfg_from_commandline
    import sys 
    sys.path.insert(0, '../')
    from config.darcy_config import Default


    config = cfg_from_commandline(Default)
    config = config.to_dict()

    # Set-up distributed communication, if using
    device, is_logger = setup(config)

    # Set up WandB logging
    wandb_args = None
    if config.wandb.log and is_logger:
        wandb.login(key=get_wandb_api_key())
        if config.wandb.name:
            wandb_name = config.wandb.name
        else:
            wandb_name = "_".join(
                f"{var}"
                for var in [
                    config.model.model_arch,
                    config.model.n_layers,
                    config.model.n_modes,
                    config.model.hidden_channels,
                ]
            )
        wandb_args =  dict(
            config=config,
            name=wandb_name,
            group=config.wandb.group,
            project=config.wandb.project,
            entity=config.wandb.entity,
        )
        if config.wandb.sweep:
            for key in wandb.config.keys():
                config.params[key] = wandb.config[key]
        wandb.init(**wandb_args)

    # Make sure we only print information when needed
    config.verbose = config.verbose and is_logger

    # Print config to screen
    if config.verbose and is_logger:
        print(f"##### CONFIG #####\n")
        print(config)
        sys.stdout.flush()

    # Loading the Darcy flow dataset
    data_root = Path(config.data.folder).expanduser()
    print (data_root)
    train_loader, test_loaders, data_processor = load_darcy_flow_small(
        data_root=data_root,
        n_train=config.data.n_train,
        batch_size=config.data.batch_size,
        test_resolutions=config.data.test_resolutions,
        n_tests=config.data.n_tests,
        test_batch_sizes=config.data.test_batch_sizes,
        encode_input=False,
        encode_output=False,
    )
    model = get_model(config)

    # convert dataprocessor to an MGPatchingDataprocessor if patching levels > 0
    if config.patching.levels > 0:
        data_processor = MGPatchingDataProcessor(model=model,
                                                in_normalizer=data_processor.in_normalizer,
                                                out_normalizer=data_processor.out_normalizer,
                                                padding_fraction=config.patching.padding,
                                                stitching=config.patching.stitching,
                                                levels=config.patching.levels,
                                                use_distributed=config.distributed.use_distributed,
                                                device=device)

    # Reconfigure DataLoaders to use a DistributedSampler 
    # if in distributed data parallel mode
    if config.distributed.use_distributed:
        train_db = train_loader.dataset
        train_sampler = DistributedSampler(train_db, rank=get_local_rank())
        train_loader = DataLoader(dataset=train_db,
                                batch_size=config.data.batch_size,
                                sampler=train_sampler)
        for (res, loader), batch_size in zip(test_loaders.items(), config.data.test_batch_sizes):
            
            test_db = loader.dataset
            test_sampler = DistributedSampler(test_db, rank=get_local_rank())
            test_loaders[res] = DataLoader(dataset=test_db,
                                batch_size=batch_size,
                                shuffle=False,
                                sampler=test_sampler)
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
    if config.opt.training_loss == "l2":
        train_loss = l2loss
    elif config.opt.training_loss == "h1":
        train_loss = h1loss
    else:
        raise ValueError(
            f'Got training_loss={config.opt.training_loss} '
            f'but expected one of ["l2", "h1"]'
        )
    eval_losses = {"h1": h1loss, "l2": l2loss}

    if config.verbose and is_logger:
        print("\n### MODEL ###\n", model)
        print("\n### OPTIMIZER ###\n", optimizer)
        print("\n### SCHEDULER ###\n", scheduler)
        print("\n### LOSSES ###")
        print(f"\n * Train: {train_loss}")
        print(f"\n * Test: {eval_losses}")
        print(f"\n### Beginning Training...\n")
        sys.stdout.flush()

    trainer = Trainer(
        model=model,
        n_epochs=config.opt.n_epochs,
        device=device,
        data_processor=data_processor,
        mixed_precision=config.opt.mixed_precision,
        wandb_log=config.wandb.log,
        eval_interval=config.opt.eval_interval,
        log_output=config.wandb.log_output,
        use_distributed=config.distributed.use_distributed,
        verbose=config.verbose and is_logger,
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

    # Train the model
    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=False,
        training_loss=train_loss,
        eval_losses=eval_losses,
    )

    if is_logger:
        model_path = os.path.join("checkpoints", "darcy_final_model.pth")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # If using DDP, get the wrapped model
        model_to_save = model.module if isinstance(model, DDP) else model

        torch.save(model_to_save.state_dict(), model_path)

        if config.verbose:
            print(f"\nModel saved to {model_path}\n")

    if config.wandb.log and is_logger:
        wandb.finish()
