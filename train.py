"""
train.py: Trainer of the LSTM model. Input pickled numpy array of shape (n_bianoids, n_events, n_features). n_events is variable for each user.
    Target is of shape (n_users, 1), i.e. floats of target conversion ratio (CVR). train data is packed mini-batch packed to avoid padding and masking.
"""

# Native Libraries
import sys
import os
# Essential Libraries
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Callable
# AI-related
import optuna
import torch
import torch.cuda.amp
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
# Internal Package Librariess
from src import DAQ
from src import tools
from src.LSTM_conversion_prediction_nn import LSTMmodel


def train_epoch(dataloader: DataLoader, model: LSTMmodel, loss_fn: Callable, optimizer: torch.optim.Optimizer, accum_iter: int = 1,
                datatype: torch.dtype = torch.float16, scaler=GradScaler()) -> float:
    """
    Train model on given data.
    :param dataloader: [DataLoader] Dataloader to iterate over for (batched) training data.
    :param model: [nn.LSTM] Model to be inferred by given data.
    :param loss_fn: [Callable] Evaluated loss function, e.g. nn.MSELoss.
    :param optimizer: [Callable] Optimizer function (resp. algorithm) for updating model parameters.
    :param accum_iter: [int] How many batches accumulate gradients before backpropagation
    :param datatype: [torch.dtype] Type of data for casting to device (limit cuda memory usage)
    :param scaler: [amp.GradScaler] scaler for loss and gradient matching to prevent mixed precision underflow
    :return: [float] Average loss of all batches for the given epoch.
    """
    size = len(dataloader.dataset)
    num_batches = size/dataloader.batch_size
    train_loss = 0
    model.train()
    for ibatch, (X, y) in enumerate(dataloader):
        X, y = X.to(dtype=datatype, non_blocking=True, copy=False, device=torch.device(model.device.type)), y.to(X.data[0], non_blocking=True, copy=False)
        # Compute prediction error
        if model.device.type == 'cuda':
            with torch.cuda.amp.autocast(dtype=datatype):
                pred = model(X)
                # Calculate loss gradient
                loss = loss_fn(pred, y)
        else:
            pred = model(X)
            loss = loss_fn(pred, y)
        print("\tOutside Model: input size", X.data.size(),
              "output size", pred.size())
        # full loss, normalization is for gradient accumulation
        train_loss += loss.item()
        # --Backpropagation--
        # account for accumulating gradients
        loss = loss / accum_iter
        # backpropagation (fraction of importance due to optimizer steps)
        scaler.scale(loss).backward()
        # update every accum_iter iterations to provide more data to find pattern before resetting grads
        if (ibatch + 1) % accum_iter == 0 or (ibatch + 1) == len(dataloader):
            # Update weights
            # calls internally scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            # Reset the gradients to None
            optimizer.zero_grad(set_to_none=True)
        if ibatch % np.round(num_batches/20) == 0 and ibatch != 0:
            current = ibatch * y.shape[0]
            logger.info(f"loss(batch)): {accum_iter * loss.item():>7f}  [{current:>5d}/{size:>5d}]")
    train_loss /= num_batches
    logger.info(f"Train Error: \n MSE loss: {train_loss:>0.7f} \n")
    return train_loss


def evaluate(dataloader: DataLoader, model: LSTMmodel, loss_fn: Callable,
             datatype: torch.dtype = torch.float16) -> float:
    """
    Evaluate (test/validate) model on given data.
    :param dataloader: [DataLoader] Dataloader to iterate over for (batched) training data.
    :param model: [nn.LSTM] Model to be inferred by given data.
    :param loss_fn: [Callable] Evaluated loss function, e.g. nn.MSELoss.
    :param datatype: [torch.dtype] Type of data for casting to device (limit cuda memory usage)
    :return: [float]] Average loss of all batches for given epoch.
    """
    num_batches = len(dataloader)
    valid_loss = 0
    # set model mode (model.training=False) to testing
    model.eval()
    # don't calculate the gradients, there is only forward call
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(dtype=datatype, non_blocking=True, copy=False), y.to(X.data[0], non_blocking=True, copy=False)
            if model.device.type == 'cuda':
                with torch.cuda.amp.autocast(dtype=datatype):
                    pred = model(X)
                    # Calculate loss gradient
                    loss = loss_fn(pred, y)
            else:
                pred = model(X)
                # Calculate loss gradient
                loss = loss_fn(pred, y)
            valid_loss += loss.item()
    # Average statistics
    valid_loss /= num_batches
    logger.info(f"Train/validation Error: \n MSE loss: {valid_loss:>0.7f} \n")
    return valid_loss


def train_and_evaluate(h_params, params, model, trial):
    logger.info('----------------------------- Training settings ---------------------------')
    folder_path_project = Path.cwd().resolve()
    learned_dir = Path(folder_path_project, 'data', 'learned')
    os.makedirs(learned_dir, exist_ok=True)

    train_daterange, test_daterange, _ = tools.get_split_dataranges(folder_path_project)

    logger.info('\n Done')
    logger.info('----------------------------- Loading training data ---------------------------')

    logger.info(f"Loading input *training* data (days {train_daterange})")
    train_dataset = tools.get_dataset(train_daterange, folder_path_project)
    logger.info(f"Loading input *test* data (days {test_daterange})")
    test_dataset = tools.get_dataset(test_daterange, folder_path_project)

    logger.debug('Train dataset length: ')
    logger.debug(f'len Y: {len(train_dataset)}')
    logger.debug('Test dataset length: ')
    logger.debug(f'len Y: {len(test_dataset)}')

    logger.info('\n Done')

    logger.info('----------------------------- Setting up training tools ---------------------------')
    logger.info('------ Dataloaders -------')

    cpu_cores = torch.multiprocessing.cpu_count()
    # always leave at least one core for basic opereation (and eventual shutdown commands)
    if cpu_cores > 1:
        cpu_cores = cpu_cores - 1
    train_dataloader = DataLoader(train_dataset, batch_size=params['training_params']['batch_size'],
                                  collate_fn=tools.collate_fn, shuffle=True,
                                  num_workers=min(cpu_cores, params['training_params']['cpu_cores']), pin_memory=False)
    test_dataloader = DataLoader(test_dataset, batch_size=params['training_params']['batch_size'],
                                 collate_fn=tools.collate_fn, shuffle=False,
                                 num_workers=min(cpu_cores, params['training_params']['cpu_cores']), pin_memory=False)

    logger.info('------ loss function, optimizer and model definition -------')
    device, parallel = tools.resolve_device(params['training_params']['enforce_cpu'])
    # move model to device and set future hidden params to be created on device directly
    if not model.device == device:
        model.change_device(device)
    # Standard MSE regression loss function, no need for L1 or so.
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(h_params['learning_rate']), amsgrad=params['training_params']['adam_amsg'])
    scaler = GradScaler()
    logger.debug('\n Model architecture:')
    logger.debug(model)
    logger.debug('\n Model parameters:')
    nparams = tools.count_parameters(model)
    logger.debug(f'Total parameters: {nparams}')

    logger.info('\n Done')

    logger.info('----------------------------- Training ---------------------------')

    model_output_path = Path(learned_dir, f"model_{params['prepare_training']['n_features']}_{h_params['hidden_size']}_{h_params['num_layers']}.pt")
    logger.add(Path(learned_dir, "epoch_logger.txt"), filter=epoch_only, mode="w")
    epoch_logger = logger.bind(specific=True, name='epoch')

    train_loss = np.zeros(params['training_params']['epochs'])
    test_loss = np.zeros(params['training_params']['epochs'])
    min_test_loss = np.inf
    for t in range(params['training_params']['epochs']):
        logger.info(f"Epoch {t + 1}\n-------------------------------")
        dtype = getattr(sys.modules['torch'], params['training_params']['dtype'])
        train_loss[t] = train_epoch(train_dataloader, model, loss_fn, optimizer,
                                    params['training_params']['accum_iter']/params['training_params']['batch_size'],
                                    datatype=dtype, scaler=scaler)
        test_loss[t] = evaluate(test_dataloader, model, loss_fn, datatype=dtype)

        trial.report(test_loss[t], t)

        logger.info(f'Epoch {t + 1} \t\t Valid_loss Loss: {test_loss[t]} \t\t Validation Loss: {test_loss[t]}')
        if min_test_loss > test_loss[t]:
            epoch_logger.info(f'Validation Loss Decreased({min_test_loss:.6f}--->{test_loss[t]:.6f}) \t Saving The Model')
            min_test_loss = test_loss[t]
            # min_test_loss State Dict
            torch.save(model.state_dict(), model_output_path)
            epoch_logger.info('saved model to: ', model_output_path)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        epoch_logger.info(f'Epoch: {t}, loss {test_loss[t]}')
    logger.info('\n Done')

    train_dir = Path(learned_dir, f"{h_params['num_layers']}_{h_params['hidden_size']}_{h_params['learning_rate']}")
    os.makedirs(train_dir, exist_ok=True)
    np.save(Path(train_dir, 'train_loss.npy'), train_loss)
    np.save(Path(train_dir, 'test_loss.npy'), test_loss)

    torch.save(model.state_dict(), model_output_path)
    logger.info('saved model to: ', model_output_path)
    return test_loss


def objective(trial):
    params = DAQ.load_params(Path(Path.cwd(), 'data', 'params.yaml').resolve())
    h_params = {
        # 'learning_rate': trial.suggest_uniform('learning_rate', float(params['hyper']['learning_rate'][0]), float(params['hyper']['learning_rate'][1])),
        # 'num_layers': trial.suggest_int("num_layers", int(params['hyper']['num_layers'][0]), int(params['hyper']['num_layers'][1])),
        # 'hidden_size': trial.suggest_categorical("hidden_size", params['hyper']['hidden_size'])
        'learning_rate': trial.suggest_categorical('learning_rate', params['hyper']['learning_rate']),
        'num_layers': trial.suggest_categorical("num_layers", params['hyper']['num_layers']),
        'hidden_size': trial.suggest_categorical("hidden_size", params['hyper']['hidden_size']),
    }

    model = LSTMmodel(params['prepare_training']['n_features'], hidden_size=h_params['hidden_size'],
                      num_layers=h_params['num_layers'], stateful=bool(params['training_params']['stateful']))

    return train_and_evaluate(h_params, params, model, trial)


def hyper_tune():
    params = DAQ.load_params(Path(Path.cwd(), 'data', 'params.yaml').resolve())
    torch.set_default_dtype(getattr(sys.modules['torch'], params['training_params']['dtype']))
    # benchmark speeds up when data shape doesn't change. Not our case and when True blocks GPU cuda at collate_fn level.
    # torch.backends.cudnn.benchmark = False
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(),
                                pruner=optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(),
                                                                    patience=params['training_params']['patience_epochs'],
                                                                    min_delta=params['training_params']['min_delta']),
                                direction="minimize")
    study.optimize(objective, n_trials=100)

    best_trial = study.best_trial

    for key, value in best_trial.params.items():
        logger.info("{}: {}".format(key, value))

    optuna.visualization.plot_intermediate_values(study)
    optuna.visualization.plot_optimization_history(study)
    optuna.visualization.plot_param_importances(study)


def epoch_only(record):
    return "epoch" in record["extra"]


if __name__ == "__main__":
    logger.add(Path(Path.cwd(), 'data', "run.log"), level="DEBUG", mode="w")
    logger.add(sys.stderr, level="ERROR")
    try:
        # required even without distributed package, if num_workers>0 (cuda tensors initialized in dataset/collate_fn for performance)
        torch.multiprocessing.set_start_method('spawn', force=True)
        hyper_tune()
    except Exception as e:
        logger.error(f"Command failed. {e}", exc_info=True)
        raise
