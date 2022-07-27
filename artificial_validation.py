import os
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src import tools
from src import DAQ
from src.LSTM_conversion_prediction_nn import LSTMmodel
from src.custom_dataset import CompleteDataset
from train import train_epoch, evaluate
from validate import validate, plot_results


def generate_dataset(n_blocks, n_users):
    X_dataset = []
    y_dataset = []
    for j in range(n_blocks):
        arr = np.empty(n_users, object)
        targets = np.zeros(n_users, dtype=np.float32)
        for i in range(n_users):
            shift = np.round(np.random.random()*10)
            len_shift = np.round(np.random.random()*400)-200
            # print('generating x points')
            np1 = np.sin(np.arange(500 + len_shift) + shift)
            np2 = np.sin(np.arange(500 + len_shift) - shift)
            # print('generating y points')
            targets[i] = np.cos(np1[-1])**2 + np.cos(np2[-1])**2
            # wo sinus
#             np1 = np1/(500+200+10)
#             np2 = np2/(500+200+10)
#             print('targets finished, stacking')
            arr[i] = np.stack((np1, np2), axis=1).astype(np.float32)
        X_dataset += [arr]
        y_dataset += [targets]
        # logger.info('Block complete')
        # with open(Path(f'/media/data/Biano/research/conversion-prediction/artificial_dataset/train_{j}.pkl'), 'wb') as f:
        #     pickle.dump(arr, f, pickle.HIGHEST_PROTOCOL)
        # with open(Path(f'/media/data/Biano/research/conversion-prediction/artificial_dataset/test_{j}.pkl'), 'wb') as f:
        #     pickle.dump(targets, f, pickle.HIGHEST_PROTOCOL)
    return X_dataset, y_dataset


def plot_dataset_check(X, y, arti_valid_folder, plotrange = 20):
    fig = plt.figure(figsize=(14, 6), dpi=200)
    prange = min(plotrange, X[0][0].shape[0])
    plt.plot(np.arange(prange), X[0][0][:prange, 0], label='dataset 1')
    plt.plot(np.arange(prange), X[0][0][:prange, 1], label='dataset 2')
    plt.scatter(prange, y[0][0], label='target')
    plt.xlabel('event_number[-]')
    plt.ylabel('value[-]')
    plt.legend()
    plt.savefig(Path(arti_valid_folder, f'Artificial_check.png'))
    plt.show()


def validate_artificial(args, artificial_folder, recalculate=False, device=torch.device('cpu')):
    model_path = Path(args[2]) if Path(args[2]).exists() else Path(artificial_folder, args[2])
    valid_loss_path = Path(args[3]) if Path(args[3]).exists() else Path(artificial_folder, args[3])
    predictions_path = Path(args[4]) if Path(args[4]).exists() else Path(artificial_folder, args[4])
    targets_path = Path(args[5]) if Path(args[5]).exists() else Path(artificial_folder, args[5])

    train_path = Path()
    test_path = Path()
    if len(args) == 8:
        targets_path = Path(args[7], 'train_loss.npy') if Path(args[7], 'train_loss.npy').exists() \
            else Path(artificial_folder, args[7], 'train_loss.npy')
        test_path = Path(args[7], 'test_loss.npy') if Path(args[7], 'test_loss.npy').exists() \
            else Path(artificial_folder, args[7], 'test_loss.npy')
    else:
        logger.info('Train and test loss files not found, plotting only validation.')
    if not model_path.exists():
        logger.error('There is no model file to load architecture from.')
        raise Exception
    if valid_loss_path.exists() and predictions_path.exists() and targets_path.exists() and not recalculate:
        logger.info('Reading already calculated validation, i.e. plotting only.')
        valid_loss = np.load(valid_loss_path)
        predictions = np.load(predictions_path)
        targets = np.load(targets_path)
    else:
        X_val, y_val = generate_dataset(params['artificial']['n_blocks'][2], params['artificial']['n_users'])
        validation_dataset = CompleteDataset(X_val, y_val, params['artificial']['names_val'])
        logger.debug('Validation dataset length: ')
        logger.debug(f'len Y: {len(validation_dataset)}')
        logger.info('\n Done')

        valid_loss, predictions, targets = validate(model_path, valid_loss_path, predictions_path, targets_path, validation_dataset, params)

    logger.info('----------------- Plotting ------------------')
    logger.info('Data obtained, plotting.')
    if train_path != Path():
        train_loss = np.load(train_path)
        test_loss = np.load(test_path)
        plot_results(valid_loss_path.parents[0], predictions, targets, valid_loss, plot_size=params['artificial']['plot_size'], train_loss=train_loss, test_loss=test_loss)
    else:
        plot_results(valid_loss_path.parents[0], predictions, targets, valid_loss, plot_size=params['artificial']['plot_size'])


def train_artificial(artificial_folder, device=torch.device('cpu')):
    cpu_cores = torch.multiprocessing.cpu_count()
    # always leave at least one core for basic opereation (and eventual shutdown commands)
    if cpu_cores > 1:
        cpu_cores = cpu_cores - 1
    X_train, y_train = generate_dataset(params['artificial']['n_blocks'][0], int(params['artificial']['n_users']))
    X_test, y_test = generate_dataset(params['artificial']['n_blocks'][1], int(params['artificial']['n_users']))
    train_dataset = CompleteDataset(X_train, y_train, params['artificial']['names_train'])
    train_loader = DataLoader(train_dataset, batch_size=params['artificial']['batch_size'], collate_fn=tools.collate_fn,
                              shuffle=True, num_workers=min(cpu_cores, params['artificial']['cpu_cores']))
    test_dataset = CompleteDataset(X_test, y_test, params['artificial']['names_test'])
    test_loader = DataLoader(test_dataset, batch_size=params['artificial']['batch_size'], collate_fn=tools.collate_fn,
                             shuffle=False, num_workers=min(cpu_cores, params['artificial']['cpu_cores']))
    plot_dataset_check(X_train, y_train, arti_valid_folder, params['artificial']['plot_size'])

    logger.info('------ loss function, optimizer and model definition -------')
    model = LSTMmodel(params['artificial']['n_features'], hidden_size=params['artificial']['hidden_size'],
                      num_layers=params['artificial']['num_layers'], stateful=bool(params['artificial']['stateful']))
    if not model.device == device:
        model.change_device(device)
    # Standart regression loss function, no need for L1 or so.
    loss_fn = nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=lrate, momentum = mom)
    optimizer = torch.optim.Adam(model.parameters(), lr=np.float32(params['artificial']['lrate']))
    scaler = GradScaler()
    logger.info('\n Done')

    logger.info('----------------------------- Training ---------------------------')

    model_output_path = Path(artificial_folder, f"model_{params['artificial']['n_features']}_{params['artificial']['hidden_size']}_{params['artificial']['num_layers']}.pt")
    logger.add(Path(artificial_folder, "epoch_logger.txt"), filter=epoch_only, mode="w")
    epoch_logger = logger.bind(specific=True, name='epoch')

    train_loss = np.zeros(params['artificial']['epochs'])
    test_loss = np.zeros(params['artificial']['epochs'])
    min_test_loss = np.inf
    for t in range(params['artificial']['epochs']):
        logger.info(f"Epoch {t + 1}\n-------------------------------")
        dtype = getattr(sys.modules['torch'], params['training_params']['dtype'])
        train_loss[t] = train_epoch(train_loader, model, loss_fn, optimizer,
                                    params['artificial']['accum_iter'] / params['artificial']['batch_size'],
                                    datatype=dtype, scaler=scaler)
        logger.info(f"Epoch {t + 1} train complete, testing.")
        test_loss[t] = evaluate(test_loader, model, loss_fn, datatype=dtype)

        logger.info(f'Epoch {t + 1} \t\t Valid_loss Loss: {test_loss[t]} \t\t Validation Loss: {test_loss[t]}')
        if min_test_loss > test_loss[t]:
            epoch_logger.info(f'Validation Loss Decreased({min_test_loss:.6f}--->{test_loss[t]:.6f}) \t Saving The Model')
            min_test_loss = test_loss[t]
            # min_test_loss State Dict
            torch.save(model.state_dict(), model_output_path)
            epoch_logger.info(f'saved model to: {model_output_path}')

        epoch_logger.info(f'Epoch: {t}, loss {test_loss[t]}')

    logger.info('\n Done')

    train_dir = Path(artificial_folder, f"{params['artificial']['num_layers']}_{params['artificial']['hidden_size']}_{params['artificial']['lrate']}")
    os.makedirs(train_dir, exist_ok=True)
    np.save(Path(train_dir, 'train_loss.npy'), train_loss)
    np.save(Path(train_dir, 'test_loss.npy'), test_loss)

    torch.save(model.state_dict(), model_output_path)
    logger.info(f'saved model to: {model_output_path}')


def epoch_only(record):
    return "epoch" in record["extra"]


if __name__ == "__main__":
    try:
        project_folder_path = Path(__file__).resolve().parent
        arti_valid_folder = Path(project_folder_path, 'data', 'artificial')
        os.makedirs(arti_valid_folder, exist_ok=True)
        logger.add(Path(arti_valid_folder, "run.log"), level="DEBUG", mode="w")
        torch.multiprocessing.set_start_method('spawn', force=True)

        params = DAQ.load_params(Path(Path.cwd(), 'data', 'params.yaml').resolve())
        device, parallel = tools.resolve_device(bool(params['artificial']['enforce_cpu']))
        if len(sys.argv) < 2:
            logger.error("Not enough parameters.")
            raise Exception
        if sys.argv[1] == 'train':
            train_artificial(arti_valid_folder, device=device)
        elif sys.argv[1] == 'validate':
            validate_artificial(sys.argv, arti_valid_folder, bool(sys.argv[6] == 'True'), device=device)
        else:
            logger.error("Parameter fail. Valid parameter is one, value 'train' or 'validate'.")
            raise Exception
    except Exception as e:
        logger.error("Runtime error, main fail.")
        raise e

