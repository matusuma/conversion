# Native Libraries
import os
import sys
# Essential Libraries
from pathlib import Path
import numpy as np
from loguru import logger
from typing import Callable, Tuple, List
# AI-related
import torch
from torch import nn
from torch.utils.data import DataLoader
# Plotting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# Internal package libs
from src import DAQ
from src import tools
from src.LSTM_conversion_prediction_nn import LSTMmodel

"""
usage

>>> python3 validate.py model.pt validation_loss.npy(output name) predictions.npy targets.npy recalculate[bool] train_losses_folder
e.g.
>>> python3 validate.py model_7_512_64.pt valid_loss.npy predictions.npy targets.npy False 64_512_5e-4
"""


def loss_only(record: dict):
    """
    logger filter function to separate from complete record only custom context ('loss'), which is sent to the logger sink.
    :param record: [dict] logger record dictionary, containing contextual information in nested record["extra"] dictionary.
    :return: attributes bound by
    """
    return "loss" in record["extra"]


def evaluate(dataloader: DataLoader, model: LSTMmodel, loss_fn: Callable,
             datatype: torch.dtype = torch.float16) -> Tuple[float, List, List]:
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
    predictions = []
    targets = []
    # set model mode (model.training=False) to testing
    model.eval()
    # don't calculate the gradients, there is only forward call
    with torch.no_grad():
        for X, y in dataloader:
            # cast datatype, non_blocking allows asynchronous data transfer.
            # Make sure y is on the same device as X (tensor as parameter copies its attributes)
            X, y = X.to(dtype=datatype, non_blocking=True), y.to(X.data[0], non_blocking=True)
            pred, hidden = model(X)
            # -- TEST START: Testing in case of target invariance is needed - to verify pairing, etc.
            # y = y[X.unsorted_indices]
            # pred = pred[X.unsorted_indices]
            # print(X.unsorted_indices, X.sorted_indices)
            # TEST END --
            # evaluate loss criterion, i.e. error between targets and predictions. See criterion fn, usual default is mean over batch.
            loss = loss_fn(pred, y)
            # accumulate batch losses to all data loss
            valid_loss += loss.item()
            # save for visualization
            predictions += [pred]
            targets += [y]
    # Average loss for prediction
    valid_loss /= num_batches
    logger.info(f"Test/validation Error: \n MSE loss: {valid_loss:>0.7f} \n")
    # flatten list accumulation to 1D tensors
    predictions = torch.cat(predictions, 0).flatten()
    targets = torch.cat(targets, 0).flatten()
    # move tensors back to cpu if cuda was used, numpy computations from now on.
    return valid_loss, predictions.cpu().detach().numpy(), targets.cpu().detach().numpy()


def validate(model_path, loss_file, predictions_file, targets_file, validation_dataset, params):
    logger.info('----------------------------- Setting up tools ---------------------------')
    logger.info('------ Dataloaders -------')

    cpu_cores = torch.multiprocessing.cpu_count()
    valid_dataloader = DataLoader(validation_dataset, batch_size=params['validation_params']['batch_size'],
                                  collate_fn=tools.collate_fn, shuffle=False, num_workers=min(cpu_cores, params['validation_params']['cpu_cores']), pin_memory=False)

    logger.info('------ loss function and model definition -------')
    device, _ = tools.resolve_device(params['validation_params']['enforce_cpu'])

    n_features, hidden_size, num_layers = model_path.name.replace('.', '_').split('_')[1:-1]
    logger.info(f"Setting up model architecture with parameters(features, hidden size, number of layers): {n_features}, {hidden_size}, {num_layers}")
    model = LSTMmodel(int(n_features), hidden_size=int(hidden_size), num_layers=int(num_layers), stateful=bool(params['training_params']['stateful']))
    logger.info(f'Loading model for validation from file: {model_path}')
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    # move model to device and set future hidden params to be created on device directly
    if not model.device == device:
        model.change_device(device)
    # Standard MSE regression loss function, no need for L1 or so.
    loss_fn = nn.MSELoss()
    logger.info('------ Validation ------')
    logger.info(f"All set up, starting validation.")

    logger.add(Path(loss_file.parents[0], "validate_loss_logger.txt"), filter=loss_only, mode="w")
    loss_logger = logger.bind(specific=True, name='loss')

    epochs = params['validation_params']['epochs']
    n_datapoints = len(valid_dataloader.dataset)
    valid_loss = np.zeros(epochs)
    predictions = np.zeros((epochs, n_datapoints))
    targets = np.zeros((epochs, n_datapoints))
    min_valid_loss = np.inf
    for epoch in range(epochs):
        dtype = getattr(sys.modules['torch'], params['validation_params']['dtype'])
        valid_loss[epoch], predictions[epoch], targets[epoch] = evaluate(valid_dataloader, model, loss_fn, dtype)
        logger.info(f'Epoch {epoch + 1} \t\t Valid_loss Loss: {valid_loss[epoch]}')
        if min_valid_loss > valid_loss[epoch]:
            loss_logger.info(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss[epoch]:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss[epoch]
            # Saving State Dict
            torch.save(model.state_dict(), Path(model_path.parents[0], 'validated_model.pt'))
    np.save(loss_file, valid_loss)
    np.save(predictions_file, predictions)
    np.save(targets_file, targets)
    return valid_loss, predictions, targets


def plot_results(validation_folder, pred, targets, valid_loss, plot_size=None, train_loss=None, test_loss=None):
    # --------- Losses (validation; train and test if available) -----------
    plt.figure(figsize=(8, 6), dpi=200)
    plt.suptitle(f"Losses through epochs")
    # minimal = {np.minimum(loss)} @ e{np.argmin(loss)}
    if train_loss:
        plt.plot(train_loss, label=f'train, min = {np.amin(train_loss)} @ e{np.argmin(train_loss)}')
    if test_loss:
        plt.plot(test_loss, label=f'test, min = {np.amin(test_loss)} @ e{np.argmin(test_loss)}')
    plt.plot(valid_loss, label=f'validation, min = {np.amin(valid_loss)} @ e{np.argmin(valid_loss)}')
    plt.legend()
    plt.xlabel('Epoch[-]')
    plt.ylabel('Loss[-]')
    plt.savefig(Path(validation_folder, "Loss.png"))
    plt.show()

    def scatter(i):
        # predictions
        X_pred = np.c_[np.arange(pred[i].shape[0])[:plot_size], pred[i][:plot_size]]
        sc_pred.set_offsets(X_pred)
        X_target = np.c_[np.arange(targets[i].shape[0])[:plot_size], targets[i][:plot_size]]
        sc_targets.set_offsets(X_target)
        fig.suptitle(f'Epoch: {i}', fontsize=20)
        # manually relim:
        # ax.set_xlim(0, 12)
        # ax.set_ylim(0, 12)

    logger.info("--- Plotting scatter sequence ---")
    fig, ax = plt.subplots()
    ax.grid()
    n_epochs = pred.shape[0]
    # plot_size = None corresponds to all elements
    sc_pred = ax.scatter(np.arange(pred[0].shape[0])[:plot_size], pred[0][:plot_size], label='predictions', s=2)
    sc_targets = ax.scatter(np.arange(targets[0].shape[0])[:plot_size], targets[0][:plot_size], label='targets', s=2)
    plt.legend()
    anim = matplotlib.animation.FuncAnimation(fig, scatter, frames=np.arange(n_epochs), interval=100)
    FFwriter = animation.FFMpegWriter(fps=0.5, extra_args=['-vcodec', 'libx264'])
    anim.save(Path(validation_folder, "predictions.mp4"), writer=FFwriter)

    # import numpy as np
    # from matplotlib import pyplot as plt
    # from matplotlib import animation
    # plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
    #
    # fig = plt.figure()
    # ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
    # line, = ax.plot([], [], lw=2)
    #
    # def init():
    #     line.set_data([], [])
    #     return line,
    #
    # def animate(i):
    #     x = np.linspace(0, 2, 1000)
    #     y = np.sin(2 * np.pi * (x - 0.01 * i))
    #     line.set_data(x, y)
    #     return line,
    #
    # anim = animation.FuncAnimation(fig, animate, init_func=init, frames=200, interval=20, blit=True)
    #
    # FFwriter = animation.FFMpegWriter(fps=30, extra_args=['-vcodec', 'libx264'])
    # anim.save('basic_animation.mp4', writer=FFwriter)


    #
    # plt.subplot(2, 1, 2)
    # plt.plot(train_acc, label='train')
    # plt.plot(valid_acc, label='validation')
    # plt.legend()
    # plt.xlabel('Epoch[-]')
    # plt.ylabel('Accuracy[%]')
    #
    # plt.tight_layout()
    # plt.show()
    # validation_loss
    # # sns_plot = sns.heatmap(preds_df.corr(), annot=True)
    # # plt.savefig("Validation_loss.png")


def validate_control(args, recalculate=True):
    validation_folder = Path(project_folder_path, 'data', 'validation')
    os.makedirs(validation_folder, exist_ok=True)

    model_path = Path(args[1]) if Path(args[1]).exists() else Path(learned_folder, args[1])
    valid_loss_path = Path(args[2]) if Path(args[2]).exists() else Path(validation_folder, args[2])
    predictions_path = Path(args[3]) if Path(args[3]).exists() else Path(validation_folder, args[3])
    targets_path = Path(args[4]) if Path(args[4]).exists() else Path(validation_folder, args[4])

    train_path = Path()
    test_path = Path()
    if len(args) == 7:
        targets_path = Path(args[6], 'train_loss.npy') if Path(args[6], 'train_loss.npy').exists() \
            else Path(learned_folder, args[6], 'train_loss.npy')
        test_path = Path(args[6], 'test_loss.npy') if Path(args[6], 'test_loss.npy').exists() \
            else Path(learned_folder, args[6], 'test_loss.npy')
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
        _, _, valid_daterange = tools.get_split_dataranges(project_folder_path)
        logger.info('----------------------------- Loading validation data ---------------------------')
        logger.info(f"Loading input *validation* data (days {valid_daterange})")
        validation_dataset = tools.get_dataset(valid_daterange, project_folder_path)
        logger.debug('Validation dataset length: ')
        logger.debug(f'len Y: {len(validation_dataset)}')
        logger.info('\n Done')

        valid_loss, predictions, targets = validate(model_path, valid_loss_path, predictions_path, targets_path, validation_dataset, params)

    logger.info('----------------- Plotting ------------------')
    logger.info('Data obtained, plotting.')
    if train_path != Path():
        train_loss = np.load(train_path)
        test_loss = np.load(test_path)
        plot_results(valid_loss_path.parents[0], predictions, targets, valid_loss, train_loss, test_loss)
    else:
        plot_results(valid_loss_path.parents[0], predictions, targets, valid_loss)

    logger.info('--- All done ---')


if __name__ == "__main__":
    logger.add(Path(Path.cwd(), 'data', "run.log"), level="DEBUG")
    logger.add(sys.stderr, level="ERROR")
    try:
        if len(sys.argv) < 6:
            logger.error('Arguments error. Usage:\n')
            logger.error('\tpython3 validate.py model.pt validation_loss.npy(output name) predictions.npy targets.npy'
                         ' recalculate[bool] train_losses_folder \n')
            raise Exception
        torch.multiprocessing.set_start_method('spawn', force=True)

        project_folder_path = Path(__file__).resolve().parent
        learned_folder = Path(project_folder_path, 'data', 'learned')
        os.makedirs(learned_folder, exist_ok=True)
        params = DAQ.load_params(Path(project_folder_path, 'data', 'params.yaml').resolve())
        validate_control(sys.argv, bool(sys.argv[5] == 'True'))
    except Exception as e:
        logger.error(f"Command failed. {e}", exc_info=True)
        raise
