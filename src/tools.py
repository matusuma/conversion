# Native Libraries
import datetime
import os
# Essential Libraries
import numpy as np
from pathlib import Path
from loguru import logger
from typing import List, Tuple
from dateutil.relativedelta import relativedelta
from prettytable import PrettyTable
# AI-related
import torch
from torch.nn.utils.rnn import pack_sequence
# Internal package libs
from src import DAQ
from src import custom_dataset


def get_split_dataranges(project_folder_path: Path) -> Tuple[List, List, List]:
    """
    Get ranges of dates to take training, test and validation datasets from.
    :param [Path] project_folder_path: [Path] root directory of the project
    :return: [Lists] Lists (by given splits - train, test, valid.) of couples of dates
        interpreted as strings %Y-%m-%d, e.g. ['2022-05-02', '2022-05-07']
    """
    params = DAQ.load_params(Path(project_folder_path, 'data', 'params.yaml').resolve())
    startdate_datalim_str = params['prepare_training']['StartDateStr']
    startdate_datalim = datetime.datetime.strptime(startdate_datalim_str, "%Y-%m-%d")
    # recommended: LastDateStr = "2022-05-11" (changed pairing column in cookies)
    # last day of data, e.g. yesterday
    lastdate_datalim_str = params['prepare_training']['LastDateStr']
    lastdate_datalim = datetime.datetime.strptime(lastdate_datalim_str, "%Y-%m-%d")

    alldays = (lastdate_datalim - startdate_datalim).days
    test_days = np.round(alldays * params['prepare_training']['split'][0])
    valid_days = np.round(alldays * params['prepare_training']['split'][1])
    train_days = alldays - test_days - valid_days
    # check ratio and nonzero sets
    if train_days < valid_days + test_days:
        raise Exception("Invalid training data split - training ratio needs to be at least 0.5")
    if test_days == 0 or valid_days == 0:
        raise Exception("Resolve ratios/training dataset. Test or validate set is now empty.")
    train_enddate = startdate_datalim + relativedelta(years=0, months=0, days=train_days - 1, hours=+0)
    # relative delta days: -1 to include the startdate, +1 for following the train_enddate -> test_days only
    test_enddate = train_enddate + relativedelta(years=0, months=0, days=test_days, hours=+0)
    valid_enddate = test_enddate + relativedelta(years=0, months=0, days=valid_days, hours=+0)

    train_daterange = [startdate_datalim.strftime('%Y-%m-%d'), train_enddate.strftime('%Y-%m-%d')]
    test_daterange = [(train_enddate + relativedelta(days=1)).strftime('%Y-%m-%d'), test_enddate.strftime('%Y-%m-%d')]
    validate_daterange = [(test_enddate + relativedelta(days=1)).strftime('%Y-%m-%d'), valid_enddate.strftime('%Y-%m-%d')]
    return train_daterange, test_daterange, validate_daterange


def get_dataset(daterange, project_folder_path):
    processed_folder = Path(project_folder_path, 'data', 'processed')
    X, y, _, dates = DAQ.load_data(daterange, processed_folder)
    return custom_dataset.CompleteDataset(X, y, dates)


def resolve_device(enforce_cpu: bool = False) -> Tuple[torch.device, bool]:
    """
    Check if device has cuda available, prefer using cuda and eventually print out its parameters.
    :param enforce_cpu: [bool] enforce cpu even with cuda available
    :return: [torch.device, bool] cuda device object or string 'cpu', bool if parallelizable
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    if enforce_cpu:
        device = torch.device('cpu')

    logger.info(f'Using device: {device}')

    # Additional Info when using cuda
    if device.type == 'cuda':
        logger.debug(f'Device count: {torch.cuda.device_count()}')
        logger.debug(torch.cuda.get_device_name(0))
        logger.debug('Memory Usage:')
        logger.debug(f'Allocated: {round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)} GB')
        logger.debug(f'Cached:    {round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)} GB')

    return device, torch.cuda.device_count() > 1


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    logger.debug(table)
    logger.debug(f"Total Trainable Params: {total_params}")
    return total_params


def collate_fn(data: List) -> Tuple:
    """
    Override of default collate function for DataLoader. Necessary to utilize packed sequence.
    :param data: [List] is a list of tuples with (example, label)
        where 'example' is a tensor of (:, features) shape and label is a scalar
    :return: packed sequence of the minibatch, tensor of targets
    """
    events, labels = zip(*data)
    if torch.cuda.is_available():
        events = tuple(block.to(torch.device('cuda')) for block in events)
        features = pack_sequence(events, enforce_sorted=False)  # events is a tupple of ndarrays. Pack requests list of tensors.
        labels = torch.as_tensor(labels, device=torch.device('cuda'))
        sorted_labels = labels[features.sorted_indices]
    else:
        features = pack_sequence(events, enforce_sorted=False)
        labels = torch.as_tensor(labels)
        sorted_labels = labels[features.sorted_indices]
    return features, sorted_labels.view(-1, 1)
