"""DAQ.py: data acquisition for preprocessing and model training (raw and processed)."""

__author__ = "Martin Matusu"

# Native Libraries
import yaml
import os
import datetime
import pickle
import ast
# Essential Libraries
from dotenv import dotenv_values
import numpy as np
import pandas as pd
from loguru import logger
from pathlib import Path
from typing import Dict, Final, Tuple, List
from dateutil.relativedelta import relativedelta


QUERY_PIXEL_EVENTS: Final[str] = Path(Path(__file__).resolve().parent.parent, 'queries', 'query_pixel.sql') \
    .resolve() \
    .read_text(encoding='utf-8') \
    .strip()

QUERY_GA_EVENTS: Final[str] = Path(Path(__file__).resolve().parent.parent, 'queries', 'query_ga.sql') \
    .resolve() \
    .read_text(encoding='utf-8') \
    .strip()


def load_config(*args) -> Dict:
    """
    Loads environment configs one by one - in provided order - and overwrites them by os.environ ones.
    Attempts to parse them so basic structures like dict and list can be parsed.
    :param args: string identifiers of .env files to load, sorted by loading order
    :return: config dictionary of all loaded environments
    """
    config = {}
    for env_name in args:
        if isinstance(env_name, str) and env_name[:4] == '.env':
            config.update(**dotenv_values(env_name))
    config.update(**os.environ) # override loaded values with environment variables
    config = ast.literal_eval(str(config))
    # TODO: string parameters, so far parsing only literals.
    for key in config:
        config[key] = ast.literal_eval(config[key])
    return config


def load_params(params_path: Path) -> Dict:
    """
    Load parameters for the preprocessor
    :param params_path: [Path, .yaml] path to the parameter .yaml file
    """
    return yaml.safe_load(open(params_path))


def get_sql(datatype: str, country: str, date: str, params: Dict) -> str:
    """
    Load appropriate formatted query for BQ client.
    :param datatype: [str] 'pixel' or 'ga'
    :param country: [str] domain for the request
    :param date: [str] day for download
    :param params: [Dict] params file for authentication and proper destination formatting
    :return: [str] Query for the client
    """
    if datatype == 'pixel':
        formatted_query: Final[str] = QUERY_PIXEL_EVENTS.format(country=country, date=date)
        return formatted_query
    elif datatype == 'ga':
        country_id = params['ga_country_id'][country]
        formatted_query: Final[str] = QUERY_GA_EVENTS.format(country_id=country_id, date=date)
        return formatted_query


def load_df_from_file(folder_path: Path, filename: str) -> pd.DataFrame:
    """
    Load pixel data from local .csv file
    :param folder_path: [Path] Path to the folder containing the datafile, e.g. os.getcwd()
    :param filename: [str] filename of the .csv datafile
    """
    save_path = folder_path.joinpath(filename).resolve()
    logger.info(f'File {filename} already exists at location {folder_path}. Reading from storage.')
    df = pd.read_csv(save_path, index_col=0)
    logger.debug(f'Dimensions: {df.shape}')
    return df


def load_data_from_bq(client: object, sql: str, save_path: Path, date: str) -> pd.DataFrame:
    """
    Load pixel data from BQ by given sql query and save it locally
    :param client: [object] BQ client taking care of authentication etc.
    :param sql: [str] sql query to run at bq to get the data
    :param save_path: [Path] resolved Path to save datafile at
    :param date: [str] date string in form "%Y%m%d"
    :return [DataFrame] BQ_data
    """
    logger.info(f'Downloading data from {date}')
    df = client.query(sql).to_dataframe()
    logger.info('Downloaded')
    logger.info(f'Saving file as: {save_path}')
    df.to_csv(save_path)
    logger.info('Saved')
    logger.debug(f'Dimensions: {df.shape}')
    return df


def get_data(client: object, params: Dict, date: str, country: str, folder_path: Path, datatype: str, overwrite: bool = False)\
        -> pd.DataFrame:
    """
    Load pixel data from BQ or local file
    :param client: [object] BQ client taking care of authentication etc.
    :param params: [Path] Path to the file containing parameters, e.g. Path(Path.cwd(), 'params.yaml')
    :param date: [str] date string in form "%Y%m%d"
    :param country: [str] country domain to download data from, e.g. 'cz', 'hu',..
    :param folder_path: [Path] Path to the folder containing the datafile, e.g. Path.cwd()
    :param datatype: [str] 'pixel' or 'ga'
    :param overwrite: [bool] whether to overwrite local datafiles (corruption, sql change,..)
    :return [DataFrame] DataFrame from 'cache' or BQ
    """
    filename = f'{datatype}_{country}_{date}.csv'
    save_path = folder_path.joinpath(filename).resolve()
    if save_path.exists() and not overwrite:
        return load_df_from_file(folder_path, filename)
    else:
        sql = get_sql(datatype, country, date, params)
        return load_data_from_bq(client, sql, save_path, date)

# ----- train.py ------------------


def read_data(pixel_path: Path, target_path: Path, ids_path: Path = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Unpickle pixel data.
    :param pixel_path: [Path] Path to pixel data pickle file
    :param target_path: [Path] Path to pickled target data file
    :param ids_path: [Path] Optional, Path to pickled IDs data file
    :return: Unpickled training and target data
    """
    with open(pixel_path, 'rb') as f:
        X = pickle.load(f)
    with open(target_path, 'rb') as f:
        y = pickle.load(f)
    if ids_path:
        with open(ids_path, 'rb') as f:
            ids = pickle.load(f)
        return X, y, ids
    else:
        return X, y, None


def filter_desired_files(files: List[str], daterange: List[datetime.datetime]) -> pd.DataFrame:
    """

    :param files: [List[str]] Split of filenames including dates. Files of pickled processed data.
    :param daterange: [List] couple of marginal dates to load data from.
    :return: [DataFrame] Filtered filenames for loading (split into df columns).
    """
    split = []
    for file in files:
        split += [file.replace('.', '_').split('_')]
    df = pd.DataFrame(split, columns=['filetype', 'date', 'extension'])
    df['date'] = df['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H'))
    df = df[(daterange[0] <= df['date']) & (daterange[1] >= df['date'])]
    df['date'] = df['date'].apply(lambda x: x.strftime('%Y-%m-%dT%H'))
    return df


def load_data(daterangestr: List[str], processed_folder: Path) -> Tuple[List[np.array], List[np.array], List[str], pd.Series]:
    """
    Load processed data within given range, making blocks of data, each corresponding to one day.
    :param daterangestr: [List[str]] Couple of date strings limiting the range of data to load.
    :param processed_folder: [Path] Path to the folder containing the processed datafiles, e.g. Path(Path.cwd(), 'processed').
    :return: Training data and target data blocks for each processed day. Series of unique dates.
    """
    startdate = datetime.datetime.strptime(daterangestr[0], '%Y-%m-%d')
    # day+1 to include partial days - day without hour is specified as midnight of it's begining.
    enddate = datetime.datetime.strptime(daterangestr[1], '%Y-%m-%d') + relativedelta(years=0, months=0, days=+1, hours=+0)
    daterange = [startdate, enddate]
    X = []
    y = []
    ids = []
    files = [f.name for f in os.scandir(processed_folder) if f.is_file()]
    df = filter_desired_files(files, daterange)
    dates = df['date'].unique()
    for date in dates:
        datefiles = df[df['date'] == date].reset_index(drop=True)
        if datefiles.shape[0] != 3:
            raise Exception("Something's wrong with pixel-target file coupling.")
        target_namebase = 'Targets'
        pixel_namebase = 'PixelNumerized'
        ids_namebase = 'IDs'
        # target_namebase = datefiles.loc[0]['filetype']
        # pixel_namebase = datefiles.loc[1]['filetype']
        # if 'target' in datefiles.loc[1]['filetype'].lower():
        #     target_namebase, pixel_namebase = pixel_namebase, target_namebase
        pixel_filename = Path(processed_folder, pixel_namebase+'_'+datefiles['date'][0]+'.'+datefiles['extension'][0])
        targets_filename = Path(processed_folder, target_namebase+'_'+datefiles['date'][0]+'.'+datefiles['extension'][0])
        ids_filename = Path(processed_folder, ids_namebase + '_' + datefiles['date'][0] + '.' + datefiles['extension'][0])
        logger.debug(f'Loading data for day {date}.')
        Xt, yt, id_block = read_data(pixel_filename, targets_filename, ids_filename)
        X += [Xt]
        y += [yt]
        ids += [id_block]
    match = sum([len(X[i]) for i in np.arange(len(X))]) == sum([len(y[i]) for i in np.arange(len(y))])
    logger.debug(f'Loaded files input counts match: {match} \n')
    return X, y, ids, dates
