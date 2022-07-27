"""
preprocess.py: Preprocessor implementing data reshape from raw to model input.
    Pairing ga and pixel date, cutting into blocks, and calculating CVR as targets.
"""
# Native libraries
import os
import pickle
import datetime
# Essential Libraries
import numpy as np
import pandas as pd
from loguru import logger
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
# Case related
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
from google.cloud import bigquery
from google.oauth2 import service_account
# Internal package libs
from src import DAQ


def normalize_time(data: pd.DataFrame, domain: str, date_range: str, recalculate: bool = False) -> pd.DataFrame:
    """
    Normalize timedeltas by its standard deviation to correspond with [-1,1] one-hot encoding
    :param data: Dataframe of one hot encoding
    :param domain:
    :param date_range:
    :param recalculate:
    :return: Dataframe with normalized timedeltas of the same shape as data input
    """
    # TODO: replace path.cwd with environ variable 'root'
    folder_path_learned = Path(Path.cwd(), 'data', 'learned').resolve()
    os.makedirs(folder_path_learned, exist_ok=True)
    norm_filename = f'norm_{domain}_{date_range}.csv'
    norm_filepath = folder_path_learned.joinpath(norm_filename).resolve()

    # if normalization already exists (as with inference in the future (function relocation) use), take training normalization file.
    if norm_filepath.exists() and not recalculate:
        norm = DAQ.load_df_from_file(folder_path_learned, norm_filename)
        logger.debug(f'Normalization factors: {norm}')
        data['timediff'] = (data['timediff'] - norm['mean']) / norm['std']
        return data
    factors = np.array([data['timediff'].mean(), data['timediff'].std()])
    norm = pd.DataFrame(factors.reshape(-1, len(factors)), columns=['mean', 'std'])
    norm.reset_index(drop=True, inplace=True)
    logger.debug(f'Normalization factors: {norm}')
    data['timediff'] = (data['timediff'] - norm['mean'].values) / norm['std'].values
    norm.to_csv(norm_filepath)
    return data


def numerize_pixel(raw_pixel_data: pd.DataFrame, country: str, date_range: str, folder_path_preprocessed: Path, params: Dict,
                   e_cutoff: int = 100, recalculate: bool = False) -> pd.DataFrame:
    """
    Reshape pixel events into hybrid onehot of events_id and 'normalized timedeltas'
    :param raw_pixel_data: Dataframe of pixel data, cols [biano_ident, event_type, timestamp]
    :param country: [str] country domain to download data from, e.g. 'cz', 'hu',..
    :param date_range: [str] Start and end of processed data. Format '%Y-%m-%d_%Y-%m-%d'.
    :param folder_path_preprocessed: [Path] Path to folder to save preprocessed data to
    :param params: [Dict] Path to folder to save preprocessed data to
    :param e_cutoff: [int] number of events to cutoff significant enough users
    :param recalculate: [bool] recalculate all parts due to corruption or changed parameters
    :return: Dataframe of preprocessed events
    """

    filename = f'pixel_{country}_{date_range}_preprocessed.csv'
    save_path = folder_path_preprocessed.joinpath(filename).resolve()
    if save_path.exists() and not recalculate:
        return DAQ.load_df_from_file(folder_path_preprocessed, filename)

    logger.info(f'----- Filtering relevant data ------')
    # filter users with enough events to be considered significant
    list_uniques = raw_pixel_data['biano_ident'].value_counts()[lambda x: x > e_cutoff].index.tolist()
    unique_csv = raw_pixel_data.loc[raw_pixel_data['biano_ident'].isin(list_uniques)]
    # convert event type into int , timestamp
    unique_csv['event_type_int'] = unique_csv['event_type'].map(params['page_type_map_to_id'])
    unique_csv['timestamp'] = pd.to_datetime(unique_csv['timestamp'], utc=True)

    logger.info(f'----- Reshaping events into one-hot ------')
    # create one hot array for each event in dictionary, [-1, 1] for better overlap with 'normalized timedeltas'
    arr = np.full((unique_csv.shape[0], len(params['page_type_map_to_id'])), -1, dtype=np.int32)
    # take event_type_int column as column index in the new array for each row
    arr[np.arange(len(arr)), [unique_csv['event_type_int'].values]] = 1
    unique_csv.reset_index(drop=True, inplace=True)
    unique_csv = unique_csv.drop(columns=['event_type_int', 'event_type'])
    unique_csv = pd.concat((unique_csv, pd.DataFrame(arr, columns=params['page_type_map_to_id'].keys())), axis=1)

    logger.info(f'----- Temporal mumbo jumbo ------')
    data_groups = unique_csv.groupby('biano_ident')
    users = []
    for user_id, user_data in tqdm(data_groups):
        # for each significant biano_ident
        user_data = user_data.sort_values(by=['timestamp'])
        user_data = user_data.reset_index(drop=True)
        # calculate conversion into seconds, take timedeltas and convert into seconds
        _norm = (user_data.loc[user_data.shape[0] - 1, 'timestamp'] - user_data.loc[0, 'timestamp']) / np.timedelta64(1, 's')
        user_data['timediff'] = user_data['timestamp'] - user_data['timestamp'].shift()
        if params['preprocess']['full_period_normalization']:
            user_data['timediff'] = user_data['timediff'].apply(lambda x: 0 if x is pd.NaT else x / np.timedelta64(1, 's')).astype(
                'int64') / _norm
        else:
            user_data['timediff'] = user_data['timediff'].apply(lambda x: 0 if x is pd.NaT else x / np.timedelta64(1, 's')).astype(
                'int64') % (24 * 60)
        users.append(user_data)

    dataframes = pd.concat(users)
    dataframes.reset_index(drop=True, inplace=True)

    logger.info(f'----- Normalization ------')
    # normalize timedeltas to its std ~ normalization factor, so the range is approx (-1,1) as the one hot encoding of events
    input_like = normalize_time(dataframes, country, date_range, recalculate)
    logger.debug(f'Saving preprocessed pixel data as: {save_path}')
    input_like.to_csv(save_path)
    logger.debug('Saved')
    return input_like


def intersection(lst1: List, lst2: List) -> Set:
    return set(lst1).intersection(lst2)


def cvr(ga_window: pd.DataFrame, outliers_cutoff: float = 10, normalize: bool = True) -> pd.DataFrame:
    """
    Calculate conversion ratio (cvr) of given window - ratio of number of purchases to number of sessions.
    :param ga_window: [pd.DataFrame] Dataframe of ga data, columns['biano_ident']
    :param outliers_cutoff: [float] Outliers limitation (no point in predicting more, only worsens training). 0 ~ no cutoff
    :param normalize: [bool] Normalize targets as well. Default = True.
    :return:
    """
    counts = ga_window['event_name'].value_counts()
    kauf = 0
    ss = 0
    if counts.get('purchase', 0):
        kauf = counts[counts.index.get_loc('purchase')]
    if counts.get('session_start', 0):
        ss = counts[counts.index.get_loc('session_start')]
    if ss == 0:
        return 0
    targets = np.float32(kauf/ss)
    if outliers_cutoff:
        targets = np.minimum(outliers_cutoff, targets)
    if normalize:
        targets /= outliers_cutoff
    return targets


def cut_ga(ga_data: pd.DataFrame, timestamp_now: pd.Timestamp, window_size: relativedelta) -> pd.DataFrame:
    """
    Cut given size window of ga data from complete ga dataframe to calculate cvr at as target value.
    :param ga_data: [pd.DataFrame] Complete ga dataframe of training data (target)
    :param timestamp_now: [pd.Timestamp] timestamp of the moment to cut from
    :param window_size: [relativedelta object] size of window to cut and calculate target on (forward), e.g. 3 days
    :return: DataFrame of subset of original complete data.
    """
    ga_data['timestamp'] = pd.to_datetime(ga_data['timestamp'], utc=True)
    ga_data = ga_data.sort_values(by=['timestamp'])
    ga_data.reset_index(drop=True, inplace=True)
    logger.debug(f'ga block: {timestamp_now} -> {timestamp_now+window_size}')
    ga_pick_idx = (timestamp_now.tz_localize('utc') <= ga_data['timestamp']) & (ga_data['timestamp'] < timestamp_now.tz_localize('utc')+window_size)
    return ga_data.loc[ga_pick_idx]


def cut_pixel(pixel_data: pd.DataFrame, timestamp_now: pd.Timestamp, window_size: relativedelta) -> pd.DataFrame:
    """
    Cut given size window of pixel data from complete pixel events dataframe.
    :param pixel_data: [pd.DataFrame] Complete pixel dataframe of training data events
    :param timestamp_now: [pd.Timestamp] timestamp of the moment to cut from
    :param window_size: [relativedelta object] size of window to cut and train model on (backward), e.g. 30 days
    :return: DataFrame of subset of original complete data.
    """
    pixel_data['timestamp'] = pd.to_datetime(pixel_data['timestamp'], utc=True)
    pixel_data = pixel_data.sort_values(by=['timestamp'])
    pixel_data.reset_index(drop=True, inplace=True)
    logger.debug(f'pixel block: {timestamp_now} -> {timestamp_now+window_size}')
    pixel_pick_idx = (timestamp_now.tz_localize('utc') >= pixel_data['timestamp']) & (pixel_data['timestamp'] > timestamp_now.tz_localize('utc')+window_size)
    return pixel_data.loc[pixel_pick_idx]


def create_blocks_at_t(pixel_data: pd.DataFrame, ga_data: pd.DataFrame, timestamp: pd.Timestamp,
                       windowB: relativedelta, windowF: relativedelta) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cut datablock for training events and target data at given timestamp.
    :param pixel_data: [pd.DataFrame] Complete pixel dataframe of training data events
    :param ga_data: [pd.DataFrame] Complete ga dataframe of training data (target)
    :param timestamp: [pd.Timestamp] timestamp of the moment to cut from
    :param windowB: [relativedelta object] size of window to cut pixel data and train model on (backward), e.g. 30 days
    :param windowF: [relativedelta object] size of window to cut ga data and calculate target on (forward), e.g. 3 days
    :return: dataframes of pixel window and ga window
    """
    return cut_pixel(pixel_data, timestamp, windowB), cut_ga(ga_data, timestamp, windowF)


def create_data_blocks(pixel_data: pd.DataFrame, ga_data: pd.DataFrame, windowB: relativedelta, windowF: relativedelta,
                       window_step: relativedelta, start: Optional[datetime.datetime] = None) -> Tuple[List, List]:
    """
    Construct datablocks for pixel and ga data by rolling window (window_step)
    :param pixel_data: [pd.DataFrame] Complete pixel dataframe of training data events
    :param ga_data: [pd.DataFrame] Complete ga dataframe of training data (target)
    :param windowB: [relativedelta object] size of window to cut pixel data and train model on (backward), e.g. 30 days
    :param windowF: [relativedelta object] size of window to cut ga data and calculate target on (forward), e.g. 3 days
    :param window_step: [relativedelta object] relativedelta to shift between each block, e.g. 1 day
    :param start: [datetime] timestamp of the moment to start cutting from and which are window_step
    :return: Two lists of Dataframes of given windows. One of pixel data, second of ga data.
    """
    y_list = []
    X_list = []
    k = 0
    if start is None:
        start = pd.to_datetime(ga_data['timestamp'].iloc[0]).replace(tzinfo=None)
    else:
        start = pd.Timestamp(start)
    end = pd.to_datetime(ga_data['timestamp'].iloc[-1]).replace(tzinfo=None)
    if start > end:
        logger.info('Invalid timestamps. Perhaps timestamps are not ordered?')
        raise
    # (k-1) so that the remainder is taken as well (sometimes not all day, can be half day).
    # allows even ga window smaller than window step
    while start + (k-1) * window_step + windowF <= end:
        now = start + k * window_step
        X0, y0 = create_blocks_at_t(pixel_data, ga_data, now, windowB, windowF)
        X_list += [X0]
        y_list += [y0]
        k += 1
    return X_list, y_list


def process_users_date(date: str, pixblock: pd.DataFrame, gablock: pd.DataFrame, folder_path_processed: Path, recalculate: bool = False) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    For given pair pixel-ga datablocks, create:
     pixel: list of arrays [n_users, n_features] for given block
     ga: calculate conversion ratio of the events, i.e. transform each users events into int.
    :param date: [str] String identifier to include in naming of the file.
    :param pixblock: [pd.DataFrame] Dataframe of pixel events to convert into model_input-only data.
    :param gablock: [pd.DataFrame] Dataframe of ga events to compute cvr from
    :param folder_path_processed: [Path] save path for final data, loaded by train.py library
    :param recalculate: [bool] whether data should be recalculated or can be loaded from disk
    :return: Tuple of lists: (list of pixel events for each user, list of ints corresponding to cvr for each user.
    """
    pixel_file = Path(folder_path_processed, f'PixelNumerized_{date}'+'.pkl')
    target_file = Path(folder_path_processed, f'Targets_{date}'+'.pkl')
    users_file = Path(folder_path_processed, f'IDs_{date}' + '.pkl')
    if Path.exists(pixel_file) and Path.exists(target_file) and Path.exists(users_file) and not recalculate:
        logger.info(f'Processed pixel data for day {date} exist at {pixel_file}. \n Skipping calculation, loading instead.')
        with open(pixel_file, 'rb') as f:
            X = pickle.load(f)
        with open(target_file, 'rb') as f:
            y = pickle.load(f)
        with open(users_file, 'rb') as f:
            users = pickle.load(f)
        return X, y, users
    X = []
    y = []
    users = []
    ga_idents = gablock.groupby(['biano_ident'])
    for bianoid, ga in tqdm(ga_idents):
        pixel = pixblock.loc[pixblock['biano_ident'] == bianoid]
        X += [pixel.drop(columns=['timestamp', 'biano_ident']).to_numpy(dtype=np.float32)]
        y += [cvr(ga)]
        users += [bianoid]
    y = np.array(y, dtype=np.float32)
    arr = np.empty(len(X), object)
    arr[:] = X
    with open(pixel_file, 'wb') as f:
        pickle.dump(arr, f, pickle.HIGHEST_PROTOCOL)
    with open(target_file, 'wb') as f:
        pickle.dump(y, f, pickle.HIGHEST_PROTOCOL)
    with open(users_file, 'wb') as f:
        pickle.dump(users, f, pickle.HIGHEST_PROTOCOL)
    return arr, y, users


def process_separate_users(pix_blocks: List, ga_blocks: List, folder_path_project: Path, recalculate: bool = False) -> Tuple[List, List, List]:
    """
    For each pair of pixel-ga blocks, do further processing towards model input data.
    :param pix_blocks: [List] list of dataframes of windowed pixel events
    :param ga_blocks: [List] list of dataframes of windowed ga events
    :param folder_path_project: [Path] Project base folder to form structure in.
    :param recalculate: [bool] whether data should be recalculated or can be loaded from disk
    :return: Tuple of form: (List of np.ndarrays of user events objects (dataframes for each user)
    """
    folder_path_processed = Path(folder_path_project, 'data', 'processed')
    os.makedirs(folder_path_processed, exist_ok=True)
    X = []
    y = []
    users = []
    for iblock, gablock in enumerate(ga_blocks):
        date = str(ga_blocks[iblock]['timestamp'].values[0].astype('datetime64[h]'))
        logger.info(f'Block {iblock+1}/{len(ga_blocks)}, {date}')
        X2, y2, u2 = process_users_date(date, pix_blocks[iblock], gablock, folder_path_processed, recalculate)
        X += [X2]
        y += [y2]
        users += [u2]
    return X, y, users


def preprocess():
    logger.info("--------------- Resolving processed dates ----------------")
    folder_path_project = Path.cwd().resolve()
    params = DAQ.load_params(Path(folder_path_project, 'data', 'params.yaml').resolve())
    credentials = service_account.Credentials.from_service_account_file(params['BQ']['SERVICE_ACCOUNT_FILE'], scopes=params['BQ']['SCOPES'])
    client = bigquery.Client(credentials=credentials, project=credentials.project_id, )

    forward = params['preprocess']['RANGE_FORWARD']
    back = params['preprocess']['RANGE_BACK']
    step = params['preprocess']['RANGE_STEP']
    range_forward = relativedelta(years=forward[0], months=forward[1], days=forward[2])
    range_back = relativedelta(years=back[0], months=back[1], days=back[2])
    range_step = relativedelta(years=step[0], months=step[1], days=step[2])

    # recommended: StartDateStr = "2022-05-02" (First paired biano_idents)
    startdate_datalim_str = params['preprocess']['StartDateStr']
    startdate_datalim = datetime.datetime.strptime(startdate_datalim_str, "%Y-%m-%d")
    # recommended: LastDateStr = "2022-05-11" (changed pairing column in cookies)
    # last day of data, e.g. yesterday
    lastdate_datalim_str = params['preprocess']['LastDateStr']
    lastdate_datalim = datetime.datetime.strptime(lastdate_datalim_str, "%Y-%m-%d")
    # end of datarange so that ga data can fit after it
    # limit days included -> +1
    enddate_datalim = (lastdate_datalim - range_forward + relativedelta(years=0, months=0, days=int(1)))
    enddate_datalim_str = enddate_datalim.strftime('%Y-%m-%d')

    # Log the range
    startdate_pixel = startdate_datalim + range_back
    logger.info(f"Pixel range: {startdate_pixel.strftime('%Y-%m-%d')}->{enddate_datalim.strftime('%Y-%m-%d')}")
    logger.info(f"GA range: {startdate_datalim.strftime('%Y-%m-%d')}->{lastdate_datalim.strftime('%Y-%m-%d')}")
    date_range = [startdate_datalim_str, enddate_datalim_str]
    logger.debug(f'Range of dates defining datablocks: {date_range}')
    forwdiff = abs(lastdate_datalim - startdate_datalim).days + 1   # limit days included -> +1
    logger.debug(f'Number of ga data days: {forwdiff}')
    backdiff = abs(enddate_datalim - startdate_pixel).days + 1   # limit days included -> +1
    logger.debug(f'Number of pixel data days: {backdiff}')

    domain = params['preprocess']['domain']
    overwrite_data = params['preprocess']['overwrite_data']
    overwrite_calc = params['preprocess']['overwrite_calc']
    significant_cutoff = params['preprocess']['significant_cutoff']

    logger.info("--------------- Pixel data acquisition --------------------")
    folder_path_raw_pixel = Path(folder_path_project, 'data', 'raw', 'pixel')
    os.makedirs(folder_path_raw_pixel, exist_ok=True)

    days_data = []
    for day_number in np.arange(backdiff):
        logger.debug(f'-----Day {day_number + 1}/{backdiff}--------')
        date = enddate_datalim + relativedelta(years=0, months=0, days=-int(day_number))
        datestr = date.strftime("%Y-%m-%d")
        dataframe = DAQ.get_data(client, params, datestr, domain, folder_path_raw_pixel, 'pixel', overwrite_data)
        days_data.append(dataframe)
    logger.debug(f'----- Merging... ------')
    pixeldata = pd.concat(days_data)
    pixeldata.reset_index(inplace=True, drop=True)
    logger.debug(f'----- Files merged ------')
    logger.info('Done')

    logger.info("--------------- GA data acquisition --------------------")

    folder_path_raw_ga = Path(folder_path_project, 'data', 'raw', 'ga')
    os.makedirs(folder_path_raw_ga, exist_ok=True)

    days_data = []
    for day_number in np.arange(forwdiff):
        logger.debug(f'-----Day {day_number + 1}/{forwdiff}--------')
        date = startdate_datalim + relativedelta(years=0, months=0, days=int(day_number))
        datestr = date.strftime("%Y%m%d")
        dataframe = DAQ.get_data(client, params, datestr, domain, folder_path_raw_ga, 'ga', overwrite_data)
        days_data.append(dataframe)
    logger.debug(f'----- Merging... ------')
    gadata = pd.concat(days_data)
    gadata.reset_index(inplace=True, drop=True)
    logger.debug(f'----- Files merged ------')
    logger.info('Done')

    logger.info("-------------------- Data processing --------------------")

    logger.debug("------ First pairing cull ------")
    paired_ga = gadata.loc[gadata['biano_ident'].isin(pixeldata['biano_ident'])]
    logger.debug(f'ga_data (all, paired, frac): {gadata.shape[0]} -> {paired_ga.shape[0]} | {paired_ga.shape[0] / gadata.shape[0]}')
    paired_pixel = pixeldata.loc[pixeldata['biano_ident'].isin(paired_ga['biano_ident'])]
    logger.debug(f'pixel_data (all, paired, frac): {pixeldata.shape[0]} -> {paired_pixel.shape[0]} | {paired_pixel.shape[0] / pixeldata.shape[0]}')
    logger.debug(f"ga_data counts: \n {gadata['event_name'].value_counts()}")
    logger.debug(f"paired_ga counts: \n {paired_ga['event_name'].value_counts()}")

    logger.info("----------- Pixel Processing ------------")
    folder_path_preprocessed = Path(folder_path_project, 'data', 'preprocessed')
    os.makedirs(folder_path_preprocessed, exist_ok=True)
    preprocessed_rangestr = f'{date_range[0]}_{date_range[1]}'
    preprocessed = numerize_pixel(paired_pixel, domain, preprocessed_rangestr, folder_path_preprocessed, params, significant_cutoff, overwrite_calc)

    logger.info("------------ GA Processing ------------")
    logger.info("-------- Second pairing cull (after pixel significant cutoff) ---------")
    paired_ga2 = paired_ga.loc[paired_ga['biano_ident'].isin(preprocessed['biano_ident'])]
    paired_ga2.reset_index(drop=True, inplace=True)
    logger.debug(f"ga_data (all, paired): {paired_ga.shape[0]} -> {paired_ga2.shape[0]} | {paired_ga2.shape[0] / paired_ga.shape[0]}")
    logger.debug(f"significant paired_ga counts: {paired_ga2['event_name'].value_counts()}")

    ga2_in_pix = paired_ga2.loc[paired_ga2['biano_ident'].isin(preprocessed['biano_ident'])]
    pix_in_ga2 = preprocessed.loc[preprocessed['biano_ident'].isin(paired_ga2['biano_ident'])]
    logger.info(f"Ids match:  {(paired_ga2.shape == ga2_in_pix.shape) and (preprocessed.shape == pix_in_ga2.shape)}")

    logger.info("-------------------- Forming training dataset --------------------")
    logger.info("--------- Creating window chunks ---------")

    start_time = datetime.datetime.strptime(f"{params['preprocess']['StartDateStr']} {params['preprocess']['hours_offset']}", "%Y-%m-%d %H")
    pix_blocks_tmp, ga_blocks_tmp = create_data_blocks(preprocessed, paired_ga2, windowB=range_back, windowF=range_forward,
                                                       window_step=range_step, start=start_time)
    pix_blocks = []
    ga_blocks = []
    logger.info("--------- Separate into blocks corresponding to each day ---------")
    logger.info("(Includes third cull by pixel significant cutoff per block)")
    for i, pix in enumerate(tqdm(pix_blocks_tmp)):
        logger.debug(f'---block {i}/{len(pix_blocks_tmp)}---')
        ga = ga_blocks_tmp[i]
        list_pix_uniques = pix['biano_ident'].value_counts()[lambda x: x > significant_cutoff].index.tolist()
        list_ga_uniques = ga['biano_ident'].value_counts().index.tolist()
        list_uniques = intersection(list_pix_uniques, list_ga_uniques)
        logger.debug(f'uniques[pix, ga | intersect]: {len(list_pix_uniques)}, {len(list_ga_uniques)} | {len(list_uniques)}')
        pix_blocks += [pix.loc[pix['biano_ident'].isin(list_uniques)]]
        ga_blocks += [ga.loc[ga['biano_ident'].isin(list_uniques)]]
        logger.debug(f'pixel cull: {pix.shape} -> {pix_blocks[-1].shape}')
        logger.debug(f'ga cull: {ga.shape} -> {ga_blocks[-1].shape}')
    logger.info('Done')

    logger.info("\n --------- Transforming chunks into per user dataset ----------")

    X, y, users = process_separate_users(pix_blocks, ga_blocks, folder_path_project, overwrite_calc)

    logger.info('\n Done')


if __name__ == "__main__":
    try:
        preprocess()
    except Exception as e:
        logger.error(f"Command failed. {e}", exc_info=True)
        raise
