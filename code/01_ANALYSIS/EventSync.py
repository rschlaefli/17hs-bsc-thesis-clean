import sys
import pathlib
import time
import os
from multiprocessing import Manager, Pool, cpu_count

# statistical libraries
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx

# visualization libraries
import matplotlib.pyplot as plt
from matplotlib import cm

# import own library functionality
import paths
from TRMM import TRMM
from Visualization import Visualization

print(str(sys.argv))

# how many years to load data for
# TODO: use 2017 as soon as december data is fully available
YEARS = range(1998, 2017)

# pre-monsoon period
MAM = [3, 4, 5]

# monsoon period
JJAS = [6, 7, 8, 9]

# post-monsoon period
OND = [10, 11, 12]

# quantiles for extreme events and climate network
# as in PhD: 0.9 for extreme events and 0.95 for network
QUANTILE_EE = 0.9
QUANTILE_GRAPH = 0.95

# which period to execute
PERIODS = [(MAM, 'MAM'), (JJAS, 'JJAS'), (OND, 'OND')]
PERIOD = PERIODS[int(sys.argv[1])]
PERIOD_NAME = PERIOD[1]
PERIOD_RANGE = PERIOD[0]

# run the aggregation algorithm over the dataframes
AGGREGATION_RESOLUTION = float(sys.argv[2]) if len(sys.argv) == 3 else None

# ------ parallel helper functions ------
# as extracted from the TRMM main class

def parallel_calculate_sync(extreme_events, i, j, q):
    """
    Setup an asynchronous worker that performs the sync calculation
    """

    sync_strength, sync_count = TRMM.calculate_sync_strength(extreme_events.iloc[i], extreme_events.iloc[j])

    result = i, j, sync_strength, sync_count
    q.put(result)
    return result

def parallel_log_results(q, df_shape, cache_id=None):
    """
    Setup an asynchronous listener that logs each sync calculation result
    """

    current_path = pathlib.Path(__file__).resolve().parent.parent
    cache_path = current_path / f'00_CACHE/{df_shape}_{cache_id}_sync_v2.txt'
    with open(cache_path, 'a') as handle:
        while 1:
            m = q.get()

            # once the kill message is received, break the loop
            if m == 'kill':
                handle.write('killed\n')
                break

            # append the calculation results to the log
            handle.write(f'{m[0]};{m[1]};{m[2]};{m[3]};{m[4]};{m[5]}\n')
            handle.flush()

# main method to be executed by the process manager
def main():
    # load the dataset
    data = TRMM.load_dataset(
        YEARS,
        PERIOD_RANGE,
        aggregation_resolution=AGGREGATION_RESOLUTION,
        invalidate=False,
        lon_slice=slice(61.125, 97.625),
        lat_slice=slice(4.125, 40.625),
        version='v3')
    print(data.info())
    print(data.iloc[0].quantile(QUANTILE_EE))

    # extract extreme events
    extreme_events = TRMM.extract_extreme_events(data, quantile=QUANTILE_EE)
    print(extreme_events.info())

    print(f'Available CPU count: {cpu_count()}')

    current_path = pathlib.Path(__file__).resolve().parent.parent
    df_shape = extreme_events.shape[0]
    sync_path = current_path / f'00_CACHE/sync_range{df_shape}_{PERIOD_NAME}_v2.pkl'
    count_path = current_path / f'00_CACHE/count_range{df_shape}_{PERIOD_NAME}_v2.pkl'
    split_path = current_path / f'00_CACHE/split_range{df_shape}_{PERIOD_NAME}_v2.pkl'

    # setup process manager, queue and pool
    manager = Manager()
    q = manager.Queue()
    pool = Pool(cpu_count())

    # initialize a logger for the results
    logger = pool.apply_async(parallel_log_results, (q, df_shape, PERIOD_NAME))

    # save the starting time for runtime calculations
    start_time = time.time()

    # setup empty sync and count matrices with the correct shape
    sync_matrix = pd.DataFrame(
        index=extreme_events.index,
        columns=extreme_events.index,
        dtype='float32')
    count_matrix = pd.DataFrame(
        -1,
        index=extreme_events.index,
        columns=extreme_events.index,
        dtype='int16')
    split_matrix = pd.DataFrame(
        -1,
        index=extreme_events.index,
        columns=extreme_events.index,
        dtype='int16')

    # setup the sync calculation jobs
    # for each combination of i and j in one half (diagonal matrix)
    # jobs = []
    initial_i = 0
    restarted = False
    cache_time = 0

    # if the logfile already exists, parse it and look for the last line
    log_path = current_path / f'00_CACHE/{df_shape}_{PERIOD_NAME}_sync_v2.txt'
    if os.path.isfile(log_path):
        with open(log_path, 'r') as handle:
            lst = list(handle.readlines())

            if len(lst) >= 1:
                restarted = True
                print('> Restarting from cache...')

                # parse the items that are already in the cache file
                for item in lst:
                    if item == 'killed\n':
                        continue

                    i, j, sync, count, i_leads, j_leads = item.split(';')

                    i = int(i)
                    j = int(j)
                    sync = float(sync)
                    count = int(count)
                    i_leads = int(i_leads)
                    j_leads = int(j_leads)

                    sync_matrix.iloc[i, j] = sync
                    sync_matrix.iloc[j, i] = sync
                    count_matrix.iloc[i, j] = count
                    count_matrix.iloc[j, i] = count
                    split_matrix.iloc[i, j] = i_leads
                    split_matrix.iloc[j, i] = j_leads

                last_item = lst[len(lst)-2]
                last_processed = last_item.split(';')[0]
                initial_i = int(last_processed) - 1

                cache_time = time.time() - start_time

    # iterator for batching
    def get_batch(length, chunk_size):
        full_array = range(0, length)
        for i in range(0, length, chunk_size):
            # if the chunk fills up the rest of the array, return the entire rest
            if i + chunk_size > len(full_array):
                yield full_array[i:]
            # otherwise return a further chunk
            else:
                yield full_array[i:i + chunk_size]

    for i in range(initial_i, sync_matrix.shape[0]):
        # as the matrix is symmetrical, only calculate the upper half
        for batch in get_batch(i+1, 1024):
            jobs = []

            # run the jobs for the current batch
            for j in batch:
                job = pool.apply_async(TRMM.parallel_calculate_sync, (extreme_events, i, j, q))
                jobs.append(job)

            print(f'>> {i} Started batch {batch} {len(jobs)}')

            # for each job, wait for results and append them to the matrices
            for job in jobs:
                i, j, sync, count, i_leads, j_leads = job.get()

                # update the matrix with the results
                sync_matrix.iloc[i, j] = sync
                sync_matrix.iloc[j, i] = sync
                count_matrix.iloc[i, j] = count
                count_matrix.iloc[j, i] = count
                split_matrix.iloc[i, j] = i_leads
                split_matrix.iloc[j, i] = j_leads

    process_time = time.time() - cache_time
    print('> Finished processing jobs...')

    # save the results to cache
    sync_matrix.to_pickle(sync_path)
    count_matrix.to_pickle(count_path)
    split_matrix.to_pickle(split_path)

    print('> Successfully saved matrices...')

    # calculate the runtime
    end_time = time.time() - start_time
    print(f'\n> Successfully finished in {end_time:f}s (cache: {cache_time:f}, processing: {process_time:f})')

    # make the logger finish the log file and exit the process
    q.put('kill')
    pool.close()


# only the main process will execute this
if __name__ == '__main__':
    main()
