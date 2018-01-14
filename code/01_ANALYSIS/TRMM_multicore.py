import os
import sys
import pathlib
import pickle
import pandas as pd
from multiprocessing import Manager, Pool, cpu_count

from TRMM import TRMM

# how many years to load data for
# TODO: use 2017 as soon as december data is fully available
EE_QUANTILE = 0.9
PERIODS = [('MAM', [3, 4, 5]), ('JJAS', [6, 7, 8, 9]), ('OND', [10, 11, 12])]
YEARS = range(1998, 2017)

# the tuning to actually train
INDEX = int(sys.argv[1])
PERIOD = PERIODS[INDEX]

# the aggregation resolution to use
AGGREGATION_RESOLUTION = float(sys.argv[2])

# calculate the current path
current_path = pathlib.Path(__file__).resolve().parent.parent
cache_path = current_path / f'00_CACHE/{AGGREGATION_RESOLUTION}_{PERIOD[0]}_sync.txt'


# setup a worker that performs the sync calculation
def calculate_sync(extreme_events, i, j, q):
    # full calculation run
    # print(f'> Starting {i}, {j}')
    sync_strength, sync_count = TRMM.calculate_sync_strength(extreme_events.iloc[i], extreme_events.iloc[j])
    # print(f'> Finished {i}, {j}')

    result = i, j, sync_strength, sync_count
    q.put(result)
    return result


# setup a listener that logs each result
def log_results(q):
    with open(cache_path, 'a') as handle:
        while 1:
            m = q.get()

            # once the kill message is received, break the loop
            if m == 'kill':
                handle.write('killed')
                break

            # append the calculation results to the log
            handle.write(f'{m[0]};{m[1]};{m[2]};{m[3]};\n')
            handle.flush()


def main():
    # load the dataset
    trmm = TRMM.load_dataset(
        YEARS,
        PERIOD[1],
        aggregation_resolution=AGGREGATION_RESOLUTION,
        invalidate=False,
        lon_slice=slice(61.125, 97.625),
        lat_slice=slice(4.125, 40.625),
        version='v3')

    print('> Loaded TRMM:', trmm.shape)

    # extract extreme events
    extreme_events = TRMM.extract_extreme_events(trmm, quantile=EE_QUANTILE)

    # setup process manager, queue and pool
    manager = Manager()
    q = manager.Queue()
    pool = Pool(cpu_count() * 2)

    # initialize a logger for the results
    logger = pool.apply_async(log_results, (q, ))

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

    # setup the sync calculation jobs
    # for each combination of i and j in one half (diagonal matrix)
    jobs = []
    initial_i = 0

    # check if the job had already been started
    # restart from the latest processed index
    if os.path.isfile(cache_path):


    for i in range(initial_i, sync_matrix.shape[0]):
        # as the matrix is symmetrical, only calculate the upper half
        for j in range(i + 1):
            job = pool.apply_async(calculate_sync, (extreme_events, i, j, q))
            jobs.append(job)

    # for each job, wait for results and append them to the matrices
    for job in jobs:
        i, j, sync, count = job.get()

        # update the matrix with the results
        sync_matrix.iloc[i, j] = sync
        sync_matrix.iloc[j, i] = sync
        count_matrix.iloc[i, j] = count
        count_matrix.iloc[j, i] = count

    # save the results to cache
    sync_path = current_path / f'00_CACHE/sync_range{str(extreme_events.shape[0])}_{PERIOD[0]}.pkl'
    count_path = current_path / f'00_CACHE/count_range{str(extreme_events.shape[0])}_{PERIOD[0]}.pkl'
    sync_matrix.to_pickle(sync_path)
    count_matrix.to_pickle(count_path)

    # make the logger finish the log file and exit the process
    q.put('kill')
    pool.close()


# only the main process will execute this
if __name__ == '__main__':
    main()
