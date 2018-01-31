import pathlib
import time
import os
import math
import sys
import itertools
import pickle
import operator
import arrow as ar
from multiprocessing import Manager, Pool, cpu_count

import numpy as np
import pandas as pd
import xarray as xr
import networkx as nx

from Dataset import Dataset


class TRMM(Dataset):
    @staticmethod
    def load_dataset_month(date_template, timestamp=True, version=None):
        """
        Load a single month of data

        :date_template: Template with year and month for date formatting (e.g., "2017-01-{}")
        :timestamp: Whether to convert dates to epoch timestamps:
        :version: The version of the TRMM dataset to load

        :return: Dataframe
        """

        df = pd.DataFrame()

        # go through 31 possible days (also for shorter months)
        # this allows us to skip dealing with different number of days
        for i in range(31):

            # format days to be 8,9,10 => 08, 09, 10...
            day = f'{i+1:02}'

            # create a date string with the given day
            date = date_template.format(day)

            # try fetching the respective dataset
            # just skip it if it doesn't exist (e.g. 31.02.xxxx)
            try:
                # try to open the corresponding data file
                current_path = pathlib.Path(__file__).resolve().parent.parent
                filename = f'00_DATA/TRMM/3B42_Daily.{date.replace("-", "")}{f"_{version}.trmm" if version else ""}'
                dataset = xr.open_dataset(current_path / filename)

                if timestamp:
                    # save the data at the timestamp
                    timestamp = ar.get(date, 'YYYY-MM-DD').datetime.timestamp()
                    df[timestamp] = dataset.to_dataframe()['precipitation']
                else:
                    date = ar.get(date, 'YYYY-MM-DD').datetime
                    df[date] = dataset.to_dataframe()['precipitation']

            except (FileNotFoundError, OSError) as e:
                # simply skip and log if a day is not available
                print(f'Failure for {date} at {filename}')
                pass

        return df

    @staticmethod
    def load_dataset(years,
                    months,
                    invalidate=False,
                    filter_fun=None,
                    aggregation_resolution=2.5,
                    bundled=True,
                    fill_na='zero',
                    timestamp=True,
                    version=None,
                    lat_slice=None,
                    lon_slice=None,
                    default_slice=False):
        """
        Load the entire TRMM dataset

        :years: A list of years to include
        :months: A list of months to include for each year
        :invalidate: Whether the cache should be invalidated
        :filter_fun: An optional filtering function to be applied after each imported year
        :aggregation_resolution: The resolution of the aggregation to be applied
        :bundled: Whether the data should be bundled into a single dataframe or into a dictionary
        :fill_na: Whether to fill NaN values with zero or column mean values ('zero' | 'mean' | None)
        :lat_slice: Optional slice of latitudes to be included in the dataframe
        :lon_slice: Optional slice of longitudes to be included in the dataframe

        :return: Dataframe
        """

        current_path = pathlib.Path(__file__).resolve().parent.parent
        pickle_path = current_path / f'00_CACHE/{str(years) + str(months)}{"_filtered" if filter_fun is not None else ""}{f"_aggregated{aggregation_resolution}" if aggregation_resolution is not None else ""}{"_bundled" if bundled else ""}{"_no-ts" if timestamp is False else ""}{f"_{lat_slice}-{lon_slice}" if lat_slice is not None and lon_slice is not None else ""}{f"_{version}" if version else ""}.pkl'

        # if the data has already been processed, use the cache
        if not invalidate and os.path.isfile(pickle_path):
            print('> Loading from cache...')

            if bundled:
                return pd.read_pickle(pickle_path)

            with open(pickle_path, 'rb') as file:
                return pickle.load(file)

        print('> Processing: ', end='')

        result = {}

        # pass through each year
        for year in years:

            df = None

            # pass through each month
            for month in months:

                # load a single month of data
                dataset = TRMM.load_dataset_month(f'{year}-{month:02}-{{}}', timestamp=timestamp, version=version)

                # combine it with the existing dataframe
                if df is None:
                    df = dataset
                else:
                    df = df.join(dataset)

            # restrict the geographic area to the slices provided
            if lat_slice is not None and lon_slice is not None:
                df = df.loc[(lat_slice, lon_slice), :]

            if default_slice:
                df = df.loc[(slice(4.125, 40.625), slice(61.125, 97.625)), :]

            # if the data is to be aggregated, run the aggregation algorithm
            if aggregation_resolution is not None:
                df = TRMM.aggregate_cells(df, aggregation_resolution, timestamp=timestamp, method='sum', prevent_invalid=False if version is None else True)

            # if a custom filter function has been passed, apply it to the dataframe
            if filter_fun is not None:
                df = filter_fun(df, year)

            # fill NaN values with the appropriate values
            if fill_na == 'mean':
                df = df.fillna(df.mean())
            elif fill_na == 'zero':
                df = df.fillna(0)

            result[year] = df

            print(str(year) + ' ', end='')

        # if the data should be bundled into a single dataframe, perform a second pass
        # otherwise, a dictionary with yearly dataframes is returned
        if bundled:
            bundle = None

            for year in years:
                if bundle is None:
                    bundle = result[year]
                else:
                    bundle = bundle.join(result[year])

            bundle.to_pickle(pickle_path)

            return bundle

        pickle.dump(result, open(pickle_path, 'wb'))

        return result

    @staticmethod
    def extract_extreme_events(df, quantile=0.9, pad=True):
        """
        Extract events above a certain quantile (extreme)

        :df: Dataframe with numerical data
        :quantile: Quantile/Threshold for extreme events
        :pad: Whether to pad the sequence with 0 and 999999999 extreme events (such that the sliding window also incorporates i=0 and i=m)

        :return: Dataframe containing only True/False for Extreme/Non-Extreme
        """
        extracted = df.apply(lambda row: row > row.quantile(quantile), axis=1)

        # pad the extracted extreme event sequence if so requested
        if pad:
            extracted.insert(0, 0, True)
            extracted.insert(len(extracted.columns), 999999999999, True)

        return extracted

    @staticmethod
    def sliding_window(iterable, window_size=3, padded=False):
        """
        Generate an iterator that serves as a sliding window of given size

        :iterable: The row to iterate over
        :window_size: The size of the sliding window
        :padded: Whether to apply padding at this stage instead of at extreme event extraction stage

        :return: Iterator
        """

        # get an iterator from the iterable (df row)
        i = iter(iterable.index)

        # prepare an empty array for the window
        win = [0] if padded else []

        # fill the window with prev, current and next elements
        for e in range(0, window_size - (1 if padded else 0)):
            win.append(next(i))

        # yield the window
        yield win

        # for all elements in the iterator
        for e in i:
            # keep the last two elements in the window
            # append the next element in the iterator
            win = win[1:] + [e]

            # return a new window
            yield win

        if padded:
            yield win[1:] + [999999999999]

    @staticmethod
    def calculate_synchronization(row1, row2):
        """
        Calculate the number of synchronous events between two rows where row1 leads row2

        :row1: First row
        :row2: Second row

        :return: The number of synchronous events (integer)
        """

        # initialize the total number of synchronous events for the two grid points
        num_sync_events = 0

        # iterate over all windows of size three in the first row
        for i_prev, i_current, i_next in TRMM.sliding_window(row1, window_size=3):
            # calculate the last timestamp of row2 that could be synchronous
            latest = i_current + 0.5 * min(i_current - i_prev, i_next - i_current)

            # pass through all events at the same or at a later time at the second location
            for j_prev, j_current, j_next in TRMM.sliding_window(row2, window_size=3):

                # calculate the difference
                current_diff = j_current - i_current

                # if the difference gets negative, continue
                # the second pass will encompass these combinations
                if current_diff < 0:
                    continue

                # if the events occur at the same time, they will be counted twice
                # thus only add half the value
                if current_diff == 0:
                    num_sync_events += 0.5
                    continue

                # break for timestamps that cannot possibly be synchronous
                # i.e. are much too late
                if j_current > latest:
                    break

                # calculate the time lag based on the current state of the two sliding windows
                time_lag = 0.5 * min(
                    i_next - i_current, i_current - i_prev,
                    j_next - j_current, j_current - j_prev)

                # if the second event lies within the time lag, it is fully synchronous
                if 0 < current_diff <= time_lag:
                    num_sync_events += 1.0

        return num_sync_events

    @staticmethod
    def calculate_sync_strength(row1, row2, v2=False):
        """
        Calculate the synchronicity coefficient between two rows

        :row1: First row
        :row2: Second row
        :v2: Whether returns should adhere to the v2 format (allowing split matrices)

        :return: The strength of synchronicity
        :return: The number of total synchronous events
        :return: The numbers of synchronous events for c(i|j) and c(j|i)
        """

        # extract only the timestamps that are actually extreme events
        row1_events = row1[row1]
        row2_events = row2[row2]

        # calculate synchronization between row1 and row2 and reverse
        row1_sync = TRMM.calculate_synchronization(row1_events, row2_events)
        row2_sync = TRMM.calculate_synchronization(row2_events, row1_events)

        # calculate the strength of synchronization according to the formula
        sync_strength = (row1_sync + row2_sync) / \
            math.sqrt((len(row1_events) - 2) * (len(row2_events) - 2))

        if v2:
            return sync_strength, row1_sync + row2_sync, row1_sync, row2_sync

        return sync_strength, row1_sync + row2_sync


    @staticmethod
    def calculate_sync_matrix(df, cache_id, invalidate=False, v2=False):
        """
        Calculate the synchronization matrix for all permutations of grid cells

        :df: Dataframe containing extreme events
        :cache_id: A unique identifier to use in the cache path
        :invalidate: Whether the cache should be invalidated
        :v2: Whether returns should adhere to the v2 format (return a split matrix)

        :return: Dataframe with synchronization coefficient
        :return: Dataframe with synchronous event count
        :return: Dataframe with an asymmetrical split matrix
        :return: Runtime for calculations
        """
        current_path = pathlib.Path(__file__).resolve().parent.parent
        sync_path = current_path / f'00_CACHE/sync_range{str(df.shape[0])}_{cache_id}{"_v2" if v2 else ""}.pkl'
        count_path = current_path / f'00_CACHE/count_range{str(df.shape[0])}_{cache_id}{"_v2" if v2 else ""}.pkl'
        split_path = current_path / f'00_CACHE/split_range{str(df.shape[0])}_{cache_id}{"_v2" if v2 else ""}.pkl'

        # if the data has already been processed, use the cache
        if not invalidate and os.path.isfile(sync_path) and os.path.isfile(count_path):
            print('> Loading from cache...')

            if v2:
                return pd.read_pickle(sync_path), pd.read_pickle(count_path), pd.read_pickle(split_path), 0

            return pd.read_pickle(sync_path), pd.read_pickle(count_path), 0

        else:
            # save the starting time for runtime calculations
            start_time = time.time()

            print('> Processing...')

            # initialize an empty matrix with rows and columns for the grid points
            sync_matrix = pd.DataFrame(index=df.index, columns=df.index, dtype='float32')
            count_matrix = pd.DataFrame(-1, index=df.index, columns=df.index, dtype='int16')
            split_matrix = pd.DataFrame(-1, index=df.index, columns=df.index, dtype='int16')

            # calculate the sync strength for each permutation of grid cells
            for i in range(sync_matrix.shape[0]):
                sys.stdout.write(f'\r>> {i}/{sync_matrix.shape[0]}')

                # as the matrix is symmetrical, only calculate the upper half
                for j in range(i + 1):
                    # calculate the synchronicity for the permutation of rows
                    sync_strength, count, i_leads, j_leads = TRMM.calculate_sync_strength(df.iloc[i], df.iloc[j], v2=True)

                    # save results in the respective matrices
                    sync_matrix.iloc[i, j] = sync_strength
                    sync_matrix.iloc[j, i] = sync_strength
                    count_matrix.iloc[i, j] = count
                    count_matrix.iloc[j, i] = count
                    split_matrix.iloc[i, j] = i_leads
                    split_matrix.iloc[j, i] = j_leads

            # save the results to cache
            sync_matrix.to_pickle(sync_path)
            count_matrix.to_pickle(count_path)
            split_matrix.to_pickle(split_path)

            # calculate the runtime
            end_time = time.time() - start_time
            print(f'\n> Successfully finished in {end_time:f}s')

            if v2:
                return sync_matrix, count_matrix, split_matrix, end_time

            return sync_matrix, count_matrix, end_time

    @staticmethod
    def generate_graph(df, quantile=0.95, set_lte=0, set_ge=1, directed=False):
        """
        Generate a graph/network from a sync/count matrix

        :df: The calculated matrix with event synchronization counts
        :quantile: The quantile to use for extraction of relevant values
        :set_geq: What value to set elements above the quantile to
        :set_lt: What value to set elements below the quantile to
        :directed: Whether the generated graph should be directed

        :return: A networkx graph representation of the adjacency matrix
        """

        # copy the df to not mutate any references
        adjacency_matrix = df.copy()

        # replace all existing ones with zeroes
        # this removes the diagonal matrix (recursive edges)
        # which should lead to a more significant quantile
        np.fill_diagonal(adjacency_matrix.values, np.nan)

        # apply the quantile twice to get a single value (not per-column)
        quantile = np.nanpercentile(adjacency_matrix.values, int(100 * quantile))

        # set all values above and below the quantile to predefined values
        ge = adjacency_matrix > quantile
        lte = adjacency_matrix <= quantile

        # if set_ge is None, the sync coefficient should be used for weighting of edges
        # otherwise, set it to the value specified (no weighting of edges)
        if set_ge is not None:
            adjacency_matrix[ge] = set_ge

        adjacency_matrix[lte] = set_lte

        # readd recursive edges in the adjacency matrix
        np.fill_diagonal(adjacency_matrix.values, 0)

        # generate a networkx graph from the adjacency matrix
        graph = nx.from_numpy_matrix(adjacency_matrix.values, create_using=nx.DiGraph() if directed else None)

        # add coordinates to each graph node
        for index in graph.nodes():
            graph.nodes[index]['coordinates'] = df.index[index]

        return graph

    @staticmethod
    def calculate_centrality(graph, weighted=False):
        """
        Calculate centrality measures for a given graph

        :graph: The full climate network representation
        :weighted: Whether weighted centrality measures should be applied

        :return: Dataframes for each centrality measure
        """

        # calculate degree, betweenness and pagerank using networkx
        # cent_degree = nx.degree_centrality(graph)
        cent_degree = graph.degree(weight='weight' if weighted else None)
        cent_between = nx.betweenness_centrality(graph, normalized=False, weight='weight' if weighted else None)
        pagerank = nx.pagerank_numpy(graph, weight='weight' if weighted else None)

        # print('cent_degree', cent_degree)

        # convert each measure to a dataframe and return them
        return TRMM.measure_to_df(graph, list(cent_degree)), \
            TRMM.measure_to_df(graph, list(cent_between.items())),  \
            TRMM.measure_to_df(graph, list(pagerank.items()))

    @staticmethod
    def measure_to_df(graph, nodes):
        """
        Convert calculated centrality measures to a clean and extended dataframe for further use

        :graph: The full climate network graph
        :nodes: A list with centrality values for each node

        :return: An extended dataframe with converted data
        """

        # prepare empty "dataframe columns"
        measure_lat = []
        measure_lon = []
        measure_val = []
        measure_text = []

        # for each node, extract relevant data and append to the respective dataframe column
        # print('nodes', nodes)
        for node in nodes:
            measure_lat.append(graph.nodes[node[0]]['coordinates'][0])
            measure_lon.append(graph.nodes[node[0]]['coordinates'][1])
            measure_val.append(node[1])
            measure_text.append(
                f'{graph.nodes[node[0]]["coordinates"]}={node[1]:.4f}')

        # create a dataframe from the calculated arrays
        result = pd.DataFrame(
            dict(
                lat=measure_lat,
                lon=measure_lon,
                val=measure_val,
                text=measure_text),
            columns=['lat', 'lon', 'val', 'text'])

        # standardize the values by scaling the min to be 0 and the max to be 1
        # this will allow to exactly define size of markers on the map
        result['val_0to1'] = ((result['val'] - result['val'].min()) /
                            (result['val'].max() - result['val'].min()))
        result['val_std'] = (result['val'] - result['val'].mean()) / result['val'].std()

        return result

    @staticmethod
    def parallel_calculate_sync(extreme_events, i, j, q):
        """
        LEGACY: see EventSync.py

        Setup an asynchronous worker that performs the sync calculation
        """

        sync_strength, sync_count, i_leads, j_leads = TRMM.calculate_sync_strength(extreme_events.iloc[i], extreme_events.iloc[j], v2=True)

        result = i, j, sync_strength, sync_count, i_leads, j_leads
        q.put(result)
        return result


    @staticmethod
    def parallel_log_results(q, df_shape, cache_id=None):
        """
        LEGACY: see EventSync.py

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

    @staticmethod
    def parallel_calculate_sync_matrix(extreme_events, cache_id, invalidate=False, hard_cpu_count=None):
        """
        LEGACY: see EventSync.py

        Parallelized version of the sync matrix calculation
        """

        current_path = pathlib.Path(__file__).resolve().parent.parent
        df_shape = extreme_events.shape[0]
        sync_path = current_path / f'00_CACHE/sync_range{df_shape}_{cache_id}_v2.pkl'
        count_path = current_path / f'00_CACHE/count_range{df_shape}_{cache_id}_v2.pkl'
        split_path = current_path / f'00_CACHE/split_range{df_shape}_{cache_id}_v2.pkl'
        # if the data has already been processed, use the cache
        if not invalidate and os.path.isfile(sync_path) and os.path.isfile(count_path) and os.path.isfile(split_path):
            print('> Loading from cache...')
            return pd.read_pickle(sync_path), pd.read_pickle(count_path), pd.read_pickle(split_path), 0

        # setup process manager, queue and pool
        manager = Manager()
        q = manager.Queue()
        pool = Pool(hard_cpu_count if hard_cpu_count is not None else cpu_count())

        # initialize a logger for the results
        logger = pool.apply_async(TRMM.parallel_log_results, (q, df_shape, cache_id))

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
        jobs = []
        initial_i = 0
        restarted = False
        cache_time = 0

        # if the logfile already exists, parse it and look for the last line
        log_path = current_path / f'00_CACHE/{df_shape}_{cache_id}_sync_v2.txt'
        if not invalidate and os.path.isfile(log_path):
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

        batch_len = 0
        for i in range(initial_i, sync_matrix.shape[0]):
            # as the matrix is symmetrical, only calculate the upper half
            for j in range(i + 1):
                job = pool.apply_async(TRMM.parallel_calculate_sync, (extreme_events, i, j, q))
                jobs.append(job)
                batch_len += 1

        setup_time = time.time() - cache_time

        print('> Started all jobs...')

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

        process_time = time.time() - setup_time

        print('> Finished processing jobs...')

        # save the results to cache
        sync_matrix.to_pickle(sync_path)
        count_matrix.to_pickle(count_path)
        split_matrix.to_pickle(split_path)

        print('> Successfully saved matrices...')

        # calculate the runtime
        end_time = time.time() - start_time
        print(f'\n> Successfully finished in {end_time:f}s (cache: {cache_time:f}, setup: {setup_time:f}, processing: {process_time:f})')

        # make the logger finish the log file and exit the process
        q.put('kill')
        pool.close()

        return sync_matrix, count_matrix, split_matrix, end_time
