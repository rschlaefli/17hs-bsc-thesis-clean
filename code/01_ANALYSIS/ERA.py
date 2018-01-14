import pathlib
import time
import os
import math
import sys
import itertools
import pickle
import operator

import arrow as ar
import numpy as np
import pandas as pd
import xarray as xr

from Dataset import Dataset


class ERA(Dataset):
    @staticmethod
    def prepare_variable(data, timestamp=True):
        """
        Prepare an ERA variable for processing
        """

        df = data.to_dataframe()
        df = df.unstack('time')
        df.columns = df.columns.droplevel(0)

        if timestamp:
            df.columns = df.columns.map(lambda c: int(ar.get(c).datetime.timestamp()))

        return df

    @staticmethod
    def load_year(year, version=None, filter_fun=None, timestamp=True):
        """
        Load a single year of the ERA dataset
        """

        current_path = pathlib.Path(__file__).resolve().parent.parent

        # interim narrow contains 01-03 until 31-05 of each year
        data = xr.open_dataset(
            current_path /
            f'00_DATA/ERA/{year}_interim_narrow{f"_{version}" if version else ""}.nc'
        )

        # resample to daily values
        # TODO: what aggregation should we use here?
        data = data.resample(time='24H').median()

        # split up dataset into temperature and humidity dataframes
        temperature = ERA.prepare_variable(data['t'], timestamp=timestamp)
        humidity = ERA.prepare_variable(data['r'], timestamp=timestamp)

        if filter_fun is not None:
            temperature = filter_fun(temperature, year)
            humidity = filter_fun(humidity, year)

        # return the two dataframes
        return temperature, humidity


    @staticmethod
    def load_dataset(years, version='v3', filter_fun=None, aggregation_resolution=None, invalidate=False, timestamp=True):
        """
        Load the ERA dataset
        """

        temp_dict = dict()
        hum_dict = dict()

        current_path = pathlib.Path(__file__).resolve().parent.parent
        temp_path = current_path / f'00_CACHE/ERA_{str(years)}{"_filtered" if filter_fun is not None else ""}{f"_aggregated{aggregation_resolution}" if aggregation_resolution is not None else ""}{"_no-ts" if timestamp is False else ""}_temp_{version}.pkl'
        hum_path = current_path / f'00_CACHE/ERA_{str(years)}{"_filtered" if filter_fun is not None else ""}{f"_aggregated{aggregation_resolution}" if aggregation_resolution is not None else ""}{"_no-ts" if timestamp is False else ""}_hum_{version}.pkl'

        # if the data has already been processed, use the cache
        if not invalidate and os.path.isfile(temp_path) and os.path.isfile(hum_path):
            print('> Loading from cache...')

            temp = None
            with open(temp_path, 'rb') as file:
                temp = pickle.load(file)

            hum = None
            with open(hum_path, 'rb') as file:
                hum = pickle.load(file)

            return temp, hum

        print('> Processing: ', end='')

        for year in years:
            temperature, humidity = ERA.load_year(year, version=version, filter_fun=filter_fun, timestamp=timestamp)

            if aggregation_resolution is not None:
                temperature = ERA.aggregate_cells(temperature, resolution=aggregation_resolution, timestamp=timestamp, method='mean')
                humidity = ERA.aggregate_cells(humidity, resolution=aggregation_resolution, timestamp=timestamp, method='mean')

            temp_dict[year] = temperature
            hum_dict[year] = humidity

            print(str(year) + ' ', end='')

        pickle.dump(temp_dict, open(temp_path, 'wb'))
        pickle.dump(hum_dict, open(hum_path, 'wb'))

        return temp_dict, hum_dict


    @staticmethod
    def load_year_v2(year, level, variables, version='v5', filter_fun=None, timestamp=True):
        """
        Load a single year of the ERA dataset (v2)
        """
        current_path = pathlib.Path(__file__).resolve().parent.parent
        var_dict = {}

        # interim contains 01.03 until 31.10 of each year
        data = xr.open_dataset(
            current_path /
            f'00_DATA/ERA/{year}_interim_{version}_{level}.nc'
        )

        # resample to daily values (using mean to aggregate)
        data = data.resample(time='24H').mean()

        # split up dataset into variable dataframes
        for variable in variables:
            var_dict[variable] = ERA.prepare_variable(data[variable], timestamp=timestamp)

            if filter_fun is not None:
                var_dict[variable] = filter_fun(var_dict[variable], year)

        # return the variable dict containing all dataframes
        return var_dict


    @staticmethod
    def load_dataset_v2(years, level, variables, version='v5', filter_fun=None, aggregation_resolution=None, invalidate=False, timestamp=True):
        """
        Load the ERA dataset (v2)
        """

        current_path = pathlib.Path(__file__).resolve().parent.parent
        file_paths = {
            variable: f'00_CACHE/ERA_{str(years)}{"_filtered" if filter_fun is not None else ""}{f"_aggregated{aggregation_resolution}" if aggregation_resolution is not None else ""}{"_no-ts" if timestamp is False else ""}_{level}_{variable}_{version}.pkl' for variable in variables
        }

        # initialize a resulting dict
        var_years = {var: {} for var in variables}

        num_cached = 0
        for variable in variables:
            var_path = current_path / file_paths[variable]
            if not invalidate and os.path.isfile(var_path):
                print('> Loading from cache...')

                with open(var_path, 'rb') as handle:
                    var_years[variable] = pickle.load(handle)

                num_cached = num_cached + 1

        if num_cached == len(variables):
            return var_years

        print('> Processing: ', end='')

        for year in years:
            # load dataframes for all variables
            var_dataframes = ERA.load_year_v2(year, level, variables, version=version, filter_fun=filter_fun, timestamp=timestamp)

            for variable in variables:
                # if data should be aggregated, aggregate all variables
                if aggregation_resolution is not None:
                    var_dataframes[variable] = ERA.aggregate_cells(var_dataframes[variable], resolution=aggregation_resolution, timestamp=timestamp, method='mean')

                var_years[variable][year] = var_dataframes[variable]

            print(str(year) + ' ', end='')

        # save all variables to cache files
        for variable in variables:
            with open(current_path / file_paths[variable], 'wb') as handle:
                pickle.dump(var_years[variable], handle)

        return var_years
