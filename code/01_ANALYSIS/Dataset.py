import numpy as np
import itertools

class Dataset:
    @staticmethod
    def aggregate_cells(df, resolution=2.5, timestamp=True, method='median', prevent_invalid=True, verbose=False):
        """
        Aggregate grid cells to a lower resolution to reduce dimensionality

        :param df: Dataframe
        :param resolution: The spatial resolution to aggregate to

        :return: A reduced and aggregated version of the dataframe
        """

        # copy the dataframe as to not mutate it
        df = df.copy()

        # extract grid borders from the multi index
        lat_index = df.index.get_level_values(level=0)
        lon_index = df.index.get_level_values(level=1)

        # print(lat_index.max(), lat_index.min(), lat_index.max() - lat_index.min())
        # print(lon_index.max(), lon_index.min(), lon_index.max() - lon_index.min())

        data_resolution = lon_index[1] - lon_index[0]

        if resolution <= data_resolution:
            return df

        if verbose:
            print('data resolution', data_resolution)

        if prevent_invalid and ((data_resolution + lat_index.max() - lat_index.min()) % resolution > 0 or (data_resolution +  lon_index.max() - lon_index.min()) % resolution > 0):
            raise ValueError('Invalid aggregation resolution.')

        alignment_coefficient = 0.5 * resolution - 0.5 * data_resolution

        latitudes = (lat_index.min(), lat_index.max())
        longitudes = (lon_index.min(), lon_index.max())

        if verbose:
            print('latitudes', latitudes, 'longitudes', longitudes)

        # create ranges with steps in resolution size based on the borders
        # this will allow to aggregate into a grid
        # !! we actually can't subtract anything for the end, as it won't be included in the ranges anyway!!
        lat_range = np.arange(latitudes[0] + alignment_coefficient, latitudes[1], resolution)
        lon_range = np.arange(longitudes[0] + alignment_coefficient, longitudes[1], resolution)

        if verbose:
            print('lat_range', lat_range, 'lon_range', lon_range)

        # initialize a new column to NaN
        df['latitude'] = np.nan
        df['longitude'] = np.nan

        # pass through each permutation of the range lists and annotate with the respective index
        for index, (lat, lon) in enumerate(itertools.product(lat_range, lon_range)):
            slc = (slice(0, lat +  + alignment_coefficient), slice(0, lon  + alignment_coefficient))

            df.loc[slc, 'latitude'] = df.loc[slc, 'latitude'].fillna(lat)
            df.loc[slc, 'longitude'] = df.loc[slc, 'longitude'].fillna(lon)

        # group and aggregate the rows by these grid cell numbers
        if method == 'median':
            df = df.groupby(['latitude', 'longitude']).median()
        elif method == 'mean':
            df = df.groupby(['latitude', 'longitude']).mean()
        elif method == 'sum':
            df = df.groupby(['latitude', 'longitude']).sum()

        if timestamp:
            # convert columns to int representations
            df.columns = df.columns.map(int)

        return df
