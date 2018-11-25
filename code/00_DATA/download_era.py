from ecmwfapi import ECMWFDataServer

server = ECMWFDataServer()

AREA = "40.5/61.5/4.5/97.5"
GRID = "0.75/0.75"
TIME = "00/06/12/18"
VERSION = 'v5'
YEARS = range(1979, 2018)

for year in YEARS:
    DATE_RANGE = f"{year}-03-01/to/{year}-10-31"

    server.retrieve({
        'resol': 'AV',
        'stream': "oper",
        # access data on surface level
        'levtype': "sfc",
        # which params to get (geopotential height)
        'param': "z",
        # use ERA-Interim
        'dataset': "interim",
        # which times to get data for
        'time': '12:00:00',
        # the resolution of the grid
        'grid': GRID,
        # restrict the area of the output
        'area': AREA,
        # specify the date to fetch data for
        'date': DATE_RANGE,
        # type = analysis
        'type': "an",
        # classification era interim
        'class': "ei",
        # save data in NetCDF format
        'format': "netcdf",
        'target': f"{year}_interim_{VERSION}_invariant.nc"
    })

    server.retrieve({
        'resol': 'AV',
        'stream': "oper",
        # access data on surface level
        'levtype': "sfc",
        # which params to get (surface pressure, mean sea level pressure)
        'param': "sp/msl",
        # use ERA-Interim
        'dataset': "interim",
        # which times to get data for
        'time': TIME,
        # the resolution of the grid
        'grid': GRID,
        # restrict the area of the output
        'area': AREA,
        # specify the date to fetch data for
        'date': DATE_RANGE,
        # type = analysis
        'type': "an",
        # classification era interim
        'class': "ei",
        # save data in NetCDF format
        'format': "netcdf",
        'target': f"{year}_interim_{VERSION}_surface.nc"
    })

    server.retrieve({
        'resol': 'AV',
        'stream': "oper",
        # access data by pressure level
        'levtype': "pl",
        # only get data at pressure level 1000
        'levelist': "1000",
        # which params to get (temperature, relative humidity)
        'param': "t/r",
        # use ERA-Interim
        'dataset': "interim",
        # which times to get data for
        'time': TIME,
        # the resolution of the grid
        'grid': GRID,
        # restrict the area of the output
        'area': AREA,
        # specify the date to fetch data for
        'date': DATE_RANGE,
        # type = analysis
        'type': "an",
        # classification era interim
        'class': "ei",
        # save data in NetCDF format
        'format': "netcdf",
        'target': f"{year}_interim_{VERSION}_1000.nc"
    })

    server.retrieve({
        'resol': 'AV',
        'stream': "oper",
        # access data by pressure level
        'levtype': "pl",
        # only get data at pressure level 700
        'levelist': "700",
        # which params to get (u and v components of wind)
        'param': "u/v",
        # use ERA-Interim
        'dataset': "interim",
        # which times to get data for
        'time': TIME,
        # the resolution of the grid
        'grid': GRID,
        # restrict the area of the output
        'area': AREA,
        # specify the date to fetch data for
        'date': DATE_RANGE,
        # type = analysis
        'type': "an",
        # classification era interim
        'class': "ei",
        # save data in NetCDF format
        'format': "netcdf",
        'target': f"{year}_interim_{VERSION}_700.nc"
    })

    server.retrieve({
        'resol': 'AV',
        'stream': "oper",
        # access data by pressure level
        'levtype': "pl",
        # only get data at pressure level 200
        'levelist': "200",
        # which params to get (geopotential / u components of wind)
        'param': "z/u",
        # use ERA-Interim
        'dataset': "interim",
        # which times to get data for
        'time': TIME,
        # the resolution of the grid
        'grid': GRID,
        # restrict the area of the output
        'area': AREA,
        # specify the date to fetch data for
        'date': DATE_RANGE,
        # type = analysis
        'type': "an",
        # classification era interim
        'class': "ei",
        # save data in NetCDF format
        'format': "netcdf",
        'target': f"{year}_interim_{VERSION}_200.nc"
    })
