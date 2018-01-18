import numpy as np
import pandas as pd
import arrow as ar
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import scipy.ndimage
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

class Visualization:
    """ Visualization components """

    @staticmethod
    def plot_losses(results):
        """ Plot model losses against validation losses over epochs of training """

        for result in results:
            result['loss'].plot(figsize=(15, 6))
            result['val_loss'].plot(figsize=(15, 6))

    @staticmethod
    def create_plot_nodes(data, color, name, size_multi=30, opacity_multi=0.7):
        """
        Calculate nodes for the climate network visualization (and their corresponding graphical representation)

        :data: The dataframe with coordinate and value / standardized value columns
        :color: The color of the trace items
        :name: The name of the trace (for the legend)
        :size_multi: The multiplicator in pixels for the size value
        :opacity_multi: The multiplicator for the opacity value

        :return: A plotly dictionary representing the trace for all nodes
        """

        return dict(
            type='scattergeo',
            lon=data['lon'],
            lat=data['lat'],
            hoverinfo='text',
            text=data['text'],
            mode='markers',
            showlegend=True,
            name=name,
            marker=dict(
                size=data['val_std'] * size_multi,
                opacity=data['val_std'] * opacity_multi,
                color=color,
                line=dict(width=3, color=color)),
        )

    @staticmethod
    def create_plot_edges(graph):
        """
        Calculate edges that connect the climate network's nodes (grid cells)

        :graph: The full climate network graph

        :return: An array of plotly edges (each a separate dictionary)
        """

        edges = graph.edges()
        result = []

        # create a line trace for each edge
        for edge in edges:
            # only create a trace if it is not an edge from the node to itself
            if edge[0] != edge[1]:
                from_node = graph.node[edge[0]]
                to_node = graph.node[edge[1]]

                result.append(
                    dict(
                        type='scattergeo',
                        lon=[
                            from_node['coordinates'][1],
                            to_node['coordinates'][1]
                        ],
                        lat=[
                            from_node['coordinates'][0],
                            to_node['coordinates'][0]
                        ],
                        mode='lines',
                        name='Edge',
                        showlegend=False,
                        line=dict(
                            width=1,
                            color='red',
                        ),
                        opacity=0.5,
                    ))

        return result

    @staticmethod
    def create_plot_layout(title='Centrality Measures (Degree, Betweenness, PageRank)'):
        """
        Create a plotly layout with basic settings for a map

        :title: The title of the plotly graph

        :return: A plotly layout dictionary
        """

        return dict(
            title=title,
            showlegend=True,
            width=1000,
            height=1000,
            legend=dict(traceorder='grouped'),
            geo=dict(
                scope='asia',
                projection=dict(type='mercator'),
                showland=True,
                landcolor='rgb(243, 243, 243)',
                lonaxis=dict(range=[62.375, 97.375], ),
                lataxis=dict(range=[5.125, 40.125], ),
            )
        )

    @staticmethod
    def create_cartopy_vis(df, ax=None, filename=None, title=None, cmap='afmhot', clabel=None, vis_type='mesh', no_cbar=False, log_norm=False, interpolation=None, gaussian_filtering=None, values_from='val', index_step=20):
        """
        Create a cartopy/matplotlib visualization from a passed in coordinate grid dataframe.
        The axes are expected to be named as "lat" and "lon" such that the df can be pivoted appropriately.
        """

        # setup a projection
        if ax is None:
            # setup a figure
            fig = plt.figure(figsize=(10, 10))
            ax = plt.axes(projection=ccrs.PlateCarree())

        # add natural features
        ax.add_feature(cfeature.BORDERS)
        ax.coastlines(resolution='50m')

        if vis_type == 'barbs':
            u = df['u'].pivot(index='lat', columns='lon', values=values_from)
            v = df['v'].pivot(index='lat', columns='lon', values=values_from)
            vis = ax.barbs(u.columns, u.index, u.values, v.values, alpha=0.8, length=5, sizes=dict(emptybarb=0.2, spacing=0.3, height=0.5), linewidth=0.8, transform=ccrs.PlateCarree(), regrid_shape=140)
            #ax.quiver(u.columns, u.index, u.values, v.values, alpha=0.5, linewidth=0.8, transform=ccrs.PlateCarree(), regrid_shape=20)

            # assign one of the dataframes to gridded_df such that the ticks can be read off
            gridded_df = u
        else:
            gridded_df = df.pivot(index='lat', columns='lon', values=values_from)

        df_cols = gridded_df.columns
        df_index = gridded_df.index

        if interpolation is not None:
            gridded_df = scipy.ndimage.zoom(gridded_df.values, interpolation)
        elif gaussian_filtering is not None:
            gridded_df = scipy.ndimage.gaussian_filter(gridded_df.values, gaussian_filtering)
        else:
            gridded_df = gridded_df.values

        # plot a color mesh on top
        if vis_type == 'mesh':
            vis = ax.pcolormesh(
                df_cols,
                df_index,
                gridded_df,
                alpha=0.8,
                cmap=cmap,
                norm=colors.LogNorm() if log_norm else None,
                transform=ccrs.PlateCarree())
        elif vis_type == 'contour':
            vis = ax.contourf(
                df_cols,
                df_index,
                gridded_df,
                alpha=0.8,
                cmap=cmap,
                norm=colors.LogNorm() if log_norm else None,
                transform=ccrs.PlateCarree())

        # ax.set_ylabel('Latitude', size=20, labelpad=10, rotation=90)
        ax.set_yticks([10.375, 22.375, 34.375])
        ax.set_yticklabels([10.375, 22.375, 34.375], fontsize=12, rotation=90)
        ax.set_xticks([67.375, 79.375, 91.375])
        ax.set_xticklabels([67.375, 79.375, 91.375], fontsize=12, rotation=0)
        # ax.set_xlabel('Longitude', size=20, labelpad=10)

        ax.set_xmargin(0)
        ax.set_ymargin(0)
        ax.autoscale_view()

        # set ticks and labels
        ax.xaxis.set_major_formatter(FormatStrFormatter('%g° E'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%g° N'))

        if vis_type != 'barbs' and not no_cbar:
            # ensure the colorbar is of equal height as the grid ("magic" fraction)
            # see https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
            cbar = plt.colorbar(vis, ax=ax, fraction=0.046, pad=0.04)

            # set a label for the colorbar if so defined
            if clabel is not None:
                cbar.set_label(clabel, labelpad=10, size=15)

        # set a title for the plot if so defined
        if title is not None:
            ttl = ax.set_title(title, size=15)
            ttl.set_position([0.5, 1.02])

        # save the plot into a png if so defined
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')

        return vis

    @staticmethod
    def prepare_cartopy_df(data_dict, month=None, day=None, onset_dates=None, offset=0):
        """
        Create a dataframe that has the appropriate format for visualization generation.
        Can be passed in any dictionary with years as keys.
        If onset dates are passed in, the result can be generated relative to them (by specifiyng an offset).
        """

        all_years = []
        for year, value in data_dict.items():
            if onset_dates is not None:
                onset_date = onset_dates[year]
                if offset != 0:
                    onset_date = onset_date.shift(days=offset)
                onset_date = onset_date.datetime
                year_data = value[f'{year}-{onset_date.month:02d}-{onset_date.day:02d}']
            else:
                year_data = value[f'{year}-{month:02d}-{day:02d}']

            year_data = year_data.reset_index()

            all_years.append(year_data.values)

        all_years = np.stack(all_years, axis=-1)
        averaged_years = pd.DataFrame(np.mean(all_years, axis=-1), columns=['lat', 'lon', 'val'])

        return averaged_years
