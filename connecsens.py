#!/usr/bin/env python3
""" connecsens.py
Utilities for reading, converting and viewing connecsens data
"""

from datetime import datetime as DT
from timeSeriesAnalysis.timeSeriesAnalysis import plot_data_plotly
from timeSeriesAnalysis.readWebObsDF import resample_df

import numpy as np
import pandas as pd



CONNECSENS = '~/FieldWork/Soufriere-Guadeloupe/connecsens/'


def get_depths(node, site, date=None, datefmt='%Y-%m-%d'):
    """
    Returns a list of depths of the probes for a given node
    """
    depths = pd.read_csv(CONNECSENS + 'node_depths.csv',
                         parse_dates=['start', 'end'])

    # Parse only the entries whose start and end dates
    if date is not None:
        depths.loc[depths['end'].isna(), 'end'] = DT.today()  # Fill NaTs
        date   = DT.strptime(date, datefmt)  
        depths = depths.query('start < @date < end')
    
    return depths.query('node == @node and site == @site')['depth'].values


def read_data(node):
    """
    Returns the data contained on the requested node as pandas.DataFrame

    Parameters
    ----------
    node : str
        Code of the node

    Returns
    -------
    data : pandas.DataFrame
    """
    return pd.read_csv(CONNECSENS + node + \
                       '/output/csv/434E535306E31%s-002B277B.csv' % node,
                       parse_dates=['nodeTimestampUTC'],
                       sep=';')


if __name__ == "__main__":
    import sys
    import plotly.express as px

    if len(sys.argv) == 1:
        node = '286'
        site = 'NAP'
    else:
        node = sys.argv[1]
        site = sys.argv[2]
        
    
    data   = read_data(node)
    depths = get_depths(int(node), str(site))

    # Remove all "empty" entries
    data.dropna(subset=data.columns[
        data.columns.str.contains('Temperature')], inplace=True)
    
    data = resample_df(data, freq='1D', datetimecol='nodeTimestampUTC')

    mapper = {}
    for match in data.columns.str.findall(r'(PT.*-([1-4])_Temp.*)'):
        if match != []:
            mapper[match[0][0]] = 'CH' + match[0][1]

    data = data.rename(mapper=mapper, axis=1)

    # make a "long" dataframe that is compatible with plotly plotting utilities
    longdf = pd.melt(data, id_vars='nodeTimestampUTC',
                     value_vars=['CH1', 'CH2', 'CH3', 'CH4'])
    
    
    plotlyargs = dict(x='nodeTimestampUTC', y='value', color='variable')

    fig = px.scatter(longdf, **plotlyargs)
    fig.update_layout(
        xaxis_tickformat = '%Y-%m-%d',
        xaxis_title = 'Date',
        yaxis_title = ']',
        font=dict(
            family='Computer Modern Roman',
            size=18,
            color='#000000',
        ),
        legend=dict(x=.05, y=.95, title='<b>Probe</b>'),
    )

    fig.write_html('temp_plot.html')
    # fig, ax = plt.subplots()
    # sca = []
    # for probe, depth in zip(
    #         data.columns[data.columns.str.contains('TemperatureDeg')], \
    #         depths):
    #     # sca.append(ax.scatter(data=data, x='nodeTimestampUTC', y=probe,
    #     #                       label='%2d cm' % depth))

    #     ax = sns.scatterplot(data=data, x='nodeTimestampUTC', y=probe,
    #                          label='%2d cm' % depth, ax=ax)

    # ax.legend()

    # ax.set_xlabel('')
    # ax.set_ylim((-5, 105))
    # ax.set_ylabel('Temperature/[°C]')
    # ax.set_title(f'Node: {node}, {site}')
    
    # fig.autofmt_xdate()
    # fig.tight_layout()
    # fig.savefig('/home/david/FieldWork/Soufriere-Guadeloupe/connecsens/'
    #             + 'plots/connecsens_%s_%s.png' % (
    #                 node, data.iloc[-1].nodeTimestampUTC.strftime('%Y%m%d')),
    #             dpi=150)
    # plt.show()

