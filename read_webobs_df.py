#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

from timeseries_analysis.dateparser import dateparser, date_parser_sismo 

# import numpy as np
# import datetime
import pandas as pd

##### NAME TUPLES #####
# Hot springs data
hs_names = ('Date','Time','Code Site','Site','Type','Tair','Teau','pH','Debit',
            'Cond','Niveau','Li','Na','K','Mg','Ca','F','Cl','Br',
            'NO3','SO4','HCO3','I','SiO2','d13C','d18O','dD','Cond25',
            'NICB','Remarques','Valider')

# Discrete fumarole temperature data
tf_names = ('Date','Time','Code Site','Site','Tfum','pH','Debit','Amp',
            'H2','He','CO','CH4','N2','H2S','Ar','CO2','SO2','O2','Rn',
            'd13C','d18O','S/C','Observations','Valider')

# Continuous temperature data
tf_cont_names  = ('y', 'm', 'd', 'H', 'M', 'S', 'CSC', 'CSN', 'CSS', 
                  'CS_57', 'CS_28', 'CS_14', 'CS_44')
tfcontnamesold = ('y', 'm', 'd', 'H', 'M', 'S', 'CSC', 'CSN', 'CSS', 
                  'CS_57', 'CS_SOUP', 'CS_28', 'CS_14')


# Permanent multigas data
mgperm_names = list(tf_cont_names)[:6] + ['SO2', 'H2S', 'CO2', 'Hygrometry', # 
                                          'T', 'p', 'Battery', 'H2O']

# Seismological data
seismo_ts_names = ('datetime', 'Nb', 'Duration', 'Amplitude', 'Magnitude', 
                   'Longitude', 'Latitude', 'Depth', 'Type', 'File',
                   'LocMode', 'LocType', 'Projection', 'Operator', 'TS', 'ID')

# Sanner meterological data
sanner_ts_names = ('year', 'month', 'day', 'hour', 'minute', 'second', 
                   'RH', 'Tair', 'Irrad', 'WDir', 'WSpd', 'Rain', 
                   'Pair', 'Bat', 'Tair2')


def read_webform_data(fname, names, sep=';'):
    '''
    Reads data from webforms (e.g. geochemistry databases) and returns a pandas
    dataframe.

    This routine deals with missing or erronous time entries.
    '''
    df = pd.read_csv(fname, sep=sep, names=names, skiprows=1)

    # Fill in null values for the time to midnight local
    df['Time'] = df['Time'].fillna('04:00')
    # Some data had incomplete timestamps.  Set these too to 04:00
    for ind, time in enumerate(df.Time.values):
        if len(time) != 5:
            df.loc[ind, 'Time'] = '04:00'  
        if time[2] != ':':
            df.loc[ind, 'Time'] = time[:2] + ':' + time[3:]

    # Create a new column that combines date and time
    df['datetime'] = df['Date'] + ' ' + df['Time']
    # ...and convert it to datetime format
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)

    return df


def read_timeseries(fname, names, sep=' ', comment='#', **kwargs):
    '''
    Returns a dataframe containing generic time series data using the standard
    dateparser routine.

    Parameters
    ----------
    fname : str
        Name of the csv file to be read
    names : tuple or array-like
        A list or tuple of column names for the dataframe
    sep : str
        Separator for the text file

    Returns
    -------
    pandas dataframe
    '''

    return pd.read_csv(fname, names=names, sep=sep, comment=comment,
                       parse_dates={'datetime':
                                    ['y','m','d','H','M','S']},
                       date_parser=dateparser, **kwargs)


def read_seismo_ts(fname, names, sep=';'):
    '''
    Returns a dataframe containing seismological time series data.  Uses a 
    particular form of the date parsing routines to account for the fractional 
    seconds encountered therein.
    '''
    return pd.read_csv(fname, sep=sep, comment='#', names=names,
                       parse_dates=[0], date_parser=date_parser_sismo) 


def write_to_workbook(df, iterkey, iterable, wb_name='workbook.xlsx'):
    '''
    Write contents of a dataframe to an excel xlsx workbook, with each 
    worksheet corresponding to a separate event.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe 
    iterkey : string
        Dataframe key (column name) whose values are the contents of "iterable"
    iterable : array-like (iterable)
        An iterable variable
    wb_name : string (optional)
        Name of the workbook to be written.  Default name is 'workbook.xlsx'
    '''
    
    with pd.ExcelWriter(wb_name) as writer: 
        for item in iterable:
            erupt_df = df[df[iterkey] == item]
            erupt_df.to_excel(writer, sheet_name=item, index=False) 
        
    return
