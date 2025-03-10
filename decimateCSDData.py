#!/usr/bin/python

# decimateCSDData.py
# Reads files from the Diaphane in a directory of the format yyyy/mm/dd/ as
# a pandas DataFrame and saves.
#
# To be run from the ./docobs/acqui/ directory

# Written by D. E. Jessop (OVSG), May 2019

from datetime import datetime as DT
from datetime import timedelta as TD
from calendar import monthrange
from timeSeriesAnalysis.readWebObsDF import (TFContNames,
                                             dateparser,
                                             resample_df)

import numpy as np
import pandas
import os


# Base path, to be modified as necessary to the directory where data are stored
path = ('/home/david/FieldWork/Soufriere-Guadeloupe/' +
        'temperatureFumaroles/data/acqui/')
frequency = '1T'        # Produce 1 sample per n minutes (T), second (s) etc.

TFContNames = list(TFContNames)
ind = 0
while len(TFContNames) < 14:
    TFContNames.append('FREE%02d' % ind)
    ind += 1


def readDataFrame(fname):
    # Wrapper to streamline the reading of dataframes
    return pandas.read_csv(fname, skiprows=1,
                           names=TFContNames,
                           parse_dates={'datetime' :
                                        ['y', 'm', 'd', 'H', 'M', 'S']},
                           date_parser=dateparser)



# Read a number of files to memory (corresponding to 1 calendar month).  Also
# need to read the last file from the last day of the previous month, if it
# exists
years = sorted([int(name) for name in os.listdir(path) if
                os.path.isdir(os.path.join(path, name))]) 
for year in years:
    yearPath = path + '%4d/' % year
    months = sorted([int(name) for name in os.listdir(yearPath) if
                os.path.isdir(os.path.join(yearPath, name))]) 
    for month in months:
        df = pandas.DataFrame()
        lastMonthsFile = ''
        if month == 1:
            try:
                # Look for the last file of the last day of the previous month 
                # of the previous year and read it if it exists.
                lastMonthsPath = path + '%4d/%02d/%2d/' % (year-1, 12, 31)
                lastMonthsFile = sorted(os.listdir(lastMonthsPath)).pop()
                df = readDataFrame(lastMonthsPath + lastMonthsFile)
            except FileNotFoundError:
                pass
        else:
            lastDay = monthrange(year, month-1)[1]
            try:
                # Look for the last file of the last day of the previous 
                # month and read it if it exists.
                lastMonthsPath = path + '%4d/%02d/%2d/' % (year,
                                                           month-1,
                                                           lastDay)
                lastMonthsFile = sorted(os.listdir(lastMonthsPath)).pop()
                df = readDataFrame(lastMonthsPath + lastMonthsFile)
            except FileNotFoundError:
                pass
            
        # Resampled files are stored under the "month" directory, so need to 
        # avoid adding them ot the "days" variable
        monthPath = path + '%4d/%02d/' % (year, month) 
        newPath, oldFile = '', ''

        days = sorted([name for name in os.listdir(monthPath) 
                       if os.path.isdir(os.path.join(monthPath, name))])
        for day in days:
            newPath = monthPath + '%s/' % day
            files   = sorted(os.listdir(newPath))
            for file in files:
                df = pandas.concat([df, readDataFrame(newPath + file)])

        # Some files contain data for several days, so produce and save to 
        # file a resampled data frame for each day of the month.
        rsFreq  = '1T'
        tFormat = '%Y-%m-%d_%H-%M-%S'
        if df.empty is not True:
            for day in range(1, 32):
                RSFName = ''
                try:
                    dt    = DT(year, month, day)
                    subDF = df[df.datetime >= dt]
                    subDF = subDF[subDF.datetime < dt + TD(1)]
                    dfResampled = resample_df(subDF, rsFreq)
                    
                    # filename from datetime of first entry in resampled data
                    RSFName = (dfResampled.iloc[0].name.strftime(tFormat) +
                               '_resampled%s.csv' % rsFreq)
                    dfResampled.to_csv(monthPath + RSFName, float_format='%.3f')
                except: #ValueError:
                    pass
                finally:
                    print(RSFName)
