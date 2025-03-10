#!/usr/bin/env python3
#
# This script reads data files from the Diaphane project (from the yyyy/mm/dd
# directory) to memory as pandas DataFrame and saves the resampled day data to
# files in a yyyy/mm/resampled directory. A year and month must be provided as
# command line argument (run it without argument to get a help screen).
#
# Written by D. E. Jessop and X. Béguin (OVSG), May-June 2019
#
# To process the whole archive, use something like this:
#   arch_dir="path/to/arch"
#   for y in $(cd $arch_dir; ls -xd ????); do 
#     for m in $(cd $arch_dir/$y; ls -xd ??); do 
#       ./decimate_CSD_month.py $y $m
#     done
#   done
#

import calendar
import datetime as dt
import os
import pandas
import re
import sys


# Base archive directory (must contains the year directory)
#archive_dir = ('/home/david/FieldWork/Soufriere-Guadeloupe/'
#               + 'temperatureFumaroles/data/acqui/')
archive_dir = 'TempFlux_CSD/'

# Resample frequency: 1 sample per n minutes (T), second (s) etc.
frequency = '1T'

# Timestamp format in output file names
#timestampFormat = '%Y-%m-%d_%H-%M-%S'
timestampFormat = '%Y-%m-%d'

# Column headers
TFContNames = ['y','m','d','H','M','S','CSC','CSN','CSS','CS_57',
               'CS_SOUP','CS_28','CS_14']
# Force 14 fields in the CSV
for i in range(len(TFContNames), 15):
    TFContNames.append('FREE%02d' % i)

# Default regular expression to match month or days directories
RE_2INT = re.compile('\d{2}$') 

# The name of the subdirectory of the month directories where the resampled
# files will be written (these directories are created if necessary)
resampled_dirname = 'resampled'


def usage():
    """
    Usage screen for the command line help.
    """
    print("""Usage: %s <year> <month>
Read data from the CSD station and resample the data for the requested month.""" \
        % sys.argv[0])


def resampleDF(df, freq='1H'):
    """
    Return a dataframe whose data has been resampled at the requested 
    frequency and averaged.  

    Note: the index of the dataframe must be a datetime.datetime object.
    """
    df.index = df.datetime
    return df.resample(freq).mean()


def dateparser(timestamp_str):
    """
    Date parser function for the pandas.read_csv function. Must parse the
    concatenated string of columns described by the parse_dates argument to
    read_csv and return a datetime.datetime object.
    (See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)
    """
    timestamp_int_list = [int(e) for e in timestamp_str.split(' ')]
    return dt.datetime(*timestamp_int_list)


def readDataFrame(fname):
    """
    Wrapper to streamline the reading of dataframes.
    """
    return pandas.read_csv(fname, skiprows=1,
                           names=TFContNames,
                           parse_dates={'datetime' :
                                        ['y', 'm', 'd', 'H', 'M', 'S']},
                           date_parser=dateparser)

def list_int_dir(path, pattern=RE_2INT):
    """
    Return the list of subdirectories of 'path' with names composed only with
    integers, as a sorted list of integers.
    """
    # Note: using os.scandir instead of os.listdir significantly increases
    # performance when testing each entry type.
    return sorted([int(entry.name) for entry in os.scandir(path)
                   if entry.is_dir() and pattern.match(entry.name)])


def list_CSV_files(path):
    """
    Return the list of *.csv files in 'path' as a sorted list of absolute
    filenames.
    """
    CSVFiles = sorted([entry.name for entry in os.scandir(path)
                       if entry.is_file() and entry.name.endswith('.csv')])
    return [os.path.join(path, f) for f in CSVFiles]



# Read the requested month from the command line arguments
try:
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    month_dt = dt.datetime(year, month, 1)
except IndexError:
    usage()
    sys.exit(1)
except ValueError:
    print('Error: cannot parse provided month date')
    usage()
    sys.exit(1)

month_path = os.path.join(archive_dir, month_dt.strftime('%Y/%m'))


# Read the last file of the last day of the previous month
# in hope to get the latest data before the requested month
prev_day_dt = month_dt - dt.timedelta(1)
prev_day_path = os.path.join(archive_dir, prev_day_dt.strftime("%Y/%m/%d"))
try:
    # Get filename of the last file of the last month
    prev_month_last_file = list_CSV_files(prev_day_path)[-1]
except (FileNotFoundError, IndexError) as e:
    # No file in last day of previous month: use an empty DataFrame
    print('WARNING: could not read previous month data: %s' % str(e))
    df = pandas.DataFrame()
else:
    # Read the last day of the previous month into a DataFrame
    print('DEBUG: reading previous data from %s' % prev_month_last_file)
    df = readDataFrame(prev_month_last_file)


# Read data files for the requested month
try:
    for day in list_int_dir(month_path):
        day_path = os.path.join(month_path, '%0.2d' % day)
        for f in list_CSV_files(day_path):
            print('Reading CSV file %s... ' % f, end='')
            df = pandas.concat([df, readDataFrame(f)])
            print('done.')
except FileNotFoundError as e:
    # No data for this month, proceed to the next
    print('ERROR: could not read month data: %s' % str(e))
    print('No data found for requested month %s. Nothing to do.' % month_dt.strftime('%Y-%m'))
    sys.exit(0)


# Some files contain data for several days, so produce and save to 
# file a resampled data frame for each day of the month.
for day in range(1, calendar.monthrange(year, month)[1] + 1):
    day_dt = dt.datetime(year, month, day)
    nextday_dt = day_dt + dt.timedelta(1)

    # Select only the data for the day
    subDF = df[(df.datetime >= day_dt) & (df.datetime < nextday_dt)]

    # Ignore this day if there is no data
    if subDF.empty:
        print('WARNING: ignoring empty source data frame for %s' \
                % day_dt.strftime('%Y-%m-%d'))
        continue

    # Resample the selected data
    dfResampled = resampleDF(subDF, frequency)

    # Ignore this day if there is no data
    if dfResampled.empty:
        print('WARNING: ignoring empty resampled data frame for %s' \
                % day_dt.strftime('%Y-%m-%d'))
        continue

    # Name the filename the date of the first entry in resampled data
    resamplFilename = '%s_resampled%s.csv' % \
        (dfResampled.iloc[0].name.strftime(timestampFormat), frequency)
    # Make sure the 'Resampled' directory in the month directory exists
    resampled_dir = os.path.join(month_path, resampled_dirname)
    os.makedirs(resampled_dir, exist_ok=True)

    # Write the resampled CSV file
    print('Writing resampled file %s' % resamplFilename)
    dfResampled.to_csv(os.path.join(resampled_dir, resamplFilename),
                       float_format='%.3f')

