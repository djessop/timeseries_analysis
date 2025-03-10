import pandas as pd
import numpy as np

from datetime import datetime as DT, timedelta as TD


def dateparser(y, m, d, H=None, M=None, S=None, s=None):
    """ 
    A wrapper that allows column separated timestamps to be reconstructed
    in pandas.  Useful for pandas >= 0.20.
    The datetime formatting string is configured to work for datetimes where
    each of the year, month, date, ... columns are separated.
    Note 2020-04-01: the pandas.datetime class will be depreciated as of 
    pandas==1.1.0 as it imported directly from the datetime module.  Now 
    importing from the datetime module

    To use, 
    df = pandas.read_csv(filename, sep=<separator>, comment=<commentchar>, 
    ...: names=('yyyy', 'mm', 'dd', 'HH', 'MM', 'SS', <variableList>),
    ...: parse_dates={'datetime' : ['yyyy', 'mm', 'dd', 'HH', 'MM', 'SS']}, 
    ...: date_parser=dateparser) 

    Lifted shamelessly from:
    https://stackoverflow.com/questions/17465045/can-pandas-automatically-
    recognize-dates

    Parameters:
    -----------
    y           year
    m           month
    d           day
    H           hours
    M           minutes
    S           seconds
    s           decimal seconds
    """
    
    # Base case dt and format string.  
    dt = y + '-' + m + '-' + d
    fmtStr = '%Y-%m-%d'
    
    # Build dt and fmtStr according to case
    if H is not None:
        dt += " " + H           # Add 'T' between parenthesis for iso format
        fmtStr += " %H"         # fmtStr is now '%Y-%m-%d %H'
        if M is not None:       # Only do this if H AND M are not None
            dt += ':' + M
            fmtStr += ":%M"     # fmtStr is now '%Y-%m-%d %H:%M'
            if S is not None:   # Only do this if H AND M AND S are not None
                dt += ':' + S
                fmtStr += ":%S" # fmtStr is now '%Y-%m-%d %H:%M:%S'
                if s is not None:
                    dt += '.' + s
                    fmtStr += ".%f" # fmtStr is now '%Y-%m-%d %H:%M:%S.%f'

    return DT.strptime(dt, fmtStr)


def date_parser_sismo(dt):
    """
    A function useful for parsing datetime strings from seismological or 
    other data where the date and time data are in %Y%m%d %H%M%S.%s format

    See help for dateparser.
    """
    fmtStr = '%Y%m%d %H%M%S.%f'
    return DT.strptime(dt, fmtStr)


def datetime2unixtime(dtarray):
    """
    Converts an array of datetime objects into unixtime (seconds ellapsed 
    since 1970-01-01 00:00:00).
    """
    return pd.DatetimeIndex(dtarray).astype(np.int64) // 10**9


def unixtime2datetime(timearray):
    """
    
    """
    return pd.to_datetime(timearray, unit='s')


def mat2pydt(matlabtime):
    '''
    Converts "matlab" datetime (number of days since 0000-01-01) to a python 
    datetime object, according to this blog post:
    sociograph.blogspot.com/2011/04/how-to-avoid-gotcha-when-converting.html

    Parameters
    ----------
    matlabtime : list
        A list of the "matlab" datenum format datetimes

    Returns
    -------
    pythontime : list
        A list of datetime objects
    '''
    return [DT.fromordinal(int(t) - 366)
            + TD(days=t%1) for t in matlabtime]
