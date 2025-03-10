#!/usr/bin/python3

from scipy.interpolate import splrep, splev

import numpy as np
import pandas as pd


logbookfname = ('/home/david/FieldWork/Soufriere-Guadeloupe/HotSprings/data/' + 
                'hotSprings_flowratesFromLogbooks.csv')

def read_logbook_data(fname=logbookfname):
    ''' Reads log book data for the hot springs and returns the weighted mean 
    flowrate and weighted standard deviation '''
    dataStruct = []
    max_iters  = 20
    iters      = 0
    keysTuple  = ('Times', 'sigVolume', 'stdevFlowrate', 'sigTime', 
                  'meanFlowrate', 'weights', 'sigFlowrate', 
                  'Flowrate', 'Volumes')

    with open(fname, 'r') as f:
        while iters < max_iters:
            key = 0
            struct = {}
            while key != '':
                # Remove leading and trailing whitespace, as well as any newline
                # characters.  Split the remaining output.
                key, *params = map(str.strip, f.readline().split(','))

                # Filter out empty strings
                params = list(filter(None, params))

                # If the list has only one entry, convert to a 'scalar'
                if len(params) == 1:
                    params = params[0]

                # If the key is in 'keysTuple', recast params to floats
                if key in keysTuple:
                    params = np.float64(params)
                struct[key] = params

            # Remove lines with an empty key from the structure
            struct = {k: v for k, v in struct.items() if k is not ''}
            if len(struct) > 0:
                dataStruct.append(struct)
            iters += 1
    
    cols = ['DateTime', 'Location', 'meanFlowrate', 'stdevFlowrate']
    # An empty dataframe to which data can later be append to
    dStruct = pd.DataFrame(columns=cols)  

    for d in dataStruct:
        df = pd.DataFrame(d).iloc[0]
        df = df[['DateTime', 'Location', 'meanFlowrate', 'stdevFlowrate']]
        dStruct = dStruct.append(df.to_frame().transpose(), ignore_index=True)
        dStruct['DateTime'] = pd.to_datetime(dStruct['DateTime'])
    return dStruct
            

def plot_for_loc(dSeries, loc):
    ''' 
    Extract the dates, flowrates and errors from the dataframe 'dSeries' 
    for the location given by 'loc'.

    Parameters:
    -----------
    dSeries     a pandas dataframe
    loc         location (e.g. 'TA', 'GA', 'PR')
    '''
    if loc not in dSeries.Location.values:
        raise AttributeError('location not in the supplied dataframe')
    else:
        dates = dSeries[dSeries.Location == loc].DateTime
        Q     = dSeries[dSeries.Location == loc].meanFlowrate
        Qerr  = dSeries[dSeries.Location == loc].stdevFlowrate
    return dates, Q, Qerr


def spl_interpolate(times, Q, Qerr, autosmooth=False, k='3'):
    ''' 
    Provides a wrapper for the interpolation of data.  
    Uses the splrep and splev routines from scipy.interpolate

    Parameters:
    -----------
    times       time instances of the interpolation grid
    Q           data to be interpolated
    Qerr        standard deviations of the data
    autosmooth  (optional) automatic smoothing of data.  Default is False
    k           order of the interpolation.  Should be between 1 <= k <= 5 and
                even numbers should be avoided.
    '''
    m = len(times)
    w = 1./Qerr                 # Weights as fn of std. dev.
    if not autosmooth:          # Maximise the smoothing
        s   = np.floor(m + np.sqrt(2*m)) 
        tck = splrep(times, Q, w, k=k, s=s)
    else:
        tck = splrep(times, Q, w)

    # Define interpolation points
    t_new = np.linspace(times[0], times[-1], 3*m+1)
    Q_new = splev(t_new, tck)
    
    return t_new, Q_new


if __name__ == '__main__':
    from matplotlib.dates import MonthLocator, DateFormatter

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt


    # Load the data.  Contains flowrate data for TA and PR
    dSeries = read_logbook_data()
    
    # Plot the flowrate timeseries data
    fig, ax = plt.subplots()
    
    # TA first
    loc   = 'TA'
    dates, Q, Qerr = plot_for_loc(dSeries, loc)
    times = mdates.date2num(pd.Index(dates).to_pydatetime())
    ax.errorbar(times, Q, yerr=Qerr, fmt='.--', c='r', label='TA')

    # Interpolate data
    t_TA, Q_TA = spl_interpolate(times, Q, Qerr)
    
    # Then PR
    loc   = 'PR'
    dates, Q, Qerr = plot_for_loc(dSeries, loc)
    times = mdates.date2num(pd.Index(dates).to_pydatetime())
    ax.errorbar(times, Q, yerr=Qerr, fmt='.--', c='g', label='PR')

    # Interpolate data
    t_PR, Q_PR = spl_interpolate(times, Q, Qerr)

    # Plot interpolated data
    ax.plot(t_TA, Q_TA, 'r-', label='TA interp.')
    ax.plot(t_PR, Q_PR, 'g-', label='PR interp.')

    ax.legend(loc=0)

    ax.set_xlabel('Date')
    ax.set_ylabel(r'Flow rate/[l/s]')

    # Display dates on the xaxis and format the dates to show each month
    ax.xaxis_date()
    ax.xaxis.set_major_locator(MonthLocator(np.arange(0, 13, 1)))
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    #fig.autofmt_xdate()
    fig.tight_layout()

    plt.show()
    
