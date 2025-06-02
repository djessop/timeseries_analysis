#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

""" 
timeseries_analysis.py

Provides routines for the analysis of time series, including plotting and 
filtering.

Last update: 2025-06-02

Created by D. E. Jessop
"""

from matplotlib.transforms import blended_transform_factory
import matplotlib.pyplot as plt
import plotly.express as px

from scipy import ndimage
from scipy.signal import (butter,
                          lfilter,
                          # freqz, 
                          firwin,
                          fftconvolve,
                          filtfilt,
                          welch,
                          # periodogram,
                          # medfilt,
                          periodogram,
                          # find_peaks,
                          correlate,
                          correlation_lags,
                          detrend)
from scipy.signal.windows import hann, tukey
# from scipy.stats import ks_2sam

from astropy.timeseries import LombScargle

import pandas as pd
import numpy as np


secs_in_day = 86400.0020
secs_in_hr  =  3600
days_in_yr  =   365.2422
days_in_mon =    28

def hz_to_yr(f_hz):
    '''
    Transform the frequency (in Hz) to a period (in yrs)
    '''
    return 1 / (f_hz * days_in_yr * secs_in_day)


def plot_peaks_periods(ax, peaks, power_spectrum, periods=None, unit='years', 
                       fs=None, yp=0.002, zorder=8, peakoffset=.8, **kwargs):
    '''
    Parameters
    ----------
    ax : matplotlib axis
    peaks : array-like
    power_spectrum : list or tuple
        Power-spectrum and frequencies 
    periods : list or array-like
    fs : scalar
    yp : scalar
        proportion of yaxis to place the annotations
    zorder : int
        order of plotting (default is 8 - high), so red dots get plotted
        over the top of other curves and annotations
    '''
    tform  = blended_transform_factory(ax.transData, ax.transAxes)
    f, Pxx = power_spectrum

    # Plot peak frequencies and annotate these points with the cooresponding
    # period in years.
    for peak in peaks:
        if peak is not None:
            T_years = hz_to_yr(f[peak])
            _ = ax.annotate('%.2f %s' % (T_years, 'yr'),
                            xy=(f[peak], Pxx[peak]),
                            # Place text slightly below data point
                            xytext=(f[peak], Pxx[peak]*peakoffset),
                            xycoords='data',
                            **kwargs)
            _ = ax.plot(f[peak], Pxx[peak], '.r', zorder=zorder)

    if 'hour' in unit:
        days_in_period = 1. / 24.
    if 'day' in unit:
        days_in_period = 1.
    if 'week' in unit:
        days_in_period = 7.
    if 'month' in unit:
        days_in_period = days_in_mon
    if 'year' in unit:
        days_in_period = days_in_yr

    # Add vertical (grey) lines to indicate certain periods in the
    # frequency space, i.e. 1 week, 1 month, 1 year etc.
    if periods is not None:
        for period in periods:
            freq = 1 / (period * days_in_period * secs_in_day)
            ax.axvline(freq, color='gray', lw=.5, zorder=0)
            
            if (period == 1) and (unit.endswith('s')):
                unit = unit[:-1]  # e.g. 'year', not 'years'
            elif (period != 1) and not unit.endswith('s'):
                unit += 's'
            text_str = '%d %s' % (period, unit)
            ax.annotate(text_str, xy=(freq, yp),
                        transform=tform,
                        color='gray', rotation=90, 
                        ha='center', fontsize=10, zorder=6,
                        bbox=dict(fc='w', ec='none', pad=.1))

    # Add annotations to these vertical bars
    if fs is not None:
        ax.axvline(fs, color='gray', lw=.5, zorder=0)
        period = round(fs * (secs_in_day * 28))
        # print(str(period) + 'months')
        text_str = '%d %s' % (period, unit)
        ax.annotate(text_str, xy=(fs, yp),
                    transform=tform,
                    color='gray', rotation=90, 
                    ha='center', fontsize=10, zorder=6,
                    bbox=dict(fc='w', ec='none', pad=.1))

    return
        

# Fit a sinusoid to the gas data
def sinusoid(x, y0=0., A=1., omega=1., phi=0.):
    '''Returns a model of an offset, phased sinusoid.
    
    Parameters
    ----------
    x : list or array-like
      independent variable
    y0 : scalar
      offset of the model, roughly the mean of the data to be modelled
    A : scalar
      amplitude of the model
    omega : scalar
      frequency of the model
    phi : scalar
      phase of the model
    
    Returns:
    --------
    y : list or array-like
      predicted values
    '''
    return y0 + A * np.sin(omega * x + phi)


def mpole_sinusoid(x, y0=0., A=1., f=2*np.pi, phi=0.):
    ''' 
    Returns a multipole sinusoid model.  Note that A, omega and phi
    MUST be of the same length.
    
    Parameters
    ----------
    x : list or array-like
      data
    y0 : scalar
        signal midpoint
    A : list or array-like
        amplitudes of each component 
    omega : list or array-like
        driving frequencies
    phi : list or array-like
        phase andgle of each component
    '''
    mpole_model = y0
    for A_, f_, phi_ in zip(A, f, phi):
        mpole_model += A_ * np.sin(2 * np.pi * f_ * x + phi_)

    return mpole_model


def spectral_density(df, colname, fs=None, index=None, method='LS',
                     window='hann'):
    '''
    '''

    if ((index == 'Date') or ('Date' in df.columns)): 
        d = df['Date']
    elif (df.index.name == 'Date'):
        d = df.index
    else:  #if index == 'index':
        d = df.index
        
    x = d.astype(int).values // 1e9
    y = df[colname]

    x = x[~y.isna()]
    d = d[~y.isna()]
    y = y[~y.isna()]
    if window == 'tukey':
        y = detrend(y) * tukey(len(y))
    else:
        y = detrend(y) * hann(len(y))
    
    

    ## RESAMPLE (INTERPOLATE) DATA TO 28-DAY PERIOD 
    ds = pd.date_range(d.min(), end=d.max(), freq='1D')
    xs = ds.astype(int).values // 1e9
    # Linear interpolation onto a strictly 28-day grid
    ys = np.interp(xs, x, y)    

    if fs == None:
        method = 'LS'
        print('No sampling frequency given.  ' +
              'Computing Lomb-Scargle periodogram')

    if method == 'LS':
        if fs is not None:
            f, Pxx = LombScargle(x, y).autopower(maximum_frequency=.5*fs,
                                                 samples_per_peak=11)
        else:
            f, Pxx = LombScargle(x, y).autopower(samples_per_peak=11)
    elif method == 'PD':
        f, Pxx = periodogram(ys,
                             fs=fs,
                             window=window,
                             scaling='spectrum',
                             detrend='linear')
    else:
        print('Unknown method.  Aborting...')
        return
    return f, Pxx


def n_highest_peaks(peaks, Pxx, npeaks=5):
    '''
    Return the npeak highest peaks from spectrum data.  Spectrum 
    is sorted in highest-lowest order.

    Parameters
    ----------
    peaks : array_like
    Pxx : array_like
        The PSD defining the peaks
    npeaks : int
        Number of peaks to analyse    
    '''
    ordered_peaks = []
    for P in sorted(Pxx[peaks], reverse=True)[:npeaks]:
        ordered_peaks.append(np.where(Pxx == P)[0][0])
    return ordered_peaks


def fir_filter(data, order, cutoff):
    coeffs = firwin(order, cutoff, window='hann')
    y = fftconvolve(data, coeffs, mode='same')
    return y


def butter_lowpass(cutoff, fs, order=5):
    '''
    Returns the coefficients of a Butterworth filter with cutoff and order as
    specified in the arguements.
    '''
    nyq  = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    ''' 
    Applies a uni-directional Butterworth filter to data.  Lag and edge effects
    can be apparent using this method
    
    Parameters
    ----------
    data : array_like
        Values of the sampled data
    cutoff : float
        Cutoff frequency for the filter
    fs : float
        Sampling frequency (i.e. twice the Nyquist frequency)
    order : int (optional)
        Order of the butterworth filter (default is 5).  

    Returns
    -------
    y : array_like
        Filtered data

    See Also
    --------
    butter_lowpass, filtfilt
    '''
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_lowpass_filtfilt(data, cutoff, fs, order=5):
    ''' 
    Applies a cascading (forward and backward) Butterworth filter to data.
    Lag and edge effects are eliminated using this method, though it is a 
    little slower (twice as slow?) than a unidirectional filter
    
    Parameters
    ----------
    data : array_like
        Values of the sampled data
    cutoff : float
        Cutoff frequency for the filter
    fs : float
        Sampling frequency (i.e. twice the Nyquist frequency)
    order : int (optional)
        Order of the butterworth filter (default is 5).  

    Returns
    -------
    y : array_like
        Filtered data

    See Also
    --------
    butter_lowpass, filtfilt
    '''
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def ts2psd(data, t0=1, method='periodogram'):
    '''
    Calculates the power-spectral density (periodogram) for a given timeseries

    Parameters
    ----------
    data : array_like
        Values of the sampled data
    t0 : float, optional
        Periodicity of the data sampling, i.e. the time interval between 
        successive data points.  Default is 1 s.
    method : string, optional
        'periodogram' or 'welch', the former is the default

    Returns:
    --------
    f : array_like
        frequencies at which PSD is calculated
    PSD : array_like
        power-spectral density of the data set

    See also:
    ---------
    scipy.signal.periodogram
    scipy.signal.welch
    '''
    fs  =  1./t0  # Sampling frequency

    # Estimate PSD
    if method == 'welch':
        return welch(data, fs, window='hann')
    else:       # Default setting
        return periodogram(data, fs, window='hann')


def moving_average(data, radius, pad_mode='mean'):
    """ Adapted from 
    https://learnopencv.com/video-stabilization-using-point-feature-matching-
    in-opencv/

    Created by D. E. Jessop, 2023-04-28
    """
    window_size = 2 * radius + 1
    # Define the filter
    f = np.ones(window_size) / window_size
    # Add padding to the boundaries
    data_pad = np.pad(data, (radius, radius), pad_mode)
    # Return convolved data with padding removed
    return np.convolve(data_pad, f, mode='valid')


def plot_data_plotly(df, **kwargs):
    '''
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing information to be plotted.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    fig : matplotlib.figure
        DESCRIPTION.

    '''
    fig = px.scatter(df, **kwargs)
    # Put y-axis values on logarithmic scale
    fig.update_layout(yaxis_type='linear')
    fig.update_layout(
        xaxis_tickformat = '%Y-%m-%d',
        xaxis_title = 'Date',
        font=dict(
            family='Computer Modern Roman',
            size=18,
            color='#000000',
        ),
        legend=dict(x=.05, y=.95, title='<b>Site</b>'),
    )

    return fig


## For working with pandas.DataFrame objects
def detrend_normalise_data(data, window=None):
    """Return detrended and normalised data for use in calculating 
    periodograms and correlations
    
    Parameters
    ----------
    data : array_like
        The data to be processed
    window : scipy.signal.windows object
        The windowing function to be used.  Default is None

    Returns
    -------
    ret : ndarray
        Detrended and normalised data, with or without windowing
    """
    d = detrend(data)
    if window is not None:
        d *= window(len(d))
    return d / np.linalg.norm(d)


def detrend_df(df, series, detrend_series='True', window=None):
    d = df.index
    x = d.astype(int).values // 1e9
    y = df[series]
    x = x[~y.isna()][::-1]
    d = d[~y.isna()][::-1]
    y = y[~y.isna()][::-1]
    if detrend_series:
        y = detrend_normalise_data(y, window)
    return d, x, y


def smoothing_timeseries(d, y, interp_kernel='28D', smoothing_kernel=2):
    """Returns interpolated and gaussian-kernel smoothed data of timeseries data

    Parameters
    ----------
    d : array_like
        Series of datetimes
    y : array_like
        The data to be smoothed
    interp_kernel : str or array_like
        Period of the interpolation (if str) or a monotonically-increasing
        list of values for interpolation.
    smoothing_kernel : int or float
        Width of the smoothing kernel (units equivalent to those 
        of interp_kernel)

    Returns
    -------
    ds : ndarray
        The base for interpolation as datetime
    xs : ndarray
        The base for interpolation
    ys : ndarray
        The smoothed and interpolation data
    """
    # d = ratio_df.index
    # y = ratio_df[ratio]
    x = d.astype(int).values // 1e9
    x = x[~y.isna()][::-1]
    d = d[~y.isna()][::-1]
    y = y[~y.isna()][::-1]
    
    ## RESAMPLE (INTERPOLATE) DATA TO 28-DAY PERIOD
    if type(interp_kernel) is str:
        ds = pd.date_range(d.min(), end=d.max(), freq=interp_kernel)
    else:
        ds = interp_kernel
    xs = ds.astype(int).values // 1e9
    yi = np.interp(xs, x, y)  # Linear interpolation to given period
    # gaussian smoothing using 2-day kernel
    ys = ndimage.gaussian_filter1d(yi, smoothing_kernel)  

    return ds, xs, ys


def resample_df(df, freq='1H', method='mean',
                datetimecol='datetime', dropna_subset=None):
    '''
    Returns a dataframe whose data has been resampled at the requested 
    freqency and averaged.  

    Parameters
    ----------
    df : pandas.DataFrame()
        Dataframe to be resampled
    freq : str
        Sampling frequency, e.g. '1H', '1D'
    method : str
        Method for resampling, one of 'mean', 'median' or 'std'
    datetimecol : str
        Name of the column containing datetime information.  Default is 
        'datetime'.
    dropna_subset : array_like
        Column names within which NaN values should be discarded.  Default
        is None, i.e. all columns.


    Notes
    -----
    The index of the dataframe is required to be a "datetime".  If this is 
    not the case, one will be created.
    '''
    # check if index is already in datetime format AND/OR that datetime exists
    if datetimecol not in df.columns:
        raise KeyError('"datetime" not in dataframe')
    df.index = df[datetimecol]

    if method == 'mean':
        new_df = df.resample(freq).mean()
    elif method == 'median':
        new_df = df.resample(freq).median()
    elif method == 'std':
        new_df = df.resample(freq).std()
    else:
        raise TypeError('unknown method: %s' % method)

    if dropna_subset is not None:
        new_df.dropna(subset=dropna_subset, inplace=True)
    new_df[datetimecol] = new_df.index
    new_df.reset_index(inplace=True, drop=True)

    return new_df


def correlation_matrix_plot(df, column_names, title, xlabel, figsize=(10, 10),
                            corr_type='full'):
    '''Returns a plot of the auto- and cross-correlations for the n DataSeries
    in the DataFrame df in the form of a n-by-n matrix, plus n-by-n matrices
    containing the maximum lags and correlations

    Parameters
    ----------
    df : pandas.DataFrame
    column_names : list of str
    title : str
    xlabel : str
    figsize : tuple
    corr_type : str
        Size of the output ('full' - full discrete linear cross-correlation of
        the inputs, default; 'valid' - the output consists only of those
        elements that do not rely on the zero-padding.  In 'valid' mode, either
        in1 or in2 must be at least as large as the other in every dimension;
        'same' - output is the same size as in1, centered with respect to the
        'full' output.)

    Returns
    -------
    fig : matplotlib.figure
    ax : matplotlib.axis
    mlags : numpy.array
    mcorr : numpy.array
    '''

    ndata = len(column_names)
    fig, ax = plt.subplots(nrows=ndata, ncols=ndata, sharex=True, sharey=True,
                           figsize=figsize)

    mlags = np.zeros((ndata, ndata), dtype=float)
    mcorr = np.zeros((ndata, ndata), dtype=float)
    for indy in range(ndata):
        series_a = detrend_normalise_data(df[column_names[indy]].interpolate())
        for indx in range(indy, ndata):  # Only "upper" triangle of matrix
            series_b = detrend_normalise_data(
                df[column_names[indx]].interpolate())
            corr = correlate(series_a, series_b, corr_type)
            lags = correlation_lags(len(series_a), len(series_b),
                                    corr_type)
            ax[indy, indx].plot(lags, corr, '-k')
            ax[indy, indx].grid()

            max_lag = np.argmax(abs(corr))

            ax[indy, indx].plot(lags[max_lag], corr[max_lag], '.r')

            mlags[indy, indx] = lags[max_lag]
            mcorr[indy, indx] = corr[max_lag]

    fig.suptitle(title)
    fig.supxlabel(xlabel)
    fig.supylabel('Correlation coefficient/[-]')

    return fig, ax, mlags, mcorr

def xcorrelation_matrix_plot(df1, df2, datecol='Date', title='',
                             figsize=(8, 6)):
    '''Plot the cross-correlations for the datasets contained within two
    dataframes, df1 and df2, in the form of a m-by-n matrix of plots where m
    is the number of datasets in df1 and n the number in df2

    See also: correlation_matrix_plot
    '''

    mdata = len(df1.columns[~df1.columns.str.contains(datecol)])
    ndata = len(df2.columns[~df2.columns.str.contains(datecol)])
    # print(mdata, ndata)
    fig, ax = plt.subplots(nrows=mdata, ncols=ndata, sharex=True, sharey=True,
                           figsize=figsize)

    mlags = np.zeros((mdata, ndata), dtype=float)
    mcorr = np.zeros((mdata, ndata), dtype=float)
    for indy in range(mdata):
        series_a = detrend_normalise_data(df1.iloc[:, indy])
        for indx in range(ndata):
            series_b = detrend_normalise_data(df2.iloc[:, indx])
            corr = correlate(series_a, series_b, 'full')
            lags = correlation_lags(len(series_a), len(series_b), 'full')
            ax[indy, indx].plot(lags, corr, '-k')
            ax[indy, indx].grid()

            max_lag = np.argmax(abs(corr))

            ax[indy, indx].plot(lags[max_lag], corr[max_lag], '.r')

            mlags[indy, indx] = lags[max_lag]
            mcorr[indy, indx] = corr[max_lag]

    fig.suptitle(title, y=.95, fontsize='large')
    fig.supylabel('Correlation function/[-]', x=.05, fontsize='large')
    fig.supxlabel('Lag/[days]', y=.05, fontsize='large')
    return fig, ax, mlags, mcorr
