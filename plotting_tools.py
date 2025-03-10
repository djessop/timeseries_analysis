import datetime


def plotQuakes(datetimes, magnitudes, axis, M0=2., width=1., alpha=.5,
               **kwargs):
    '''
    Plot earthquakes as vertical lines on background, where width is an
    indicator of magnitude.  Do this for all events >= M0.

    Parameters
    ----------
    datetimes : array_like
    magnitudes : array_like
    axis : matplotlib axis
    M0 : scalar
    width : scalar
    alpha : scalar
    kwargs : parameters to be passed to matplotlib.pyplot.axvline

    Returns
    -------
    None

    '''
    datetimes  = datetimes[magnitudes >= M0]
    magnitudes = magnitudes[magnitudes >= M0]
    for d, m in zip(datetimes, magnitudes):
        axis.axvline(d, alpha=.5, color='gray', linewidth=m * width, zorder=0)
    return


def plotMaria(ax, alpha=.5, linewidth=1):
    '''
    Put a red vertical stripe when Maria occurred
    '''
    ax.axvline(datetime.date(2017,9,18),
               color='r', linewidth=linewidth, alpha=alpha, zorder=0)
    return
        
