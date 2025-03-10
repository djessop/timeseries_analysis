import numpy as np
import pandas
import os

from scipy.io.matlab import loadmat


def datenum_to_datetime(datenum): 
    """     
    Convert Matlab datenum into Python datetime.     
    
    Parameters
    ----------
    datenum: array like
        Date in datenum format     

    Returns
    -------
    dt
        Datetime object corresponding to datenum.     
    """  
    from datetime import datetime as dt, timedelta as td

    days = datenum % 1  
    return dt.fromordinal(int(datenum)) + td(days=days) - td(days=366)


if __name__ == "__main__":
    SITE = 'NAPE'
    path = ('/home/david/FieldWork/Soufriere-Guadeloupe/temperatureFumaroles/'
            + 'data/MatFiles/')

    time, fractionOfSecond, Tdata = [], [], []
    for file in sorted(os.listdir(path)): # Look for files in current directory
        if SITE in file:                  # Ignore files not containing SITE
            # read matlab .mat files.  Returns a dict containing the keys
            # "__header__", "__version__", "__globals__", "udbfHeader" and
            # "udbfData".  The latter, a numpy structured array, is the bit 
            # we're really interested in, and this contains the fields "time",
            # "fractionOfSecond" and "data".  We can access the data within 
            # each field via the ".item()" method
            data = loadmat(path + file, squeeze_me=False)
            # Loop through the values of returned arrays and append to the
            # lists defined above.  I've not yet found a better way to
            # restructure the individual data from many datasets.
            for t, fs, T in zip(data['udbfData']['time'].item(), 
                                data['udbfData']['fractionOfSecond'].item(),
                                data['udbfData']['data'].item()):
                time.append(t)
                fractionOfSecond.append(fs)
                Tdata.append(T)

    # Flatten (ravel) the data after conversion to numpy arrays.
    time             = np.array(time).ravel()
    fractionOfSecond = np.array(fractionOfSecond).ravel()
    Tdata            = np.array(Tdata).ravel()
    dt               = np.array([datenum_to_datetime(t) for t in time])
