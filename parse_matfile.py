import numpy as np
import pandas

from scipy.io import loadmat


def parseMatFile(filename, header='udbfHeader', data='udbfData'):
    '''
    Read a matlab *.mat file and return it as a pandas dataframe, and return
    metadata as a dict

    By default, the optional arguments 'header' and 'data' are set to read the 
    UDBF format used by the DIAPHANE Q-Station digitiser, particularly for 
    temperature data.  

    Parameters
    ----------
    filename : str
    header : str, optional
    data : str, optional

    Returns
    -------
    df : pandas dataframe
    metadata : dict
    '''

    mat = loadmat(filename)
    
    # Read data
    mdata = mat[data]
    names = mdata.dtype.names
    ndata = {n : mdata[n][0, 0] for n in names}

    # Create column names according to the number of variables associated
    # with each 'key' in ndata
    cols  = []
    for name in names: 
        val = ndata[name] 
        vs  = val.shape[-1] 
        if vs > 1: 
            for i in range(vs):  
                cols.append(name + '%02d' % i) 
        else: 
            cols.append(name)
    df = pandas.DataFrame(np.concatenate([ndata[n] for n in names],
                                         axis=1),
                          columns=cols,
                          index=range(len(val)))

    # If header/metadata description is not requested, return only the dataframe
    # else return both.
    if header is None:
        return df
    else:
        mdata    = mat[header]
        metadata = {n : mdata[n][0, 0] for n in mdata.dtype.names}
        return df, metadata


def parseMultipleFiles(filelist, header='udbfHeader', data='udbfData'):
    dfs = []
    for file in filelist:
        df, _ = parseMatFile(file, header=header, data=data)
        dfs.append(df)

    dfs = pandas.concat(dfs)
    dfs.index = range(len(dfs))

    return dfs

