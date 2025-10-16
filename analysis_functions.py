import numpy as np
from scipy.signal import correlate
from scipy.stats import circmean, circstd

import parameters as pm
import snet as sn
import connectome as cn

def find_first_min(y,win_size=3):
    window = []
    for l in range(win_size):
        window.append(l+1)
        window.append(-1-l)
    window = np.array(window,dtype=np.int32)
    window.sort()
    
    for i in range(win_size,len(y)-win_size):
        if np.all( y[window+i] > y[i] ):
            return i
    return -1
    
def autocorrelation_freq_detection_local(y, sr=10000):
    '''
    Find the peak in the autocorrelation of a time series to determine the 
    dominant frequency.
    
    Parameters
    --
    y : A 1D time series.
    
    sr : The sampling rate for the time series in samples per second (int).
        
    Returns
    --
    pitch : The dominant frequency (in Hz) of the time series
    '''
    
    # Compute autocorrelation of the signal
    corr = correlate(y, y, mode='full')
    corr = corr[len(corr) // 2:]

    # Find the peak in the autocorrelation after the global minimum
    # start_lag = np.argmin(corr)
    start_lag = find_first_min(corr)
    if start_lag < 0:
        raise ValueError("Frequency detector failed to find first local min in the data")
    peak_lag = np.argmax(corr[start_lag:]) + start_lag
    pitch = sr / peak_lag

    return pitch

def autocorrelation_freq_detection_global(y, sr=10000):
    '''
    Find the peak in the autocorrelation of a time series to determine the 
    dominant frequency.
    
    Parameters
    --
    y : A 1D time series.
    
    sr : The sampling rate for the time series in samples per second (int).
        
    Returns
    --
    pitch : The dominant frequency (in Hz) of the time series
    '''
    
    # Compute autocorrelation of the signal
    corr = correlate(y, y, mode='full')
    corr = corr[len(corr) // 2:]

    # Find the peak in the autocorrelation after the global minimum
    # start_lag = np.argmin(corr)
    start_lag = np.argmin(corr)
    peak_lag = np.argmax(corr[start_lag:])
    peak_lag = peak_lag + start_lag
    pitch = sr / peak_lag

    return pitch

# Extract frequency from a time series
def get_freq(series):
    r = series.copy()
    r = r[len(r)//6:]
    r -= np.mean(r)
    return autocorrelation_freq_detection_global(r, sr=10000)

# "Local" version of frequency extraction. Used for checking the coherency of oscillations
def get_freq_local(series):
    r = series.copy()
    r = r[len(r)//6:]
    r -= np.mean(r)
    return autocorrelation_freq_detection_local(r, sr=10000)

# Measure amplitude from a time series
def get_amp(series):
    r = series.copy()
    r = r[len(r)//6:]
    amp = np.max(r) - np.min(r)
    return amp

# Find phase of a time series, requires the dominant frequency f (usually extracted by get_freq())
def get_phase(series,f):
    sr = 10000
    r = series.copy()
    r = r[len(r)//6:]
    nu = 2.0*np.pi*f # angular frequency in hertz
    nu = nu / sr # convert to inverse timesteps
    FF = np.exp(-1.0j * nu * np.arange(len(r)))
    coef = np.dot(FF,r) # find fourier transform coefficient at frequency nu (only care about the argument)
    return np.angle(coef)

# Measure all quantities from a saved file
def get_freqs_amps_phases_fromfile(filename):
    with open(filename, "rb") as input:
        Spine = pickle.load(input)
    
    n_timesteps, n_segments, n_sides, n_populations = Spine.fullrates.shape
    freqs = np.zeros((n_segments,n_sides,n_populations))
    amps = np.zeros((n_segments,n_sides,n_populations))
    phases = np.zeros((n_segments,n_sides,n_populations))

    for seg in range(n_segments):
        for side in range(n_sides):
            for pop in range(n_populations):
                r = Spine.fullrates[:,seg,side,pop]
                fg = get_freq(r)
                freqs[seg,side,pop] = fg
                amps[seg,side,pop] = get_amp(r)
                phases[seg,side,pop] = get_phase(r,fg)
    return freqs, amps, phases

if __name__ == '__main__':
    import pickle
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-f','--filename',action='store',type=str,dest='filename',default='',help='Name of file to measure quantities from')
    args = parser.parse_args()
    filename = args.filename

    # Load in the data file
    FREQS, AMPS, PHASES = get_freqs_amps_phases_fromfile(filename)

    # Calculate amplitude-weighted average frequency
    FA = FREQS * AMPS
    FREQS_AMPAVG = np.sum(FA,axis=2)/np.sum(AMPS,axis=2)
    FREQS_AVG = np.mean(FREQS_AMPAVG)
    FREQS_STD = np.std(FREQS_AMPAVG)

    # Calculate average amplitude
    AMPS_AVG = np.mean(AMPS)
    AMPS_STD = np.std(AMPS)

    # Calculate amplitude-weighted average phase differences (LR = left-to-right, SEG = between nearest neighbour segments)
    PHDIF_SEG = np.angle(np.exp(-1.0j * (PHASES[1:,:,:] - PHASES[:-1,:,:])))
    PSEGA_SEG = PHDIF_SEG * AMPS[1:,:,:]
    PHDIF_SEG_AMPAVG = np.sum(PSEGA_SEG,axis=2) / np.sum(AMPS[1:,:,:],axis=2)
    PHDIF_SEG_AVG = circmean(PHDIF_SEG_AMPAVG) / (2.0 * np.pi) # normalize to 0-1
    PHDIF_SEG_STD = circstd(PHDIF_SEG_AMPAVG) / (2.0 * np.pi)

    PHDIF_LR = np.angle(np.exp(-1.0j * (PHASES[:,0,:] - PHASES[:,1,:])))
    PSEGA_LR = PHDIF_LR * AMPS[:,0,:]
    PHDIF_LR_AMPAVG = np.sum(PSEGA_LR,axis=1) / np.sum(AMPS[:,0,:],axis=1)
    PHDIF_LR_AVG = circmean(PHDIF_LR_AMPAVG) / (2.0 * np.pi) # normalize to 0-1
    PHDIF_LR_STD = circstd(PHDIF_LR_AMPAVG) / (2.0 * np.pi)

    print("Results:")
    print(f"Frequency: {FREQS_AVG} +/- {FREQS_STD} Hz")
    print(f"Amplitude: {AMPS_AVG} +/- {AMPS_STD}")
    print(f"Left-right phase difference: {PHDIF_LR_AVG} +/- {PHDIF_LR_STD}")
    print(f"Intersegmental phase difference: {PHDIF_SEG_AVG} +/- {PHDIF_SEG_STD}")
