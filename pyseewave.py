import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.signal import get_window, spectrogram
from scipy.fftpack import fft, ifft
from scipy.stats import skew, kurtosis

def afilter(wave, f, channel=1, threshold=5, plot=True):
    """
    Apply an amplitude threshold filter to a signal.

    Parameters:
        wave (array): Input signal (1D or 2D array).
        f (int): Sampling frequency.
        channel (int): Channel to process (default: 1).
        threshold (float): Amplitude threshold as a percentage of the maximum amplitude (default: 5).
        plot (bool): Whether to plot the original and filtered signals (default: True).
    
    Returns:
        array: Filtered signal.
    """
    # Ensure the input is a numpy array
    wave = np.array(wave)

    # Handle multi-channel signals
    if wave.ndim == 2:
        wave = wave[:, channel - 1]  # Select the specified channel (0-based index)

    # Calculate the threshold value
    t1 = np.max(np.abs(wave)) * (threshold / 100)

    # Apply the threshold filter
    filtered_wave = np.where(np.abs(wave) <= t1, 0, wave)

    # Plot the original and filtered signals
    if plot:
      plt.figure(figsize=(12, 6))

      # Plot original signal
      plt.subplot(2, 1, 1)
      plt.plot(wave, color='blue')
      plt.title("Original Signal")
      plt.xlabel("Sample")
      plt.ylabel("Amplitude")

      # Plot filtered signal
      plt.subplot(2, 1, 2)
      plt.plot(filtered_wave, color='red')
      plt.title(f"Filtered Signal (Threshold = {threshold}%)")
      plt.xlabel("Sample")
      plt.ylabel("Amplitude")

      plt.tight_layout()
      plt.show()

    return filtered_wave

def dBweight(f, dBref=None):
  # dBA
  num = (12200**2 * f**4)
  den = (f**2 + 20.6**2) * np.sqrt((f**2 + 107.7**2) * (f**2 + 737.9**2))*(f**2 + 12200**2)
  A = 2 + 20*np.log10(num/den)
  A = np.where(np.isinf(A), np.nan, A)

  # dBB
  num = (12200**2*f**3)
  den = (f**2 + 20.6**2)*np.sqrt((f**2 + 158.5**2))*(f**2 + 12200**2)
  B = 0.17 + 20*np.log10(num/den)
  B = np.where(np.isinf(B), np.nan, B)

  # dbC
  num = (12200**2*f**2)
  den = (f**2 + 20.6**2)*np.sqrt(f**2 + 12200**2)
  C = 0.06 + 20*np.log10(num/den)
  C = np.where(np.isinf(C), np.nan, C)

    # dBD
  a = f/(6.8966888496476*10**-5);
  h = ((1037918.48 - f**2)**2 + (1080768.16*f**2))/((9837328 - f**2)**2 + (11723776*f**2))
  b = np.sqrt(h/((f**2 + 79919.29)*(f**2 + 1345600)))
  D = 20*np.log10(a*b)
  D = np.where(np.isinf(D), np.nan, D)

  h1 = -4.737338981378384 * 10**(-24) * f**6 + \
      2.04382833606125 * 10**(-15) * f**4 - \
      1.363894795463638 * 10**(-7) * f**2 + 1
  h2 = 1.306612257412824 * 10**(-19) * f**5 - \
      2.118150887518656 * 10**(-11) * f**3 + \
      5.559488023498642 * 10**(-4) * f
  R_ITU = 1.246332637532143 * 10**(-4) * f / np.sqrt(h1**2 + h2**2)
  ITU = 18.2 + 20 * np.log10(R_ITU)

  result = {"A": A, "B": B, "C": C, "D": D, "ITU": ITU}
  if dBref is not None:
    result = {key: value + dBref for key, value in result.items()}
  return result

from scipy.signal import get_window, spectrogram

def dfreq(wave, f, channel=1, wl=512, wn="hann", ovlp=0, fftw=False, at=None,
          tlim=None, threshold=None, bandpass=None, clip=None, plot=True,
          xlab="Time (s)", ylab="Frequency (kHz)", ylim=None, **kwargs):
    """
    Compute the dominant frequency of a signal over time using the Short-Time Fourier Transform (STFT).

    Parameters:
        wave (np.array): Input signal.
        f (int): Sampling frequency.
        channel (int): Channel to process (default: 1).
        wl (int): Window length (default: 512).
        wn (str): Window type (default: "hann").
        ovlp (int): Overlap percentage between windows (default: 0).
        fftw (bool): Whether to use FFTW for FFT computation (default: False).
        at (float or list): Specific time(s) to analyze (default: None).
        tlim (tuple): Time limits for segment selection (default: None).
        threshold (float): Amplitude threshold for filtering (default: None).
        bandpass (tuple): Frequency bandpass limits (default: None).
        clip (float): Clip value for discarding low-amplitude peaks (default: None).
        plot (bool): Whether to plot the results (default: True).
        xlab (str): Label for the x-axis (default: "Time (s)").
        ylab (str): Label for the y-axis (default: "Frequency (kHz)").
        ylim (tuple): Limits for the y-axis (default: (0, f/2000)).
        **kwargs: Additional arguments for plotting.

    Returns:
        np.array: Time and frequency data.
    """
    # Error messages
    if at is not None and ovlp != 0:
        raise ValueError("The 'ovlp' argument cannot be used in conjunction with the argument 'at'.")
    if clip is not None and (clip <= 0 or clip >= 1):
        raise ValueError("'clip' value has to be greater than 0 and less than 1.")

    # Input handling
    if wave.ndim > 1:
        wave = wave[:, channel - 1]  # Select the specified channel

    # Time limits
    if tlim is not None:
        start = int(tlim[0] * f)
        end = int(tlim[1] * f)
        wave = wave[start:end]

    # Amplitude threshold
    if threshold is not None:
        wave = np.where(np.abs(wave) < threshold, 0, wave)

    # Position(s)
    n = len(wave)
    if at is not None:
        step = np.round(np.array(at) * f).astype(int)
        N = len(step)
        step[step < 0] = 0
        step[step + wl // 2 >= n] = n - wl
        x = np.concatenate(([0], at, [n / f]))
    else:
        step = np.arange(0, n - wl + 1, wl - (ovlp * wl // 100))
        N = len(step)
        x = np.linspace(0, n / f, N)

    # STFT (Short-Time Fourier Transform)
    _, _, y1 = spectrogram(wave, fs=f, window=get_window(wn, wl), nperseg=wl,
                           noverlap=wl - (wl - (ovlp * wl // 100)), scaling="spectrum")

    # Bandpass filter
    if bandpass is not None:
        if len(bandpass) != 2:
            raise ValueError("The argument 'bandpass' should be a numeric vector of length 2.")
        if bandpass[0] >= bandpass[1]:
            raise ValueError("The first element of 'bandpass' has to be less than the second element.")
        lowlimit = int((wl * bandpass[0]) / f)
        upperlimit = int((wl * bandpass[1]) / f)
        y1[:lowlimit, :] = 0
        y1[upperlimit:, :] = 0

    # Maximum search
    maxi = np.max(y1, axis=0)
    y2 = np.argmax(y1, axis=0)
    y2[maxi == 0] = np.nan

    # Discard peaks with amplitude lower than the clip value
    if clip is not None:
        y2[maxi < clip] = np.nan

    # Convert to frequency
    y = (f * y2) / (1000 * wl)

    if at is not None:
        y = np.concatenate(([np.nan], y, [np.nan]))

    # Plot
    if plot:
        plt.figure()
        plt.plot(x, y, **kwargs)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        if ylim is not None:
            plt.ylim(ylim)
        plt.show()

    return np.column_stack((x, y))

def fund(wave, f, channel=1, wl=512, ovlp=0, fmax=None, threshold=None, at=None,
         from_=None, to=None, plot=True, xlab="Time (s)", ylab="Frequency (kHz)",
         ylim=None, pb=False, **kwargs):
    """
    Estimate the fundamental frequency of a signal over time using the cepstrum method.

    Parameters:
        wave (np.array): Input signal.
        f (int): Sampling frequency.
        channel (int): Channel to process (default: 1).
        wl (int): Window length (default: 512).
        ovlp (int): Overlap percentage between windows (default: 0).
        fmax (float): Maximum frequency to consider (default: f/2).
        threshold (float): Amplitude threshold for filtering (default: None).
        at (float): Specific time to analyze (default: None).
        from_ (float): Start time for segment selection (default: None).
        to (float): End time for segment selection (default: None).
        plot (bool): Whether to plot the results (default: True).
        xlab (str): Label for the x-axis (default: "Time (s)").
        ylab (str): Label for the y-axis (default: "Frequency (kHz)").
        ylim (tuple): Limits for the y-axis (default: (0, f/2000)).
        pb (bool): Whether to show a progress bar (default: False).
        **kwargs: Additional arguments for plotting.

    Returns:
        np.array: Time and frequency data.
    """
    # Error Handling
    if at is not None:
        if threshold is not None:
            raise ValueError("The 'threshold' argument cannot be used with the argument 'at'.")
        if ovlp != 0:
            raise ValueError("The 'overlap' argument should be 0 when using the argument 'at'.")
        if from_ is not None or to is not None:
            raise ValueError("The 'from_' and/or 'to' arguments cannot be used with 'at'.")
        if pb:
            raise ValueError("No progress bar can be displayed when using the argument 'at'.")
        if plot:
            plot = False
            print("When the argument 'at' is used, the argument 'plot' is automatically turned to 'False'.")

    # Set fmax if not provided
    if fmax is None:
        fmax = f / 2

    # Input handling (assuming wave is already a numpy array)
    if len(wave.shape) > 1:
        wave = wave[:, channel - 1]  # Select the specified channel
    WL = wl // 2

    # FROM-TO SELECTION
    if from_ is not None or to is not None:
        a = int(from_ * f) if from_ is not None else 0
        b = int(to * f) if to is not None else len(wave)
        wave = wave[a:b]

    # AT SELECTION
    if at is not None:
        c = int(at * f)
        wave = wave[c - WL:c + WL]

    # THRESHOLD
    if threshold is not None:
        wave = np.where(np.abs(wave) < threshold, 1e-6, wave)

    # SLIDING WINDOW
    wave = np.where(wave == 0, 1e-6, wave)  # Avoid log(0)
    n = len(wave)
    step = np.arange(0, n - wl + 1, wl - (ovlp * wl // 100))
    N = len(step)
    z1 = np.zeros((wl, N))

    # CEPSTRUM CALCULATION
    for i, start in enumerate(step):
        segment = wave[start:start + wl]
        spectrum = np.abs(fft(segment))
        cepstrum = np.real(ifft(np.log(spectrum)))
        z1[:, i] = cepstrum

    # FUNDAMENTAL FREQUENCY TRACKING
    z2 = z1[:WL, :]
    z = np.where(np.isnan(z2) | np.isinf(z2), 0, z2)
    fmaxi = int(f // fmax)
    tfund = np.zeros(N)
    for k in range(N):
        tfund[k] = np.argmax(z[fmaxi:, k]) + fmaxi
    tfund = np.where(tfund == 0, np.nan, tfund)
    ffund = f / (tfund + fmaxi - 1)
    x = at if at is not None else np.linspace(0, n / f, N)
    y = ffund / 1000
    res = np.column_stack((x, y))

    # PLOT
    if plot:
        plt.figure()
        plt.plot(x, y, **kwargs)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        if ylim is not None:
            plt.ylim(ylim)
        plt.show()

    return res

def sfm(spec):
  spec = np.array(spec)

  if np.ndim(spec) == 2:
    spec = spec[1]

  if np.any(spec < 0):
    raise ValueError("Data do not have to be in dB")

  if np.sum(spec) == 0:
    return np.nan

  if len(spec) > 4000:
    step = np.linspace(1, len(spec) - 1, 256, dtype=int)
    spec = spec[step]

  spec = np.where(spec == 0, 1e-5, spec)

  n = len(spec)
  geo = np.prod(spec**(1/n))
  ari = np.mean(spec)

  flat = geo/ari
  return flat

def sh(spec, alpha='shannon'):
  spec = np.mean(spec, axis=0) if np.ndim(spec) == 2 else spec

  if np.any(np.isnan(spec)):
    return np.nan
  else:
    if np.any(spec<0):
      raise ValueError("Data do not have to be in dB")
    if np.sum(spec) == 0:
      warnings.warn("Caution! This is a null spectrum. The spectral entropy is null!")
      z = 0
    else:
      N = len(spec)
      spec = np.where(spec==0, 1e-7, spec)
      spec = spec / np.sum(spec) # PMF

      if alpha == 'shannon':
        z = -np.sum(spec * np.log(spec)) / np.log(N)

      elif alpha == 'simpson':
        z = 1 - np.sum(spec ** 2)

      else:

        if alpha < 0:
          raise ValueError("'alpha' cannot be negative")
        if alpha == 1:
          raise ValueError("'alpha' cannot be set to 1.")

        z = (1/(1-alpha)) * np.log(np.sum(spec**alpha))/np.log(N)

  return z

def spec(wave, f, channel=1, wl=512, wn="hanning", norm=True, scaled=False,
         PSD=False, PMF=False, correction="none", dB=None, dBref=None,
         at=None, start=None, end=None, identify=False, col="black",
         cex=1, plot=1, flab="Frequency (kHz)", alab="Amplitude", flim=None, alim=None, type="1"):
  
  # Error Handling
  if not norm and PMF:
    raise Exception("'PMF' can be computed only if 'norm' is True") 
  if not norm and dB is not None:
    raise Exception("dB are computed on normalised spectra only, 'norm' should be turned to True")
  if norm and scaled:  
    raise Exception("'norm' and 'scaled' cannot be both set to True")
  if scaled and PMF:
    raise Exception("'scaled' and 'PMF' cannot be both set to True")
  if dB is not None and PMF:   
    raise Exception("PMF cannot be in dB")
  if dB is None and dBref is not None: 
    raise Exception("'dB' cannot be None when 'dBref' is not None") 
  if dB is not None and dB not in ['max0', 'A', 'B', 'C', 'D']:
    raise Exception("'dB' must be one of: 'max0', 'A', 'B', 'C', or 'D'")

  # From - To Selection
  if start is not None or end is not None:
    a = 0 if start is None else int(round(start * f))
    b = len(wave) if end is None else int(round(end * f))
    if start is not None and end is not None and start > end:
      raise ValueError("'start' cannot be greater than 'end'")
    
    wave = np.array(wave[a:b])

  # At Selection
  if at is not None:
    c = int(round(at * f))
    wl2 = wl // 2
    wave = np.array(wave[c - wl2:c + wl2])

  # FFT
  n = len(wave)
  W = get_window(wn, n)  # Apply windowing function
  wave = wave * W

  y = np.abs(np.fft.fft(wave))  # Fixed FFT computation
  if scaled: 
    y = y / len(y)

  y = 2 * y[:n // 2]

  # PSD, NORM, PMF, SCALED OPTIONS
  if PSD:
    y = y ** 2
  if norm:
    y = y / np.max(y)
  if PMF:
    y = y / np.sum(y)

  # FREQUENCY DATA
  x = (np.arange(n) * f / n / 1000)[:n // 2]

  # DB
  if dB is not None:
    y = np.where(y == 0, 1e-6, y)
    if dBref is None:
      y = 20 * np.log10(y)
    else:
      y = 20 * np.log10(y / dBref)
    if dB != 'max0':
      if dB == 'A':
        y = dBweight(x * 1000, dBref=y)['A']
      elif dB == 'B':
        y = dBweight(x * 1000, dBref=y)['B']
      elif dB == 'C':
        y = dBweight(x * 1000, dBref=y)['C']
      elif dB == 'D':
        y = dBweight(x * 1000, dBref=y)["D"]

  # Limits of the Amplitude Axis
  if alim is None:
    if dB is None:
      alim = (0, 1.1)
    else:
      alim = (np.nanmin(y), np.nanmax(y) + 20)
    if PMF or not norm:
      alim = (0, np.nanmax(y))

  # Plotting
  if plot:
    plt.plot(x, y)
    if flim is not None:
      plt.xlim(flim)
    plt.ylim(alim)
    plt.xlabel(flab)
    plt.ylabel(alab)
    plt.show()
  else:
    return x, y

def specprop(spec, f=None, str_output=False, flim=None, mel=False, plot=False, plot_type="l",
             xlab=None, ylab=None, col_mode=2, col_quartiles=4, **kwargs):
    """
    Compute spectral properties of a frequency spectrum.

    Parameters:
        spec (np.array or list): Frequency spectrum (amplitude or power).
        f (float): Sampling frequency (default: None).
        str_output (bool): Whether to print results as a string (default: False).
        flim (tuple): Frequency limits for analysis (default: None).
        mel (bool): Whether to convert frequencies to the mel scale (default: False).
        plot (bool or int): Whether to plot the results (0: no plot, 1: probability plot, 2: cumulative plot).
        plot_type (str): Plot type ("l" for line, etc.) (default: "l").
        xlab (str): Label for the x-axis (default: None).
        ylab (str): Label for the y-axis (default: None).
        col_mode (int): Color for the mode line (default: 2).
        col_quartiles (int): Color for the quartile lines (default: 4).
        **kwargs: Additional arguments for plotting.

    Returns:
        dict: Spectral properties.
    """
    # Input handling
    if isinstance(spec, np.ndarray) and spec.ndim == 2:
        freq = spec[:, 0] * 1000  # Convert kHz to Hz
        spec = spec[:, 1]
    else:
        freq = np.linspace(0, f / 2, len(spec))  # Frequency axis in Hz

    if mel:
        from scipy.signal import mel_scale
        freq = mel_scale(freq / 1000) * 1000  # Convert Hz to mel scale

    # Frequency limits
    if flim is None:
        flim = (0, f / 2 / 1000)  # Convert Hz to kHz
    else:
        if flim[0] < 0 or flim[1] > f / 2 / 1000:
            raise ValueError("'flim' should range between 0 and f/2")

    # Filter spectrum within frequency limits
    mask = (freq >= flim[0] * 1000) & (freq <= flim[1] * 1000)
    freq = freq[mask]
    spec = spec[mask]
    L = len(spec)

    # Normalize amplitude
    amp = spec / np.sum(spec)
    cumamp = np.cumsum(amp)

    # Spectral properties
    mean = np.sum(amp * freq)
    sd = np.sqrt(np.sum(amp * (freq - mean) ** 2))
    sem = sd / np.sqrt(L)
    median = freq[np.argmax(cumamp >= 0.5)]
    mode = freq[np.argmax(amp)]
    Q25 = freq[np.argmax(cumamp >= 0.25)]
    Q75 = freq[np.argmax(cumamp >= 0.75)]
    IQR = Q75 - Q25
    cent = np.sum(freq * amp)
    z = amp - np.mean(amp)
    w = np.std(amp)
    skewness = skew(amp)
    kurt = kurtosis(amp)
    flatness = sfm(amp)  # Spectral flatness
    entropy = sh(amp)  # Spectral entropy
    prec = f / (2 * L)  # Frequency precision

    # Results
    results = {
        "mean": mean,
        "sd": sd,
        "median": median,
        "sem": sem,
        "mode": mode,
        "Q25": Q25,
        "Q75": Q75,
        "IQR": IQR,
        "cent": cent,
        "skewness": skewness,
        "kurtosis": kurt,
        "sfm": flatness,
        "sh": entropy,
        "prec": prec,
    }

    if str_output:
        print(results)

    # Plotting
    if plot == 1:
        if xlab is None:
            xlab = "Frequency (kmel)" if mel else "Frequency (kHz)"
        if ylab is None:
            ylab = "Probability"
        plt.figure()
        plt.plot(freq / 1000, amp, plot_type, **kwargs)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.axvline(mode / 1000, color=f"C{col_mode}", label="Mode")
        plt.axvline(median / 1000, color=f"C{col_quartiles}", label="Median")
        plt.axvline(Q25 / 1000, color=f"C{col_quartiles}", linestyle="--", label="Q25")
        plt.axvline(Q75 / 1000, color=f"C{col_quartiles}", linestyle=":", label="Q75")
        plt.legend()
        plt.show()
    elif plot == 2:
        if xlab is None:
            xlab = "Frequency (kmel)" if mel else "Frequency (kHz)"
        if ylab is None:
            ylab = "Cumulated Probability"
        plt.figure()
        plt.plot(freq / 1000, cumamp, plot_type, **kwargs)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.axvline(mode / 1000, color=f"C{col_mode}", label="Mode")
        plt.axhline(cumamp[np.argmax(freq == mode)], color=f"C{col_mode}")
        plt.axvline(median / 1000, color=f"C{col_quartiles}", label="Median")
        plt.axhline(0.5, color=f"C{col_quartiles}")
        plt.axvline(Q25 / 1000, color=f"C{col_quartiles}", linestyle="--", label="Q25")
        plt.axhline(0.25, color=f"C{col_quartiles}", linestyle="--")
        plt.axvline(Q75 / 1000, color=f"C{col_quartiles}", linestyle=":", label="Q75")
        plt.axhline(0.75, color=f"C{col_quartiles}", linestyle=":")
        plt.legend()
        plt.show()

    return results
