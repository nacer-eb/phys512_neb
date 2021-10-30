import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import matplotlib

# Define a bigger font for the next plots
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)

# Define the number of x points and frequency samples to use
N = 10000
N_k = N*100

# Obtain a random walk and it's autocorrelation
rnd_walk = np.cumsum(np.random.randn(4*N))[N:-N]
rnd_walk_corr = np.correlate(rnd_walk, rnd_walk, mode='same')

# Apply a hanning window to the random walk correlation
y = np.hanning(2*N)*rnd_walk_corr

# Obtain the FFT for the rw autocorrelation and the frequencies used
y_fft = np.abs(np.fft.rfft(y, n=N_k)[1:int(N_k/100)])
y_fft_freq = np.fft.rfftfreq(N_k)[1:int(N_k/100)]

# Plot the windowed rw autocorrelation (to show it's roughly abs(x))
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
plt.plot(y)
plt.title("Windowed Correlation")
plt.savefig("windowed_correlation")

# Plot the power spectra - FFT of the rw autocorr.
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
plt.plot(y_fft_freq, y_fft, color="orange")
plt.title("Fourier transform of Windowed-Correlation")
plt.xlabel("$k^2$ Frequency")
plt.ylabel("Fourier amplitude")
plt.savefig("rw_ps")


# Plot the power spectra * k² to show that is constant (roughly)
# This implies the power spectra goes like k^-2
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
plt.plot(y_fft_freq, y_fft*y_fft_freq**2, color="orange")

mean_coef = np.mean(y_fft*y_fft_freq**2)
plt.plot([np.min(y_fft_freq), np.max(y_fft_freq)],
         [mean_coef, mean_coef], color="red")

plt.ylabel("Fourier transform times $k^2$")
plt.xlabel("Frequency")
plt.savefig("fft_fourier_ksqrd_analysis")
