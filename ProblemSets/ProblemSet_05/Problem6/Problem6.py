import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import matplotlib

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)

N = 10000
N_k = N*100

rnd_walk = np.cumsum(np.random.randn(4*N))[N:-N]
rnd_walk_corr = np.correlate(rnd_walk, rnd_walk, mode='same')

y = np.hanning(2*N)*rnd_walk_corr

y_fft = np.abs(np.fft.rfft(y, n=N_k)[1:int(N_k/100)])
y_fft_freq = np.fft.rfftfreq(N_k)[1:int(N_k/100)]

fig, ax = plt.subplots(1, 1, figsize=(16, 9))
plt.plot(y)
plt.title("Windowed Correlation")
plt.savefig("windowed_correlation")

fig, ax = plt.subplots(1, 1, figsize=(16, 9))
plt.plot(y_fft_freq, y_fft, color="orange")
plt.title("Fourier transform of Windowed-Correlation")
plt.xlabel("$k^2$ Frequency")
plt.ylabel("Fourier amplitude")
plt.savefig("rw_ps")


mean_coef = np.mean(y_fft*y_fft_freq**2)

fig, ax = plt.subplots(1, 1, figsize=(16, 9))
plt.plot(y_fft_freq, y_fft*y_fft_freq**2, color="orange")
plt.plot([np.min(y_fft_freq), np.max(y_fft_freq)],
         [mean_coef, mean_coef], color="red")
plt.ylabel("Fourier transform times $k^2$")
plt.xlabel("Frequency")
plt.savefig("fft_fourier_ksqrd_analysis")
