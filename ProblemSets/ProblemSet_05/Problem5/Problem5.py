import numpy as np
import matplotlib.pyplot as plt
import matplotlib

N_x = 50
N_k = N_x*50
k_s = 3.2

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)

def analytic_prediction(k, k_s, N_x):
    i = complex(0, 1)
    n_1 = 1.0-np.exp(-2*np.pi*i*(k-k_s))
    d_1 = 1.0-np.exp(-2*np.pi*i*(k-k_s)/N_x)

    n_2 = 1.0-np.exp(-2*np.pi*i*(k+k_s))
    d_2 = 1.0-np.exp(-2*np.pi*i*(k+k_s)/N_x)

    return (n_1/d_1 - n_2/d_2)/(2.0*i)


def plot_unwindowed_FFT_analytic():
    x = np.arange(0, N_x, 1)
    y = np.sin(2*np.pi*k_s*x/N_x)
    y_fft = np.abs(np.fft.rfft(y, n=N_k))
    y_fft_freq = np.fft.rfftfreq(N_k)
    
    y_fft_full = np.concatenate((np.flip(y_fft), y_fft))
    y_fft_freq_full = np.concatenate((np.flip(-y_fft_freq), y_fft_freq))

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    plt.scatter(y_fft_freq_full*N_x, np.abs(analytic_prediction(y_fft_freq_full*N_x, k_s, N_x)),
                color="blue", s=1, label="Analytic")
    plt.scatter(y_fft_freq_full*N_x, y_fft_full, color="orange", s=1, label="FFT")
    plt.xlabel("k-frequency")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig("DFT_Analyt_v_FFT")
    plt.cla()
    plt.clf()
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    plt.scatter(y_fft_freq_full*N_x, np.abs(y_fft_full-np.abs(analytic_prediction(y_fft_freq_full*N_x, k_s, N_x))),
                                color="orange", s=1)
    plt.xlabel("k-frequency")
    plt.ylabel("Residuals")
    plt.tight_layout()
    plt.savefig("DFT_Analyt_v_FFT_residuals")
    plt.cla()
    plt.clf()
    plt.close()
    
### With the window ###

def plot_windowed_FFT():
    x = np.arange(0, N_x, 1)
    y = np.sin(2*np.pi*k_s*x/N_x)

    y_window = 0.5*(1-np.cos(2*np.pi*x/N_x))
    y *= y_window

    y_fft = np.abs(np.fft.rfft(y, n=N_k))
    y_fft_freq = np.fft.rfftfreq(N_k)

    y_fft_full = np.concatenate((np.flip(y_fft), y_fft))
    y_fft_freq_full = np.concatenate((np.flip(-y_fft_freq), y_fft_freq))

    y_fft_freq_full_win = y_fft_freq_full
    y_fft_full_win = y_fft_full

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    plt.scatter(y_fft_freq_full*N_x, np.abs(analytic_prediction(y_fft_freq_full*N_x, k_s, N_x)),
                color="blue", s=1, label="unwindowed analytical")
    plt.scatter(y_fft_freq_full*N_x, 2*y_fft_full, color="orange", s=1, label="windowed fft")
    plt.xlabel("k-frequency")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig("DFFT_with_window")
    plt.cla()
    plt.clf()
    plt.close()

   
### Analytical Windowed ###

def plot_windowed_analytic():
    x = np.arange(0, N_x, 1)
    y = np.sin(2*np.pi*k_s*x/N_x)  
    
    y_window = 0.5*(1-np.cos(2*np.pi*x/N_x))
    y *= y_window
    
    y_fft = np.abs(np.fft.rfft(y, n=N_k))
    y_fft_freq = np.fft.rfftfreq(N_k)

    y_fft_full = np.concatenate((np.flip(y_fft), y_fft))
    y_fft_freq_full = np.concatenate((np.flip(-y_fft_freq), y_fft_freq))
    
    k = y_fft_freq_full*N_x

    y_dft_window = 2*(0.5*analytic_prediction(-k, k_s, N_x)
               - 0.25*analytic_prediction(-k-1, k_s, N_x)
               - 0.25*analytic_prediction(-k+1, k_s, N_x))

    y_dft_no_window = np.abs(analytic_prediction(y_fft_freq_full*N_x, k_s, N_x))

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    plt.scatter(k, y_dft_no_window, color="blue", s=1, label="Analytical unwindowed")
    plt.plot(k, np.abs(y_dft_window), color="orange", label="Analytical windowed")
    plt.xlabel("k-frequency")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig("DFT_window_analytic")
    plt.cla()
    plt.clf()
    plt.close()


plot_unwindowed_FFT_analytic()
plot_windowed_FFT()
plot_windowed_analytic()

