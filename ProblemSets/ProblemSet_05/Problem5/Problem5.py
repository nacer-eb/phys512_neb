import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Declare the necessary parameters
# (Number of x points, frequency samples and the sinusoid's frequency)
N_x = 50
N_k = N_x*50
k_s = 3.2

# Define a bigger font for the plots
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}

# Apply the custom font size as default
matplotlib.rc('font', **font)

def analytic_prediction(k, k_s, N_x):
    """
    Computes the analytic formula for the DFT of sin(2pik_s x/N)
    (As derived in the pdf)
    
    :param k: The range of frequencies to compute
    :param k_s: The frequency of the sine wave
    :param N_x: The number of points we have for the sine wave

    :return: The Discrete Fourier Transform (DFT) of a pure sine.
    """
    i = complex(0, 1)
    n_1 = 1.0-np.exp(-2*np.pi*i*(k-k_s))
    d_1 = 1.0-np.exp(-2*np.pi*i*(k-k_s)/N_x)

    n_2 = 1.0-np.exp(-2*np.pi*i*(k+k_s))
    d_2 = 1.0-np.exp(-2*np.pi*i*(k+k_s)/N_x)

    return (n_1/d_1 - n_2/d_2)/(2.0*i)


def plot_unwindowed_FFT_analytic():
    # Obtain the y (sine) values for a discrete x range
    x = np.arange(0, N_x, 1)
    y = np.sin(2*np.pi*k_s*x/N_x)

    # Compute the (fast) fourier transform (This gives N_k data)
    y_fft = np.abs(np.fft.rfft(y, n=N_k))

    # Get the frequencies our fft has values for
    y_fft_freq = np.fft.rfftfreq(N_k)

    # Mirror our data (Nyquist) from -N to N (N = N_k/2)
    y_fft_full = np.concatenate((np.flip(y_fft), y_fft))
    y_fft_freq_full = np.concatenate((np.flip(-y_fft_freq), y_fft_freq))

    # Plot the analytical prediction and the FFT curve
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

    # Plot the residuals (analytic - FFT)
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    plt.scatter(y_fft_freq_full*N_x,
                np.abs(y_fft_full-np.abs(analytic_prediction(y_fft_freq_full*N_x, k_s, N_x))),
                color="orange", s=1)
    plt.xlabel("k-frequency")
    plt.ylabel("Residuals")
    plt.tight_layout()
    plt.savefig("DFT_Analyt_v_FFT_residuals")
    plt.cla()
    plt.clf()
    plt.close()

    
def plot_windowed_FFT():
    # Obtain the y (sine) values for a discrete x range
    x = np.arange(0, N_x, 1)
    y = np.sin(2*np.pi*k_s*x/N_x)

    # Define and apply the window function
    y_window = 0.5*(1-np.cos(2*np.pi*x/N_x))
    y *= y_window

    # Compute the fft 
    y_fft = np.abs(np.fft.rfft(y, n=N_k))
    y_fft_freq = np.fft.rfftfreq(N_k)

    # Mirror the fft about zero
    y_fft_full = np.concatenate((np.flip(y_fft), y_fft))
    y_fft_freq_full = np.concatenate((np.flip(-y_fft_freq), y_fft_freq))

    #Plot the (windowed) FFT 
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

   
def plot_windowed_analytic():
    # Obtain the y (sine) values for a discrete x range
    x = np.arange(0, N_x, 1)
    y = np.sin(2*np.pi*k_s*x/N_x)  

    # Define and apply the window function
    y_window = 0.5*(1-np.cos(2*np.pi*x/N_x))
    y *= y_window

    # Compute the fft 
    y_fft = np.abs(np.fft.rfft(y, n=N_k))
    y_fft_freq = np.fft.rfftfreq(N_k)

    # Mirror the fft
    y_fft_full = np.concatenate((np.flip(y_fft), y_fft))
    y_fft_freq_full = np.concatenate((np.flip(-y_fft_freq), y_fft_freq))

    # Define a shorthand variable for the frequency
    k = y_fft_freq_full*N_x

    # Combine neighbour points of the analytical prediction
    # Predict the 'analytical' effect of windowing (see pdf for derivation)
    y_dft_window = 2*(0.5*analytic_prediction(-k, k_s, N_x)
               - 0.25*analytic_prediction(-k-1, k_s, N_x)
               - 0.25*analytic_prediction(-k+1, k_s, N_x))

    # The unwindowed version for comparison/debugging
    y_dft_no_window = np.abs(analytic_prediction(y_fft_freq_full*N_x, k_s, N_x))

    # Plot the windowed and unwindowed analytical predictions
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


# Answers to part c, d and e
plot_unwindowed_FFT_analytic()
plot_windowed_FFT()
plot_windowed_analytic()

