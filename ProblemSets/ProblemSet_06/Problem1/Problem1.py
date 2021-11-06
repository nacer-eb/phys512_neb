import numpy as np
import matplotlib.pyplot as plt
import h5py
import json
import scipy.ndimage
import scipy.signal
import matplotlib
import warnings

# Ignores matplotlib warnings
warnings.filterwarnings("ignore")

# Number of data points we are supplied per strain
DATA_LENGTH = 4096*32

# Changing font size
matplotlib.rcParams.update({'font.size': 22})

# IMPORTANT: DIRECTORY TO THE LIGO DATA
dir = "../../../../LOSC_Event_tutorial/LOSC_Event_tutorial/"

# The names of the events and detectors.
bbh_events_names = ["GW150914", "LVT151012", "GW151226", "GW170104"]
bbh_detectors_names = ["Livingston", "Hanford"]

# Obtaining the event info json file and loading it
bbh_events_info_file = open(dir+"BBH_events_v3.json")
bbh_events_info = json.load(bbh_events_info_file)
bbh_events_info_file.close()


def getTemplates(event_index):
    """
    Obtains the templates data.

    :param event_index: The event number [0-4]

    :return: Returns the templates of the Livingston and Hanford detectors 
             for the specified event (in one array)
    """
    bbh_event = bbh_events_info[bbh_events_names[event_index]]
    bbh_event_template_filename = bbh_event["fn_template"]
    bbh_event_template_file = h5py.File(dir + bbh_event_template_filename, 'r')
    bbh_event_templates = bbh_event_template_file['template']

    return bbh_event_templates


def getStrains(event_index):
    """
    Obtains the strain data.

    :param event_index: The event number [0-4]

    :return: Returns the strains of the Livingston and Hanford detectors 
             for the specified event. (in one tuple)
    """
    bbh_event = bbh_events_info[bbh_events_names[event_index]]
    bbh_event_data_l_filename = bbh_event["fn_L1"]
    bbh_event_data_l_file = h5py.File(dir + bbh_event_data_l_filename, 'r')
    bbh_event_data_l = bbh_event_data_l_file['strain']['Strain'][:]

    bbh_event_data_h_filename = bbh_event["fn_H1"]
    bbh_event_data_h_file = h5py.File(dir + bbh_event_data_h_filename, 'r')
    bbh_event_data_h = bbh_event_data_h_file['strain']['Strain'][:]

    return bbh_event_data_l, bbh_event_data_h


def flat_window(N_total, flat_proportion):
    """
    Calculates the flat windowing function. 
    Creates a hanning and adds a flat part in the middle. Hanning start - Flat - Hanning end

    :param N_total: The total output length
    :param flat_proportion: The proportion of the window which must be flat (centered)

    :return: Returns the y-values for the flat window instance
    """
    N_flat = int(N_total*flat_proportion)
    N_hanning = N_total - N_flat
    hanning_part = np.hanning(N_hanning)
    
    flat_window = np.ones(N_total)
    flat_window[:N_hanning//2] = hanning_part[0:N_hanning//2]
    flat_window[-N_hanning//2:] = hanning_part[-N_hanning//2:]

    return flat_window


# Calculates the RMS of an array
def calcRms(x):
    return np.sqrt(np.mean(np.square(x)))


# Smoothen data by averaging over a range multiple times. (Like in class but the range can be modified)
def smoothen(x, smoothing_steps, average_range):
    for s in range(0, smoothing_steps):
        x_tmp = np.copy(x)
        for t in range(1, average_range+1):
            x_tmp += np.roll(x, (t)) + np.roll(x, -(t))
        x = x_tmp/(2.0*t+1)
    return x


def getWindowedFFTTemplates(e):
    """
    Obtains the templates data applies a window and a rFFT.

    :param e: The event number [0-4]

    :return: Returns the windowed FFT templates of the Livingston and Hanford detectors 
             for the specified event (in one array)
    """
    bbh_event_templates = getTemplates(e)
    
    template_window = flat_window(len(bbh_event_templates[0]), 0.95)
    template_window /= calcRms(template_window) # Renormalize the template (In this case the normalization constant is practically 1)
    bbh_event_templates_windowed = bbh_event_templates*template_window

    bbh_event_templates_windowed_fft = np.fft.rfft(bbh_event_templates_windowed)
    return bbh_event_templates_windowed_fft


def getWindowedFFTData(e):
    """
    Obtains the strain data applies a window and a rFFT.

    :param e: The event number [0-4]

    :return: Returns the windowed FFT strains of the Livingston and Hanford detectors 
             for the specified event (in one array)
    """
    bbh_event_data_lh = getStrains(e)
    
    data_window = flat_window(len(bbh_event_data_lh[0]), 0.95)
    data_window /= calcRms(data_window) # Renormalize the template (In this case the normalization constant is practically 1)
    bbh_event_data_lh_windowed = bbh_event_data_lh*data_window

    bbh_event_data_lh_windowed_fft = np.fft.rfft(bbh_event_data_lh_windowed)
    return bbh_event_data_lh_windowed_fft
    
    
def getNoiseModel():
    """
    Obtains the noise model.

    :return: Returns the smooth noise model for the Livingston and Hanford detectors.
    """
    # Create an array to hold the model
    noise_model_complete_fft = np.zeros((2,1))

    # Average PSD of the four events (keeping detectors seperate)
    for e in range(4):
        bbh_event_data_lh_windowed_fft = getWindowedFFTData(e)

        if e == 0:
            noise_model_complete_fft = np.abs(bbh_event_data_lh_windowed_fft)**2
        else:
            noise_model_complete_fft += np.abs(bbh_event_data_lh_windowed_fft)**2

    noise_model_complete_fft /= (e+1)

    # Then smoothen the noise model by doing multiple local averages
    noise_model_complete_fft_smooth = smoothen(noise_model_complete_fft, 12, 2)
    
    return noise_model_complete_fft_smooth
    

def searchEvent(e, noise_model_smooth_fft):
    """
    Search for an event using the matched filter method.

    :param e: The event number [0-4]
    :param noise_model_smooth_fft: The smoothed noise model

    :return: Returns the signal amplitude m, the pre-whitened templates for each e and detectors(d)
             and the pre-whitened strain data for each e and d.
    """

    # Obtain the templates and strain windowed ffts 
    bbh_event_templates_windowed_fft = getWindowedFFTTemplates(e)
    bbh_event_data_lh_windowed_fft = getWindowedFFTData(e)

    # Pre-whitened both in the fourier space
    bbh_event_templates_windowed_white_fft = bbh_event_templates_windowed_fft/np.sqrt(noise_model_smooth_fft)
    bbh_event_data_lh_windowed_white_fft = bbh_event_data_lh_windowed_fft/np.sqrt(noise_model_smooth_fft)

    # Obtain the normalized signal amplitude m.
    f = np.fft.irfft(bbh_event_data_lh_windowed_white_fft*np.conj(bbh_event_templates_windowed_white_fft))
    H = np.dot(np.fft.irfft(bbh_event_templates_windowed_white_fft[0]), np.fft.irfft(bbh_event_templates_windowed_white_fft[0]))
    m = f/H

    # Switch back the templates and strains from fourier space to real space
    bbh_event_templates_windowed_white = np.fft.irfft(bbh_event_templates_windowed_white_fft)
    bbh_event_data_lh_windowed_white = np.fft.irfft(bbh_event_data_lh_windowed_white_fft)
    
    return  m, bbh_event_templates_windowed_white, bbh_event_data_lh_windowed_white


# Plotting the window as an example.
def window_demo():
    window_res = 4096*32
    window_sample = flat_window(window_res, 0.95)
    
    window_sample_x = np.linspace(0, 1, window_res)

    # Plots the window function 
    fig, ax = plt.subplots(1, figsize=(16, 9))
    plt.plot(window_sample_x, window_sample)
    plt.title("Window function")
    plt.tight_layout()
    plt.savefig("window_sample_full")
    plt.cla()
    plt.clf()
    plt.close()

    # Zooms in to show continuity
    zoomed_window_sample = window_sample[int(window_res*0.0248): int(window_res*0.0251)]
    zoomed_window_sample_x = window_sample_x[int(window_res*0.0248): int(window_res*0.0251)]
    fig, ax = plt.subplots(1, figsize=(16, 9))
    plt.plot(zoomed_window_sample_x, zoomed_window_sample)
    plt.title("Window function - Zoomed in")
    plt.tight_layout()
    plt.savefig("window_sample_zoom")
    plt.cla()
    plt.clf()
    plt.close()


# A few example plots relating the PSD and obtaining the noise model
def PSD_smoothing_demo():
    # Plotting the PSD to show the flattening process
    # Loop over events
    for e in range(4):
        fig, ax = plt.subplots(1, figsize=(16, 9))
        
    #Detectors
    for d in range(2):
        plt.loglog(np.abs(getWindowedFFTData(e)[d])**2, label=bbh_detectors_names[d])
        
    plt.title("Initial PSD of event " + bbh_events_names[e])
    plt.legend()
    plt.tight_layout()
    plt.savefig("Initial_PSD_event"+str(e))
    plt.cla()
    plt.clf()
    plt.close()

    # Plotting average PSD
    average_PSD = np.abs(getWindowedFFTData(0))**2
    for e in range(3):
        average_PSD += np.abs(getWindowedFFTData(e))**2
        
        average_PSD /= 4
        
        fig, ax = plt.subplots(1, figsize=(16, 9))

    #Detectors
    for d in range(2):
        plt.loglog(average_PSD[d], label=bbh_detectors_names[d])
        
    plt.title("Average PSD of all events")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Average_PSD")
    plt.cla()
    plt.clf()
    plt.close()


    # Plotting final noise model
    noise_model_smooth_fft = getNoiseModel()
    fig, ax = plt.subplots(1, figsize=(16, 9))
   
    #Detectors
    for d in range(2):
        plt.loglog(noise_model_smooth_fft[d], label=bbh_detectors_names[d])

        plt.title("Final noise model (squared)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("final_noise_model_squared")
        plt.cla()
        plt.clf()
        plt.close()


######## THE DEMOS ##########

#window_demo()
#PSD_smoothing_demo()

# Only need to be called once so are commented out here - they are not required for the program to work
# These just supply plots to illustrate my answers in the pdf.
        
        
####################### GET THE NOISE MODEL AND SEARCH EVENTS #######################

# Get the nosie model
noise_model_smooth_fft = getNoiseModel()

# Prepare arrays to store the signal amplitude m, the windowed pre-whitened templates and strains
m = np.zeros((4, 2, DATA_LENGTH))
bbh_event_templates_windowed_white = np.zeros((4, 2, DATA_LENGTH))
bbh_event_data_lh_windowed_white = np.zeros((4, 2, DATA_LENGTH))


# Loop over events
for e in range(0, 4):

    # Search each event for a GW signal
    m[e], bbh_event_templates_windowed_white[e], bbh_event_data_lh_windowed_white[e] = searchEvent(e, noise_model_smooth_fft)

    # Plot the normalized matched filter response m - which is also the signal amplitude.
    fig, ax = plt.subplots(1, figsize=(16, 9))
    for d in range(0, 2):
        plt.plot(m[e][d], ".-", linewidth=1, markersize=1, label=bbh_detectors_names[d], alpha=0.5)
        
    plt.title("Matched Filter - Event Amplitude - Event " + bbh_events_names[e])
    plt.legend()
    plt.tight_layout()
    plt.savefig("mf_event_amplitude_e"+str(e))
    plt.cla()
    plt.clf()
    plt.close()


####################### OBTAIN THE EVENT AMPLITUDES - STD - TIMES AND SNR #######################

# Prepare arrays to store the data
events_amplitudes = np.zeros((4, 2))
events_std = np.zeros((4, 2))
events_times = np.zeros((4, 2))

# Loop over events
for e in range(0, 4):
    # Loop over detectors
    for d in range(0, 2):
        # Obtain the max amplitude, std and time of the max amplitude
        events_amplitudes[e][d] = max(np.min(m[e][d]), np.max(m[e][d]), key=abs)
        events_std[e][d] = np.std(m[e][d])
        events_times[e][d] = np.argwhere(m[e][d] == events_amplitudes[e][d])


# Tmp range of events [0, ..., 4]
e_range = np.arange(0, 4, 1)

# Plot the signal amplitude (m) with error bars based on the scatter of the matched filter.
fig, ax = plt.subplots(1, figsize=(16, 9))
plt.errorbar(e_range-0.1, np.abs(events_amplitudes[:,0]), events_std[e][0], fmt=".", label=bbh_detectors_names[0])
plt.errorbar(e_range+0.1, np.abs(events_amplitudes[:,1]), events_std[e][1], fmt=".", label=bbh_detectors_names[1])

# tick x-axis with the event names rather than event indices
labels = [item.get_text() for item in ax.get_xticklabels()]
for e in range(0, 4):
    labels[2*e+1] = bbh_events_names[e]

# Axes and whatnot
ax.set_xticklabels(labels)
plt.title("GW Events signals")
plt.xlabel("GW Event")
plt.ylabel("Normalized Signal amplitude")
plt.legend()
plt.tight_layout()
plt.savefig("Sig_Amplitude")
plt.cla()
plt.clf()
plt.close()

# Print our data.
print("Events amplitudes (m)")
print(events_amplitudes)
print("-------------")
print("Events std (sigma_m scatter)")
print(events_std)
print("-------------")

print("Events SNR")
print(np.divide(events_amplitudes, events_std))
print("-------------")

print("Events Time")
print(events_times)
print("-------------")


####################### PLOT THE SNR #######################

# Self explanatory
e_range = np.arange(0, 4, 1)

fig, ax = plt.subplots(1, figsize=(16, 9))
plt.scatter(e_range-0.1, np.abs(events_amplitudes[:,0]/events_std[:,0]), label=bbh_detectors_names[0])
plt.scatter(e_range+0.1, np.abs(events_amplitudes[:,1]/events_std[:,1]), label=bbh_detectors_names[1])

# Display event names in the x-axis
labels = [item.get_text() for item in ax.get_xticklabels()]
for e in range(0, 4):
    labels[2*e+1] = bbh_events_names[e]

ax.set_xticklabels(labels)
plt.title("GW Events SNR")
plt.xlabel("GW Event")
plt.ylabel("SNR")
plt.legend()
plt.tight_layout()
plt.savefig("GW_SNR")
plt.cla()
plt.clf()
plt.close()

####################### OBTAIN AND PLOT THE COMBINED SNRs #######################

# Arrays to store our data
events_amplitudes_weighted = np.zeros(4)
events_std_weighted = np.zeros(4)
norm_fac = np.zeros(4)

# A weighted average of our data - more weight to the data with a smaller uncertainty.
for d in range(0, 2):
    events_amplitudes_weighted += np.abs(events_amplitudes.T[d])/np.abs(events_std.T[d])**2
    norm_fac += 1/np.abs(events_std.T[d])**2

events_amplitudes_weighted /= norm_fac
events_std_weighted = np.sqrt(1.0/norm_fac)
events_SNR_weighted = events_amplitudes_weighted/events_std_weighted

# Plot the SNR
e_range = np.arange(0, 4, 1)
fig, ax = plt.subplots(1, figsize=(16, 9))

plt.scatter(e_range, events_SNR_weighted)

labels = [item.get_text() for item in ax.get_xticklabels()]
for e in range(0, 4):
    labels[2*e+1] = bbh_events_names[e]

ax.set_xticklabels(labels)
plt.title("GW Events SNR")
plt.xlabel("GW Event")
plt.ylabel("SNR")
plt.tight_layout()
plt.savefig("GW_SNR_weighted")
plt.cla()
plt.clf()
plt.close()


####################### OBTAIN THE THEORETICAL SNR #######################

# See the pdf. This is mostly analytical with 1 calculation to obtain the theoretical sigma_m.

theoretical_sigma_m = np.zeros((4, 2))

for e in range(4):
    for d in range(2):
        theoretical_sigma_m[e][d] = np.sqrt((1.0/np.dot(bbh_event_templates_windowed_white[e][d], bbh_event_templates_windowed_white[e][d].T))/DATA_LENGTH)

print("Events std (sigma_m theoretical)")
print(theoretical_sigma_m)
print("-------------")

print("Theoretical SNR")
print(np.divide(events_amplitudes, theoretical_sigma_m))
print("-------------")


####################### OBTAIN THE FREQUENCY OF THE SIGNAL #######################

print("Amplitude, Frequency")

# Loop over events
for e in range(4):
    print(bbh_events_names[e])

    # Loop over detectors
    for d in range(2):
        # Fourier transform the whitened template to see the frequencies of our signal
        bbh_event_templates_windowed_white_fft = np.fft.rfft(bbh_event_templates_windowed_white[e][d], n=DATA_LENGTH*2)
        bbh_event_templates_windowed_white_fft_freq = np.fft.rfftfreq(DATA_LENGTH*2)*4096 

        # Print the frequency with the largest amplitude (This should range from 35-350Hz)
        print(bbh_event_templates_windowed_white_fft_freq[np.argmax(bbh_event_templates_windowed_white_fft)])        
print("-------------")
        
####################### ESTIMATE THE UNCERTAINTY IN EVENT TIME DETECTION #######################

# This is mostly for analysis - done in the pdf - it prints the 10 highest peaks in the matched filter
# These are distributed around the maximum point in the mf.
# Using this you can estimate sigma_time
# And propagate to get sigma_dt (Essentially 2*sigma_time - uncertainties are added in subtr. and addit.)
# Then you can get the positional uncertainty of the signal.

time = np.arange(0, DATA_LENGTH, 1)

for e in range(4):
    print(bbh_events_names[e])
    for d in range(2):
        m_data_tmp = sorted(zip(np.abs(m[e][d]), time), reverse=True)[:10]
        m_amplitudes, m_positions = np.transpose(m_data_tmp)

        print("The SNR of the 10 largest amplitudes")
        print(m_amplitudes/theoretical_sigma_m[e][d])

        print("Positions of the points")
        print(m_positions)
        print(" ")


# End.
