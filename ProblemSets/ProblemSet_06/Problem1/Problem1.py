import numpy as np
import matplotlib.pyplot as plt
import h5py
import json

dir = "../../../../LOSC_Event_tutorial/LOSC_Event_tutorial/"

bbh_events_names = ["GW150914", "LVT151012", "GW151226", "GW170104"]

bbh_events_info_file = open(dir+"BBH_events_v3.json")
bbh_events_info = json.load(bbh_events_info_file)
bbh_events_info_file.close()

def getTemplates(event_index):
    bbh_event = bbh_events_info[bbh_events_names[event_index]]
    bbh_event_template_filename = bbh_event["fn_template"]
    bbh_event_template_file = h5py.File(dir + bbh_event_template_filename, 'r')
    bbh_event_templates = bbh_event_template_file['template']

    return bbh_event_templates


def getStrains(event_index):
    bbh_event = bbh_events_info[bbh_events_names[event_index]]
    bbh_event_data_l_filename = bbh_event["fn_L1"]
    bbh_event_data_l_file = h5py.File(dir + bbh_event_data_l_filename, 'r')
    bbh_event_data_l = bbh_event_data_l_file['strain']['Strain'][:]

    bbh_event_data_h_filename = bbh_event["fn_H1"]
    bbh_event_data_h_file = h5py.File(dir + bbh_event_data_h_filename, 'r')
    bbh_event_data_h = bbh_event_data_h_file['strain']['Strain'][:]

    return bbh_event_data_l, bbh_event_data_h


def flat_window(N_total, flat_proportion):
    N_flat = int(N_total*flat_proportion)
    N_hanning = N_total - N_flat
    hanning_part = np.hanning(N_hanning)
    
    flat_window = np.ones(N_total)
    flat_window[:N_hanning//2] = hanning_part[0:N_hanning//2]
    flat_window[-N_hanning//2:] = hanning_part[-N_hanning//2:]

    return flat_window


def calcRms(x):
    return np.sqrt(np.mean(np.square(x)))


def smoothen(x, smoothing_steps, average_range):
    for s in range(0, smoothing_steps):
        x_tmp = np.copy(x)
        for t in range(1, average_range+1):
            x_tmp += np.roll(x, t) + np.roll(x, -t)
        x = x_tmp/(2.0*t+1)
    return x

bbh_event_templates = getTemplates(0)
bbh_event_data_lh = getStrains(0)

template_window = flat_window(len(bbh_event_templates[0]), 0.95)
template_window /= calcRms(template_window) # Renormalize the template
bbh_event_templates_windowed = bbh_event_templates*template_window

data_window = flat_window(len(bbh_event_data_lh[0]), 0.95)
data_window /= calcRms(data_window) # Renormalize the template
bbh_event_data_lh_windowed = bbh_event_data_lh*data_window

bbh_event_templates_windowed_fft = np.fft.rfft(bbh_event_templates_windowed)
bbh_event_data_lh_windowed_fft = np.fft.rfft(bbh_event_data_lh_windowed)

Noise_model_fft = np.abs(bbh_event_data_lh_windowed_fft)**2

Noise_model_fft_smooth = smoothen(Noise_model_fft, 10, 5)

plt.loglog(Noise_model_fft[0])
plt.loglog(Noise_model_fft_smooth[0])
plt.show()

bbh_event_templates_windowed_fft_white = bbh_event_templates_windowed_fft/np.sqrt(Noise_model_fft_smooth)
bbh_event_data_lh_windowed_fft_white = bbh_event_data_lh_windowed_fft/np.sqrt(Noise_model_fft_smooth)



mf = np.fft.irfft(bbh_event_data_lh_windowed_fft_white*np.conj(bbh_event_templates_windowed_fft_white))

norm_fac = np.dot(np.fft.irfft(bbh_event_templates_windowed_fft_white[0]), np.fft.irfft(bbh_event_templates_windowed_fft_white[0]))

mf /= norm_fac

plt.plot(mf[0], ".")
plt.show()
plt.plot(mf[1], ".")
plt.show()

noise_only_std = np.std(mf[0][80000:]), np.std(mf[1][80000:])

sigmas_away = np.max(mf[0]/noise_only_std[0]), np.max(mf[1]/noise_only_std[1])
print(sigmas_away)

print(np.argmax(mf, axis=1))

plt.plot(np.fft.irfft(bbh_event_data_lh_windowed_fft_white[0]))

plt.plot(np.max(np.abs(mf), axis=1)[0]*np.roll(np.fft.irfft(bbh_event_templates_windowed_fft_white[0]), np.argmax(np.abs(mf), axis=1)[0]), color="red")
plt.show()


plt.plot(np.fft.irfft(bbh_event_data_lh_windowed_fft_white[1]))

plt.plot(-np.max(np.abs(mf), axis=1)[1]*np.roll(np.fft.irfft(bbh_event_templates_windowed_fft_white[1]), np.argmax(np.abs(mf), axis=1)[1]), color="red") # don't forget the np.abs it's important
# However the sign is also important which is why i added a minus, as it tells you if the wave is flipped or not in the fit ! this is good tho
plt.show()
