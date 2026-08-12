import numpy as np
import json
import matplotlib.pyplot as plt
import mne
from scipy import signal
from pathlib import Path
from mne_connectivity import spectral_connectivity_epochs
from Procesamiento_Funciones import (
    cargar_npy_eeg,
    aplicar_notch_multiple,
    aplicar_wavelet_dwt_multicanal,
    ica_eliminar_componente,
    aplicar_pasabanda_butter,
    rectificar_senal,
    resamplear_por_promedio,
    resamplear_profesional,
    emg_envelope
)

plt.close('all')
with open("C:/Users/lrl13/Downloads/Entrenamiento imaginacion2/Entrenamiento imaginacion/Fieldtrip_Archivos/trials_crudos_subjectcmc/metadata_trials.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

fs = meta["sfreq"]

FS_EEG = int(fs)
FS_EMG = int(fs)
FREQS_NOTCH = (50.0, 100.0)
Q_NOTCH = 5
PAD = 5 * FS_EEG            # PADDING PARA FILTROS
PAD_MUESTRAS = 5*FS_EMG
# Wavelet
WAVELET = 'db4'
LEVEL = 6
NIVELES_CERO = [1, 6]   # QUITAR D1 y D6
REMOVE_APPROX = True    # QUITAR A6
WAVE_MODE = 'reflect'

# ICA
IC_TO_REMOVE = 0        # IC A QUITAR [X,X]

# PASA BANDA
BP_LOW =    5.0         
BP_HIGH =   45.0
BP_ORDER =  3
concatenar = True
envolvente = True

# Segmentación (Welch)
SEG_DUR   = 1.0     
OVERLAP   = 0.5     


def procesar_todos_los_trials(meg_trials_raw, emg_trials_raw, fs, formaSig = "envolvente",concatenar=False ):
    """
    Aplica tu preprocesamiento a cada trial por separado.

    Entrada:
        meg_trials_raw: n_trials x n_meg x n_samples
        emg_trials_raw: n_trials x n_emg x n_samples

    Salida:
        meg_trials_proc: n_trials x n_meg x n_samples
        emg_trials_proc: n_trials x n_emg x n_samples
    """

    meg_proc = []
    emg_proc = []

    n_trials = meg_trials_raw.shape[0]

    for i in range(n_trials):
        print(f"Procesando trial {i + 1}/{n_trials}")

        meg_i = meg_trials_raw[i]   # n_meg x n_samples
        emg_i = emg_trials_raw[i]   # n_emg x n_samples

        meg_i_proc = procesar_eeg(meg_i, fs=fs, concatenar = concatenar)
        emg_i_proc = procesar_emg(emg_i, fs=fs, forma=formaSig, concatenar = concatenar)

        meg_proc.append(meg_i_proc)
        emg_proc.append(emg_i_proc)
    
    if not concatenar:
        
        meg_proc = np.stack(meg_proc, axis=0)
        emg_proc = np.stack(emg_proc, axis=0)

    return meg_proc, emg_proc

def calcula_coherencia_todos_trials(
    emg_trials,
    meg_trials,
    fs,
    emg_channel=0,
    fmin=5.0,
    fmax=80.0,
    mt_bandwidth=5.0,
):
    """
    Calcula coherencia EMG-MEG usando todos los trials.

    Parámetros
    ----------
    emg_trials : np.ndarray
        Forma: n_trials x n_emg x n_samples

    meg_trials : np.ndarray
        Forma: n_trials x n_meg x n_samples

    fs : float
        Frecuencia de muestreo.

    emg_channel : int
        Índice del canal EMG a usar.
        0 suele ser EMGlft si emg_ch_names = ['EMGlft', 'EMGrgt'].

    Retorna
    -------
    freqs : np.ndarray
        Frecuencias.

    coherencia : np.ndarray
        Forma: n_freqs x n_meg.
        coherencia[:, i] corresponde a EMG vs canal MEG i.
    """

    n_trials, n_emg, n_samples = emg_trials.shape
    n_trials_meg, n_meg, n_samples_meg = meg_trials.shape

    if n_trials != n_trials_meg:
        raise ValueError("EMG y MEG no tienen el mismo número de trials.")

    if n_samples != n_samples_meg:
        raise ValueError("EMG y MEG no tienen el mismo número de muestras.")

    # Construir datos para mne-connectivity:
    # n_trials x n_signals x n_samples
    #
    # Señal 0 = EMG seleccionado
    # Señales 1...n_meg = canales MEG
    emg_sel = emg_trials[:, emg_channel:emg_channel + 1, :]

    data_conn = np.concatenate(
        [emg_sel, meg_trials],
        axis=1
    )

    # Conexiones: EMG seleccionado contra cada canal MEG
    seed = np.zeros(n_meg, dtype=int)
    targets = np.arange(1, n_meg + 1, dtype=int)
    indices = (seed, targets)

    con = spectral_connectivity_epochs(
        data_conn,
        method="coh",
        mode="multitaper",
        sfreq=fs,
        fmin=fmin,
        fmax=fmax,
        faverage=False,
        mt_bandwidth=mt_bandwidth,
        indices=indices,
        verbose=True,
    )

    freqs = con.freqs

    coh = con.get_data()

    # Usualmente con indices: n_connections x n_freqs
    # Queremos: n_freqs x n_meg
    if coh.shape[0] == n_meg:
        coherencia = coh.T
    else:
        coherencia = coh.squeeze().T

    return freqs, coherencia


def procesar_eeg(eeg_data: np.ndarray, fs: float = FS_EEG, sujeto = "S5",concatenar = True):
    eeg = eeg_data
    print(f"DIMENSIONES DEL EEG A PROCESAR : {eeg.shape}")
    eeg = aplicar_notch_multiple(eeg, fs=FS_EEG, freqs=FREQS_NOTCH, Q=Q_NOTCH, pad=PAD)
    # if sujeto == "S5":
    #     eeg, _ica = ica_eliminar_componente(eeg, fs=FS_EEG, ch_names=EEG_NAMES, ic_to_remove=IC_TO_REMOVE)

    eeg_final = aplicar_pasabanda_butter(eeg, fs=FS_EEG, f_low=BP_LOW, f_high=BP_HIGH, orden=BP_ORDER, pad=PAD) 
    if concatenar:    
        eeg_final = resamplear_profesional(eeg_final, 1200, 200)
    return eeg_final

def procesar_emg(emg_data: np.ndarray, fs: float = FS_EMG, forma:str = None,concatenar = True):
    FS = FS_EMG
    ORDEN_BP = 3
    PAD_MUESTRAS = 5 * FS
    Q_NOTCH = 200
    emg_data = emg_data - emg_data.mean()
    # emg_data = emg_data[0]
    # print(f"FORMA DEL EMG A PROCESAR: {emg_data.shape}")
    
    emg_proc = aplicar_pasabanda_butter(
              emg_data,
              fs=FS,
              f_low=5.0,
              f_high=200.0,
              orden=ORDEN_BP,
              pad=PAD_MUESTRAS,
          )
    emg_proc = aplicar_notch_multiple(
          emg_proc,
          fs=FS,
          freqs=tuple(FREQS_NOTCH),
          Q=Q_NOTCH,
          pad=PAD_MUESTRAS,
      )
      # 6) Rectificar Señales
    # emg_final = rectificar_senal(emg_final, tipo='completa')
    
    if forma == 'envolvente':
        #emg_final = rectificar_senal(emg_proc, tipo='completa') 
        for i in range(emg_proc.shape[0]):
            emg_proc[i, :] = emg_envelope(emg_proc[i, :], fs=FS, tc_ms=20, rectify="full")
            emg_final = emg_proc
    elif forma == 'rectificar':
        emg_final = rectificar_senal(emg_proc, tipo='completa')  
    else:
        emg_final = emg_proc
    
    if concatenar:
        emg_final = resamplear_profesional(emg_final, 1200, 200)
   
    return emg_final

def concatenate_eeg_list(signals_list, fs=250, ms=40):
    """
    Concatena una lista de arrays (7, muestras) usando cross-fade.
    """
    # if not signals_list:
    #     return None
    
    # 1. Definir parámetros del cross-fade
    n_samples = int((ms / 1000) * fs)
    t = np.linspace(0, np.pi/2, n_samples)
    fade_in = (np.sin(t)**2)
    fade_out = (np.cos(t)**2)
    
    # 2. Tomar la primera señal como base
    combined = signals_list[0]
    
    # 3. Iterar sobre el resto de las señales
    for next_sig in signals_list[1:]:
        # Zona de solapamiento entre el acumulado y la siguiente señal
        overlap = (combined[:, -n_samples:] * fade_out) + (next_sig[:, :n_samples] * fade_in)
        
        # Unir
        combined = np.concatenate([
            combined[:, :-n_samples], 
            overlap, 
            next_sig[:, n_samples:]
        ], axis=1)
        
    return combined

def calcula_coherencia_welch(emg, eeg, fs=1200, fmin=5.0, fmax=45.0, mt_bandwidth=5.0, metodo_espectro = 'Multitapers'):
    """
    Calcula coherencia EMG-EEG con multitaper, en un estilo similar a FieldTrip.

    Acepta:
    - 2D: (n_muestras, n_canales)
    - 3D: (n_epochs, n_muestras, n_canales)

    Retorna:
    - freqs: (n_freqs,)
    - coherencia: (n_freqs, n_emg, n_eeg)
    """
    emg = np.asarray(emg)
    eeg = np.asarray(eeg)   
    
    if metodo_espectro == 'Welch':
        n_emg = emg.shape[1]
        n_eeg = eeg.shape[1]
    
        freqs, _ = signal.coherence(emg[:,0],eeg[:,0], fs = fs, nperseg = int(fs*SEG_DUR))
        coherencia = np.zeros((len(freqs), n_emg,n_eeg))
    
        for ei in range(n_emg):
            for yi in range(n_eeg):
                _ ,coherencia[:,ei,yi] = signal.coherence(emg[:,ei],eeg[:,yi], fs=fs,nperseg=int(fs*SEG_DUR),
                                                          noverlap=int(fs*SEG_DUR*OVERLAP))
    
        return freqs, coherencia
    
    elif metodo_espectro == 'Multitapers': 
    
        if fmax is None:
            fmax = min(100.0, fs / 2.0)
    
        # Compatibilidad con el pipeline original: entrada 2D (n_muestras, n_canales)
        if emg.ndim == 2:
            emg = emg[None, ...]
        if eeg.ndim == 2:
            eeg = eeg[None, ...]
    
        if emg.ndim != 3 or eeg.ndim != 3:
            raise ValueError('EMG y EEG deben ser 2D (n_muestras, n_canales) o 3D (n_epochs, n_muestras, n_canales).')
        if emg.shape[0] != eeg.shape[0]:
            raise ValueError('EMG y EEG deben tener el mismo numero de epochs/trials.')
        if emg.shape[1] != eeg.shape[1]:
            raise ValueError('EMG y EEG deben tener el mismo numero de muestras por epoch/trial.')
    
        n_epochs, n_times, n_emg = emg.shape
        _, _, n_eeg = eeg.shape
    
        print('CALCULA COHERENCIA (MULTITAPER TIPO FIELDTRIP)')
        print(f'{n_epochs} epoch(s) / trial(s)')
        print(f'{n_emg} canales de EMG')
        print(f'{n_eeg} canales de EEG')
        print(f'{n_times} muestras por epoch')
        print(f'Suavizado multitaper: {mt_bandwidth} Hz')
    
        # MNE-Connectivity espera (n_epochs, n_signals, n_times)
        emg_x = np.transpose(emg, (0, 2, 1))
        eeg_x = np.transpose(eeg, (0, 2, 1))
        X = np.concatenate([emg_x, eeg_x], axis=1)
    
        seed, target = [], []
        for ei in range(n_emg):
            for yi in range(n_eeg):
                seed.append(ei)
                target.append(n_emg + yi)
    
        con = spectral_connectivity_epochs(
            X,
            method='coh',
            indices=(np.asarray(seed), np.asarray(target)),
            sfreq=fs,
            mode='multitaper',
            #cwt_freqs = np.arange(5.0,45.0),
            fmin=fmin,
            fmax=fmax,
            mt_bandwidth=mt_bandwidth,
            faverage=False,
            verbose=False,
            fdecim= 2
        )
    
        freqs = np.asarray(con.freqs)
        dat = np.asarray(con.get_data())
    
        coherencia = np.zeros((len(freqs), n_emg, n_eeg), dtype=float)
        pair = 0
        for ei in range(n_emg):
            for yi in range(n_eeg):
                coherencia[:, ei, yi] = dat[pair, :]
                pair += 1

    return freqs, coherencia
def calcula_coherencia(emg, eeg, fs=FS_EEG, fmin=5.0, fmax=80.0, mt_bandwidth=5.0):
    """
    Calcula coherencia EMG-EEG con multitaper, en un estilo similar a FieldTrip.

    Acepta:
    - 2D: (n_muestras, n_canales)
    - 3D: (n_epochs, n_muestras, n_canales)

    Retorna:
    - freqs: (n_freqs,)
    - coherencia: (n_freqs, n_emg, n_eeg)
    """
    #emg = np.asarray(emg)
    #eeg = np.asarray(eeg)
    emg = emg.T
    eeg = eeg.T
    print(f"Dimensiones de EMG: {emg.shape}")
    print(f"Dimensiones de EMG: {emg.shape[0]}")
    print(f"Dimensiones de EEG: {eeg.shape}")
    print(f"Dimensiones de EEG: {eeg.shape[0]}")
    

    if fmax is None:
        fmax = min(100.0, fs / 2.0)

    # Compatibilidad con el pipeline original: entrada 2D (n_muestras, n_canales)
    if emg.ndim == 2:
        emg = emg[None, ...]
    if eeg.ndim == 2:
        eeg = eeg[None, ...]

    if emg.ndim != 3 or eeg.ndim != 3:
        raise ValueError('EMG y EEG deben ser 2D (n_muestras, n_canales) o 3D (n_epochs, n_muestras, n_canales).')
    if emg.shape[0] != eeg.shape[0]:
        raise ValueError('EMG y EEG deben tener el mismo numero de epochs/trials.')
    if emg.shape[1] != eeg.shape[1]:
        raise ValueError('EMG y EEG deben tener el mismo numero de muestras por epoch/trial.')

    n_epochs, n_times, n_emg = emg.shape
    _, _, n_eeg = eeg.shape

    print('CALCULA COHERENCIA (MULTITAPER TIPO FIELDTRIP)')
    print(f'{n_epochs} epoch(s) / trial(s)')
    print(f'{n_emg} canales de EMG')
    print(f'{n_eeg} canales de EEG')
    print(f'{n_times} muestras por epoch')
    print(f'Suavizado multitaper: {mt_bandwidth} Hz')

    # MNE-Connectivity espera (n_epochs, n_signals, n_times)
    emg_x = np.transpose(emg, (0, 2, 1))
    eeg_x = np.transpose(eeg, (0, 2, 1))
    X = np.concatenate([emg_x, eeg_x], axis=1)

    seed, target = [], []
    for ei in range(n_emg):
        for yi in range(n_eeg):
            seed.append(ei)
            target.append(n_emg + yi)

    con = spectral_connectivity_epochs(
        X,
        method='coh',
        indices=(np.asarray(seed), np.asarray(target)),
        sfreq=fs,
        mode='multitaper',
        #cwt_freqs = np.arange(5.0,45.0),
        fmin=fmin,
        fmax=fmax,
        mt_bandwidth=mt_bandwidth,
        faverage=False,
        verbose=False,
        fdecim= 2
    )

    freqs = np.asarray(con.freqs)
    dat = np.asarray(con.get_data())

    coherencia = np.zeros((len(freqs), n_emg, n_eeg), dtype=float)
    pair = 0
    for ei in range(n_emg):
        for yi in range(n_eeg):
            coherencia[:, ei, yi] = dat[pair, :]
            pair += 1

    return freqs, coherencia


emg = np.load(r"C:\Users\lrl13\Downloads\Entrenamiento imaginacion2\Entrenamiento imaginacion\Fieldtrip_Archivos\trials_crudos_subjectcmc\trial_01_raw_emg.npy")
eeg = np.load(r"C:\Users\lrl13\Downloads\Entrenamiento imaginacion2\Entrenamiento imaginacion\Fieldtrip_Archivos\trials_crudos_subjectcmc\trial_01_raw_meg.npy") 


concatenar = False

ds_path = Path(r"C:\Users\lrl13\Downloads\SubjectCMC\SubjectCMC.ds")

raw = mne.io.read_raw_ctf(
    ds_path,
    preload=True,
    clean_names=True,
    system_clock="truncate"  # default; si hay problemas prueba "ignore"
)

# indice = raw.ch_names.index('MRC21')
# print("índice de MRC21:")
# print(indice)

# emg_final = procesar_emg(emg,fs, forma = 'rectificar')
# eeg_final = procesar_eeg(eeg,fs = FS_EEG)

# #plt.plot(emg[0,:]-emg[0,:].mean(),label = 'crudo')
# plt.plot(emg_final[0,:],label = 'filtrado')
# plt.plot(emg_final[1,:],label = 'filtrado')

# #plt.plot(emg_rect[0,:],label = 'rectificado')
# plt.legend()

# plt.figure()
# plt.plot(eeg[0,:]-eeg[0,:].mean(),label = 'crudo')
# plt.plot(eeg_final[0,:],label = 'filtrado')
# #plt.plot(emg_rect[0,:],label = 'rectificado')
# plt.legend()

# freqs, coherencia = calcula_coherencia(emg_final,eeg_final,fs = 200)

# plt.figure()
# plt.plot(freqs, coherencia[:,0,indice])

#-------------------------------------------------------------------------------

# ============================================================
# CARGAR TODOS LOS TRIALS DESDE EL NPZ
# ============================================================

npz_path = r"C:\Users\lrl13\Downloads\Entrenamiento imaginacion2\Entrenamiento imaginacion\Fieldtrip_Archivos\trials_crudos_subjectcmc\subjectcmc_trials_raw_all.npz"

data_npz = np.load(npz_path, allow_pickle=True)

trials_raw = data_npz["trials"]          # n_trials x n_channels x n_samples
fs = float(data_npz["sfreq"])
ch_names = list(data_npz["ch_names"])
meg_ch_names = list(data_npz["meg_ch_names"])

meg_ch_names = [
    ch for ch in meg_ch_names
    if ch.startswith(("ML", "MR", "MZ"))
]

meg_idx = [ch_names.index(ch) for ch in meg_ch_names]
emg_ch_names = list(data_npz["emg_ch_names"])
time = data_npz["time"]

print("Forma de trials_raw:", trials_raw.shape)
print("fs:", fs)
print("Canales EMG:", emg_ch_names)
print("Número de canales MEG:", len(meg_ch_names))

# Índices de canales
meg_trials_raw = trials_raw[:, meg_idx, :]

#meg_idx = [ch_names.index(ch) for ch in meg_ch_names]
emg_idx = [ch_names.index(ch) for ch in emg_ch_names]

# Extraer arreglos
#meg_trials_raw = trials_raw[:, meg_idx, :]   # n_trials x n_meg x n_samples
emg_trials_raw = trials_raw[:, emg_idx, :]   # n_trials x n_emg x n_samples

print("MEG:", meg_trials_raw.shape)
print("EMG:", emg_trials_raw.shape)

meg_trials_proc, emg_trials_proc = procesar_todos_los_trials(
    meg_trials_raw,
    emg_trials_raw,
    fs,
    formaSig = "rectificar",
    concatenar = concatenar
)

print("MEG procesado:", meg_trials_proc.shape)
print("EMG procesado:", emg_trials_proc.shape)

#Coherencia con músculo del lado izquierdo
emg_channel = emg_ch_names.index("EMGlft")

if concatenar:
    # emg_trials_sel = emg_trials_proc[:, emg_channel:emg_channel + 1, :]
    emg_trials_sel = emg_trials_proc[:, :, :]
    emg_trials_sel = concatenate_eeg_list(emg_trials_sel,fs = 200,ms = 10)
    meg_trials_proc = concatenate_eeg_list(meg_trials_proc,fs = 200,ms = 10)
    
    plt.figure()
    plt.plot(emg_trials_sel.T)
    plt.title("EMG concatenado")
    plt.figure()
    plt.plot(meg_trials_proc[5,:].T)
    plt.title("MEG concatenado")
    
    freqs,results = calcula_coherencia_welch(emg_trials_sel.T,meg_trials_proc.T,metodo_espectro = 'Welch',fs = 200)
    
    

else:

    freqs, coherencia_all_left = calcula_coherencia_todos_trials(
        emg_trials_proc,
        meg_trials_proc,
        fs=fs,
        emg_channel=emg_channel,
        fmin=5.0,
        fmax=80.0,
        mt_bandwidth=5.0,
        )
    
#Calcula coherencia con músculo del lado derecho
emg_channel = emg_ch_names.index("EMGrgt")

freqs, coherencia_all_right = calcula_coherencia_todos_trials(
    emg_trials_proc,
    meg_trials_proc,
    fs=fs,
    emg_channel=emg_channel,
    fmin=5.0,
    fmax=80.0,
    mt_bandwidth=5.0,
)

# from scipy.signal import welch
# # import matplotlib.pyplot as plt

# mrc21_idx = meg_ch_names.index("MRC21")
# emglft_idx = emg_ch_names.index("EMGlft")

# x_meg = meg_trials_proc[:, mrc21_idx, :].reshape(-1)
# x_emg = emg_trials_proc[:, emglft_idx, :].reshape(-1)

# f_meg, p_meg = welch(x_meg, fs=fs, nperseg=int(fs*2))
# f_emg, p_emg = welch(x_emg, fs=fs, nperseg=int(fs*2))

# plt.figure()
# plt.semilogy(f_meg, p_meg, label="MRC21")
# plt.semilogy(f_emg, p_emg, label="EMGlft")
# plt.axvline(50, color="r", linestyle="--")
# plt.xlim(0, 100)
# plt.legend()
# plt.title("PSD después del preprocesamiento")
# plt.show()

# print("Forma coherencia:", coherencia_all_left.shape)

# mrc21_idx = meg_ch_names.index("MRC21")

# plt.figure()
# plt.plot(freqs, coherencia_all_left[:, mrc21_idx])
# plt.xlabel("Frecuencia [Hz]")
# plt.ylabel("Coherencia")
# plt.title("CMC: MRC21 - EMGlft, todos los trials")
# plt.grid(True)
# plt.show()


# mrc21_idx = meg_ch_names.index("MRT32")

# plt.figure()
# plt.plot(freqs, coherencia_all_left[:, mrc21_idx])
# plt.xlabel("Frecuencia [Hz]")
# plt.ylabel("Coherencia")
# plt.title("CMC: MRC21 - EMGlft, todos los trials")
# plt.grid(True)
# plt.show()


#Finalmente, concatenamos los dos arreglos de coherencia de los músculos

ruta_salidaReporte = r"C:\Users\lrl13\Downloads\Entrenamiento imaginacion2\Entrenamiento imaginacion\Fieldtrip_Archivos\Reportes_Base_datos_multitapers_rectificada"

if not concatenar:
    coherencia_final = np.stack((coherencia_all_left,coherencia_all_right),axis = 1)
    results = coherencia_final

general = {}
# general['SDataBase'] = {'coherencia':coherencia_final,
#                       'freqs': np.array(freqs)}
general['SDataBase'] = {'coherencia':results,
                      'freqs': np.array(freqs)}

np.save(ruta_salidaReporte,general)