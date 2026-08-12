import os
import contextlib
import numpy as np
#import pandas as pd
from scipy import signal
from mne_connectivity import spectral_connectivity_epochs
import matplotlib.pyplot as plt
from pathlib import Path
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
# ───────────────────
# PARÁMETROS GLOBALES
# ───────────────────
FS_EMG   = 1000          
FS_EEG   = 250           
FS_FINAL = 1000#FS_EEG        

# Filtros
EMG_BAND  = (20.0,300.0)#(2.0,100.0)   
EEG_BAND  = (5,  45.0)   
NOTCH_F   = 60.0            

# Segmentación (Welch)
SEG_DUR   = 1.0     
OVERLAP   = 0.5     

BANDAS = {
    'δ' :       (0,  4),
    'θ' :       (4,  8),
    'μ' :       (8, 13),
    'β' :       (13,30),
    'γ' :       (30,45),
    "Otros":  (45,1000)
}
CONDICION_MAP = {
    'OV': 'Ojos Abiertos - Movimiento Visual',
    'OK': 'Ojos Abiertos - Movimiento Cinestésico',
    'CK': 'Ojos Cerrados - Movimiento Cinestésico',
    'CV': 'Ojos Cerrados - Movimiento Visual',
}

ESTADO_MAP = {
    'ME': 'Movimiento Ejecutado',
    'MI': 'Movimiento Imaginado',
    'RE': 'Reposo',
}
# COLORES = [   '#6FC7B7', '#A1DAD1', '#D0EDE8',
#               '#FFFFFF',  '#BFEEF6', '#81DCEE', '#41CCE3']

# BG_FIG      = '#0a0a0a'
# BG_AX       = '#0d0d0d'
# COLOR_GRID  = '#404040'
# COLOR_TICK  = '#a5a5a5'
# COLOR_LABEL = '#d1d1d1'
# COLOR_TITLE = '#ffffff'
# PIE_FIGURA = " "

# --- ESTOS COLORES SON PARA VERSION CLARA

COLORES = [    '#477e75', '#4b7168', '#4d615f',
                '#515051',  '#475e63', '#3d6e74', '#307c8a']

BG_FIG      = '#ffffff'
BG_AX       = '#ffffff'
COLOR_GRID  = '#bfbfbf'
COLOR_TICK  = '#5a5a5a'
COLOR_LABEL = '#2e2e2e'
COLOR_TITLE = '#1e1e1e'

# ══════════════════════════════════════════════
#  CARGA DE SEÑALES
# ══════════════════════════════════════════════

def load_emg(RUTA_EMG):
    datos_emg = np.load(RUTA_EMG)
    if not (datos_emg.ndim == 2 and datos_emg.shape[0] == 3):
        raise ValueError(f"Shape inesperado: {datos_emg.shape}. Se esperaba (3, n_muestras).")

    n_canales, n_muestras = datos_emg.shape
    tiempo = np.arange(n_muestras) / FS_EMG
    print(f"\nDatos cargados → {datos_emg.shape}  |  Duración: {n_muestras/FS_EMG:.2f} s")
    return datos_emg

def load_eeg(RUTA_EEG):
    datos_eeg = np.load(RUTA_EEG)
    if not (datos_eeg.ndim == 2 and datos_eeg.shape[0] == 7):
        raise ValueError(f"Shape inesperado: {datos_eeg.shape}. Se esperaba (7, n_muestras).")

    n_canales, n_muestras = datos_eeg.shape
    tiempo = np.arange(n_muestras) / FS_EEG
    print(f"\nDatos cargados → {datos_eeg.shape}  |  Duración: {n_muestras/FS_EEG:.2f} s")

    return datos_eeg

def parsear_nombre_archivo(ruta):
    nombre = os.path.splitext(os.path.basename(ruta))[0]        # 'S17_260310_EEG_CK_3_RE'
    partes = nombre.split('_')                                  # ['S17','260310','EEG','CK','3','RE']

    sujeto    = partes[0]                                       # 'S17'
    condicion = partes[3]                                       # 'CK'
    estado    = partes[5]                                       # 'RE'

    num_sujeto   = sujeto[1:]                                   # '17'
    desc_cond    = CONDICION_MAP.get(condicion, condicion)
    desc_estado  = ESTADO_MAP.get(estado, estado)

    descripcion = f"Sujeto {num_sujeto}  |  {desc_cond}  |  {desc_estado}"
    tag_archivo = f"{sujeto}_{condicion}_{estado}"

    return descripcion, tag_archivo, [sujeto,condicion,estado]
def lee_carpeta(ruta_base,tipos, sujetos):
    rutas = {}
    for tipo in tipos:
        rutas[tipo] = {}
        for sujeto in sujetos :
            archivos_sujeto = []                        
            carpeta_objetivo = os.path.join(ruta_base, sujeto, tipo)        
            
            for carpeta_raiz,_, archivos in os.walk(carpeta_objetivo):
                for archivo in archivos:
                    ruta_completa = os.path.join(carpeta_raiz,archivo)
                    _,_,estado =  parsear_nombre_archivo(ruta_completa)
                    if ((os.path.splitext(os.path.basename(ruta_completa))[1] == '.npy') &
                        (estado[2] == 'ME')):
                    #archivos_sujeto.append((ruta_completa,archivo))
                        archivos_sujeto.append(ruta_completa)
            
            rutas[tipo][sujeto] = archivos_sujeto
    return rutas
def generar_pares(ruta_base,tipos,sujetos):
    rutas = lee_carpeta(ruta_base,tipos,sujetos)
    print(rutas)
    analisis = {}

    for sujeto in sujetos:
        analisis_sujeto = []
        muestras = len(rutas["EEG"][sujeto])
        if muestras == len(rutas["EMG"][sujeto]):
            for i in range(muestras):
                for j in range(muestras):
                    if parsear_nombre_archivo(rutas["EEG"][sujeto][i]) == parsear_nombre_archivo(rutas["EMG"][sujeto][j]):
                        analisis_sujeto.append((rutas["EEG"][sujeto][i],rutas["EMG"][sujeto][j]))
        analisis[sujeto] = analisis_sujeto
    return analisis



# ══════════════════════════════════════════════
# LÍMITES DE CONFIANZA
# ══════════════════════════════════════════════

def confidence_limit(n_segmentos, alpha=0.05):
    """
    Limite de confianza aproximado para coherencia de Welch.
    Devuelve el limite para coherencia en magnitud.
    """
    n_segmentos = int(n_segmentos)
    if n_segmentos <= 1:
        return np.nan
    return np.sqrt(1.0 - alpha ** (1.0 / (n_segmentos - 1.0)))


def n_segmentos(n_muestras, fs=FS_FINAL):
    """
    Numero de segmentos de Welch usados segun SEG_DUR y OVERLAP.
    """
    seg_len = int(fs * SEG_DUR)
    step = int(seg_len * (1 - OVERLAP))
    if seg_len <= 0 or step <= 0 or n_muestras < seg_len:
        return 0
    return len(np.arange(0, n_muestras - seg_len + 1, step))


def confidence_limit_multitaper_aprox(n_epochs, n_times, fs=FS_FINAL,
                                      mt_bandwidth=5.0, alpha=0.05):
    """
    Limite de confianza aproximado para coherencia multitaper en magnitud.

    La coherencia devuelta por MNE para method='coh' esta en magnitud, no en
    coherencia cuadratica. Por eso se calcula primero el limite para C^2 y
    despues se toma la raiz cuadrada.

    Aproximacion:
        Ccrit^2 = 1 - alpha ** (1 / (L - 1))
        Ccrit   = sqrt(Ccrit^2)

    donde:
        L ~= n_epochs * K
        K ~= floor(2*TW - 1)
        TW = mt_bandwidth * T / 2
        T  = n_times / fs

    Notas:
    - Es una aproximacion analitica util como referencia grafica.
    - Para inferencia mas robusta en CMC, se recomienda usar permutaciones
      o desplazamientos circulares como contraste no parametrico.
    """
    n_epochs = int(n_epochs)
    n_times = int(n_times)
    T = n_times / fs

    if n_epochs <= 0 or n_times <= 0 or T <= 0:
        return np.nan, 0, 0, np.nan

    TW = mt_bandwidth * T / 2.0
    K = int(np.floor(2.0 * TW - 1.0))
    K = max(K, 1)

    L = n_epochs * K
    if L <= 1:
        return np.nan, K, L, TW

    crit_squared = 1.0 - alpha ** (1.0 / (L - 1.0))
    crit_squared = np.clip(crit_squared, 0.0, 1.0)
    crit = np.sqrt(crit_squared)

    return crit, K, L, TW


def test_permutacion_coherencia_multitaper(
    emg,
    eeg,
    coherencia_obs=None,
    fs=FS_FINAL,
    fmin=5.0,
    fmax=45.0,
    mt_bandwidth=5.0,
    n_perm=500,
    alpha=0.05,
    surrogate='circular_shift',
    min_shift_s=1.0,
    random_state=None,
    guardar_null=False
):
    """
    Prueba no parametrica por permutaciones/surrogates para coherencia multitaper.

    Entrada esperada:
      emg, eeg: (n_epochs, n_muestras, n_canales)

    Surrogates disponibles:
      - 'circular_shift': desplaza circularmente EEG dentro de cada epoch.
        Conserva el espectro aproximado de cada epoch pero rompe la sincronía temporal EEG-EMG.
      - 'shuffle_epochs': permuta el orden de epochs EEG frente a EMG.
        Conserva cada epoch completo pero rompe la correspondencia trial-a-trial.

    Retorna:
      - pvals: p-values punto a punto, sin corrección múltiple.
      - pvals_fwer: p-values corregidos por máximo estadístico/FWER.
      - umbral_punto_a_punto: percentil 1-alpha de la nula por frecuencia/canal.
      - umbral_global_fwer: umbral escalar basado en el máximo de la coherencia nula.
    """
    rng = np.random.default_rng(random_state)

    emg = np.asarray(emg)
    eeg = np.asarray(eeg)

    if emg.ndim == 2:
        emg = emg[None, ...]
    if eeg.ndim == 2:
        eeg = eeg[None, ...]

    if emg.ndim != 3 or eeg.ndim != 3:
        raise ValueError('EMG y EEG deben ser 2D o 3D para la prueba de permutacion.')
    if emg.shape[0] != eeg.shape[0]:
        raise ValueError('EMG y EEG deben tener el mismo numero de epochs para permutaciones.')
    if emg.shape[1] != eeg.shape[1]:
        raise ValueError('EMG y EEG deben tener el mismo numero de muestras por epoch.')

    n_epochs, n_times, _ = emg.shape

    if coherencia_obs is None:
        freqs, coherencia_obs = calcula_coherencia(
            emg, eeg, fs=fs, fmin=fmin, fmax=fmax,
            mt_bandwidth=mt_bandwidth, metodo_espectro='Multitapers'
        )
    else:
        # Se calcula una vez solo para recuperar freqs con los mismos parametros.
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
            freqs, _ = calcula_coherencia(
                emg, eeg, fs=fs, fmin=fmin, fmax=fmax,
                mt_bandwidth=mt_bandwidth, metodo_espectro='Multitapers'
            )

    null_shape = (n_perm, len(freqs), coherencia_obs.shape[1], coherencia_obs.shape[2])
    if guardar_null:
        null_dist = np.zeros(null_shape, dtype=float)
    else:
        null_dist = None

    # Para ahorrar memoria, tambien acumulamos conteos y maximos.
    conteo_punto = np.zeros_like(coherencia_obs, dtype=int)
    max_null = np.zeros(n_perm, dtype=float)

    # Para el umbral punto a punto sí necesitamos guardar la nula completa.
    # Si guardar_null=False, se guarda de todos modos temporalmente para percentiles,
    # pero no se devuelve en el diccionario final.
    null_tmp = np.zeros(null_shape, dtype=float)

    min_shift = int(min_shift_s * fs)

    print('\nPrueba de permutacion/surrogates para multitapers')
    print(f'  surrogate      : {surrogate}')
    print(f'  n_perm         : {n_perm}')
    print(f'  alpha          : {alpha}')
    print(f'  epochs         : {n_epochs}')
    print(f'  muestras/epoch : {n_times}')

    for iperm in range(n_perm):
        if surrogate == 'shuffle_epochs':
            if n_epochs < 2:
                raise ValueError("surrogate='shuffle_epochs' requiere al menos 2 epochs.")
            perm = rng.permutation(n_epochs)
            eeg_surr = eeg[perm, :, :]

        elif surrogate == 'circular_shift':
            eeg_surr = eeg.copy()
            for ep in range(n_epochs):
                if n_times <= 2 * min_shift:
                    shift = int(rng.integers(1, n_times))
                else:
                    shift = int(rng.integers(min_shift, n_times - min_shift))
                eeg_surr[ep, :, :] = np.roll(eeg_surr[ep, :, :], shift, axis=0)

        else:
            raise ValueError("surrogate debe ser 'circular_shift' o 'shuffle_epochs'.")

        # Evita imprimir el diagnóstico de calcula_coherencia en cada permutación.
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
            _, coh_surr = calcula_coherencia(
                emg, eeg_surr, fs=fs, fmin=fmin, fmax=fmax,
                mt_bandwidth=mt_bandwidth, metodo_espectro='Multitapers'
            )

        null_tmp[iperm] = coh_surr
        conteo_punto += (coh_surr >= coherencia_obs)
        #max_null[iperm] = np.nanmax(coh_surr)
        mask_banda = (freqs >= 13)&(freqs <= 30)
        max_null[iperm] = np.nanmax(coh_surr[mask_banda,:,:])

        if (iperm + 1) % max(1, n_perm // 10) == 0:
            print(f'  Permutaciones completadas: {iperm + 1}/{n_perm}')
    
    umbral_global_fwer = np.percentile(max_null, 100 * (1 - alpha))

    pvals = (1 + conteo_punto) / (n_perm + 1)
    umbral_punto_a_punto = np.percentile(null_tmp, 100 * (1 - alpha), axis=0)
    umbral_global_fwer = np.percentile(max_null, 100 * (1 - alpha))

    # p-value corregido FWER para cada punto: proporcion de maximos nulos >= coherencia observada.
    pvals_fwer = (1 + np.sum(max_null[:, None, None, None] >= coherencia_obs[None, :, :, :], axis=0)) / (n_perm + 1)

    if guardar_null:
        null_dist = null_tmp

    print(f'  Umbral global FWER aprox: {umbral_global_fwer:.4f}')

    return {
        'freqs': freqs,
        'pvals': pvals,
        'pvals_fwer': pvals_fwer,
        'umbral_punto_a_punto': umbral_punto_a_punto,
        'umbral_global_fwer': umbral_global_fwer,
        'max_null': max_null,
        'null_dist': null_dist,
        'n_perm': n_perm,
        'surrogate': surrogate,
        'alpha': alpha,
        'random_state': random_state,
    }


def procesar_eeg(eeg_data: np.ndarray, fs: float = FS_EEG, sujeto="S5", filtro_espacial=''):
    """
    Preprocesa EEG manteniendo una convención clara de dimensiones:
    entrada/salida = (n_canales, n_muestras).

    Si filtro_espacial es 'CAR' o 'Laplaciano', la salida queda como
    un solo canal derivado: (1, n_muestras). Por eso, fuera de esta
    función también deben actualizarse eeg_names y eeg_to_plot a un canal.
    """
    eeg = np.asarray(eeg_data, dtype=float).copy()
    if eeg.ndim != 2:
        raise ValueError(f"EEG debe ser 2D (n_canales, n_muestras). Recibido: {eeg.shape}")

    if sujeto == 'S4':
        idx_lateral_dominante = 6   # C4
        idx_NN1 = 3                # FC4
        idx_NN2 = 4                # Cz
    else:
        idx_lateral_dominante = 5   # C3
        idx_NN1 = 1                # FC3
        idx_NN2 = 4                # Cz

    # 1) Filtros temporales/artifact rejection sobre los canales originales.
    eeg = eeg - eeg.mean(axis=1, keepdims=True)
    eeg = aplicar_notch_multiple(eeg, fs=fs, freqs=FREQS_NOTCH, Q=Q_NOTCH, pad=PAD)

    # ICA no es válido/útil si ya redujiste a un solo canal. Se aplica solo con el montaje completo.
    if sujeto == "S5" and eeg.shape[0] == len(EEG_NAMES) and filtro_espacial not in ('CAR', 'Laplaciano'):
        eeg, _ica = ica_eliminar_componente(eeg, fs=fs, ch_names=EEG_NAMES, ic_to_remove=IC_TO_REMOVE)
    elif sujeto == "S5" and filtro_espacial in ('CAR', 'Laplaciano'):
        print("Aviso: se omite ICA porque se usará un filtro espacial de un solo canal derivado.")

    # 2) Filtro espacial. Mantener siempre forma (1, n_muestras) cuando se reduce a un canal.
    if filtro_espacial == 'CAR':
        canal_dom = eeg[idx_lateral_dominante:idx_lateral_dominante + 1, :]
        car_ref = eeg.mean(axis=0, keepdims=True)
        eeg = canal_dom - car_ref
        eeg = eeg - eeg.mean(axis=1, keepdims=True)

    elif filtro_espacial == 'Laplaciano':
        canal_dom = eeg[idx_lateral_dominante:idx_lateral_dominante + 1, :]
        ref_laplaciano = (
            eeg[idx_NN1:idx_NN1 + 1, :] +
            eeg[idx_NN2:idx_NN2 + 1, :]
        ) / 2.0
        eeg = canal_dom - ref_laplaciano
        eeg = eeg - eeg.mean(axis=1, keepdims=True)

    elif filtro_espacial in ('', None):
        pass
    else:
        raise ValueError("filtro_espacial debe ser '', None, 'CAR' o 'Laplaciano'.")

    eeg = np.atleast_2d(eeg)
    print(f"DIMENSIONES DEL EEG A PROCESAR : {eeg.shape}")

    eeg_final = aplicar_pasabanda_butter(
        eeg, fs=fs, f_low=BP_LOW, f_high=BP_HIGH, orden=BP_ORDER, pad=PAD
    )
    eeg_final = resamplear_profesional(eeg_final, fs, FS_FINAL)
    return eeg_final



def procesar_emg(emg_data: np.ndarray, fs: float = FS_EMG, forma:str = ''):
    FS = FS_EMG
    ORDEN_BP = BP_ORDER
    PAD_MUESTRAS = 5 * FS
    Q_NOTCH = 200
    emg_data = emg_data - emg_data.mean()
    # emg_data = emg_data[0]
    # print(f"FORMA DEL EMG A PROCESAR: {emg_data.shape}")
    
    emg_proc = aplicar_pasabanda_butter(
              emg_data,
              fs=FS,
              f_low = EMG_BAND[0],
              f_high = EMG_BAND[1],
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
    if forma == 'rectificada':
        # emg_proc = aplicar_pasabanda_butter(
        #           emg_proc,
        #           fs=FS,
        #           f_low= 0.5,
        #           f_high=22.5,
        #           orden=ORDEN_BP,
        #           pad=PAD_MUESTRAS,
        #       )
        emg_final = rectificar_senal(emg_proc, tipo='completa')
    
    elif forma == 'envolvente':
        #emg_final = rectificar_senal(emg_proc, tipo='completa') 
        emg_final = emg_proc
        for i in range(emg_final.shape[0]):
            emg_final[i, :] = emg_envelope(emg_final[i, :], fs=FS, tc_ms=20, rectify="full")
    elif forma == 'completa':
        emg_final = emg_proc
        
    #emg_final = resamplear_profesional(emg_final, 1000, 250)
  
 
    return emg_final


# ══════════════════════════════════════════════
# 8. COHERENCIA, LÍMITE DE CONFIANZA 
# ══════════════════════════════════════════════
def calcula_coherencia(emg, eeg, fs=FS_FINAL, fmin=5.0, fmax=45.0, mt_bandwidth=5.0, metodo_espectro = 'Multitapers'):
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



def segmentar_epochs_fijos(sig, fs, epoch_dur=2.0, overlap=0.0):
    """
    Convierte una señal 2D (n_muestras, n_canales) en epochs 3D
    (n_epochs, n_muestras_epoch, n_canales).
    """
    sig = np.asarray(sig)
    if sig.ndim != 2:
        raise ValueError('La señal debe tener forma (n_muestras, n_canales).')

    epoch_len = int(fs * epoch_dur)
    if epoch_len <= 0:
        raise ValueError('epoch_dur debe ser mayor que 0.')

    step = int(epoch_len * (1 - overlap))
    if step <= 0:
        raise ValueError('El overlap produce un step <= 0.')

    n_samples = sig.shape[0]
    if n_samples < epoch_len:
        raise ValueError('La señal es mas corta que la longitud del epoch solicitada.')

    starts = np.arange(0, n_samples - epoch_len + 1, step)
    epochs = np.stack([sig[s:s+epoch_len, :] for s in starts], axis=0)
    return epochs


def asegurar_muestras_canales(sig, nombres_canales=None, nombre_signal='señal'):
    """
    Devuelve una señal 2D como (n_muestras, n_canales).

    Tus funciones de carga/procesamiento suelen producir (n_canales, n_muestras),
    mientras que calcula_coherencia/segmentar_epochs_fijos esperan
    (n_muestras, n_canales). Esta función corrige automáticamente esa orientación.
    """
    sig = np.asarray(sig)

    if sig.ndim != 2:
        raise ValueError(f"{nombre_signal} debe ser 2D. Recibido: {sig.shape}")

    n_ch = len(nombres_canales) if nombres_canales is not None else None

    if n_ch is not None:
        if sig.shape[0] == n_ch and sig.shape[1] != n_ch:
            print(f"{nombre_signal}: transponiendo de {sig.shape} a {sig.T.shape} para usar (muestras, canales)")
            return sig.T
        if sig.shape[1] == n_ch:
            return sig

    # Regla práctica: si la primera dimensión es pequeña, probablemente son canales.
    if sig.shape[0] <= 64 and sig.shape[0] < sig.shape[1]:
        print(f"{nombre_signal}: transponiendo de {sig.shape} a {sig.T.shape} para usar (muestras, canales)")
        return sig.T

    return sig


def asegurar_epochs_muestras_canales(sig, nombres_canales=None, nombre_signal='señal'):
    """
    Devuelve una señal para multitaper trial_completo como:
      - 2D: (n_muestras, n_canales)
      - 3D: (n_epochs, n_muestras, n_canales)
    Corrige automáticamente si entra como (n_canales, n_muestras) o
    (n_epochs, n_canales, n_muestras).
    """
    sig = np.asarray(sig)

    if sig.ndim == 2:
        return asegurar_muestras_canales(sig, nombres_canales, nombre_signal)

    if sig.ndim == 3:
        n_ch = len(nombres_canales) if nombres_canales is not None else None

        if n_ch is not None:
            if sig.shape[1] == n_ch and sig.shape[2] != n_ch:
                print(f"{nombre_signal}: transponiendo de {sig.shape} a {(sig.transpose(0, 2, 1)).shape} para usar (epochs, muestras, canales)")
                return sig.transpose(0, 2, 1)
            if sig.shape[2] == n_ch:
                return sig

        # Regla práctica para datos 3D: (epochs, canales, muestras)
        if sig.shape[1] <= 64 and sig.shape[1] < sig.shape[2]:
            print(f"{nombre_signal}: transponiendo de {sig.shape} a {(sig.transpose(0, 2, 1)).shape} para usar (epochs, muestras, canales)")
            return sig.transpose(0, 2, 1)

        return sig

    raise ValueError(f"{nombre_signal} debe ser 2D o 3D. Recibido: {sig.shape}")


def segmentar_subepochs_2d_o_3d(sig, fs, epoch_dur=2.0, overlap=0.0,
                                nombres_canales=None, nombre_signal='señal'):
    """
    Segmenta en subepochs aceptando entrada 2D o 3D.

    Entrada 2D:
      (n_muestras, n_canales) o (n_canales, n_muestras)
    Entrada 3D:
      (n_epochs, n_muestras, n_canales) o (n_epochs, n_canales, n_muestras)

    Salida:
      (n_subepochs_total, n_muestras_subepoch, n_canales)
    """
    sig = asegurar_epochs_muestras_canales(sig, nombres_canales, nombre_signal)

    if sig.ndim == 2:
        return segmentar_epochs_fijos(sig, fs=fs, epoch_dur=epoch_dur, overlap=overlap)

    epochs = []
    for ep in range(sig.shape[0]):
        epochs_ep = segmentar_epochs_fijos(sig[ep], fs=fs, epoch_dur=epoch_dur, overlap=overlap)
        epochs.append(epochs_ep)

    return np.concatenate(epochs, axis=0)
# ══════════════════════════════════════════════
# VISUALIZACIÓN
# ══════════════════════════════════════════════
def pantalla_completa(fig):
    try:
        manager = fig.canvas.manager
        try:    manager.window.state('zoomed');  return
        except: pass
        try:    manager.window.showMaximized();  return
        except: pass
        try:    manager.frame.Maximize(False);    return
        except: pass
        fig.set_size_inches(9, 6)
    except:
        fig.set_size_inches(9, 6)
def estilo_ax(ax):
    ax.set_facecolor(BG_AX)
    ax.tick_params(colors=COLOR_TICK, labelsize=7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_color('#30363d')
    ax.spines['bottom'].set_color('#30363d')

def plot_coherencia(
    freqs,
    coherencia,
    emg_names,
    eeg_names,
    fmax=45,
    pie='',
    confidence_limit=None,
    eeg_to_plot=None
):
    """
    Grafica la coherencia EMG-EEG.

    Parámetros
    ----------
    freqs : array, shape (n_freqs,)
        Vector de frecuencias.
    coherencia : array, shape (n_freqs, n_emg, n_eeg)
        Matriz de coherencia.
    emg_names : list[str]
        Nombres de canales EMG.
    eeg_names : list[str]
        Nombres de canales EEG.
    fmax : float
        Frecuencia máxima a mostrar.
    pie : str
        Texto al pie de la figura.
    confidence_limit : float o None
        Límite de confianza a mostrar.
    eeg_to_plot : None, str, int, list[str], list[int]
        Canales EEG a graficar.
        - None: todos
        - "Fz": solo ese canal
        - ["Fz", "C3"]: esos canales
        - [0, 2]: por índices
    """
    mask = (freqs >= 5.0) & (freqs <= fmax)
    colores_bandas = ['#9B3461', '#C15579', '#E77692', '#f0B8AB', '#FAFAC4']

    # -----------------------------
    # Normalizar selección de canales EEG
    # -----------------------------
    if eeg_to_plot is None:
        eeg_idx = list(range(len(eeg_names)))
    else:
        # Si viene un solo elemento, convertirlo a lista
        if isinstance(eeg_to_plot, (str, int, np.integer)):
            eeg_to_plot = [eeg_to_plot]

        eeg_idx = []
        for ch in eeg_to_plot:
            if isinstance(ch, str):
                if ch not in eeg_names:
                    raise ValueError(
                        f'El canal EEG "{ch}" no existe. '
                        f'Canales disponibles: {eeg_names}'
                    )
                eeg_idx.append(eeg_names.index(ch))

            elif isinstance(ch, (int, np.integer)):
                if ch < 0 or ch >= len(eeg_names):
                    raise ValueError(
                        f'Índice EEG fuera de rango: {ch}. '
                        f'Debe estar entre 0 y {len(eeg_names)-1}'
                    )
                eeg_idx.append(int(ch))

            else:
                raise TypeError(
                    "eeg_to_plot debe ser None, str, int, list[str] o list[int]"
                )

        # Quitar duplicados conservando el orden
        eeg_idx = list(dict.fromkeys(eeg_idx))

    eeg_names_sel = [eeg_names[i] for i in eeg_idx]
    n_eeg = len(eeg_idx)

    if n_eeg == 0:
        raise ValueError("No se seleccionó ningún canal EEG para graficar.")

    # Calcular límite Y solo con los canales seleccionados
    coh_sel = coherencia[mask][:, :, eeg_idx]
    ylimit = np.nanmax(coh_sel)
    if not np.isfinite(ylimit) or ylimit <= 0:
        ylimit = 1.0

    # -----------------------------
    # Una figura por músculo EMG
    # -----------------------------
    for ei, ename in enumerate(emg_names):

        fig, axes = plt.subplots(
            n_eeg, 1,
            figsize=(14, 2.8 * n_eeg),
            sharex=True,
            squeeze=False
        )
        axes = axes.ravel()

        fig.patch.set_facecolor(BG_FIG)
        fig.subplots_adjust(
            left=0.09, right=0.98,
            top=0.93, bottom=0.07,
            hspace=0.12
        )

        for row, (yi, yname, ax) in enumerate(zip(eeg_idx, eeg_names_sel, axes)):
            color = COLORES[row % len(COLORES)]

            estilo_ax(ax)
            ax.plot(
                freqs[mask],
                coherencia[mask, ei, yi],
                color=color,
                lw=1.5,
                alpha=0.95
            )

            # if confidence_limit is not None and np.isfinite(confidence_limit):
            #     ax.axhline(
            #         confidence_limit,
            #         color='#FF6B6B',
            #         lw=1.0,
            #         linestyle='--',
            #         alpha=0.8,
            #         label=f'CL 95% = {confidence_limit:.3f}'
            #     )
            #     if row == 0:
            #         ax.legend(loc="upper right", fontsize=7)
            
            
            
            # Definición de las bandas que quieres incluir
            bandas_deseadas = ['θ', 'μ', 'β', 'γ']
            
            # Diccionario filtrado
            bandas_filtradas = {k: v for k, v in BANDAS.items() if k in bandas_deseadas}
          
            
            
            

            for (bname, (f0, f1)), col in zip(bandas_filtradas.items(), colores_bandas):
                if f0 < fmax:
                    ax.axvspan(f0, min(f1, fmax), alpha=0.15, color=col, zorder=0)
                    if row == 0:
                        ax.text(
                            (f0 + min(f1, fmax)) / 2,
                            ylimit * 1.02,
                            bname,
                            color='k',
                            fontsize=20,
                            ha='center',
                            va='bottom'
                        )

            ax.set_ylabel(
                yname,
                color=color,
                fontsize=25,
                rotation=0,
                labelpad=36,
                va='center',
                fontweight='bold'
                
            )
            ax.set_ylim(0, ylimit * 1.15)
            ax.set_xlim(5.0, fmax)
            ax.grid(
                True,
                color=COLOR_GRID,
                linewidth=0.5,
                linestyle='--',
                axis='y'
            )

            # if row == 0:
            #     ax.set_title(
            #         'Coherencia corticomuscular (CMC)',
            #         color=COLOR_LABEL,
            #         fontsize=20,
            #         pad=3
            #     )
                
            ax.tick_params(axis='both', labelsize=20)    

            if row < n_eeg - 1:
                ax.tick_params(axis = 'x',labelbottom=False)
            else:
                ax.set_xlabel('Frecuencia (Hz)', color=COLOR_LABEL, fontsize=25)
                
            

        # fig.suptitle(
        #     f'CMC EEG  —  EMG: {ename}  |  fs = {FS_FINAL} Hz',
        #     color=COLOR_TITLE,
        #     fontsize=11,
        #     fontweight='bold'
        # )

        # fig.text(
        #     0.5, 0.005, pie,
        #     ha='center',
        #     va='bottom',
        #     fontsize=8,
        #     color=COLOR_LABEL,
        #     style='italic',
        #     fontweight='bold'
        # )
        
        fig.subplots_adjust(
            left = 0.06,
            right = 0.94,
            top = 0.94,
            bottom = 0.1
        )
        
        #plt.savefig('grafica_'+pie+'.png', dpi=300, bbox_inches='tight')

        pantalla_completa(fig)        


def _extraer_espectro_desde_resultado(resultado, emg_idx=0, eeg_idx=0):
    """
    Extrae freqs y un vector de coherencia desde un resultado de calcula_todo().
    Por defecto toma EMG_INSTAR y el primer canal EEG disponible.
    """
    if not isinstance(resultado, dict):
        raise TypeError("Cada entrada de general debe ser un diccionario de resultados.")

    freqs = np.asarray(resultado["freqs"])
    coherencia = np.asarray(resultado["coherencia"])

    if coherencia.ndim == 1:
        espectro = coherencia
    elif coherencia.ndim == 2:
        espectro = coherencia[:, eeg_idx]
    elif coherencia.ndim == 3:
        emg_idx = min(int(emg_idx), coherencia.shape[1] - 1)
        eeg_idx = min(int(eeg_idx), coherencia.shape[2] - 1)
        espectro = coherencia[:, emg_idx, eeg_idx]
    else:
        raise ValueError(f"Forma inesperada de coherencia: {coherencia.shape}")

    return freqs, np.asarray(espectro, dtype=float)


def plot_coherencia_general(
    general,
    sujetos_orden=None,
    disposicion='horizontal',
    fmax=45,
    fmin=5,
    emg_idx=0,
    eeg_idx=0,
    guardar=None,
    mostrar=True,
    dpi=300
):
    """
    Grafica los espectros de coherencia corticomuscular de todos los sujetos
    almacenados en el diccionario `general`.

    Parámetros
    ----------
    general : dict
        Diccionario con una entrada por sujeto. Cada entrada debe contener,
        por lo menos, las claves 'freqs' y 'coherencia' generadas por calcula_todo().
    sujetos_orden : list[str] o None
        Orden de sujetos a graficar. Si es None usa el orden de inserción de general.
    disposicion : {'vertical', 'horizontal'}
        - 'vertical': 6 filas x 1 columna.
        - 'horizontal': 3 filas x 2 columnas, con 3 sujetos del lado izquierdo
          y 3 del lado derecho.
    fmax, fmin : float
        Rango de frecuencia a mostrar.
    emg_idx, eeg_idx : int
        Índices del canal EMG y EEG a extraer cuando la coherencia es 3D.
    guardar : str, pathlib.Path o None
        Ruta para guardar la figura. Si es None no se guarda.
    mostrar : bool
        Si True, llama a plt.show() al final.
    dpi : int
        Resolución de guardado.

    Notas
    -----
    - Debajo de cada gráfica se escribe el ID del sujeto.
    - Para S4, la etiqueta vertical se fuerza a 'C4'. En los demás sujetos se usa 'C3'.
    - Las etiquetas de bandas θ, μ, β y γ se comparten por columna: solo aparecen
      en el primer eje de cada columna.
    """
    if sujetos_orden is None:
        sujetos_orden = list(general.keys())
    mapa_ids_visual = {
        "S17": "SV1",
        "S6":  "SV2",
        "S7":  "SV3",
        "S18": "SK1",
        "S4":  "SK2",
        "S5":  "SK3"
    }

    sujetos_orden = [s for s in sujetos_orden if s in general]
    if len(sujetos_orden) == 0:
        raise ValueError("No hay sujetos válidos en el diccionario general.")

    disposicion = str(disposicion).lower().strip()
    if disposicion not in ('vertical', 'horizontal'):
        raise ValueError("disposicion debe ser 'vertical' u 'horizontal'.")

    n_sujetos = len(sujetos_orden)
    if disposicion == 'vertical':
        nrows, ncols = n_sujetos, 1
        # figsize = (8, max(2.0 * n_sujetos, 8))
        figsize = (8, 12)
    else:
        ncols = 2
        nrows = int(np.ceil(n_sujetos / ncols))
        # figsize = (16, max(2.6 * nrows, 7.5))
        figsize = (16, 8)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex='col', squeeze=False)
    fig.patch.set_facecolor(BG_FIG)

    bandas_deseadas = ['θ', 'μ', 'β', 'γ']
    bandas_filtradas = {k: v for k, v in BANDAS.items() if k in bandas_deseadas}
    colores_bandas = ['#9B3461', '#C15579', '#E77692', '#f0B8AB']

    # Extraer espectros primero para usar un mismo límite Y y que las figuras sean comparables.
    espectros = []
    y_max = 0.0
    for sujeto in sujetos_orden:
        freqs, coh = _extraer_espectro_desde_resultado(general[sujeto], emg_idx=emg_idx, eeg_idx=eeg_idx)
        mask = (freqs >= fmin) & (freqs <= fmax)
        if not np.any(mask):
            raise ValueError(f"El sujeto {sujeto} no tiene frecuencias dentro de {fmin}-{fmax} Hz.")
        y_local = np.nanmax(coh[mask])
        if np.isfinite(y_local):
            y_max = max(y_max, float(y_local))
        espectros.append((sujeto, freqs, coh, mask))

    if not np.isfinite(y_max) or y_max <= 0:
        y_max = 1.0
    y_lim = y_max * 1.15

    # Ubicación: en horizontal se llena por columnas para obtener 3 izq. y 3 der.
    def posicion(idx):
        if disposicion == 'vertical':
            return idx, 0
        mitad = int(np.ceil(n_sujetos / 2))
        if idx < mitad:
            return idx, 0
        return idx - mitad, 1

    for idx, (sujeto, freqs, coh, mask) in enumerate(espectros):
        row, col = posicion(idx)
        ax = axes[row, col]
        estilo_ax(ax)

        #ax.plot(freqs[mask], coh[mask], color=COLORES[0], lw=1.5, alpha=0.95)
        id_visual = mapa_ids_visual.get(str(sujeto), str(sujeto))
        ax.plot(
            freqs[mask],
            coh[mask],
            color=COLORES[0],
            lw=1.5,
            alpha=0.95,
            label=id_visual#str(sujeto)
        )
        
        ax.legend(
            loc='upper right',
            fontsize=11,
            frameon=True,
            facecolor='white',
            edgecolor=COLORES[0],
            framealpha=0.85
        )

        # Sombreado de bandas en todos los ejes; nombres solo arriba de cada columna.
        for (bname, (f0, f1)), color_banda in zip(bandas_filtradas.items(), colores_bandas):
            if f0 < fmax and f1 > fmin:
                ax.axvspan(max(f0, fmin), min(f1, fmax), alpha=0.15, color=color_banda, zorder=0)
                if row == 0:
                    ax.text(
                        (max(f0, fmin) + min(f1, fmax)) / 2,
                        y_lim * 0.98,
                        bname,
                        color='k',
                        fontsize=18,
                        ha='center',
                        va='top'
                    )

        etiqueta_eeg = 'C4' if str(sujeto).upper() == 'S4' else 'C3'
        ax.set_ylabel(
            etiqueta_eeg,
            color=COLORES[0],
            fontsize=20,
            rotation=0,
            labelpad=34,
            va='center',
            fontweight='bold'
        )
        ax.set_xlim(fmin, fmax)
        ax.set_ylim(0, y_lim)
        ax.grid(True, color=COLOR_GRID, linewidth=0.5, linestyle='--', axis='y')
        ax.tick_params(axis='both', labelsize=15)


        # La etiqueta de frecuencia se comparte por columna: solo en la fila inferior usada.
        es_ultima_fila_col = row == max(posicion(i)[0] for i in range(n_sujetos) if posicion(i)[1] == col)
        if es_ultima_fila_col:
            ax.set_xlabel('Frecuencia (Hz)', color=COLOR_LABEL, fontsize=20, )
        else:
            ax.tick_params(axis='x', labelbottom=False)

    # Apagar ejes sobrantes si n_sujetos no llena la matriz.
    usados = {posicion(i) for i in range(n_sujetos)}
    for r in range(nrows):
        for c in range(ncols):
            if (r, c) not in usados:
                axes[r, c].axis('off')

    fig.subplots_adjust(left=0.18, right=0.98, top=0.96, bottom=0.10, hspace=0.15, wspace=0.22)

    # if guardar is not None:
    #     guardar = Path(guardar)
    #     guardar.parent.mkdir(parents=True, exist_ok=True)
    #     fig.savefig(guardar, dpi=dpi, facecolor=fig.get_facecolor())#bbox_inches='tight'
    #     print(f"Figura general guardada en: {guardar}")

    if mostrar:
        plt.show()

    return fig, axes


def plot_señales_psd(emg_rs, eeg_filt, emg_names, eeg_names,
                     canal_eeg=5, canal_emg=0, fmax=100):
    """
    Figura con 4 subplots:
      - Señal temporal EMG (un canal)
      - PSD del EMG
      - Señal temporal EEG (un canal)
      - PSD del EEG
    """
    # FS_EEGAUX = FS_EEG
    # FS_EEG = 250
    #FS_FINAL = 1000
    t_emg = np.arange(emg_rs.shape[0])  / FS_FINAL
    t_eeg = np.arange(eeg_filt.shape[0]) / FS_EEG

    f_emg, p_emg = signal.welch(emg_rs[:,  canal_emg], fs=FS_FINAL,
                                 nperseg=int(FS_FINAL * SEG_DUR))
    f_eeg, p_eeg = signal.welch(eeg_filt[:, canal_eeg], fs=FS_FINAL,
                                 nperseg=int(FS_EEG  * SEG_DUR))

    mask_emg = f_emg <= fmax
    mask_eeg = f_eeg <= fmax

    nombre_emg = emg_names[canal_emg] if emg_names else f"EMG_{canal_emg}"
    nombre_eeg = eeg_names[canal_eeg] if eeg_names else f"EEG_{canal_eeg}"

    # FS_EEG = FS_EEGAUX


    colores_bandas = ['#9B3461', '#C15579', '#E77692', '#f0B8AB', '#FAFAC4']

    fig, axes = plt.subplots(2, 2, figsize=(16, 7))
    fig.patch.set_facecolor(BG_FIG)
    fig.subplots_adjust(left=0.07, right=0.97,
                        top=0.91, bottom=0.09,
                        hspace=0.35, wspace=0.25)

    # ── Señal temporal EMG ───────────────────────────────────────────
    ax = axes[0, 0]
    estilo_ax(ax)
    ax.plot(t_emg, emg_rs[:, canal_emg], lw=0.6, color='#6FC7B7', alpha=0.9)
    ax.set_title(f"EMG filtrado y resampleado  —  {nombre_emg}",
                 color=COLOR_LABEL, fontsize=9)
    ax.set_xlabel("Tiempo (s)", color=COLOR_LABEL, fontsize=8)
    ax.set_ylabel("Amplitud", color=COLOR_LABEL, fontsize=8)
    ax.grid(True, color=COLOR_GRID, linewidth=0.4, linestyle='--')

    # ── PSD EMG ──────────────────────────────────────────────────────
    ax = axes[0, 1]
    estilo_ax(ax)
    ax.plot(f_emg[mask_emg], p_emg[mask_emg], lw=1.4, color='#6FC7B7')
    for (bname, (f0, f1)), col in zip(BANDAS.items(), colores_bandas):
        ax.axvspan(f0, min(f1, fmax), alpha=0.15, color=col, zorder=0)
        ax.text((f0 + f1) / 2, p_emg[mask_emg].max() * 1.5, bname,
                color=col, fontsize=6, ha='center', va='bottom')
    ax.set_title(f"PSD  —  {nombre_emg}", color=COLOR_LABEL, fontsize=9)
    ax.set_xlabel("Frecuencia (Hz)", color=COLOR_LABEL, fontsize=8)
    ax.set_ylabel("PSD", color=COLOR_LABEL, fontsize=8)
    ax.set_xlim(0, fmax)
    ax.grid(True, color=COLOR_GRID, linewidth=0.4, linestyle='--')

    # ── Señal temporal EEG ───────────────────────────────────────────
    ax = axes[1, 0]
    estilo_ax(ax)
    ax.plot(t_eeg, eeg_filt[:, canal_eeg], lw=0.6, color='#81DCEE', alpha=0.9)
    ax.set_title(f"EEG filtrado  —  {nombre_eeg}",
                 color=COLOR_LABEL, fontsize=9)
    ax.set_xlabel("Tiempo (s)", color=COLOR_LABEL, fontsize=8)
    ax.set_ylabel("Amplitud (μV)", color=COLOR_LABEL, fontsize=8)
    ax.grid(True, color=COLOR_GRID, linewidth=0.4, linestyle='--')

    # ── PSD EEG ──────────────────────────────────────────────────────
    ax = axes[1, 1]
    estilo_ax(ax)
    ax.plot(f_eeg[mask_eeg], p_eeg[mask_eeg], lw=1.4, color='#81DCEE')
    for (bname, (f0, f1)), col in zip(BANDAS.items(), colores_bandas):
        ax.axvspan(f0, min(f1, fmax), alpha=0.15, color=col, zorder=0)
        ax.text((f0 + f1) / 2, p_eeg[mask_eeg].max() * 1.5, bname,
                color=col, fontsize=6, ha='center', va='bottom')
    ax.set_title(f"PSD  —  {nombre_eeg}", color=COLOR_LABEL, fontsize=9)
    ax.set_xlabel("Frecuencia (Hz)", color=COLOR_LABEL, fontsize=8)
    ax.set_ylabel("PSD (μV²/Hz)", color=COLOR_LABEL, fontsize=8)
    ax.set_xlim(0, fmax)
    ax.grid(True, color=COLOR_GRID, linewidth=0.4, linestyle='--')

    fig.suptitle("Señales filtradas y densidades espectrales de potencia",
                 color=COLOR_TITLE, fontsize=11, fontweight='bold')    

def concatenate_eeg_list(signals_list, fs=FS_FINAL, ms=40):
    """
    Concatena una lista de arrays (7, muestras) usando cross-fade.
    """
    if not signals_list:
        return None
    
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

# =========================
# FUNCIÓN HARDLIMIT
# =========================
def hardlim(x):
    return np.where(x >= 0, 1, 0)


# =========================
# FUNCIÓN DE ENTRENAMIENTO INSTAR
# =========================

def entrenar_instar(*vectores, epochs=1000, alpha=0.1, bias=50, random_state=42):
    data = [np.asarray(v) for v in vectores]

    for v in data:
        if v.ndim != 1:
            raise ValueError("Todos los datos de entrada deben ser vectores de una dimensión.")

    longitudes = [len(v) for v in data]
    if len(set(longitudes)) != 1:
        raise ValueError("Todos los vectores deben tener la misma longitud.")

    data = np.array(data)
    num_features = data.shape[1]

    # rng = np.random.default_rng(random_state)
    # weights = rng.random(num_features)
    weights = np.zeros(num_features)
    for epoch in range(epochs):
        for sample in data:
            z = np.dot(weights, sample) + bias
            a = hardlim(z)
            weights = weights + alpha * a * (sample - weights)

    return weights


def obtener_emg_instar(emg, epochs=1000, alpha=0.01, bias=50):
    """
    Convierte 3 canales EMG en una sola señal representativa usando Instar.

    Acepta:
      - 2D: (3, muestras) o (muestras, 3)
      - 3D: (epochs, 3, muestras) o (epochs, muestras, 3)

    Devuelve:
      - si entra 2D: (1, muestras), compatible con Welch
      - si entra 3D: (epochs, muestras, 1), compatible con Multitapers
    """
    emg = np.asarray(emg)

    if emg.ndim == 2:
        if emg.shape[0] == 3:
            emg_ch = emg
        elif emg.shape[1] == 3:
            emg_ch = emg.T
        else:
            raise ValueError(
                f"Para Instar se esperaban 3 canales EMG. Forma recibida: {emg.shape}"
            )

        emg_instar = entrenar_instar(
            emg_ch[0, :],
            emg_ch[1, :],
            emg_ch[2, :],
            epochs=epochs,
            alpha=alpha,
            bias=bias
        )
        return emg_instar[np.newaxis, :]

    if emg.ndim == 3:
        # Normalizar a (epochs, muestras, 3)
        if emg.shape[1] == 3 and emg.shape[2] != 3:
            emg = emg.transpose(0, 2, 1)
        elif emg.shape[2] == 3:
            pass
        else:
            raise ValueError(
                f"Para Instar se esperaban 3 canales EMG en alguna dimensión. Forma recibida: {emg.shape}"
            )

        emg_instar_epochs = []
        for ep in range(emg.shape[0]):
            emg_ep = emg[ep]  # (muestras, 3)
            emg_instar = entrenar_instar(
                emg_ep[:, 0],
                emg_ep[:, 1],
                emg_ep[:, 2],
                epochs=epochs,
                alpha=alpha,
                bias=bias
            )
            emg_instar_epochs.append(emg_instar[:, np.newaxis])

        return np.stack(emg_instar_epochs, axis=0)

    raise ValueError(
        f"EMG debe ser 2D o 3D para calcular Instar. Forma recibida: {emg.shape}"
    )


# ══════════════════════════════════════════════
# PRINCIPAL
# ══════════════════════════════════════════════

def calcula_todo(emg, eeg, RUTA_SALIDA, emg_channels=None, eeg_channels=None, t_start=None, t_end=None,
                 emg_names=None, eeg_names=None, graficar=False, PIE_FIGURA="", envolvente=True,
                 epoch_mode='trial_completo', epoch_dur=2.0, epoch_overlap=0.0,
                 mt_bandwidth=5.0, eeg_to_plot=None, metodo_espectro='Multitapers',
                 alpha_confianza=0.05, usar_permutaciones=False, n_perm=500,
                 surrogate='circular_shift', min_shift_s=1.0, random_state=None,
                 guardar_null=False):
    """
    Calcula coherencia por Welch o Multitapers.

    Convención interna corregida:
      - Welch usa señales 2D (n_muestras, n_canales)
      - Multitapers acepta 2D (n_muestras, n_canales) o 3D
        (n_epochs, n_muestras, n_canales)
      - Si epoch_mode='subepochs', segmenta señales 2D o cada trial de señales 3D.

    Esta versión corrige automáticamente entradas tipo (canales, muestras),
    que eran la causa del error "La señal es mas corta...".
    """
    print(f"\nMetadatos:\n  {PIE_FIGURA}")

    emg_names = emg_names if emg_names is not None else [f"EMG_{i}" for i in range(np.asarray(emg).shape[-1])]
    eeg_names = eeg_names if eeg_names is not None else [f"EEG_{i}" for i in range(np.asarray(eeg).shape[-1])]

    CL = np.nan
    K_mt = None
    L_mt = None
    TW_mt = None
    stats_perm = None
    CL_analitico = np.nan
    CL_permutacion = np.nan

    if metodo_espectro == 'Welch':
        emg_for_coh = asegurar_muestras_canales(emg, emg_names, 'EMG')
        eeg_for_coh = asegurar_muestras_canales(eeg, eeg_names, 'EEG')

        print('\nModo de coherencia: Welch')
        print(f'EMG para Welch: {emg_for_coh.shape}')
        print(f'EEG para Welch: {eeg_for_coh.shape}')

        freqs, coherencia = calcula_coherencia(
            emg_for_coh,
            eeg_for_coh,
            fs=FS_FINAL,
            metodo_espectro=metodo_espectro
        )

        # Limite de confianza aproximado original para Welch.
        L = n_segmentos(min(emg_for_coh.shape[0], eeg_for_coh.shape[0]), fs=FS_FINAL)
        #CL = confidence_limit(L) if L > 1 else np.nan
        #CL = NaN
        print(f"Segmentos Welch estimados: {L}; CL aprox: {CL}")

    elif metodo_espectro == 'Multitapers':
        if epoch_mode == 'trial_completo':
            emg_for_coh = asegurar_epochs_muestras_canales(emg, emg_names, 'EMG')
            eeg_for_coh = asegurar_epochs_muestras_canales(eeg, eeg_names, 'EEG')

            print('\nModo de coherencia: multitapers, trial completo')
            print(f'EMG para coherencia: {emg_for_coh.shape}')
            print(f'EEG para coherencia: {eeg_for_coh.shape}')

        elif epoch_mode == 'subepochs':
            emg_for_coh = segmentar_subepochs_2d_o_3d(
                emg, fs=FS_FINAL, epoch_dur=epoch_dur, overlap=epoch_overlap,
                nombres_canales=emg_names, nombre_signal='EMG'
            )
            eeg_for_coh = segmentar_subepochs_2d_o_3d(
                eeg, fs=FS_FINAL, epoch_dur=epoch_dur, overlap=epoch_overlap,
                nombres_canales=eeg_names, nombre_signal='EEG'
            )

            print(f'\nModo de coherencia: multitapers, subepochs de {epoch_dur:.2f} s con overlap={epoch_overlap:.2f}')
            print(f'EMG epochs: {emg_for_coh.shape}')
            print(f'EEG epochs: {eeg_for_coh.shape}')

        else:
            raise ValueError("epoch_mode debe ser 'trial_completo' o 'subepochs'.")

        freqs, coherencia = calcula_coherencia(
            emg_for_coh,
            eeg_for_coh,
            fs=FS_FINAL,
            fmin=5,
            fmax=45,
            mt_bandwidth=mt_bandwidth,
            metodo_espectro=metodo_espectro
        )

        # Limite de confianza aproximado para multitapers.
        # MNE devuelve coherencia en magnitud; la funcion calcula Ccrit para magnitud.
        n_epochs_mt = emg_for_coh.shape[0] if np.asarray(emg_for_coh).ndim == 3 else 1
        n_times_mt = emg_for_coh.shape[1] if np.asarray(emg_for_coh).ndim == 3 else emg_for_coh.shape[0]
        CL, K_mt, L_mt, TW_mt = confidence_limit_multitaper_aprox(
            n_epochs=n_epochs_mt,
            n_times=n_times_mt,
            fs=FS_FINAL,
            mt_bandwidth=mt_bandwidth,
            alpha=alpha_confianza
        )

        print("\nLimite de confianza multitaper aproximado")
        print(f"  alpha          : {alpha_confianza}")
        print(f"  n_epochs       : {n_epochs_mt}")
        print(f"  n_times/epoch  : {n_times_mt}")
        print(f"  duracion epoch : {n_times_mt / FS_FINAL:.3f} s")
        print(f"  mt_bandwidth   : {mt_bandwidth} Hz")
        print(f"  TW aprox       : {TW_mt:.3f}")
        print(f"  K tapers aprox : {K_mt}")
        print(f"  L efectivo     : {L_mt}")
        print(f"  CL aprox       : {CL:.4f}")
        print("  Nota: para inferencia formal en CMC, se recomienda validar con permutaciones/surrogates.")
        CL_analitico = CL

        if usar_permutaciones:
            stats_perm = test_permutacion_coherencia_multitaper(
                emg_for_coh,
                eeg_for_coh,
                coherencia_obs=coherencia,
                fs=FS_FINAL,
                fmin=5,
                fmax=45,
                mt_bandwidth=mt_bandwidth,
                n_perm=n_perm,
                alpha=alpha_confianza,
                surrogate=surrogate,
                min_shift_s=min_shift_s,
                random_state=random_state,
                guardar_null=guardar_null
            )
            CL_permutacion = stats_perm['umbral_global_fwer']
            CL = CL_permutacion
            print(f"  CL usado para graficar: umbral global por permutaciones = {CL:.4f}")

    else:
        raise ValueError("metodo_espectro debe ser 'Welch' o 'Multitapers'.")

    # Ajustar nombres si no coinciden con la dimensionalidad real de la coherencia.
    n_emg_real = coherencia.shape[1]
    n_eeg_real = coherencia.shape[2]

    if len(emg_names) != n_emg_real:
        print(f"Aviso: emg_names tiene {len(emg_names)} nombres, pero la coherencia tiene {n_emg_real} canal(es) EMG. Se reajustan nombres.")
        emg_names = [f"EMG_{i}" for i in range(n_emg_real)]

    if len(eeg_names) != n_eeg_real:
        print(f"Aviso: eeg_names tiene {len(eeg_names)} nombres, pero la coherencia tiene {n_eeg_real} canal(es) EEG. Se reajustan nombres.")
        if n_eeg_real == 1:
            if isinstance(eeg_to_plot, str):
                etiqueta = eeg_to_plot
            elif isinstance(eeg_to_plot, (list, tuple)) and len(eeg_to_plot) == 1 and isinstance(eeg_to_plot[0], str):
                etiqueta = eeg_to_plot[0]
            else:
                etiqueta = "EEG_derivado"
            eeg_names = [etiqueta]
            eeg_to_plot = [etiqueta]
        else:
            eeg_names = [f"EEG_{i}" for i in range(n_eeg_real)]
            eeg_to_plot = None

    # Encuentra la coherencia máxima registrada excluyendo bandas no deseadas.
    excluye_freqs = ["δ", "θ", "μ", "γ", "Otros"]
    mask_excluye = np.zeros(len(freqs), dtype=bool)

    for banda in excluye_freqs:
        fl, fh = BANDAS[banda]
        mask_excluye |= (freqs >= fl) & (freqs <= fh)

    coherencia_excluye = coherencia.copy()
    coherencia_excluye[mask_excluye, :, :] = 0

    idx = coherencia_excluye.argmax()
    i_freq, i_emg, i_eeg = np.unravel_index(idx, coherencia_excluye.shape)
    print(f" Coherencia maxima: {coherencia[i_freq, i_emg, i_eeg]:.4f}")
    print(f" Frecuencia       : {freqs[i_freq]:.1f}")
    print(f" Canal EMG        : {emg_names[i_emg]}")
    print(f" Canal EEG        : {eeg_names[i_eeg]}")
    print(f"Shape coherencia  : {coherencia.shape}")

    if graficar:
        # Graficar una señal 2D representativa en forma (muestras, canales).
        if metodo_espectro == 'Multitapers' and np.asarray(emg_for_coh).ndim == 3:
            emg_plot = emg_for_coh[0]
        else:
            emg_plot = asegurar_muestras_canales(emg, emg_names, 'EMG plot') if np.asarray(emg).ndim == 2 else emg_for_coh[0]

        if metodo_espectro == 'Multitapers' and np.asarray(eeg_for_coh).ndim == 3:
            eeg_plot = eeg_for_coh[0]
        else:
            eeg_plot = asegurar_muestras_canales(eeg, eeg_names, 'EEG plot') if np.asarray(eeg).ndim == 2 else eeg_for_coh[0]

        plot_señales_psd(emg_plot, eeg_plot, emg_names, eeg_names, canal_eeg=0, canal_emg=0)
        plot_coherencia(
            freqs, coherencia, emg_names, eeg_names,
            fmax=45, pie=PIE_FIGURA, confidence_limit= 0,
            eeg_to_plot=eeg_to_plot
        )

       # plt.show()

    return {
        "freqs": freqs,
        "coherencia": coherencia,
        "limiteC": CL,
        "limiteC_analitico": CL_analitico,
        "limiteC_permutacion": CL_permutacion,
        "stats_perm": stats_perm,
        "K_mt": K_mt,
        "L_mt": L_mt,
        "TW_mt": TW_mt,
        "alpha_confianza": alpha_confianza,
        "emg_shape_usada": emg_for_coh.shape,
        "eeg_shape_usada": eeg_for_coh.shape,
        "epoch_mode": epoch_mode,
        "metodo_espectro": metodo_espectro
    }

# ══════════════════════════════════════════════
# PUNTO DE ENTRADA
# ══════════════════════════════════════════════
if __name__ == "__main__":
    
    ruta_base = r"C:\Users\lrl13\Downloads\DATA CMC 2"
    RUTA_SALIDA = r"C:\Users\lrl13\Downloads\Entrenamiento imaginacion2\Entrenamiento imaginacion\Reportes todos los metodos"
    tipos = ["EEG","EMG"]
    sujetos = ["S4","S5","S6","S7","S17","S18"]
    #EMG_NAMES = ["Flexor carpi ulnaris","Right Extensor carpi radialis longus","Extensor carpi ulnaris"]#Nombres en inglés 
    EMG_NAMES = ["Flexor Cubital del Carpo","Extensor Radial Largo del Carpo","Extensor Cubital del Carpo"]#Nombres en Español
    EEG_NAMES = ["Fz","FC3","FCz","FC4","Cz","C3","C4"]
    eeg_to_plot = ["C3"] #Canal o canales que se van a graficar
    FREQS_NOTCH = (60.0, 120.0)
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
    BP_ORDER =  4
    
    #Configuración de Obtención de CMC
    metodo_espectro = 'Multitapers'#'Multitapers'#'Welch'
    concatenar = False
    forma = 'completa'# 'rectificada' 'envolvente'
    bootstrap = True
    filtro_espacial = 'Laplaciano' #'CAR'

    # Configuracion para multitapers
    epoch_mode = 'subepochs'  # 'trial_completo' o 'subepochs'
    epoch_dur = 2.0
    epoch_overlap = 0.0
    mt_bandwidth = 3.0#3.0
    alpha_confianza = 0.05  # 0.05 = limite 95%

    # Configuracion de permutaciones/surrogates para multitapers
    usar_permutaciones = False  # Cambia a True para activar la prueba no parametrica
    n_perm = 100
    surrogate = 'shuffle_epochs'  # 'circular_shift' o 'shuffle_epochs'
    min_shift_s = 1.0
    random_state = 42
    guardar_null = False
    graficar = True

    # Configuración de la figura final con los 6 sujetos del diccionario general.
    # Cambia a 'vertical' para una columna con los 6 espectros,
    # o a 'horizontal' para 3 espectros a la izquierda y 3 a la derecha.
    disposicion_figura_general = 'vertical'#'horizontal'
    graficar_general_al_final = True
    guardar_figura_general = True
    
    #np.random.seed(random_state)
    
    nombre_reporte = metodo_espectro + '_' + forma
    if metodo_espectro == 'Multitapers':
        nombre_reporte = nombre_reporte + '_' + epoch_mode
    if concatenar:
        nombre_reporte = nombre_reporte + '_' + 'concatenado'
    if bootstrap:
        nombre_reporte = nombre_reporte + '_' + 'bootstrap'
    if filtro_espacial != '':
        nombre_reporte = nombre_reporte + '_' + filtro_espacial
    
        
        
    print('EL ARCHIVO SE LLAMARÁ: '+ nombre_reporte)

    def nombres_eeg_procesados(sujeto_actual, filtro_actual):
        # Si CAR/Laplaciano dejan un solo canal derivado, las etiquetas y la selección
        # de canales para graficar deben coincidir con la dimensión real de coherencia.
        if filtro_actual in ('CAR', 'Laplaciano'):
            idx_eeg_derivado = 6 if sujeto_actual == 'S4' else 5
            nombres = [EEG_NAMES[idx_eeg_derivado]]
            return nombres, nombres
        return EEG_NAMES, eeg_to_plot


    pares_analisis = generar_pares(ruta_base,tipos,sujetos)

    print("\nPARES PARA ANALISIS\n")
    print(pares_analisis)
   
    general = {}

    #if concatenar:
    for sujeto in sujetos:
        eeg_list = []
        emg_list = []
        general[sujeto] = {}
        EEG_NAMES_PROC, eeg_to_plot_proc = nombres_eeg_procesados(sujeto, filtro_espacial)
        
        #Cargar las señales y preprocesarlas
        for i in range(4):
            eeg = load_eeg(pares_analisis[sujeto][i][0])
            eeg = eeg - eeg.mean()
            eeg_final = procesar_eeg(eeg, fs = FS_EEG, sujeto = sujeto,filtro_espacial = filtro_espacial)
            eeg_list.append(eeg_final)

            if i < 4:
                
                emg_tmp = load_emg(pares_analisis[sujeto][i][1])
                emg_tmp = procesar_emg(emg_tmp,fs = FS_EMG, forma = forma)
                emg_list.append(emg_tmp)
                
        if bootstrap:
            #Repite las señales
            for k in range(3):
                for j in range(4):
                    emg_list.append(emg_list[j])
                    eeg_list.append(eeg_list[j])
                    
                    
        if (metodo_espectro == 'Welch') & (concatenar | bootstrap):  
            emg = concatenate_eeg_list(emg_list,fs = FS_FINAL)    
            eeg = concatenate_eeg_list(eeg_list,fs = FS_FINAL)
            fig, ax = plt.subplots()
            fig.patch.set_facecolor(BG_FIG)
            ax.plot(eeg[0,:], lw=0.6, color='#6FC7B7', alpha=0.9)
            ax.set_title(f"EEG preprocesado y concatenado",
                    color=COLOR_LABEL, fontsize=9)
            ax.set_xlabel("Número de muestras", color=COLOR_LABEL, fontsize=8)
            ax.set_ylabel("Amplitud", color=COLOR_LABEL, fontsize=8)
            ax.grid(True, color=COLOR_GRID, linewidth=0.4, linestyle='--')
            estilo_ax(ax)
#            plt.show()

            emg_instar = obtener_emg_instar(
                emg,
                epochs=100,
                alpha=0.01,
                bias=0
            )
            plt.figure()
            plt.plot(emg_instar[0, :])
            plt.title("EMG representativo por Instar - Welch")
            #plt.show()

            results = calcula_todo(
                emg    = emg_instar,
                eeg    = eeg,
                RUTA_SALIDA = RUTA_SALIDA,
                emg_names = ["EMG_INSTAR"],
                eeg_names = EEG_NAMES_PROC,
                t_start      = None,          # None = solapamiento maximo
                t_end        = None,
                graficar     = graficar,
                PIE_FIGURA = sujeto,
                #envolvente = envolvente,
                metodo_espectro =  'Welch'
            )
            general[sujeto] = results
        #elif metodo_espectro == 'Welch' & not(concatenar):


        elif (metodo_espectro == 'Multitapers'): #& (not (concatenar | bootstrap)):
            if bootstrap: 
                emg_epochs = concatenate_eeg_list(emg_list,fs = FS_FINAL)#,fs=FS_EMG)    
                eeg_epochs = concatenate_eeg_list(eeg_list)
            
            else:
                emg_epochs = np.stack(emg_list, axis=0)
                eeg_epochs = np.stack(eeg_list, axis=0)
    
                emg_epochs = np.transpose(emg_epochs, (0, 2, 1))
                eeg_epochs = np.transpose(eeg_epochs, (0, 2, 1))

            emg_instar_epochs = obtener_emg_instar(
                emg_epochs,
                epochs=100,
                alpha=0.05,
                bias= 50
            )
            
            plt.figure()
            plt.plot(emg_instar_epochs[0, :])
            plt.title("EMG representativo por Instar - Welch")
            
            
            print("EMG Instar epochs shape:", emg_instar_epochs.shape)
            print("EEG epochs shape:", eeg_epochs.shape)

            results = calcula_todo(
                emg = emg_instar_epochs,
                eeg = eeg_epochs,
                RUTA_SALIDA = RUTA_SALIDA,
                emg_names=["EMG_INSTAR"],
                eeg_names=EEG_NAMES_PROC,
                graficar=graficar,
                PIE_FIGURA=sujeto,
                #envolvente=True,
                mt_bandwidth=mt_bandwidth,
                epoch_mode=epoch_mode,
                epoch_dur=epoch_dur,
                epoch_overlap=epoch_overlap,
                eeg_to_plot=eeg_to_plot_proc,
                metodo_espectro = 'Multitapers',
                alpha_confianza=alpha_confianza,
                usar_permutaciones=usar_permutaciones,
                n_perm=n_perm,
                surrogate=surrogate,
                min_shift_s=min_shift_s,
                random_state=random_state,
                guardar_null=guardar_null,

                )
            general[sujeto] = results

    nombre_archivo = f'Reporte_GENNERAL_{nombre_reporte}.npy'
    RUTA_SALIDA = Path(RUTA_SALIDA)
    RUTA_SALIDA = RUTA_SALIDA / nombre_archivo
    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    #np.save(RUTA_SALIDA, general)

    # Figura final: espectros de coherencia de todos los sujetos en general.
    if graficar_general_al_final:
        ruta_figura_general = None
        if guardar_figura_general:
            ruta_figura_general = RUTA_SALIDA.parent / f'CMC_6_sujetos_{nombre_reporte}_{disposicion_figura_general}.png'

        plot_coherencia_general(
            general=general,
            sujetos_orden=["S17","S6","S7","S18","S4","S5"],
            disposicion=disposicion_figura_general,
            fmin=5,
            fmax=45,
            emg_idx=0,
            eeg_idx=0,
            guardar=ruta_figura_general,
            mostrar=True,
            dpi=300
        )

   

