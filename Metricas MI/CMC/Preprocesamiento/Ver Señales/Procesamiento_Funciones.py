import numpy as np
import pywt

import mne
from mne.preprocessing import ICA

from scipy.signal import iirnotch, sosfiltfilt, tf2sos, butter, resample_poly



# %% CARGAR DATOS EEG

def cargar_npy_eeg(ruta_npy: str, n_canales_esperados: int | None = None) -> np.ndarray:
    datos = np.load(ruta_npy)
    if datos.ndim != 2:
        raise ValueError(f"Se esperaba un arreglo 2D (canales x muestras). Recibido: {datos.shape}")
    if n_canales_esperados is not None and datos.shape[0] != n_canales_esperados:
        raise ValueError(
            f"Se esperaban {n_canales_esperados} canales, pero el archivo trae {datos.shape[0]} canales: {datos.shape}"
        )
    return np.asarray(datos, dtype=float)


# %% CARGAR DATOS EMG

def cargar_npy_emg(
    ruta_npy: str,
    n_canales_esperados: int | None = None,
    permitir_transponer: bool = True,
) -> np.ndarray:

    datos = np.load(ruta_npy)

    if datos.ndim == 1:
        datos = datos.reshape(1, -1)
    elif datos.ndim != 2:
        raise ValueError(f"Se esperaba 1D o 2D para EMG. Recibido: {datos.shape}")

    # Si especificas canales esperados, usamos eso para decidir transposición
    if permitir_transponer and n_canales_esperados is not None:
        if datos.shape[0] != n_canales_esperados and datos.shape[1] == n_canales_esperados:
            datos = datos.T

    # Si no especificas canales esperados, heurística: usualmente muestras >> canales
    if permitir_transponer and n_canales_esperados is None:
        if datos.shape[0] > datos.shape[1]:
            datos = datos.T

    if n_canales_esperados is not None and datos.shape[0] != n_canales_esperados:
        raise ValueError(
            f"Se esperaban {n_canales_esperados} canales EMG, pero el archivo trae {datos.shape[0]}: {datos.shape}"
        )

    return np.asarray(datos, dtype=float)


# %% CARGAR DATOS DINAM

def cargar_npy_dinam(
    ruta_npy: str,
    permitir_multicanal: bool = False,
) -> np.ndarray:

    datos = np.load(ruta_npy)

    if datos.ndim == 1:
        vec = datos
    elif datos.ndim == 2:
        if 1 in datos.shape:
            vec = datos.reshape(-1)
        else:
            if not permitir_multicanal:
                raise ValueError(
                    f"DINAM debe ser 1D (o 2D con una dimensión = 1). Recibido: {datos.shape}"
                )
            vec = datos[0].reshape(-1)
    else:
        raise ValueError(f"DINAM debe ser 1D o 2D. Recibido: {datos.shape}")

    return np.asarray(vec, dtype=float)


# %% FILTRADO DE NOTCH

def _filtrar_con_padding_reflect(x: np.ndarray, sos: np.ndarray, pad: int) -> np.ndarray:
    if pad is None or pad <= 0:
        return sosfiltfilt(sos, x)
    x_pad = np.pad(x, pad_width=pad, mode="reflect")
    y_pad = sosfiltfilt(sos, x_pad)
    return y_pad[pad:-pad]


def aplicar_notch_multiple(
    datos: np.ndarray,
    fs: float,
    freqs: list[float] | tuple[float, ...] = (60.0, 120.0),
    Q: float = 30.0,
    pad: int = 0,
) -> np.ndarray:
    salida = np.asarray(datos, dtype=float).copy()

    for f0 in freqs:
        b, a = iirnotch(w0=f0, Q=Q, fs=fs)
        sos = tf2sos(b, a)
        salida = np.array([_filtrar_con_padding_reflect(canal, sos, pad) for canal in salida])

    return salida


# %% FILTRO POR WAVELETS

def _filtrar_dwt(
    signal: np.ndarray,
    wavelet: str,
    level: int,
    niveles_cero: list[int] | None = None,
    remove_approx: bool = False,
    mode: str = "symmetric",
) -> np.ndarray:
    coeffs = pywt.wavedec(signal, wavelet, level=level, mode=mode)
    coeffs_filt = [c.copy() for c in coeffs]

    if remove_approx:
        coeffs_filt[0] = np.zeros_like(coeffs_filt[0])

    if niveles_cero is not None:
        for j in niveles_cero:
            if 1 <= j <= level:
                coeffs_filt[j] = np.zeros_like(coeffs_filt[j])

    y = pywt.waverec(coeffs_filt, wavelet, mode=mode)
    return y[: len(signal)]


def _filtrar_dwt_con_padding_reflect(
    x: np.ndarray,
    wavelet: str,
    level: int,
    niveles_cero: list[int] | None = None,
    remove_approx: bool = False,
    pad: int = 0,
    mode: str = "symmetric",
) -> np.ndarray:
    if pad is None or pad <= 0:
        return _filtrar_dwt(
            x,
            wavelet=wavelet,
            level=level,
            niveles_cero=niveles_cero,
            remove_approx=remove_approx,
            mode=mode,
        )

    x_pad = np.pad(x, pad_width=pad, mode="reflect")
    y_pad = _filtrar_dwt(
        x_pad,
        wavelet=wavelet,
        level=level,
        niveles_cero=niveles_cero,
        remove_approx=remove_approx,
        mode=mode,
    )
    return y_pad[pad:-pad]


def aplicar_wavelet_dwt_multicanal(
    datos: np.ndarray,
    wavelet: str = "db4",
    level: int = 6,
    niveles_cero: list[int] | None = None,
    remove_approx: bool = False,
    pad: int = 0,
    mode: str = "reflect",
) -> np.ndarray:
    datos = np.asarray(datos, dtype=float)
    return np.array(
        [
            _filtrar_dwt_con_padding_reflect(
                canal,
                wavelet=wavelet,
                level=level,
                niveles_cero=niveles_cero,
                remove_approx=remove_approx,
                pad=pad,
                mode=mode,
            )
            for canal in datos
        ]
    )


# %% ICA

def ica_eliminar_componente(
    datos: np.ndarray,
    fs: float,
    ch_names: list[str],
    ic_to_remove: int | None = 0,
    montage: str | None = "standard_1020",
    random_state: int = 97,
) -> tuple[np.ndarray, ICA]:
    datos = np.asarray(datos, dtype=float)

    if datos.shape[0] != len(ch_names):
        raise ValueError(
            f"ch_names tiene {len(ch_names)} nombres, pero datos tiene {datos.shape[0]} canales." 
        )

    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=["eeg"] * len(ch_names))
    raw = mne.io.RawArray(datos, info, verbose=False)

    if montage:
        try:
            mont = mne.channels.make_standard_montage(montage)
            raw.set_montage(mont, match_case=False)
        except Exception:
            pass

    rank = mne.compute_rank(raw, verbose=False)
    n_comp = min(rank.get("eeg", len(ch_names)), len(ch_names))

    ica = ICA(
        n_components=n_comp,
        method="infomax",
        fit_params=dict(extended=True),
        random_state=random_state,
        max_iter="auto",
    )
    ica.fit(raw)

    if ic_to_remove is None:
        ica.exclude = []
    else:
        ica.exclude = [int(ic_to_remove)]

    raw_clean = raw.copy()
    ica.apply(raw_clean)

    return raw_clean.get_data(), ica


# %% PASA BANDA

def aplicar_pasabanda_butter(
    datos: np.ndarray,
    fs: float,
    f_low: float,
    f_high: float,
    orden: int = 4,
    pad: int = 0,
) -> np.ndarray:

    datos = np.asarray(datos, dtype=float)

    if f_low <= 0:
        raise ValueError("f_low debe ser > 0")
    if f_high >= fs / 2:
        raise ValueError(f"f_high debe ser < fs/2 = {fs/2:.2f} Hz")
    if f_low >= f_high:
        raise ValueError("Se requiere f_low < f_high")
    if orden < 1:
        raise ValueError("El orden debe ser >= 1")

    sos = butter(
        N=orden,
        Wn=[f_low, f_high],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    salida = np.array([_filtrar_con_padding_reflect(canal, sos, pad) for canal in datos])
    return salida


# %% RESAMPLEAR POR PROMEDIO

def resamplear_por_promedio(senal, factor=4):
    senal = np.asarray(senal)
    longitud_truncada = (senal.shape[1] // factor) * factor
    senal_resampleada = (
        senal[:, :longitud_truncada]
        .reshape(senal.shape[0], -1, factor)
        .mean(axis=2)
    )

    return senal_resampleada


# %% RESAMPLEAR POR FILTRO CHIDO

def resamplear_profesional(senal, fs_original, fs_nueva):
    gcd = np.gcd(int(fs_original), int(fs_nueva))
    up = int(fs_nueva // gcd)
    down = int(fs_original // gcd)

    senal_resampleada = resample_poly(
        senal,
        up=up,
        down=down,
        axis=1
    )

    return senal_resampleada

# %% RECTIFICAR

def rectificar_senal(senal, tipo='completa'):
    if tipo == 'completa':
        return np.abs(senal)
        
    elif tipo == 'media':
        return np.maximum(0, senal)
        
    else:
        raise ValueError("El tipo de rectificación debe ser 'completa' o 'media'.")
        
# %%  ENVOLVENTE

def emg_envelope(x, fs=None, t=None, tc_ms=20.0, rectify="full"):
    """
    Calcula envolvente tipo: rectificación + 'integración' (promedio móvil).
    Basado en zero_phase_rectify_integrate() de tus scripts.

    Parámetros
    ----------
    x : array-like
        Vector de señal EMG.
    fs : float, opcional
        Frecuencia de muestreo en Hz. (recomendado si la tienes)
    t : array-like, opcional
        Vector de tiempo en segundos (mismo largo que x). Se usa si fs es None.
    tc_ms : float
        "Constante de tiempo" en ms (realmente longitud de ventana del promedio móvil).
    rectify : {"half","full",None}
        - "half": media onda -> max(x,0)  (tal cual los scripts)
        - "full": onda completa -> abs(x) (más común en EMG)
        - None : sin rectificación

    Retorna
    -------
    env : np.ndarray
        Envolvente (misma longitud que x).
    """
    x = np.asarray(x, dtype=float)

    # --- determinar fs ---
    if fs is None:
        if t is None:
            raise ValueError("Debes proporcionar fs (Hz) o t (vector de tiempo en s).")
        t = np.asarray(t, dtype=float)
        if len(t) < 2:
            return x.copy()
        dt = float(np.mean(np.diff(t)))
        if dt <= 0:
            return x.copy()
        fs = 1.0 / dt

    # --- rectificación ---
    if rectify == "half":
        x_rect = np.maximum(x, 0.0)  # como en tus scripts
    elif rectify == "full":
        x_rect = np.abs(x)
    elif rectify is None:
        x_rect = x
    else:
        raise ValueError("rectify debe ser 'half', 'full' o None.")

    # --- ventana en muestras ---
    window_samples = int((tc_ms / 1000.0) * float(fs))
    if window_samples < 3:
        return x_rect.copy()

    # forzar impar (ventana simétrica) como en tus scripts
    if window_samples % 2 == 0:
        window_samples += 1

    kernel = np.ones(window_samples, dtype=float) / window_samples
    env = np.convolve(x_rect, kernel, mode="same")  # centrado ("zero-phase" aproximado)
    return env

# %% LAPLACIANO

def filtro_laplaciano(
        datos,
        canales,
        canal_interes,
        canales_vecinos):

    idx_interes = canales.index(canal_interes)

    indices_vecinos = [
        canales.index(ch)
        for ch in canales_vecinos
    ]

    senal_interes = datos[idx_interes]

    promedio_vecinos = np.mean(
        datos[indices_vecinos, :],
        axis=0
    )

    senal_laplaciana = senal_interes - promedio_vecinos

    return senal_laplaciana

# %% QUITAR COMPONENTE DC

def quitar_componente_dc(datos):

    datos = np.asarray(datos)

    datos_sin_dc = datos - np.mean(
        datos,
        axis=1,
        keepdims=True
    )

    return datos_sin_dc

# %% PANTALLA COMPLETA

def pantalla_completa(fig):

    try:
        manager = fig.canvas.manager

        try:
            manager.window.state('zoomed')
            return
        except:
            pass

        try:
            manager.window.showMaximized()
            return
        except:
            pass

        try:
            manager.frame.Maximize(True)
            return
        except:
            pass

        fig.set_size_inches(19, 10.5)

    except:
        fig.set_size_inches(19, 10.5)
        
# %% INSTAR

def hardlim(x):
    return np.where(x >= 0, 1, 0)

def entrenar_instar(*vectores, epochs=1000, alpha=0.1, bias=50):

    data = [np.asarray(v) for v in vectores]

    for v in data:
        if v.ndim != 1:
            raise ValueError("Todos los datos de entrada deben ser vectores de una dimensión.")

    longitudes = [len(v) for v in data]

    if len(set(longitudes)) != 1:
        raise ValueError("Todos los vectores deben tener la misma longitud.")

    data = np.array(data)
    num_features = data.shape[1]
    weights = np.random.rand(num_features)

    for epoch in range(epochs):
        for sample in data:

            z = np.dot(weights, sample) + bias
            a = hardlim(z)

            weights = weights + alpha * a * (sample - weights)

    return weights