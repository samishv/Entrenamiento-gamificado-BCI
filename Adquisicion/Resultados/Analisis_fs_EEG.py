# -*- coding: utf-8 -*-
"""
Análisis de frecuencia de muestreo real — EEG Unicorn Hybrid Black
===================================================================
Calcula la frecuencia de muestreo efectiva, jitter y genera figuras
de validación a partir de los CSV generados por la interfaz EEG.

Uso:
    python analisis_fs_eeg.py                        # pide el archivo
    python analisis_fs_eeg.py ruta/al/archivo.csv    # directo

Dependencias:
    pip install pandas numpy matplotlib scipy
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, filtfilt


# ─── Configuración ────────────────────────────────────────────────────────────
CANALES   = ['Fz', 'FC3', 'FCz', 'FC4', 'Cz', 'C3', 'C4']
FS_NOMINAL = 250.0  # Hz

COLORES = {
    'Fz' : '#FF1B6B',
    'FC3': '#E03884',
    'FCz': '#C1559C',
    'FC4': '#A273B5',
    'Cz' : '#8390CE',
    'C3' : '#64ADE6',
    'C4' : '#45CAFF',
}

BG  = '#0d1117'
FG  = '#f0f6fc'
MUT = '#8b949e'
GRD = '#30363d'


# ─── Utilidades ───────────────────────────────────────────────────────────────
def filtrar(y, lowcut=1, highcut=40, fs=250, order=4):
    nyq = fs * 0.5
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, y)


def estilo_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRD)
    ax.xaxis.label.set_color(MUT)
    ax.yaxis.label.set_color(MUT)
    ax.title.set_color(FG)


# ─── Carga del CSV ────────────────────────────────────────────────────────────
def cargar_csv(ruta):
    """
    Lee el CSV generado por VentanaEEG.
    El archivo puede tener más columnas que el header si el Unicorn
    transmite canales adicionales (acelerómetro, giroscopio, etc.).
    Solo se usan Timestamp + 7 canales EEG.
    """
    df = pd.read_csv(
        ruta,
        usecols=range(8),
        names=['Timestamp'] + CANALES,
        skiprows=1
    )
    df['ts'] = pd.to_datetime(df['Timestamp'], format='%H:%M:%S.%f')
    df['t']  = (df['ts'] - df['ts'].iloc[0]).dt.total_seconds()
    return df


# ─── Análisis de frecuencia ───────────────────────────────────────────────────
def analizar_fs(df):
    intervalos_ms = df['ts'].diff().dt.total_seconds().dropna() * 1000
    duracion      = df['t'].iloc[-1]
    fs_global     = (len(df) - 1) / duracion
    jitter_std    = intervalos_ms.std()
    mediana_int   = intervalos_ms.median()

    resultados = {
        'n_muestras'  : len(df),
        'duracion_s'  : duracion,
        'fs_global'   : fs_global,
        'jitter_std'  : jitter_std,
        'mediana_int' : mediana_int,
        'intervalos'  : intervalos_ms,
    }
    return resultados


def imprimir_resumen(r, nombre_archivo):
    sep = '─' * 50
    print(f'\n{sep}')
    print(f'  Archivo  : {nombre_archivo}')
    print(sep)
    print(f'  Muestras         : {r["n_muestras"]}')
    print(f'  Duración real    : {r["duracion_s"]:.3f} s')
    print(f'  fs nominal       : {FS_NOMINAL:.1f} Hz')
    print(f'  fs global (real) : {r["fs_global"]:.2f} Hz  '
          f'({100*(r["fs_global"]-FS_NOMINAL)/FS_NOMINAL:+.2f}%)')
    print(f'  Jitter (std)     : {r["jitter_std"]:.2f} ms')
    print(f'  Mediana intervalo: {r["mediana_int"]:.3f} ms')
    print(sep)
    print()
    print('  NOTA: el jitter refleja la resolución del timestamp de')
    print('  escritura en Python, no inestabilidad real del ADC.')
    print(f'{sep}\n')


# ─── Figura ───────────────────────────────────────────────────────────────────
def generar_figura(df, r, nombre_archivo):
    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)

    # ── Panel 1: los 7 canales con offset ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    estilo_ax(ax1)

    escala   = 1e-3          # µV → mV (ajustar si tus unidades son distintas)
    offset_v = 150           # separación vertical entre canales (mV)

    for i, ch in enumerate(CANALES):
        y = df[ch].values * escala
        y = y - np.mean(y)
        try:
            y = filtrar(y)
        except Exception:
            pass
        ax1.plot(df['t'], y + i * offset_v, color=COLORES[ch], lw=0.7, label=ch)

    ax1.set_xlabel('Tiempo (s)', fontsize=10)
    ax1.set_ylabel('Amplitud (mV + offset)', fontsize=10)
    ax1.set_title('Señal EEG — 7 canales (Butterworth 1–40 Hz)', fontsize=11)
    ax1.set_xlim([0, df['t'].iloc[-1]])
    ax1.legend(fontsize=8, facecolor='#161b22', labelcolor='white',
               framealpha=0.7, loc='upper right', ncol=7)

    # ── Panel 2: histograma de intervalos ──────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    estilo_ax(ax2)

    ivs   = r['intervalos']
    bins  = np.linspace(0, ivs.quantile(0.99), 60)
    ax2.hist(ivs[ivs <= ivs.quantile(0.99)], bins=bins,
             color='#8390CE', edgecolor='none', alpha=0.85)
    ax2.axvline(1000 / FS_NOMINAL, color='#45CAFF', lw=1.2, ls='--',
                label=f'{1000/FS_NOMINAL:.1f} ms ({FS_NOMINAL:.0f} Hz nominal)')
    ax2.axvline(r['mediana_int'], color='#FF1B6B', lw=1.2, ls='--',
                label=f'Mediana = {r["mediana_int"]:.2f} ms')
    ax2.set_xlabel('Intervalo entre muestras (ms)', fontsize=10)
    ax2.set_ylabel('Frecuencia', fontsize=10)
    ax2.set_title('Distribución de intervalos', fontsize=11)
    ax2.legend(fontsize=8, facecolor='#161b22', labelcolor='white', framealpha=0.7)

    # ── Panel 3: intervalos en el tiempo ───────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    estilo_ax(ax3)

    t_int = df['t'].iloc[1:].values
    ax3.scatter(t_int, ivs.values, s=1.5, color='#64ADE6', alpha=0.35)
    ax3.axhline(1000 / FS_NOMINAL, color='#45CAFF', lw=1.0, ls='--',
                alpha=0.8, label=f'{1000/FS_NOMINAL:.1f} ms nominal')
    ax3.set_ylim([0, ivs.quantile(0.98) * 2])
    ax3.set_xlabel('Tiempo (s)', fontsize=10)
    ax3.set_ylabel('Intervalo (ms)', fontsize=10)
    ax3.set_title('Estabilidad temporal del muestreo', fontsize=11)
    ax3.legend(fontsize=8, facecolor='#161b22', labelcolor='white', framealpha=0.7)

    # ── Panel 4: densidad espectral (canal Cz) ─────────────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    estilo_ax(ax4)

    y_cz = df['Cz'].values * 1e-3
    y_cz = y_cz - np.mean(y_cz)
    try:
        y_cz_f = filtrar(y_cz)
    except Exception:
        y_cz_f = y_cz

    from scipy.signal import welch
    f, psd = welch(y_cz_f, fs=r['fs_global'], nperseg=512)
    mask   = f <= 60
    ax4.semilogy(f[mask], psd[mask], color='#8390CE', lw=1.0)
    ax4.axvspan(8, 13,  alpha=0.12, color='#45CAFF',  label='Alpha (8–13 Hz)')
    ax4.axvspan(13, 30, alpha=0.10, color='#FF1B6B',  label='Beta (13–30 Hz)')
    ax4.axvspan(1, 4,   alpha=0.10, color='#64ADE6',  label='Delta (1–4 Hz)')
    ax4.set_xlabel('Frecuencia (Hz)', fontsize=10)
    ax4.set_ylabel('PSD (mV²/Hz)', fontsize=10)
    ax4.set_title('Densidad espectral de potencia — canal Cz', fontsize=11)
    ax4.legend(fontsize=8, facecolor='#161b22', labelcolor='white',
               framealpha=0.7, loc='upper right')

    # ── Pie de figura ──────────────────────────────────────────────────────
    metricas = (f"Archivo: {nombre_archivo}  |  "
                f"Muestras: {r['n_muestras']}  |  "
                f"Duración: {r['duracion_s']:.2f} s  |  "
                f"$f_s$ global: {r['fs_global']:.1f} Hz  |  "
                f"Jitter (std): {r['jitter_std']:.2f} ms")
    fig.text(0.5, 0.005, metricas, ha='center', color=MUT, fontsize=8.5)

    return fig


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    # Determinar ruta del archivo
    if len(sys.argv) > 1:
        ruta = sys.argv[1]
    else:
        ruta = input('Ruta del archivo CSV: ').strip().strip('"')

    if not os.path.exists(ruta):
        print(f'Error: no se encontró el archivo "{ruta}"')
        sys.exit(1)

    nombre = os.path.basename(ruta)

    print(f'\nCargando {nombre} ...')
    df = cargar_csv(ruta)

    print('Analizando frecuencia de muestreo ...')
    r  = analizar_fs(df)
    imprimir_resumen(r, nombre)

    print('Generando figura ...')
    fig = generar_figura(df, r, nombre)

    # Guardar en la misma carpeta que el CSV
    carpeta  = os.path.dirname(os.path.abspath(ruta))
    nombre_fig = nombre.replace('.csv', '_analisis_fs.png')
    ruta_fig = os.path.join(carpeta, nombre_fig)
    fig.savefig(ruta_fig, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f'Figura guardada en:\n  {ruta_fig}\n')
    plt.show()


if __name__ == '__main__':
    main()
