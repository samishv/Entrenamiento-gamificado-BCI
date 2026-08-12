import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.viz import plot_topomap
from datetime import datetime
from mne.datasets import fieldtrip_cmc

data_path = fieldtrip_cmc.data_path()
raw = mne.io.read_raw_ctf(data_path / "SubjectCMC.ds", preload=False)
info_ref = raw.info
info_ref.rename_channels(lambda ch: ch.split("-")[0])  

# Filtrar solo canales MEG tipo 'mag' una sola vez
picks_mag = mne.pick_types(info_ref, meg='mag', ref_meg=False, eeg=False,
                            eog=False, emg=False, stim=False, misc=False)
info_meg_only = mne.pick_info(info_ref, picks_mag)
layout = mne.channels.read_layout('CTF151')
layout_names_clean = [n.split('-')[0] for n in layout.names]  # por si trae sufijos

ahora = datetime.now()
df = pd.read_excel("Reportes_Base_datos_multitapers_rectificada_beta1520.xlsx")
testName = "FieldTrip_multitaper1520"
metricas = ["Area_Beta"]
#metricas = ["Area_Beta", "Area_Gamma", "Area_Mu", "Max_Beta", "Max_Gamma", "Max_Mu",
#            "Area_Beta_S", "Area_Gamma_S", "Area_Mu_S", "Max_Beta_S", "Max_Gamma_S", "Max_Mu_S",
#            "Area_Beta_M", "Area_Gamma_M", "Area_Mu_M", "Max_Beta_M", "Max_Gamma_M", "Max_Mu_M"]            

N_TOP = 1  # cuántos canales de mayor valor etiquetar

# --- Cálculo de la proporción real del casco (una sola vez, fuera del bucle de métricas) ---
centro_layout = layout.pos[:, :2].mean(axis=0)
radio_x_layout = np.abs(layout.pos[:, 0] - centro_layout[0]).max()
radio_y_layout = np.abs(layout.pos[:, 1] - centro_layout[1]).max()
aspecto_ovalo = radio_y_layout / radio_x_layout  # ej. ~0.55 para CTF151

ancho_celda = 5
alto_celda = ancho_celda * aspecto_ovalo  # margen para título/labels


for metrica in metricas:
    sujetos = df["Sujeto"].unique()
    musculos = df["Musculo"].unique()
    musculos = musculos[musculos == "LECRL"]  # filtrar solo el músculo de interés
    vmin = df[metrica].min()
    vmax = df[metrica].max()

    fig, axes = plt.subplots(
        len(musculos), len(sujetos),
        figsize=(ancho_celda * len(sujetos), alto_celda * len(musculos))
    )

    if len(musculos) == 1 and len(sujetos) == 1:
        axes = np.array([[axes]])
    elif len(musculos) == 1:
        axes = axes[np.newaxis, :]
    elif len(sujetos) == 1:
        axes = axes[:, np.newaxis]

    for a, m in enumerate(musculos):
        for i, s in enumerate(sujetos):

            subset = df[(df["Sujeto"] == s) & (df["Musculo"] == m)]

            if subset.empty:
                axes[a, i].set_title(f"(sin datos)" if a == 0 else "")
                axes[a, i].axis("off")
                continue

            subset = subset.copy()
            subset["Canal_EEG"] = subset["Canal_EEG"].astype(str).str.split("-").str[0]

            canales = subset["Canal_EEG"].values

            canales_validos = [c for c in canales if c in layout_names_clean]
            faltantes = set(canales) - set(canales_validos)
            if faltantes:
                print(f"[{s}-{m}-{metrica}] Canales no encontrados en layout CTF151:", faltantes)

            # índices del layout correspondientes a tus canales válidos, en el orden del layout
            idx = [layout_names_clean.index(c) for c in canales_validos]
            layout_center = layout.pos[:, :2].mean(axis=0)

            pos2d = layout.pos[idx, :2].copy()
            pos2d -= layout_center
            names_layout = [layout_names_clean[i] for i in idx]

            # reordenar tus valores para que coincidan con ese mismo orden
            subset_idx = subset.set_index("Canal_EEG")
            valores_ordenados = subset_idx.loc[names_layout, metrica].values

            # --- Top-N para etiquetas ---
            top_idx = np.argsort(valores_ordenados)[-N_TOP:]
            top_canales = set(np.array(names_layout)[top_idx])
            names_to_show = [ch if ch in top_canales else '' for ch in names_layout]

            # --- Contorno ovalado, calculado a partir de los sensores de esta celda ---
            centro_2d = np.array([0., 0.])
            radio_x = np.abs(pos2d[:, 0] - centro_2d[0]).max() * 1.15  # margen 15%
            radio_y = np.abs(pos2d[:, 1] - centro_2d[1]).max() * 1.15

            theta = np.linspace(0, 2*np.pi, 200)

            # Cabeza
            head_x = radio_x * np.cos(theta) + centro_2d[0]
            head_y = radio_y * np.sin(theta) + centro_2d[1]

            # Nariz
            nose_x = np.array([
                -0.08*radio_x,
                0.00,
                0.08*radio_x
            ]) + centro_2d[0]

            nose_y = np.array([
                radio_y,
                1.15*radio_y,
                radio_y
            ]) + centro_2d[1]

            # Oreja izquierda
            ear_left_x = np.array([
                -radio_x,
                -1.08*radio_x,
                -1.10*radio_x,
                -1.08*radio_x,
                -radio_x
            ]) + centro_2d[0]

            ear_left_y = np.array([
                0.25*radio_y,
                0.15*radio_y,
                0.00,
                -0.15*radio_y,
                -0.25*radio_y
            ]) + centro_2d[1]

            # Oreja derecha
            ear_right_x = -ear_left_x + 2*centro_2d[0]

            ear_right_y = ear_left_y.copy()

            outlines = dict(
                head=(head_x, head_y),
                nose=(nose_x, nose_y),
                ear_left=(ear_left_x, ear_left_y),
                ear_right=(ear_right_x, ear_right_y),
                mask_pos=(head_x, head_y),
                clip_radius=(radio_x, radio_y),
            )

            # sphere sigue haciendo falta para el posicionamiento/interpolación interna
            sphere = (centro_2d[0], centro_2d[1], 0, max(radio_x, radio_y))
            
            im, _ = plot_topomap(
                valores_ordenados,
                pos2d,                # <-- posiciones 2D del layout, no 'info'
                axes=axes[a, i],
                extrapolate='local',
                contours=6,
                sensors=True,
                cmap='jet',
                vlim=(vmin, vmax),
                names=names_to_show,
                #sphere=sphere,
                outlines=outlines,    # <-- contorno ovalado en lugar de circular
                show=False
            )

            for txt in axes[a, i].texts:
                txt.set_color('white')
                txt.set_fontweight('bold')
                txt.set_fontsize(9)
            #if a == 0:
            #    axes[a, i].set_title(f"CMC con el método de Welch (15-20 Hz)", fontsize=12)
            #else:
            #    axes[a, i].set_title("")

            #if i == 0:
            #    axes[a, i].set_ylabel(f"{m}", fontsize=12, labelpad=10)

    # --- Reducir espacio entre filas/columnas ---
    plt.subplots_adjust(wspace=0.1, hspace=0.05)

    # --- Alinear las etiquetas de fila (RECRL / LECRL) ---
    fig.align_ylabels(axes[:, 0])

    # --- Título general ---
    #plt.suptitle(metrica, fontsize=14, y=0.98)

    # --- Colorbar en eje dedicado (no reordena el grid completo) ---
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax)

    plt.savefig(f"topomap_{testName}_{metrica.replace(' ', '_')}_{ahora.strftime('%y-%m-%d_%H-%M')}.png",
                dpi=300, bbox_inches='tight')