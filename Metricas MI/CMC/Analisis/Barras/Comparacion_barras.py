
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
import sys
from pathlib import Path
# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURACIÓN  
# ─────────────────────────────────────────────────────────────────────────────

# True  → genera UNA gráfica por CADA combinación músculo × electrodo del archivo
# False → genera solo la combinación indicada en MUSCULO / ELECTRODO
carpeta = Path(r"Reportes_excel_instar\Reportes_excel_AreaCompleta")
# ── Carpeta de salida para las imágenes ───────────────────────────────────────
CARPETA_SALIDA = "graficas_coherencia_instar"
TODAS_LAS_COMBINACIONES = True

MUSCULO   = "FCU"   # Solo se usa si TODAS_LAS_COMBINACIONES = False
ELECTRODO = "Fz"    # Solo se usa si TODAS_LAS_COMBINACIONES = False

# ── Métricas ──────────────────────────────────────────────────────────────────
METRICAS = ["Max_Beta","Area_Beta"]

ETIQUETAS_METRICAS = ["Máximo", "Área"]

# Orden deseado de los sujetos (deja vacío [] para orden alfabético automático)
ORDEN_SUJETOS = ["S17", "S6", "S7", "S18", "S4", "S5"]  # ejemplo, usa tus nombres reales
# ── Nombres de columnas ───────────────────────────────────────────────────────
COL_SUJETO  = "Sujeto"
COL_MUSCULO = "Musculo"
COL_CANAL   = "Canal_EEG"



# ── Colores por sujeto (se extienden automáticamente si hay más sujetos) ──────
PALETA_BASE = ["#4C72B0", "#21A1B3", "#73D973", "#CDD757",
               "#8172B3", "#FB568A"]



# ─────────────────────────────────────────────────────────────────────────────
#  CARGA Y VALIDACIÓN
# ─────────────────────────────────────────────────────────────────────────────
def cargar_datos(ruta: str) -> pd.DataFrame:
    ext = os.path.splitext(ruta)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(ruta)
    elif ext == ".csv":
        try:
            df = pd.read_csv(ruta, sep=",")
            if df.shape[1] < 3:
                df = pd.read_csv(ruta, sep=";")
        except Exception:
            df = pd.read_csv(ruta, sep=";")
        return df
    else:
        sys.exit(f"Formato no soportado: '{ext}'. Usa .csv o .xlsx")


def validar_columnas(df: pd.DataFrame):
    requeridas = {COL_SUJETO, COL_MUSCULO, COL_CANAL} | set(METRICAS)
    faltantes = requeridas - set(df.columns)
    if faltantes:
        sys.exit(f"    Columnas faltantes en el archivo: {faltantes}\n"
                 f"   Columnas disponibles: {list(df.columns)}")


# ─────────────────────────────────────────────────────────────────────────────
#  GRÁFICA
# ─────────────────────────────────────────────────────────────────────────────
def graficar(df_filtrado: pd.DataFrame,
             musculo: str,
             electrodo: str,
             sujetos_globales: list,
             colores_globales: dict,
             carpeta: str,
             metodo: str):
    """
    Dibuja y guarda la gráfica de barras para una combinación músculo/electrodo.
    sujetos_globales y colores_globales garantizan colores consistentes en
    todas las gráficas del mismo run.
    """
    sujetos_presentes = [s for s in sujetos_globales if s in df_filtrado[COL_SUJETO].unique()]
    n_sujetos  = len(sujetos_presentes)
    n_metricas = len(METRICAS)

    x       = np.arange(n_metricas)*0.9
    ancho   = 0.75 / n_sujetos
    offsets = np.linspace(-(n_sujetos - 1) / 2, (n_sujetos - 1) / 2, n_sujetos) * ancho

    fig, ax = plt.subplots(figsize=(max(7, n_metricas * 1.8), 6))
    #fig.text(0.01, 0.02, f"Método: {metodo}", ha="left", fontsize=8, color="gray", style="italic")
    fig.subplots_adjust(bottom=0.15)
    for idx, sujeto in enumerate(sujetos_presentes):
        fila = df_filtrado[df_filtrado[COL_SUJETO] == sujeto]
        if fila.empty:
            continue
        valores = fila[METRICAS].values.flatten().astype(float)
        bars = ax.bar(x + offsets[idx], valores,
                      width=ancho * 0.9,
                      color=colores_globales[sujeto],
                      label=str(sujeto),
                      zorder=3,
                      edgecolor="white",
                      linewidth=0.5)

        # Valor encima de cada barra
        y_max = np.nanmax(valores) if np.any(~np.isnan(valores)) else 1
        for bar, val in zip(bars, valores):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + y_max * 0.012,
                        f"{val:.4f}",
                        ha="center", va="bottom",
                        fontsize=13, rotation=90,
                        color="#222222")

    # ── Estética ──────────────────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(ETIQUETAS_METRICAS, fontsize=14)
    ax.set_ylabel("Valor de Coherencia", fontsize=14)
    ax.set_xlabel("Métrica", fontsize=14)
    # ax.set_title(
    #     f"Comparación entre Sujetos CMC",
    #     fontsize=13, fontweight="bold", pad=14
    # )

    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax.yaxis.grid(True, linestyle="--", alpha=0.55, zorder=0)
    ax.set_axisbelow(True)

    # Leyenda con todos los sujetos del dataset (coherencia visual entre gráficas)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colores_globales[s])
               for s in sujetos_globales]
    ax.legend(handles, [str(s) for s in sujetos_globales],
              title="Sujeto",
              bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=13, title_fontsize=13)

    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    nombre = f"{metodo}_{musculo}_{electrodo}.png"
    ruta_salida = os.path.join(carpeta, nombre)
    plt.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"     Guardada → {ruta_salida}")

def extraer_desde_tercera_parte(nombre_archivo):
    nombre = Path(nombre_archivo).stem
    partes = nombre.split("_")
    
    if len(partes) >= 3:
        return "_".join(partes[2:])  # une desde la tercera parte en adelante
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Crear carpeta de salida
    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    for ARCHIVO in carpeta.glob("*.xls*"):
        metodo = extraer_desde_tercera_parte(ARCHIVO.name)
        print(f"\n  Cargando archivo: {ARCHIVO}")
        df = cargar_datos(ARCHIVO)
        validar_columnas(df)

        # Limpiar espacios en columnas de texto
        for col in [COL_SUJETO, COL_MUSCULO, COL_CANAL]:
            df[col] = df[col].astype(str).str.strip()

        MAPEO_SUJETOS = {
            "S17": "SV1",
            "S6": "SV2",
            "S7": "SV3",
            "S18": "SK1",
            "S4": "SK2",
            "S5": "SK3",
        }

        df[COL_SUJETO] = df[COL_SUJETO].replace(MAPEO_SUJETOS)
        ORDEN_SUJETOS = ["S1", "S2", "S3", "S4", "S5", "S6"]

        # Asignar color fijo a cada sujeto (consistente en todas las gráficas)
        sujetos_unicos = df[COL_SUJETO].unique().tolist()
        print("VALORES REALES en columna Sujeto:", sujetos_unicos)
        print("ORDEN_SUJETOS configurado:       ", ORDEN_SUJETOS)
        if ORDEN_SUJETOS:
            # Usa el orden indicado, pero solo con los sujetos que sí existen en el archivo
            sujetos_globales = [s for s in ORDEN_SUJETOS if s in sujetos_unicos]
            # Agrega al final cualquier sujeto no listado en ORDEN_SUJETOS (por si acaso)
            faltantes = [s for s in sujetos_unicos if s not in ORDEN_SUJETOS]
            sujetos_globales += sorted(faltantes)
        else:
            sujetos_globales = sorted(sujetos_unicos)

        paleta_ext = (PALETA_BASE * ((len(sujetos_globales) // len(PALETA_BASE)) + 1))
        colores_globales = {s: paleta_ext[i] for i, s in enumerate(sujetos_globales)}
        print(f"    Sujetos en el dataset: {sujetos_globales}")

        # Determinar combinaciones a graficar
        if TODAS_LAS_COMBINACIONES:
            combinaciones = (
                df[[COL_MUSCULO, COL_CANAL]]
                .drop_duplicates()
                .sort_values([COL_MUSCULO, COL_CANAL])
                .values.tolist()
            )
            print(f"\n Se generarán {len(combinaciones)} gráficas "
                f"({len(df[COL_MUSCULO].unique())} músculos × "
                f"{len(df[COL_CANAL].unique())} electrodos)\n")
        else:
            combinaciones = [[MUSCULO, ELECTRODO]]
            print(f"\n Generando 1 gráfica: {MUSCULO} / {ELECTRODO}\n")

        # Generar gráficas
        generadas = 0
        for musculo, electrodo in combinaciones:
            mask = (df[COL_MUSCULO] == musculo) & (df[COL_CANAL] == electrodo)
            df_sub = df[mask]
            if df_sub.empty:
                print(f"     Sin datos: {musculo} / {electrodo} — omitido")
                continue
            print(f"    {musculo:8s} | {electrodo:5s}  "
                f"→ {len(df_sub)} fila(s), sujetos: "
                f"{sorted(df_sub[COL_SUJETO].unique())}")
            graficar(df_sub, musculo, electrodo,
                    sujetos_globales, colores_globales, CARPETA_SALIDA, metodo)
            generadas += 1

        print(f"\n Listo. {generadas} gráfica(s) guardadas en '{CARPETA_SALIDA}/'")


if __name__ == "__main__":
    main()
