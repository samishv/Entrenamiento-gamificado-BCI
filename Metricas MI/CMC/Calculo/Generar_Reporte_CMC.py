import numpy as np
import pandas as pd
from pathlib import Path
import mne 

# =========================================================
# CONFIGURACION
# =========================================================

SUJETOS = ["S4", "S5", "S6", "S7", "S17", "S18"]

# Canales que quieres reportar. Para CAR/Laplaciano normalmente sera ["C3"] o ["C4"].
# Para sin filtro puede ser la lista completa o un subconjunto, por ejemplo ["C3"].
CANALES_EEG = ["C3"]

# MUSCULO_ABREV = {
#     "Flexor carpi ulnaris": "FCU",
#     "Right Extensor carpi radialis longus": "RECRL",
#     "Extensor carpi ulnaris": "ECU",
# }

MUSCULO_ABREV = {
    "EMG":"instar"
}

# Orden original esperado cuando el NPY viene con todos los canales/musculos.
# Si el NPY tiene exactamente len(CANALES_EEG), se asume que ya viene filtrado/reducido
# y se indexa 0..n-1, no por el indice de esta lista base.
EEG_BASE_ORDER = ["Fz", "FC3", "FCz", "FC4", "Cz", "C3", "C4"]
MUSCLE_BASE_ORDER = [
    "Flexor carpi ulnaris",
    "Right Extensor carpi radialis longus",
    "Extensor carpi ulnaris",
]




#----------------------------------------------------------------------------
# SUJETOS =['SDataBase']

# ds_path = Path(r"C:\Users\lrl13\Downloads\SubjectCMC\SubjectCMC.ds")

# raw = mne.io.read_raw_ctf(
#     ds_path,
#     preload=True,
#     clean_names=True,
#     system_clock="truncate"  # default; si hay problemas prueba "ignore"
# )

# # canales MEG 
# CANALES_EEG = [ch for ch in raw.ch_names if ch.startswith("M")]

# # Músculos del diccionario -> abreviatura solicitada
# MUSCULO_ABREV = {
#     "Left Extensor carpi radialis longus": "LECRL",
#     "Right Extensor carpi radialis longus": "RECRL"
# }

# EEG_BASE_ORDER = CANALES_EEG
# MUSCLE_BASE_ORDER = [
#     "Left Extensor carpi radialis longus",
#     "Right Extensor carpi radialis longus"
# ]
#----------------------------------------------------------------------------
BANDAS = {
    "mu": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

# Area beta:
#   "completa"       -> integra toda la coherencia en beta.
#   "significativa"  -> integra solo valores por encima de limiteC.
MODO_AREA_BETA = "completa"  # "completa" o "significativa"

# Si MODO_AREA_BETA = "significativa":
#   "exceso" -> integra max(coh - CL, 0)
#   "bruta"  -> integra coh solo donde coh > CL
AREA_SIGNIFICATIVA_TIPO = "exceso"  # "exceso" o "bruta"

# =========================================================
# HELPERS
# =========================================================

def _minmax_series(x: pd.Series) -> pd.Series:
    mn, mx = x.min(), x.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - mn) / (mx - mn)


def normalizar_minmax(df: pd.DataFrame, cols, group_cols, sufijo):
    if df.empty:
        return df
    for c in cols:
        df[f"{c}_{sufijo}"] = df.groupby(group_cols)[c].transform(_minmax_series)
    return df


def _es_array_cmc(arr):
    arr = np.asarray(arr)
    return arr.ndim == 3 and arr.shape[0] >= 2


def _as_dict_if_np_object(x):
    if isinstance(x, np.ndarray) and x.shape == () and x.dtype == object:
        return x.item()
    return x


def cargar_npy(path_npy):
    data = np.load(path_npy, allow_pickle=True)
    data = _as_dict_if_np_object(data)
    if not isinstance(data, dict):
        raise TypeError(f"El archivo NPY debe contener un diccionario. Tipo encontrado: {type(data)}")
    return data


def obtener_freqs(item, cmc, data_global=None):
    """Obtiene freqs del item, del global, o reconstruye linealmente 5-45 Hz."""
    cand = None
    if isinstance(item, dict):
        for k in ["freqs", "frecuencias", "f"]:
            if k in item:
                cand = item[k]
                break
    if cand is None and isinstance(data_global, dict):
        for k in ["freqs", "frecuencias", "f"]:
            if k in data_global:
                cand = data_global[k]
                break
    if cand is not None:
        freqs = np.asarray(cand, dtype=float).ravel()
        if len(freqs) == cmc.shape[0]:
            return freqs
        print(f"Aviso: freqs tiene longitud {len(freqs)} pero CMC tiene {cmc.shape[0]}; se reconstruyen frecuencias.")
    return np.linspace(5.0, 45.0, cmc.shape[0], endpoint=True)


def obtener_limiteC(item, data_global=None):
    """Extrae limiteC si existe; si no existe devuelve NaN."""
    if isinstance(item, dict):
        for k in ["limiteC", "limiteC_permutacion", "limiteC_analitico", "CL", "confidence_limit"]:
            if k in item:
                try:
                    return float(np.asarray(item[k]).squeeze())
                except Exception:
                    pass
    if isinstance(data_global, dict):
        for k in ["limiteC", "limiteC_permutacion", "limiteC_analitico", "CL", "confidence_limit"]:
            if k in data_global:
                try:
                    return float(np.asarray(data_global[k]).squeeze())
                except Exception:
                    pass
    return np.nan


def encontrar_cmc_en_item(item):
    """Busca primero llaves conocidas y luego recursivamente una matriz 3D."""
    item = _as_dict_if_np_object(item)
    if isinstance(item, dict):
        for k in ["coherencia", "CMC", "cmc", "coherence", "coh"]:
            if k in item and _es_array_cmc(item[k]):
                return np.asarray(item[k], dtype=float), k
        for k, v in item.items():
            found, key = encontrar_cmc_en_item(v)
            if found is not None:
                return found, key or k
    elif isinstance(item, (list, tuple)):
        for v in item:
            found, key = encontrar_cmc_en_item(v)
            if found is not None:
                return found, key
    elif _es_array_cmc(item):
        return np.asarray(item, dtype=float), None
    return None, None


def iterar_resultados_por_sujeto(data, sujetos):
    """
    Genera tuplas: sujeto, etiqueta, item, cmc.
    Soporta:
      data[sujeto] = results_con_coherencia
      data[sujeto][condicion] = results_con_coherencia
    """
    for sujeto in sujetos:
        if sujeto not in data:
            raise KeyError(f"El sujeto {sujeto} no existe en el NPY. Sujetos disponibles: {list(data.keys())}")
        bloque = _as_dict_if_np_object(data[sujeto])

        cmc, _ = encontrar_cmc_en_item(bloque)
        if cmc is not None:
            yield sujeto, "", bloque, cmc
            continue

        if isinstance(bloque, dict):
            encontrados = 0
            for etiqueta, item in bloque.items():
                item = _as_dict_if_np_object(item)
                cmc, _ = encontrar_cmc_en_item(item)
                if cmc is not None:
                    encontrados += 1
                    yield sujeto, str(etiqueta), item, cmc
            if encontrados == 0:
                raise KeyError(f"No encontré matriz de coherencia/CMC 3D para el sujeto {sujeto}.")
        else:
            raise TypeError(f"El bloque del sujeto {sujeto} debe ser dict o contener una CMC 3D.")


def orientar_cmc(cmc):
    """
    Devuelve CMC como (freq, musculos_total, canales_total).
    Acepta (freq, musculos, canales) o (freq, canales, musculos).
    """
    cmc = np.asarray(cmc, dtype=float)
    if cmc.ndim != 3:
        raise ValueError(f"CMC debe ser 3D. Forma encontrada: {cmc.shape}")

    nm_req = len(MUSCULO_ABREV)
    ne_req = len(CANALES_EEG)
    nm_base = len(MUSCLE_BASE_ORDER)
    ne_base = len(EEG_BASE_ORDER)

    posibles_m = {nm_req, nm_base}
    posibles_e = {ne_req, ne_base}

    if cmc.shape[1] in posibles_m and cmc.shape[2] in posibles_e:
        return cmc
    if cmc.shape[1] in posibles_e and cmc.shape[2] in posibles_m:
        return np.transpose(cmc, (0, 2, 1))

    raise ValueError(
        f"No reconozco orientación de CMC {cmc.shape}. "
        f"Se esperaba eje de musculos en {sorted(posibles_m)} y eje EEG en {sorted(posibles_e)}, "
        f"o la forma transpuesta."
    )


def indices_solicitados(n_total, solicitados, orden_base, nombre_eje):
    """
    Si el NPY ya viene reducido a len(solicitados), usa indices 0..n-1.
    Si el NPY viene completo con len(orden_base), selecciona por nombre.
    """
    if n_total == len(solicitados):
        return list(range(n_total))

    if n_total == len(orden_base):
        idx = []
        for nombre in solicitados:
            if nombre not in orden_base:
                raise ValueError(f"{nombre_eje} solicitado '{nombre}' no está en orden_base: {orden_base}")
            idx.append(orden_base.index(nombre))
        return idx

    raise ValueError(
        f"El NPY tiene {n_total} elementos en eje {nombre_eje}, pero solicitaste {len(solicitados)}. "
        f"Solo puedo validar si n_total == len(solicitados) o n_total == len(orden_base)={len(orden_base)}."
    )


def validar_y_filtrar_cmc(cmc):
    cmc = orientar_cmc(cmc)
    musculos_largos = list(MUSCULO_ABREV.keys())
    idx_m = indices_solicitados(cmc.shape[1], musculos_largos, MUSCLE_BASE_ORDER, "musculo")
    idx_e = indices_solicitados(cmc.shape[2], CANALES_EEG, EEG_BASE_ORDER, "canal EEG")
    return cmc[:, idx_m, :][:, :, idx_e]


def area_bajo_curva(y, x):
    if len(x) < 2:
        return 0.0
    return float(np.trapz(y, x))


def metricas_por_banda(cmc, freqs, limiteC=np.nan):
    """Calcula maximos y areas; beta puede ser completa o significativa."""
    out = {}
    for banda, (fl, fh) in BANDAS.items():
        mask = (freqs >= fl) & (freqs <= fh)
        if not np.any(mask):
            raise ValueError(f"No hay frecuencias para la banda {banda}: {fl}-{fh} Hz")

        sub = cmc[mask, :, :]
        fsub = freqs[mask]
        nombre = banda.capitalize()

        out[f"Max {nombre}"] = np.nanmax(sub, axis=0)

        if banda == "beta" and MODO_AREA_BETA == "significativa":
            if not np.isfinite(limiteC):
                raise ValueError("MODO_AREA_BETA='significativa' requiere que el NPY contenga limiteC válido.")
            if AREA_SIGNIFICATIVA_TIPO == "exceso":
                integ = np.maximum(sub - limiteC, 0.0)
            elif AREA_SIGNIFICATIVA_TIPO == "bruta":
                integ = np.where(sub > limiteC, sub, 0.0)
            else:
                raise ValueError("AREA_SIGNIFICATIVA_TIPO debe ser 'exceso' o 'bruta'.")
            out[f"Area {nombre}"] = np.trapz(integ, fsub, axis=0)
        else:
            out[f"Area {nombre}"] = np.trapz(sub, fsub, axis=0)

    return out


def construir_df_desde_data(data: dict, sujetos=SUJETOS):
    filas = []
    musculos_largos = list(MUSCULO_ABREV.keys())

    for sujeto, etiqueta, item, cmc_raw in iterar_resultados_por_sujeto(data, sujetos):
        cmc = validar_y_filtrar_cmc(cmc_raw)
        freqs = obtener_freqs(item, cmc, data_global=data)
        if len(freqs) != cmc.shape[0]:
            raise ValueError(f"Freqs y CMC no coinciden para {sujeto}/{etiqueta}: {len(freqs)} vs {cmc.shape[0]}")

        limiteC = obtener_limiteC(item, data_global=data)
        metricas = metricas_por_banda(cmc, freqs, limiteC=limiteC)

        for i, musc_largo in enumerate(musculos_largos):
            musc = MUSCULO_ABREV[musc_largo]
            for j, canal in enumerate(CANALES_EEG):
                filas.append({
                    "Sujeto": sujeto,
                    "Condicion": etiqueta,
                    "Musculo": musc,
                    "Canal_EEG": canal,
                    "LimiteC": limiteC,
                    "Area_Beta_Modo": MODO_AREA_BETA,
                    "Area_Beta_Tipo": AREA_SIGNIFICATIVA_TIPO if MODO_AREA_BETA == "significativa" else "NA",
                    "Max_Mu": float(metricas["Max Mu"][i, j]),
                    "Max_Beta": float(metricas["Max Beta"][i, j]),
                    "Max_Gamma": float(metricas["Max Gamma"][i, j]),
                    "Area_Mu": float(metricas["Area Mu"][i, j]),
                    "Area_Beta": float(metricas["Area Beta"][i, j]),
                    "Area_Gamma": float(metricas["Area Gamma"][i, j]),
                })

    if not filas:
        raise ValueError("No se generaron filas. Revisa SUJETOS, CANALES_EEG, MUSCULO_ABREV y la estructura del NPY.")

    df = pd.DataFrame(filas)
    metric_cols = ["Max_Mu", "Max_Beta", "Max_Gamma", "Area_Mu", "Area_Beta", "Area_Gamma"]
    df = normalizar_minmax(df, metric_cols, group_cols=["Sujeto"], sufijo="S")
    df = normalizar_minmax(df, metric_cols, group_cols=["Sujeto", "Musculo"], sufijo="M")

    orden = (
        ["Sujeto", "Condicion", "Musculo", "Canal_EEG", "LimiteC", "Area_Beta_Modo", "Area_Beta_Tipo"]
        + metric_cols
        + [f"{c}_S" for c in metric_cols]
        + [f"{c}_M" for c in metric_cols]
    )
    return df[orden]


def resolver_ruta_salida(path_npy, path_salida=None):
    path_npy = Path(path_npy)
    if path_salida is None or str(path_salida).strip() == "":
        return path_npy.with_suffix(".xlsx")
    path_salida = Path(path_salida)
    # Si path_salida es carpeta o no tiene sufijo, usar mismo nombre del NPY.
    if path_salida.suffix.lower() != ".xlsx":
        return path_salida / f"{path_npy.stem}.xlsx"
    return path_salida


def escribir_excel(df, path_xlsx):
    path_xlsx = Path(path_xlsx)
    path_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Metricas_CMC", index=False)
        ws = writer.book["Metricas_CMC"]
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for col_cells in ws.columns:
            max_len = max(len(str(c.value)) if c.value is not None else 0 for c in col_cells)
            ws.column_dimensions[col_cells[0].column_letter].width = min(max(max_len + 2, 10), 24)
        for cell in ws[1]:
            cell.font = cell.font.copy(bold=True)
    return path_xlsx


def generar_excel_metricas(path_npy: str, path_salida: str | None = None):
    data = cargar_npy(path_npy)
    print("Llaves principales en data:", list(data.keys()))
    df = construir_df_desde_data(data, sujetos=SUJETOS)
    ruta_excel = resolver_ruta_salida(path_npy, path_salida)
    escribir_excel(df, ruta_excel)
    print(f"Reporte generado: {ruta_excel}")
    return ruta_excel


def generar_exceles_desde_carpeta(
    carpeta_npy: str,
    carpeta_salida: str | None = None,
    patron: str = "*.npy",
    recursivo: bool = False,
    continuar_si_error: bool = True,
):
    """
    Lee todos los archivos .npy de una carpeta y genera un Excel por cada archivo.

    Parametros
    ----------
    carpeta_npy : str
        Carpeta donde estan los reportes .npy.
    carpeta_salida : str | None
        Carpeta donde se guardaran los .xlsx. Si es None, cada Excel se guarda
        junto a su .npy correspondiente.
    patron : str
        Patron de busqueda. Por defecto "*.npy".
    recursivo : bool
        Si True, busca tambien en subcarpetas usando rglob.
    continuar_si_error : bool
        Si True, si un archivo falla, imprime el error y continua con los demas.
        Si False, detiene el proceso en el primer error.

    Retorna
    -------
    dict
        Diccionario con las rutas generadas y los errores encontrados.
    """
    carpeta_npy = Path(carpeta_npy)
    if not carpeta_npy.exists():
        raise FileNotFoundError(f"La carpeta no existe: {carpeta_npy}")
    if not carpeta_npy.is_dir():
        raise NotADirectoryError(f"La ruta no es una carpeta: {carpeta_npy}")

    archivos_npy = sorted(carpeta_npy.rglob(patron) if recursivo else carpeta_npy.glob(patron))

    if not archivos_npy:
        raise FileNotFoundError(f"No se encontraron archivos con patron '{patron}' en: {carpeta_npy}")

    print(f"Archivos NPY encontrados: {len(archivos_npy)}")

    generados = []
    errores = []

    for i, path_npy in enumerate(archivos_npy, start=1):
        print("\n" + "=" * 70)
        print(f"Procesando {i}/{len(archivos_npy)}: {path_npy.name}")
        print("=" * 70)

        try:
            ruta_excel = generar_excel_metricas(str(path_npy), carpeta_salida)
            generados.append(ruta_excel)
        except Exception as exc:
            mensaje = f"Error procesando {path_npy}: {type(exc).__name__}: {exc}"
            errores.append((path_npy, exc))
            print(mensaje)
            if not continuar_si_error:
                raise

    print("\n" + "=" * 70)
    print("Resumen de generacion")
    print("=" * 70)
    print(f"Excel generados correctamente: {len(generados)}")
    print(f"Archivos con error: {len(errores)}")

    if errores:
        print("\nArchivos con error:")
        for path_npy, exc in errores:
            print(f"- {path_npy.name}: {type(exc).__name__}: {exc}")

    return {
        "generados": generados,
        "errores": errores,
    }


if __name__ == "__main__":
    CARPETA_NPY = r"C:\Users\lrl13\Downloads\Entrenamiento imaginacion2\Entrenamiento imaginacion\Reportes todos los metodos\Reportes Excel reproducibles"
    RUTA_SALIDA = r"C:\Users\lrl13\Downloads\Entrenamiento imaginacion2\Entrenamiento imaginacion\Reportes todos los metodos\Reportes Excel reproducibles"

    generar_exceles_desde_carpeta(
        carpeta_npy=CARPETA_NPY,
        carpeta_salida=RUTA_SALIDA,
        patron="*.npy",
        recursivo=False,
        continuar_si_error=True,
    )
