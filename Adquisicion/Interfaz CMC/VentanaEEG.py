# -*- coding: utf-8 -*-
"""
Ventana EEG independiente con PyQtGraph.
Se lanza como subproceso desde la interfaz principal.
"""

import sys
import os
import csv
import time
import queue
import datetime
import numpy as np
from scipy.signal import butter, filtfilt

import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore

from eeg_uhb import EEGAcquisitionManager

# ─── Configuración ────────────────────────────────────────────────────────────
EEG_VENTANA  = 2000  # 8 s × 250 Hz
EEG_REFRESCO = 50

FLAG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'eeg_grabar.flag')

CANALES = ['Fz', 'FC3', 'FCz', 'FC4', 'Cz', 'C3', 'C4']
INDICES = [0, 1, 2, 3, 4, 5, 7]  # 6=NC omitido

COLORES = {
    'Fz' : '#FF1B6B',
    'FC3': '#E03884',
    'FCz': '#C1559C',
    'FC4': '#A273B5',
    'Cz' : '#8390CE',
    'C3' : '#64ADE6',
    'C4' : '#45CAFF',
}

BG_COLOR   = '#0d1117'
AXIS_COLOR = '#8b949e'
GRID_COLOR = '#30363d'


# ─── Subclase EEG — solo adquisición y graficación, sin guardar ───────────────
class EEGConGrafica(EEGAcquisitionManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grafica_queue    = queue.Queue()
        self._lsl_offset      = 0
        self._first_timestamp = None

    def _acquisition_loop(self):
        try:
            from pylsl import local_clock
            self._lsl_offset      = local_clock() - time.time()
            self._first_timestamp = None
            while self.running:
                sample, timestamp = self.stream_inlet.pull_sample(timeout=0.1)
                if sample:
                    if self._first_timestamp is None:
                        self._first_timestamp = timestamp
                    self.data.append(sample)
                    self.timestamps.append(timestamp)
                    self.process_queue.put((sample, timestamp))
                    self.grafica_queue.put((sample, timestamp))
        except Exception as e:
            import logging
            logging.error(f"Acquisition error: {str(e)}")

    def _storage_loop(self):
        # Desactivado — el guardado lo maneja VentanaEEG directamente
        pass


# ─── Filtro ───────────────────────────────────────────────────────────────────
def filtrar(y, lowcut=1, highcut=40, fs=250, order=4):
    nyq = fs * 0.5
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, y)


# ─── Ventana principal ────────────────────────────────────────────────────────
class VentanaEEG(QtWidgets.QMainWindow):
    def __init__(self, stream_name, save_path, condicion, pid, indice, sid):
        super().__init__()
        self.stream_name = stream_name
        self.save_path   = save_path
        self.condicion   = condicion
        self.pid         = pid
        self.sid         = sid
        self.indice      = indice
        self.buffer      = []
        self.eeg         = EEGConGrafica()
        self._grabando   = False
        self._csv_file   = None
        self._csv_writer = None

        self._build_ui()
        self._iniciar_eeg()

        self.timer = QtCore.QTimer()
        self.timer.setInterval(EEG_REFRESCO)
        self.timer.timeout.connect(self._actualizar)
        self.timer.start()

        self.timer_flag = QtCore.QTimer()
        self.timer_flag.setInterval(100)
        self.timer_flag.timeout.connect(self._revisar_flag)
        self.timer_flag.start()
        print(f"[EEG] timer_flag activo: {self.timer_flag.isActive()}")
        print(f"[EEG] FLAG_PATH = {FLAG_PATH}")

        if os.path.exists(FLAG_PATH):
            os.remove(FLAG_PATH)

    def _build_ui(self):
        self.setWindowTitle("EEG en tiempo real")
        self.setStyleSheet(f"background-color: {BG_COLOR};")

        escritorio = QtWidgets.QDesktopWidget()
        if escritorio.screenCount() > 1:
            pantalla = escritorio.screenGeometry(0)
            self.setGeometry(pantalla)
        else:
            self.showMaximized()

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 5)
        layout.setSpacing(2)

        titulo = QtWidgets.QLabel(
            'Señales EEG Crudas — Fz, FC3, FCz, FC4, Cz, C3, C4')
        titulo.setStyleSheet(
            'color: #f0f6fc; font-size: 13px; font-weight: bold;')
        layout.addWidget(titulo)

        self.graficas = pg.GraphicsLayoutWidget()
        self.graficas.setBackground(BG_COLOR)
        layout.addWidget(self.graficas)

        btn = QtWidgets.QPushButton("Detener EEG")
        btn.setStyleSheet(
            'background-color: #E05555; color: white; '
            'font-size: 13px; font-weight: bold; '
            'padding: 6px 16px; border: none;')
        btn.clicked.connect(self._detener)
        layout.addWidget(btn, alignment=QtCore.Qt.AlignCenter)

        self.curvas = {}
        ultimo_plot = None
        for i, nombre in enumerate(CANALES):
            p = self.graficas.addPlot(row=i, col=0)
            p.setMenuEnabled(False)
            p.hideButtons()
            p.setXRange(0, 8.0, padding=0)  # 8 s fijos
            p.getAxis('left').setTextPen(color=COLORES[nombre])
            p.getAxis('left').setLabel(nombre, color=COLORES[nombre], size='15pt')
            p.getAxis('bottom').setTextPen(AXIS_COLOR)
            p.getAxis('left').setPen(color=COLORES[nombre], width=1)
            p.getAxis('bottom').setPen(GRID_COLOR)
            p.showGrid(x=False, y=True, alpha=0.15)

            if i < len(CANALES) - 1:
                p.hideAxis('bottom')
            else:
                p.getAxis('bottom').setLabel('Tiempo (s)', color=AXIS_COLOR, size='10pt')

            if ultimo_plot:
                p.setXLink(ultimo_plot)
            ultimo_plot = p

            curva = p.plot(pen=pg.mkPen(color=COLORES[nombre], width=1))
            self.curvas[nombre] = (curva, p)

    def _iniciar_eeg(self):
        self.eeg.start_acquisition(
            stream_name=self.stream_name,
            save=False
        )

    def _revisar_flag(self):
        print(f"[FLAG] revisando... grabando={self._grabando} flag={os.path.exists(FLAG_PATH)}")
        """Vigila el flag y abre/cierra el CSV directamente."""
        flag_existe = os.path.exists(FLAG_PATH)

        if flag_existe and not self._grabando:
            try:
                with open(FLAG_PATH, 'r') as f:
                    partes = f.read().strip().split(',')
                self.condicion = partes[0] if len(partes) > 0 else self.condicion
                self.pid       = partes[1] if len(partes) > 1 else self.pid
                self.indice    = partes[2] if len(partes) > 2 else self.indice
                self.sid       = partes[3] if len(partes) > 3 else self.sid
            except Exception:
                pass

            os.makedirs(self.save_path, exist_ok=True)
            fecha  = datetime.datetime.now().strftime("%y%m%d")
            nombre = f"S{self.pid}_{fecha}_EEG_{self.sid}_{self.indice}_{self.condicion}.csv"
            ruta   = os.path.join(self.save_path, nombre)
            self._csv_file   = open(ruta, 'w', newline='', encoding='utf-8')
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(['Timestamp'] + CANALES)
            self._grabando = True
            print(f"[EEG] Grabando → {nombre} — {datetime.datetime.now().strftime('%H:%M:%S.%f')}")

        elif not flag_existe and self._grabando:
            self._grabando = False
            self._cerrar_csv()
            print(f"[EEG] CSV cerrado — {datetime.datetime.now().strftime('%H:%M:%S.%f')}")

    def _cerrar_csv(self):
        print("[EEG] _cerrar_csv llamado")
        if self._csv_file:
            try:
                self._csv_file.flush()
                self._csv_file.close()
            except Exception as e:
                print(f"[EEG] Error cerrando CSV: {e}")
            self._csv_file   = None
            self._csv_writer = None

    def _actualizar(self):
        """Vacía la cola, escribe al CSV si está grabando, actualiza gráfica."""
        nuevos = []
        while not self.eeg.grafica_queue.empty():
            try:
                nuevos.append(self.eeg.grafica_queue.get_nowait())
            except Exception:
                break

        if not nuevos:
            return

        # Escribir al CSV si está grabando
        if self._grabando and self._csv_writer:
            lsl_offset = self.eeg._lsl_offset
            for sample, timestamp in nuevos:
                try:
                    unix_time  = timestamp - lsl_offset
                    hora_local = datetime.datetime.fromtimestamp(unix_time).strftime(
                        '%H:%M:%S.%f')
                    self._csv_writer.writerow([hora_local] + sample)
                except Exception as e:
                    print(f"[EEG] Error escribiendo muestra: {e}")

        # Actualizar buffer de gráfica
        self.buffer.extend([s for s, _ in nuevos])
        ultimos = self.buffer[-EEG_VENTANA:]
        n = len(ultimos)
        x = np.linspace(max(0, 8.0 - n / 250.0), 8.0, n)

        for nombre, idx in zip(CANALES, INDICES):
            y = np.array([m[idx] for m in ultimos])
            if len(y) >= 15:
                try:
                    y = filtrar(y)
                except Exception:
                    pass

            curva, p = self.curvas[nombre]
            curva.setData(x, y)

            if len(y) > 0:
                p5, p95 = np.percentile(y, [5, 95])
                margin = (p95 - p5) * 0.2 or 1
                p.setYRange(p5 - margin, p95 + margin, padding=0)

    def _detener(self):
        self.timer.stop()
        self.timer_flag.stop()
        if os.path.exists(FLAG_PATH):
            os.remove(FLAG_PATH)
        self._grabando = False
        self._cerrar_csv()
        self.eeg.stop_acquisition()
        self.close()

    def closeEvent(self, event):
        print("[EEG] closeEvent disparado")
        self.timer.stop()
        self.timer_flag.stop()
        if os.path.exists(FLAG_PATH):
            os.remove(FLAG_PATH)
        self._grabando = False
        self._cerrar_csv()
        self.eeg.stop_acquisition()
        event.accept()


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    stream_name = sys.argv[1] if len(sys.argv) > 1 else 'streamTEST'
    save_path   = sys.argv[2] if len(sys.argv) > 2 else './Data'
    condicion   = sys.argv[3] if len(sys.argv) > 3 else 'EEG'
    pid         = sys.argv[4] if len(sys.argv) > 4 else 'X'
    indice      = sys.argv[5] if len(sys.argv) > 5 else '0'
    sid         = sys.argv[6] if len(sys.argv) > 6 else 'XX'

    app = QtWidgets.QApplication(sys.argv)
    win = VentanaEEG(stream_name, save_path, condicion, pid, indice, sid)
    win.show()
    sys.exit(app.exec_())