import asyncio
import threading
import pygame
from bleak import BleakClient, BleakScanner

DEVICE_NAME      = "ESP32-DRV2605"
CHAR_WRITE_UUID  = "abcd1234-ab12-cd34-ef56-abcdef123456"
CHAR_NOTIFY_UUID = "abcd1234-ab12-cd34-ef56-abcdef123457"

INTENSIDAD_ESPACIO = 150
DELAY_ESPACIO      = 1000

cliente_ble = None
loop        = None

def on_notify(sender, data):
    mensaje = data.decode("utf-8")
    print(f"[ESP32] {mensaje}")

    if mensaje.startswith("BAT:"):
        partes = mensaje.split(",")
        for p in partes:
            if "BAT:" in p:
                print(f"  Batería: {p.replace('BAT:', '')}")
            elif "V:" in p:
                print(f"  Voltaje: {p.replace('V:', '')}")
            elif "R:" in p:
                tasa = float(p.replace("R:", "").replace("%/h", ""))
                estado = "descargando" if tasa < 0 else "cargando"
                print(f"  Tasa: {p.replace('R:', '')} ({estado})")

    elif mensaje.startswith("CALIB:"):
        partes = mensaje.split(",")
        for p in partes:
            if "FREQ:" in p:
                print(f"  Frecuencia de resonancia: {p.replace('FREQ:', '')}")
            elif "CALIB:" in p:
                estado = p.replace("CALIB:", "")
                print(f"  Resultado: {'Exitosa' if estado == 'OK' else 'Fallida'}")

def enviar_vibracion(intensidad, delay_ms):
    """Envía desde el hilo de pygame al loop asyncio de BLE."""
    if cliente_ble is None:
        return
    cmd = f"{intensidad},{delay_ms}"
    asyncio.run_coroutine_threadsafe(
        cliente_ble.write_gatt_char(CHAR_WRITE_UUID, cmd.encode()),
        loop
    )

async def ble_task():
    global cliente_ble
    print("Buscando ESP32...")
    device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=10)
    if device is None:
        print("No se encontró el dispositivo")
        return

    async with BleakClient(device) as client:
        cliente_ble = client
        print(f"Conectado a {device.name}")
        await client.start_notify(CHAR_NOTIFY_UUID, on_notify)

        # Mantiene la conexión abierta hasta que pygame cierre
        while cliente_ble is not None:
            await asyncio.sleep(0.1)

        await client.stop_notify(CHAR_NOTIFY_UUID)

def ble_thread():
    """Corre el loop de asyncio en un hilo separado."""
    global loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(ble_task())

def main():
    global cliente_ble

    # Arranca BLE en hilo separado
    t = threading.Thread(target=ble_thread, daemon=True)
    t.start()

    # Espera a que se conecte antes de arrancar pygame
    print("Esperando conexión BLE...")
    while cliente_ble is None:
        import time; time.sleep(0.1)

    # ── Pygame ──────────────────────────────────────────────
    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("ESP32 Vibración")
    font  = pygame.font.SysFont(None, 36)
    clock = pygame.time.Clock()

    COLOR_FONDO   = (30, 30, 30)
    COLOR_TEXTO   = (255, 255, 255)
    COLOR_ACTIVO  = (0, 200, 100)

    espacio_presionado = False

    running = True
    while running:
        screen.fill(COLOR_FONDO)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_SPACE:
                    espacio_presionado = True
                    enviar_vibracion(INTENSIDAD_ESPACIO, DELAY_ESPACIO)
                    print(f"[ESPACIO] Enviando {INTENSIDAD_ESPACIO},{DELAY_ESPACIO}")

            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    espacio_presionado = False

        # UI simple
        color = COLOR_ACTIVO if espacio_presionado else COLOR_TEXTO
        texto = font.render("ESPACIO → vibrar", True, color)
        screen.blit(texto, (80, 120))

        info = font.render(f"I:{INTENSIDAD_ESPACIO}  D:{DELAY_ESPACIO}ms", True, (150, 150, 150))
        screen.blit(info, (120, 170))

        pygame.display.flip()
        clock.tick(60)

    cliente_ble = None  # señal para cerrar el loop BLE
    pygame.quit()

if __name__ == "__main__":
    main()