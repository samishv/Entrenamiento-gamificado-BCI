import asyncio
from bleak import BleakClient, BleakScanner

DEVICE_NAME       = "ESP32-DRV2605"
CHAR_WRITE_UUID   = "abcd1234-ab12-cd34-ef56-abcdef123456"
CHAR_NOTIFY_UUID  = "abcd1234-ab12-cd34-ef56-abcdef123457"

def on_notify(sender, data):
    mensaje = data.decode("utf-8")
    print(f"[ESP32] {mensaje}")

    if mensaje.startswith("CALIB:"):
        partes = mensaje.split(",")
        for p in partes:
            if "FREQ:" in p:
                print(f"  → Frecuencia de resonancia: {p.replace('FREQ:', '')}")
            elif "CALIB:" in p:
                estado = p.replace("CALIB:", "")
                print(f"  → Resultado: {'Exitosa' if estado == 'OK' else 'Fallida'}")

async def main():
    print("Buscando ESP32...")
    device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=10)
    if device is None:
        print("No se encontró el dispositivo")
        return

    async with BleakClient(device) as client:
        print(f"Conectado a {device.name}")
        await client.start_notify(CHAR_NOTIFY_UUID, on_notify)

        print("\nComandos disponibles:")
        print("  CALIBRAR    — calibra si no hay datos guardados")
        print("  INT,DELAY   — vibra (ej: 150,2000)")
        print("  salir       — desconecta\n")

        while True:
            cmd = input("Comando: ").strip()
            if cmd.lower() == "salir":
                break
            await client.write_gatt_char(CHAR_WRITE_UUID, cmd.encode())

            # Si es calibración espera la respuesta
            if cmd in ("CALIBRAR"):
                await asyncio.sleep(8)

        await client.stop_notify(CHAR_NOTIFY_UUID)

if __name__ == "__main__":
    asyncio.run(main())