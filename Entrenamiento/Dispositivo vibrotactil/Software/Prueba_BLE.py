import asyncio
from bleak import BleakClient, BleakScanner

DEVICE_NAME      = "ESP32-DRV2605"
CHARACTERISTIC_UUID = "abcd1234-ab12-cd34-ef56-abcdef123456"

async def enviar_parametros(intensidad: int, delay_ms: int):
    print("Buscando dispositivo BLE...")
    device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=10)
    if device is None:
        print(f"No se encontró '{DEVICE_NAME}'")
        return

    async with BleakClient(device) as client:
        print(f"Conectado a {device.name}")
        mensaje = f"{intensidad},{delay_ms}"
        await client.write_gatt_char(CHARACTERISTIC_UUID, mensaje.encode())
        print(f"Enviado → intensidad={intensidad}, delay={delay_ms}ms")

if __name__ == "__main__":
    # Cambia estos valores a gusto
    INTENSIDAD = 150   # 0–255
    DELAY_MS   = 3000  # milisegundos

    asyncio.run(enviar_parametros(INTENSIDAD, DELAY_MS))