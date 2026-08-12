#include <Wire.h>
#include "Adafruit_DRV2605.h"
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <Preferences.h>

#define enaB 4
#define LED_NARANJITA 2
#define LED_VERDE 3

#define SERVICE_UUID          "12345678-1234-1234-1234-123456789abc"
#define CHARACTERISTIC_WRITE  "abcd1234-ab12-cd34-ef56-abcdef123456"
#define CHARACTERISTIC_NOTIFY "abcd1234-ab12-cd34-ef56-abcdef123457"

Adafruit_DRV2605 drv;
Preferences prefs;
BLECharacteristic *pCharWrite;
BLECharacteristic *pCharNotify;

bool deviceConnected     = false;
bool parametrosRecibidos = false;
bool calibrar            = false;

uint8_t  intensidad = 0;
uint32_t delayMs    = 0;

// BLE helpers 

void enviarMensaje(String msg) {
  if (!deviceConnected) return;
  pCharNotify->setValue(msg.c_str());
  pCharNotify->notify();
  delay(10);
}

// Carga o guarda la calibracion en memoria

void guardarCalibracion() {
  uint8_t comp       = drv.readRegister8(0x18);
  uint8_t bemf       = drv.readRegister8(0x19);
  uint8_t lra_period = drv.readRegister8(0x22);

  prefs.begin("drv2605", false);
  prefs.putUChar("comp", comp);
  prefs.putUChar("bemf", bemf);
  prefs.putUChar("period", lra_period);
  prefs.putBool("calibrado", true);
  prefs.end();

  enviarMensaje("Calibración guardada en memoria");
}

bool cargarCalibracion() {
  prefs.begin("drv2605", true);
  bool calibrado = prefs.getBool("calibrado", false);

  if (calibrado) {
    uint8_t comp       = prefs.getUChar("comp", 0);
    uint8_t bemf       = prefs.getUChar("bemf", 0);
    uint8_t lra_period = prefs.getUChar("period", 0);
    prefs.end();

    drv.writeRegister8(0x18, comp);
    drv.writeRegister8(0x19, bemf);
    drv.writeRegister8(0x22, lra_period);
    return true;
  }

  prefs.end();
  return false;
}

// Calibracion

void calibrarLRA() {
  enviarMensaje("Iniciando calibración...");

  drv.setMode(DRV2605_MODE_AUTOCAL);
  drv.writeRegister8(DRV2605_REG_FEEDBACK, 0xB6);
  drv.writeRegister8(DRV2605_REG_CONTROL1, 0x13);
  drv.writeRegister8(DRV2605_REG_CONTROL2, 0xF5);
  drv.writeRegister8(DRV2605_REG_CONTROL3, 0xA0);
  drv.writeRegister8(0x16, 0x53);  // RATED_VOLTAGE  
  drv.writeRegister8(0x17, 0x89);  // OD_CLAMP       
  drv.go();

  uint8_t go      = 1;
  uint8_t timeout = 0;
  while (go && timeout < 50) {
    delay(100);
    go = drv.readRegister8(DRV2605_REG_GO) & 0x01;
    timeout++;
  }

  if (timeout >= 50) {
    enviarMensaje("ERROR: calibración tardó demasiado");
    drv.setMode(DRV2605_MODE_REALTIME);
    drv.setRealtimeValue(0);
    return;
  }

  uint8_t status  = drv.readRegister8(DRV2605_REG_STATUS);
  bool    diag_ok = !(status & 0x08);

  uint8_t lra_period = drv.readRegister8(0x22);
  float   freqHz     = 0;
  if (lra_period > 0) {
    freqHz = 1.0 / (lra_period * 0.0000984615);
  }

  uint8_t comp = drv.readRegister8(0x18);
  uint8_t bemf = drv.readRegister8(0x19);

  String resultado = "CALIB:";
  resultado += diag_ok ? "OK" : "FAIL";
  resultado += ",FREQ:" + String(freqHz, 2) + "Hz";
  resultado += ",PERIOD:" + String(lra_period);
  resultado += ",COMP:" + String(comp);
  resultado += ",BEMF:" + String(bemf);
  enviarMensaje(resultado);

  if (diag_ok) {
    guardarCalibracion();
  } else {
    enviarMensaje("ADVERTENCIA: calibración fallida, no se guardó");
  }

  drv.setMode(DRV2605_MODE_REALTIME);
  drv.setRealtimeValue(0);
}

// 

class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) {
    deviceConnected = true;
    digitalWrite(LED_VERDE, HIGH);
  }
  void onDisconnect(BLEServer* pServer) {
    deviceConnected     = false;
    parametrosRecibidos = false;
    calibrar            = false;
    drv.setRealtimeValue(0);
    digitalWrite(LED_VERDE, LOW);
    pServer->startAdvertising();
  }
};

class MyCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *pChar) {
    String value = pChar->getValue().c_str();
    value.trim();

    if (value == "CALIBRAR") {
      calibrar = true;
    } else {
      int commaIndex = value.indexOf(',');
      if (commaIndex != -1) {
        intensidad          = value.substring(0, commaIndex).toInt();
        delayMs             = value.substring(commaIndex + 1).toInt();
        parametrosRecibidos = true;
      } else {
        enviarMensaje("ERROR: formato inválido, usa intensidad,delay o CALIBRAR o RECALIBRAR");
      }
    }
  }
};


void setup() {
  Serial.begin(9600);
  pinMode(LED_NARANJITA, OUTPUT);
  pinMode(LED_VERDE, OUTPUT);

  digitalWrite(LED_NARANJITA, HIGH);
  digitalWrite(LED_VERDE, LOW);
  delay(1000);
  digitalWrite(LED_NARANJITA, LOW);

  pinMode(enaB, OUTPUT);
  digitalWrite(enaB, HIGH);

  if (!drv.begin()) {
    while (1) delay(10);
  }
  drv.selectLibrary(6);
  drv.setMode(DRV2605_MODE_REALTIME);
  drv.setRealtimeValue(0);

  // Cargar calibración si es que hay
  if(cargarCalibracion()){
    enviarMensaje("Calibración cargada desde memoria");
  }

  // BLE
  BLEDevice::init("ESP32-DRV2605");
  BLEServer *pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);

  pCharWrite = pService->createCharacteristic(
    CHARACTERISTIC_WRITE,
    BLECharacteristic::PROPERTY_WRITE
  );
  pCharWrite->setCallbacks(new MyCallbacks());

  pCharNotify = pService->createCharacteristic(
    CHARACTERISTIC_NOTIFY,
    BLECharacteristic::PROPERTY_NOTIFY
  );
  pCharNotify->addDescriptor(new BLE2902());

  pService->start();
  pServer->getAdvertising()->start();
}


void loop() {

  // Calibración normal (solo si no hay guardada)
  if (calibrar) {
    calibrar = false;
    if (!cargarCalibracion()) {  // solo calibra si no hay datos guardados
        prefs.begin("drv2605", false);
        prefs.putBool("calibrado", false);
        prefs.end();
        calibrarLRA();
    }
    return;
  }

  if (!parametrosRecibidos) {
    delay(100);
    return;
  }

  drv.setRealtimeValue(intensidad);
  delay(delayMs);
  drv.setRealtimeValue(0);
  delay(1000);
  parametrosRecibidos = false;
}