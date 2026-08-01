// Código para Hand Dynamometer Vernier HD-BTA (HD462DR2)
// 100 Hz con media móvil de 100 datos y calibración de cero

// Configurar prescaler del ADC
#define PS_16 (1 << ADPS2)
#define PS_32 (1 << ADPS2) | (1 << ADPS0)
#define PS_64 (1 << ADPS2) | (1 << ADPS1)
#define PS_128 (1 << ADPS2) | (1 << ADPS1) | (1 << ADPS0)

const int SENSOR_PIN = A0;        // Pin analógico donde conectas el sensor
const unsigned long BAUD_RATE = 115200;  // Velocidad de comunicación serie

// CONFIGURACIÓN DE FRECUENCIA DE MUESTREO
const float SAMPLE_RATE_HZ = 100.0;  // 100 Hz (100 muestras por segundo)
const unsigned long SAMPLE_INTERVAL_MICROS = (unsigned long)(1000000.0 / SAMPLE_RATE_HZ);

// CONFIGURACIÓN DE MEDIA MÓVIL
const int MOVING_AVERAGE_SIZE = 100;  // 100 datos para la media móvil
float movingAverageBuffer[MOVING_AVERAGE_SIZE];
int bufferIndex = 0;
bool bufferFull = false;
float movingAverageSum = 0.0;

// Variables para calibración del HD-BTA
const float VOLTAGE_REF = 5.0;    // Voltaje de referencia (5V para Arduino Uno)
const float ADC_RESOLUTION = 1023.0;  // Resolución del ADC (10 bits)

// Parámetros de calibración específicos para HD-BTA Hand Dynamometer
const float SLOPE = 176.8325;     // Pendiente de calibración (N/V)
const float INTERCEPT = -19.5057; // Intercepto de calibración (N)

// Variables para calibración de cero
float zeroOffset = 0.0;           // Offset para establecer cero
bool zeroCalibrated = false;      // Flag para saber si ya se calibró el cero

// Variables para control de envío de datos
bool sendingData = false;         // Flag para controlar envío de datos

// Variables para control de tiempo
unsigned long lastSampleTime = 0;
unsigned long sampleCount = 0;
unsigned long startTime = 0;

void setup() {
  // Configurar comunicación serie a alta velocidad
  Serial.begin(BAUD_RATE);
  
  // Configurar el pin analógico como entrada
  pinMode(SENSOR_PIN, INPUT);
  
  // Configurar ADC para máxima velocidad
  ADCSRA &= ~PS_128;
  ADCSRA |= PS_16;
  
  // Inicializar buffer de media móvil con ceros
  for(int i = 0; i < MOVING_AVERAGE_SIZE; i++) {
    movingAverageBuffer[i] = 0.0;
  }
  
  // Calibración automática del cero
  calibrarCero();
  // Inicializar tiempo de referencia
  startTime = micros();
  lastSampleTime = startTime;
}

void loop() {
  unsigned long currentTime = micros();
  
  // Verificar si se recibió comando por puerto serie
  if (Serial.available() > 0) {
    char comando = Serial.read();
    procesarComando(comando);
  }
  
  // Solo procesar datos si está habilitado el envío
  if (sendingData && (currentTime - lastSampleTime >= SAMPLE_INTERVAL_MICROS)) {
    
    // Leer el sensor y procesar
    float instantaneousForce = leerFuerza();
    
    // Agregar a la media móvil
    float averageForce = agregarAMediaMovil(instantaneousForce);
    
    // Incrementar contador de muestras
    sampleCount++;
    
    // Enviar solo la media móvil (formato simple para Python)
    Serial.println(averageForce, 3);  // 3 decimales de precisión
    
    // Actualizar tiempo de la última muestra
    lastSampleTime = currentTime;
  }
}

// Función para leer fuerza con calibración de cero
float leerFuerza() {
  // Leer valor ADC
  int rawValue = analogRead(SENSOR_PIN);
  
  // Convertir a voltaje
  float voltage = (rawValue * VOLTAGE_REF) / ADC_RESOLUTION;
  
  // Convertir a fuerza usando calibración oficial HD-BTA
  float force = (voltage * SLOPE) + INTERCEPT;
  
  // Aplicar offset de cero
  force = force - zeroOffset;
  
  // Mantener valores negativos en 0
  if (force < 0.0) {
    force = 0.0;
  }
  
  return force;
}

// Función para agregar valor a la media móvil
float agregarAMediaMovil(float nuevoValor) {
  // Restar el valor que vamos a reemplazar de la suma
  movingAverageSum -= movingAverageBuffer[bufferIndex];
  
  // Agregar el nuevo valor
  movingAverageBuffer[bufferIndex] = nuevoValor;
  movingAverageSum += nuevoValor;
  
  // Avanzar el índice del buffer (circular)
  bufferIndex++;
  if (bufferIndex >= MOVING_AVERAGE_SIZE) {
    bufferIndex = 0;
    bufferFull = true;  // Ahora el buffer está lleno
  }
  
  // Calcular y retornar la media móvil
  if (bufferFull) {
    return movingAverageSum / MOVING_AVERAGE_SIZE;
  } else {
    // Si el buffer no está lleno, usar solo los datos disponibles
    return movingAverageSum / (bufferIndex);
  }
}

// Función para procesar comandos del puerto serie
void procesarComando(char comando) {
  switch(comando) {
    case 's':
    case 'S':
      if (!sendingData) {
        sendingData = true;
        sampleCount = 0;  // Reiniciar contador
        startTime = micros();  // Reiniciar tiempo de referencia
        lastSampleTime = startTime;
      }
      break;
      
    case 'p':
    case 'P':
      if (sendingData) {
        sendingData = false;
      }
      break;
      
    case 'z':
    case 'Z':
      bool wasRunning = sendingData;
      if (wasRunning) {
        sendingData = false;  // Pausar envío durante calibración
      }
      calibrarCero();
      if (wasRunning) {
      }
      break;
      
    default:
      break;
  }
}

// Función para calibrar el cero
void calibrarCero() {
  delay(3000);
  
  const int numMuestras = 50;
  float suma = 0.0;
   
  for(int i = 0; i < numMuestras; i++) {
    int rawValue = analogRead(SENSOR_PIN);
    float voltage = (rawValue * VOLTAGE_REF) / ADC_RESOLUTION;
    float force = (voltage * SLOPE) + INTERCEPT;
    suma += force;
    delay(20);
  }
  
  zeroOffset = suma / numMuestras;
  zeroCalibrated = true;

  // Limpiar buffer de media móvil después de calibrar
  for(int i = 0; i < MOVING_AVERAGE_SIZE; i++) {
    movingAverageBuffer[i] = 0.0;
  }
  movingAverageSum = 0.0;
  bufferIndex = 0;
  bufferFull = false;
}
