from machine import Pin, ADC
import time
import sys
import select

# Sensor pins
soil_pin = ADC(Pin(5))
conductivity_pin = ADC(Pin(6))
soil_pin.atten(ADC.ATTN_11DB)
conductivity_pin.atten(ADC.ATTN_11DB)

# Actuator pins
water = Pin(7, Pin.OUT)
fertiliser = Pin(15, Pin.OUT)

def handle_command(cmd):
    cmd = cmd.strip()
    if cmd == "PUMP_ON":    pump.value(1)
    elif cmd == "PUMP_OFF": pump.value(0)
    elif cmd == "FAN_ON":   fan.value(1)
    elif cmd == "FAN_OFF":  fan.value(0)
    elif cmd == "LIGHT_ON": light.value(1)
    elif cmd == "LIGHT_OFF":light.value(0)
    print("CMD_OK: " + cmd)

while True:
    soil = soil_pin.read()
    conductivity = conductivity_pin.read()
    print("{},{}".format(soil, conductivity))

    # Non-blocking check for incoming commands
    if select.select([sys.stdin], [], [], 0)[0]:
        cmd = sys.stdin.readline()
        handle_command(cmd)

    time.sleep(1)