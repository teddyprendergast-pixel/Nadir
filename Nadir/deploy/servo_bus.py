import serial
import struct
import time
from typing import List, Optional, Dict, Tuple

# STS3215 Protocol Constants
STS_HEADER = 0xFF
STS_INST_PING = 0x01
STS_INST_READ = 0x02
STS_INST_WRITE = 0x03
STS_INST_SYNC_WRITE = 0x83
STS_INST_SYNC_READ = 0x82

# Register addresses
STS_GOAL_POSITION = 42      # 2 bytes, 0-4095 (0.088 deg resolution)
STS_PRESENT_POSITION = 56   # 2 bytes
STS_PRESENT_SPEED = 58      # 2 bytes (signed)
STS_PRESENT_LOAD = 60       # 2 bytes (signed, ~proportional to torque)
STS_PRESENT_TEMPERATURE = 63 # 1 byte, degrees C
STS_TORQUE_ENABLE = 40      # 1 byte

class STS3215Bus:
    """Driver for Feetech STS3215 serial bus servos.
    
    Half-duplex TTL serial: TX and RX share the same wire.
    Uses GPIO to switch direction on Pi (or a UART with half-duplex support).
    """
    
    def __init__(self, port: str = '/dev/ttyAMA0', baudrate: int = 1000000,
                 servo_ids: List[int] = None, direction_pin: int = None):
        """Initialize the servo bus.
        
        Args:
            port: Serial port path
            baudrate: Bus speed (typically 500000 or 1000000)
            servo_ids: List of servo IDs on the bus (e.g., [1,2,3,4,5,6,7,8,9,10])
            direction_pin: GPIO pin for half-duplex direction control (None = auto)
        """
        self.port = port
        self.baudrate = baudrate
        self.servo_ids = servo_ids if servo_ids is not None else list(range(1, 11))
        self.direction_pin = direction_pin
        
        self.serial = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=0.01,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE
        )
        
        if self.direction_pin is not None:
            try:
                import RPi.GPIO as GPIO
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self.direction_pin, GPIO.OUT)
                GPIO.output(self.direction_pin, GPIO.LOW)
                self._set_tx = lambda: GPIO.output(self.direction_pin, GPIO.HIGH)
                self._set_rx = lambda: GPIO.output(self.direction_pin, GPIO.LOW)
            except ImportError:
                print("Warning: RPi.GPIO not found. Ignoring direction_pin.")
                self._set_tx = lambda: None
                self._set_rx = lambda: None
        else:
            self._set_tx = lambda: None
            self._set_rx = lambda: None
    
    def _send_packet(self, servo_id: int, instruction: int, params: bytes = b'') -> bytes:
        """Build and send an STS protocol packet, return response."""
        length = len(params) + 2
        checksum = ~(servo_id + length + instruction + sum(params)) & 0xFF
        packet = bytes([STS_HEADER, STS_HEADER, servo_id, length, instruction]) + params + bytes([checksum])
        
        self.serial.reset_input_buffer()
        self._set_tx()
        self.serial.write(packet)
        self.serial.flush()
        self._set_rx()
        
        # Read response if it's a read or ping command
        if instruction in (STS_INST_READ, STS_INST_PING):
            header = self.serial.read(4)
            if len(header) < 4 or header[0] != STS_HEADER or header[1] != STS_HEADER:
                return b''
            
            resp_id = header[2]
            resp_length = header[3]
            
            if resp_id != servo_id:
                return b''
                
            payload = self.serial.read(resp_length)
            if len(payload) != resp_length:
                return b''
                
            error = payload[0]
            resp_params = payload[1:-1]
            resp_checksum = payload[-1]
            
            calc_checksum = ~(resp_id + resp_length + error + sum(resp_params)) & 0xFF
            if calc_checksum != resp_checksum:
                return b''
                
            return resp_params
            
        return b''
    
    def ping(self, servo_id: int) -> bool:
        """Ping a servo, return True if it responds."""
        resp = self._send_packet(servo_id, STS_INST_PING)
        # For ping, any valid response means it is alive
        return len(self._send_packet(servo_id, STS_INST_PING, b'')) == 0 if resp is not None else False
    
    def read_position(self, servo_id: int) -> Optional[float]:
        """Read current position in radians."""
        # read 2 bytes starting from STS_PRESENT_POSITION
        resp = self._send_packet(servo_id, STS_INST_READ, bytes([STS_PRESENT_POSITION, 2]))
        if len(resp) == 2:
            raw_pos = resp[0] | (resp[1] << 8)
            return self._raw_to_rad(raw_pos)
        return None
    
    def read_all_positions(self) -> Dict[int, float]:
        """Read positions of all servos on the bus."""
        positions = {}
        for sid in self.servo_ids:
            pos = self.read_position(sid)
            if pos is not None:
                positions[sid] = pos
        return positions
    
    def read_all_state(self) -> Dict[int, Dict]:
        """Read position, speed, load, temperature of all servos."""
        states = {}
        for sid in self.servo_ids:
            resp = self._send_packet(sid, STS_INST_READ, bytes([STS_PRESENT_POSITION, 8]))
            if len(resp) == 8:
                raw_pos = resp[0] | (resp[1] << 8)
                raw_speed = resp[2] | (resp[3] << 8)
                raw_load = resp[4] | (resp[5] << 8)
                temp = resp[7]
                
                # Handling signs for speed and load
                if raw_speed & 0x8000:
                    raw_speed = -(raw_speed & 0x7FFF)
                if raw_load & 0x8000:
                    raw_load = -(raw_load & 0x7FFF)
                    
                states[sid] = {
                    'position': self._raw_to_rad(raw_pos),
                    'speed': raw_speed,
                    'load': raw_load,
                    'temperature': temp
                }
        return states
    
    def write_position(self, servo_id: int, position_rad: float) -> None:
        """Write goal position in radians to a single servo."""
        raw_pos = self._rad_to_raw(position_rad)
        params = bytes([STS_GOAL_POSITION, raw_pos & 0xFF, (raw_pos >> 8) & 0xFF])
        self._send_packet(servo_id, STS_INST_WRITE, params)
    
    def sync_write_positions(self, positions: Dict[int, float]) -> None:
        """Write goal positions to multiple servos in a single bus transaction.
        This is the fast path used by the 50 Hz control loop."""
        # Uses SYNC_WRITE instruction for minimal bus time
        if not positions:
            return
            
        data_len = 2
        params = bytes([STS_GOAL_POSITION, data_len])
        for sid, pos_rad in positions.items():
            raw_pos = self._rad_to_raw(pos_rad)
            params += bytes([sid, raw_pos & 0xFF, (raw_pos >> 8) & 0xFF])
            
        # broadcast id is 0xFE
        self._send_packet(0xFE, STS_INST_SYNC_WRITE, params)
    
    def enable_torque(self, servo_id: int, enable: bool = True) -> None:
        """Enable or disable torque for a specific servo."""
        val = 1 if enable else 0
        self._send_packet(servo_id, STS_INST_WRITE, bytes([STS_TORQUE_ENABLE, val]))
        
    def enable_all_torque(self, enable: bool = True) -> None:
        """Enable or disable torque for all servos."""
        for sid in self.servo_ids:
            self.enable_torque(sid, enable)
            time.sleep(0.002)  # small delay
    
    def _rad_to_raw(self, rad: float) -> int:
        """Convert radians to STS3215 raw position (0-4095)."""
        # STS3215: 4096 steps over ~300 degrees
        # Center position = 2048 = 0 radians
        raw = int(2048 + rad * (4096 / (300 * 3.14159265 / 180)))
        return max(0, min(4095, raw))
    
    def _raw_to_rad(self, raw: int) -> float:
        """Convert STS3215 raw position to radians."""
        return (raw - 2048) * (300 * 3.14159265 / 180) / 4096
    
    def close(self) -> None:
        """Close the serial connection."""
        self.serial.close()
