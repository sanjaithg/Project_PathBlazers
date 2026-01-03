import serial
import time

# Function to send command to start the Lidar rotation
def start_lidar(serial_port='/dev/ttyUSB0', baudrate=230400):
    try:
        # Open the serial port
        lidar = serial.Serial(serial_port, baudrate, timeout=1)
        time.sleep(2)  # Wait for the connection to be established
        
        # LDS01 has specific command to stop scanning (rotation). The stop command is usually:
        # 'b' (Start), 'e' (Stop)
        start_command = b'b'

        # Send the start command
        lidar.write(start_command)

        print("Start command sent to Lidar.")

        # Close the connection to the Lidar
        lidar.close()

    except Exception as e:
        print(f"Error communicating with the Lidar: {e}")

if __name__ == '__main__':
    # Replace with the correct serial port
    start_lidar(serial_port='/dev/ttyUSB0')
