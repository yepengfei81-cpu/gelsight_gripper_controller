import time
import serial
 
LOBOT__FRAME_HEADER              = 0x55
LOBOT_CMD_SERVO_MOVE             = 3
LOBOT_CMD_ACTION_GROUP_RUN       = 6
LOBOT_CMD_ACTION_GROUP_STOP      = 7
LOBOT_CMD_ACTION_GROUP_SPEED     = 0x0B
LOBOT_CMD_SET_ID                 = 0x0D
LOBOT_CMD_GET_BATTERY_VOLTAGE    = 0x0F
LOBOT_SERVO_ANGLE_LIMIT_WRITE    = 0x14
 
serialHandle = serial.Serial("/dev/ttyACM0", 115200)  # 初始化串口， 波特率为9600
 
def checksum(buf):
    # 计算校验和
    sum = 0x00
    for b in buf:  # 求和
        sum += b
    sum = sum - 0x55 - 0x55  # 去掉命令开头的两个 0x55
    sum = ~sum  # 取反
    return sum & 0xff
 
# 设置舵机转动范围
def setBusServoLimit(servo_id, low, high):
    buf = bytearray(b'\x55\x55')
    servo_id = 254 if (servo_id < 0 or servo_id > 254) else servo_id
    buf.append(servo_id)

    buf.append(0x07)
    buf.append(LOBOT_SERVO_ANGLE_LIMIT_WRITE)
    
    low_low = low & 0xFF
    low_high = (low >> 8) & 0xFF
    buf.append(low_low)
    buf.append(low_high)
    
    high_low = high & 0xFF
    high_high = (high >> 8) & 0xFF
    buf.append(high_low)
    buf.append(high_high)

    buf.append(checksum(buf))
    serialHandle.write(buf)

# 设置舵机ID
def setBusServoID(old_id, new_id):
    buf = bytearray(b'\x55\x55')
    old_id = 254 if (old_id < 0 or old_id > 254) else old_id
    buf.append(old_id)

    buf.append(0x04) # lenth    
    buf.append(LOBOT_CMD_SET_ID)

    new_id = 254 if (new_id < 0 or new_id > 254) else new_id
    buf.append(new_id)

    buf.append(checksum(buf))
    serialHandle.write(buf)

# 控制单个舵机旋转
def setBusServoPos(servo_id, angle, time):
    buf = bytearray(b'\x55\x55')
    servo_id = 254 if (servo_id < 0 or servo_id > 254) else servo_id
    buf.append(servo_id)

    buf.append(0x07)
    buf.append(0x01)
    
    low_low = angle & 0xFF
    low_high = (angle >> 8) & 0xFF
    buf.append(low_low)
    buf.append(low_high)
    
    high_low = time & 0xFF
    high_high = (time >> 8) & 0xFF
    buf.append(high_low)
    buf.append(high_high)

    buf.append(checksum(buf))
    serialHandle.write(buf)    

    