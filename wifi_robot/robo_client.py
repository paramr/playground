import os
from socket import *
from time import ctime
import binascii
import RPi.GPIO as GPIO
import time
import threading
from smbus import SMBus
import numpy as np

XRservo = SMBus(1)

# Signal pin definition
GPIO.setmode(GPIO.BCM)

# Motor drive interface definition
ENA = 13 # L298 Enable A
ENB = 20 # L298 Enable B
IN1 = 19 # Motor connection 1
IN2 = 16 # Motor connection 2
IN3 = 21 # Motor connection 3
IN4 = 26 # Motor connection 4


# Pin type setting and initialization
GPIO.setwarnings(False)

# The motor is initialized to LOW
GPIO.setup(ENA, GPIO.OUT, initial=GPIO.LOW)
Left_pwm=GPIO.PWM(ENA, 1000) 
Left_pwm.start(0) 
Left_pwm.ChangeDutyCycle(100)
GPIO.setup(IN1, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(IN2, GPIO.OUT, initial=GPIO.LOW)

GPIO.setup(ENB, GPIO.OUT, initial=GPIO.LOW)
Right_pwm=GPIO.PWM(ENB, 1000) 
Right_pwm.start(0) 
Right_pwm.ChangeDutyCycle(100)
GPIO.setup(IN3, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(IN4, GPIO.OUT, initial=GPIO.LOW)

def Motor_Forward():
  print 'motor forward'
  GPIO.output(ENA,True)
  GPIO.output(ENB,True)
  GPIO.output(IN1,True)
  GPIO.output(IN2,False)
  GPIO.output(IN3,True)
  GPIO.output(IN4,False)

def Motor_Backward():
  print 'motor_backward'
  GPIO.output(ENA,True)
  GPIO.output(ENB,True)
  GPIO.output(IN1,False)
  GPIO.output(IN2,True)
  GPIO.output(IN3,False)
  GPIO.output(IN4,True)

def Motor_TurnLeft():
  print 'motor_turnleft'
  GPIO.output(ENA,True)
  GPIO.output(ENB,True)
  GPIO.output(IN1,True)
  GPIO.output(IN2,False)
  GPIO.output(IN3,False)
  GPIO.output(IN4,True)

def Motor_TurnRight():
  print 'motor_turnright'
  GPIO.output(ENA,True)
  GPIO.output(ENB,True)
  GPIO.output(IN1,False)
  GPIO.output(IN2,True)
  GPIO.output(IN3,True)
  GPIO.output(IN4,False)

def Motor_Stop():
  print 'motor_stop'
  GPIO.output(ENA,False)
  GPIO.output(ENB,False)
  GPIO.output(IN1,False)
  GPIO.output(IN2,False)
  GPIO.output(IN3,False)
  GPIO.output(IN4,False)

def LeftSpeed(left_speed):
  print 'left Speed: %d'%left_speed
  ENA_pwm.ChangeDutyCycle(left_speed)

def RightSpeed(right_speed):
  print 'right Speed: %d'%right_speed
  ENB_pwm.ChangeDutyCycle(right_speed)

def Angle_cal(angle_from_protocol):
  angle=hex(eval('0x'+angle_from_protocol))
  angle=int(angle,16)
  if angle > 160:
    angle=160
  elif angle < 15:
    angle=15
  return angle

def SetHorizontalServoAngle(ServoNum,angle_from_protocol):
  XRservo.XiaoRGEEK_SetServo(0x07, Angle_cal(angle_from_protocol))

def SetVerticalServoAngle(ServoNum,angle_from_protocol):
  XRservo.XiaoRGEEK_SetServo(0x08, Angle_cal(angle_from_protocol))

'''
HOST=''
PORT=2001
ADDR=(HOST,PORT)
tcpSerSock=socket(AF_INET,SOCK_STREAM)
tcpSerSock.bind(ADDR)
tcpSerSock.listen(1)
BUFSIZ=1
rec_flag=0
i=0
buffer = ['00','00','00','00','00','00']

while True:
  print 'waitting for connection...'
  tcpCliSock,addr=tcpSerSock.accept()
  print '...connected from:',addr
  while True:
    try:
      data=tcpCliSock.recv(BUFSIZ)
      data=binascii.b2a_hex(data)
    except:
      print "Error receiving:"
      break
        
    if not data:
      break
    if rec_flag==0:
      if data=='ff':  
        buffer[:]=[]
        rec_flag=1
        i=0
    else:
      if data=='ff':
        rec_flag=0
        if i==3:
          print 'Got data',str(buffer)[1:len(str(buffer)) - 1],"\r"
          Communication_Decode();
          i=0
        else:
          buffer.append(data)
          i+=1
    tcpCliSock.close()
Motor_Stop()
tcpSerSock.close()
'''

Motor_Backward()
time.sleep(1)
Motor_Stop()
