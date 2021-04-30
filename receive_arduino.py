#-*- coding: utf-8 -*- 
#收到arduino的訊號

import serial  # 引用pySerial模組

COM_PORT = r'/dev/ttyACM0'    # 指定通訊埠名稱
BAUD_RATES = 9600    # 設定傳輸速率
ser = serial.Serial(COM_PORT, BAUD_RATES)   # 初始化序列通訊埠

try:
    while True:
        while ser.in_waiting:          # 若收到序列資料…
            data_raw = ser.readline()  # 讀取一行
            data = data_raw.decode()   # 用預設的UTF-8解碼
            print('接收到的原始資料：', data_raw)
            print('receive：', data)

            data_int = float(data)
            if data_int > 50:
                print('DANGEROUS!!')
            if data_int < 50:
                print('Normal temperature.')
except KeyboardInterrupt:
    ser.close()    # 清除序列通訊物件
    print('再見！')
