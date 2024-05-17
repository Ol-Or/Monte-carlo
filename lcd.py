# sudo raspi-config

#i2cdetect -y 1  # 연결된 장치의 주소 확인

#git clone http://gist.github.com/DenisFromHR/cc863375a6e19dce359d  //LCD 라이브러리 설치

#cd cc863375a6e19dce359d  //이동

import RPi_I2C_driver
import time

# LCD 객체 생성
mylcd = RPi_I2C_driver.lcd()

if person_count >= 5:
    mylcd.lcd_display_string("Temperature is", 1)
    mylcd.lcd_display_string("24.0", 2)
else:
    mylcd.lcd_display_string("Temperature is", 1)
    mylcd.lcd_display_string("26.0", 2)

time.sleep(3)

# 화면 지우기
mylcd.lcd_clear()
time.sleep(1)

# 백라이트 끄기
mylcd.backlight(0)

