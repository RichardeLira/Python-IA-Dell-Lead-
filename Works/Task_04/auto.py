import pyautogui
import time 


time_duration =5
time.sleep(time_duration)

i=1
while i<=2239:
    pyautogui.PAUSE = 0.000000001
    pyautogui.write(',')
    pyautogui.press('down')
    pyautogui.press('left')
    i+=1