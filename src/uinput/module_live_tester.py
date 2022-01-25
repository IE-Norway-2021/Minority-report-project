"""module_live_tester.py
Emits 3 inputs in an eternal loop using the uinput python module

To use this, make sure to compile the python module first (make module). Then launch this python script. 
It will send a zoom in, a zoom out and a scroll right continiously. Then switch to an open google earth or google maps window. The map should react to the inputs
"""
from python_module import *
import time


if __name__ == '__main__':
   try:
      fd = init_uinput_device()
      if fd < 0:
         print('Error fd wrong val')
         raise ValueError
      while True:
         print('Sending zoom in...')
         ret_val = send_movement(fd, M_ZOOM_IN)
         if ret_val < 0:
            print('Error during zoom in')
            raise ValueError
         time.sleep(4)
         print('Sending zoom out...')
         ret_val = send_movement(fd, M_ZOOM_OUT)
         if ret_val < 0:
            print('Error during zoom out')
            raise ValueError
         time.sleep(4)
         print('Sending scroll right...')
         ret_val = send_movement(fd, M_SCROLL_RIGHT)
         if ret_val < 0:
            print('Error during scroll right')
            raise ValueError
         time.sleep(4)
   finally:
      ret_val = close_uinput_device(fd)
      if ret_val < 0:
         print('Error while closing device')
         raise ValueError
