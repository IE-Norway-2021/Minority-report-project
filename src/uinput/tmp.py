from python_module import *
import time


if __name__ == '__main__':
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
   ret_val = close_uinput_device(fd)
   if ret_val < 0:
      print('Error while closing device')
      raise ValueError
