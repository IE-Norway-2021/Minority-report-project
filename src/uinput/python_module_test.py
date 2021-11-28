from python_module import *

# movements = {M_SCROLL_RIGHT, M_SCROLL_LEFT, M_SCROLL_UP, M_SCROLL_DOWN, M_ZOOM_IN, M_ZOOM_OUT}
# extern int init_uinput_device();
# extern int send_movement(int fd, enum movements mouvement_id);
# extern int close_uinput_device(int fd);

if __name__ == '__main__':
    fd = init_uinput_device()
    if fd < 0:
        print('Error fd wrong val')
        raise ValueError
    for movement in [M_SCROLL_RIGHT, M_SCROLL_LEFT, M_SCROLL_UP, M_SCROLL_DOWN, M_ZOOM_IN, M_ZOOM_OUT]:
        print(f'Sending {movement}...')
        ret_val = send_movement(fd, movement)
        if ret_val < 0:
            print(f'Error while issuing {movement} key')
            raise ValueError
    ret_val = close_uinput_device(fd)
    if ret_val < 0:
        print('Error while closing device')
        raise ValueError
