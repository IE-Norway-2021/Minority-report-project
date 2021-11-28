%module python_module
%{
   /* Put header files here or function declarations like below */
   extern enum movements { M_SCROLL_RIGHT, M_SCROLL_LEFT, M_SCROLL_UP, M_SCROLL_DOWN, M_ZOOM_IN, M_ZOOM_OUT };
   extern struct libevdev_uinput* init_uinput_device();
   extern int send_movement(struct libevdev_uinput *uidev, enum movements mouvement_id);
   extern void close_uinput_device(struct libevdev_uinput *uidev);
%}

%include <libevdev/libevdev-uinput.h>

extern enum movements { M_SCROLL_RIGHT, M_SCROLL_LEFT, M_SCROLL_UP, M_SCROLL_DOWN, M_ZOOM_IN, M_ZOOM_OUT };
extern struct libevdev_uinput* init_uinput_device();
extern int send_movement(struct libevdev_uinput *uidev, enum movements mouvement_id);
extern void close_uinput_device(struct libevdev_uinput *uidev);
