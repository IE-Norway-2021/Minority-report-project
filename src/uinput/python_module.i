%module python_module
%{
   extern enum movements { M_SCROLL_RIGHT, M_SCROLL_LEFT, M_SCROLL_UP, M_SCROLL_DOWN, M_ZOOM_IN, M_ZOOM_OUT };
   extern int init_uinput_device();
   extern int send_movement(int fd, enum movements mouvement_id);
   extern int close_uinput_device(int fd);
%}

   extern enum movements { M_SCROLL_RIGHT, M_SCROLL_LEFT, M_SCROLL_UP, M_SCROLL_DOWN, M_ZOOM_IN, M_ZOOM_OUT };
   extern int init_uinput_device();
   extern int send_movement(int fd, enum movements mouvement_id);
   extern int close_uinput_device(int fd);
