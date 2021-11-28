#include "event_codes.h"
#include <libevdev/libevdev-uinput.h>
#include <libevdev/libevdev.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h> 
// Externe
// enum for mouvements
enum movements { M_SCROLL_RIGHT, M_SCROLL_LEFT, M_SCROLL_UP, M_SCROLL_DOWN, M_ZOOM_IN, M_ZOOM_OUT };
// init
struct libevdev_uinput *init_uinput_device();
// passer le mouvement
int send_movement(struct libevdev_uinput *uidev, enum movements mouvement_id);
// fermer
void close_uinput_device(struct libevdev_uinput *uidev);

// interne
int get_movement_value(enum movements mouvement_id);
const int EVENT_TAB[] = {SCROLL_UP, SCROLL_DOWN, SCROLL_RIGHT, SCROLL_LEFT, ZOOM_IN, ZOOM_OUT};

int get_movement_value(enum movements mouvement_id) {
   switch (mouvement_id) {
   case M_SCROLL_RIGHT:
      return SCROLL_RIGHT;
   case M_SCROLL_LEFT:
      return SCROLL_LEFT;
   case M_SCROLL_UP:
      return SCROLL_UP;
   case M_SCROLL_DOWN:
      return SCROLL_DOWN;
   case M_ZOOM_IN:
      return ZOOM_IN;
   case M_ZOOM_OUT:
      return ZOOM_OUT;
   default:
      return 0;
   }
}

struct libevdev_uinput *init_uinput_device() {
   int err;
   struct libevdev *dev;
   struct libevdev_uinput *uidev;

   dev = libevdev_new();
   libevdev_set_name(dev, "project_device");

   libevdev_enable_event_type(dev, EV_KEY);

   for (int i = 0; i < NUMBER_OF_EVENTS_HANDLED; ++i) {
      int res = libevdev_enable_event_code(dev, EV_KEY, EVENT_TAB[i], NULL);
      if (res != 0) {
         printf("init_uinput_device : Error while opening event %d, at position %d\n", EVENT_TAB[i], i);
         exit(res);
      }
   }

   err = libevdev_uinput_create_from_device(dev, LIBEVDEV_UINPUT_OPEN_MANAGED, &uidev);

   if (err != 0)
      exit(err);

   return uidev;
}

int send_movement(struct libevdev_uinput *uidev, enum movements mouvement_id) {

   if (libevdev_uinput_write_event(uidev, EV_KEY, get_movement_value(mouvement_id), KEY_PRESSED) == -1) {
      return -1;
   }
   if (libevdev_uinput_write_event(uidev, EV_SYN, SYN_REPORT, KEY_RELEASED) == -1) {
      return -1;
   }
   if (libevdev_uinput_write_event(uidev, EV_KEY, get_movement_value(mouvement_id), KEY_RELEASED) == -1) {
      return -1;
   }
   if (libevdev_uinput_write_event(uidev, EV_SYN, SYN_REPORT, KEY_RELEASED) == -1) {
      return -1;
   }
   return 0;
}

void close_uinput_device(struct libevdev_uinput *uidev) {libevdev_uinput_destroy(uidev); }