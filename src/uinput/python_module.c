/**
 * @file python_module.c
 * @author David González León
 * @brief
 * @version 0.1
 * @date 14-11-2021
 *
 * @copyright Copyright (c) 2021
 *
 */
#include "event_codes.h"
#include "uinput_api.h"
#include <stdio.h>
#include <string.h>
// Externe
// enum for mouvements
enum movements { M_SCROLL_RIGHT, M_SCROLL_LEFT, M_SCROLL_UP, M_SCROLL_DOWN, M_ZOOM_IN, M_ZOOM_OUT };
// init
int init_uinput_device();
// passer le mouvement
int send_movement(int fd, enum movements mouvement_id);
// fermer
int close_uinput_device(int fd);

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

int init_uinput_device() {
   int fd = uinput_open();
   if (fd == -1) {
      printf("init_uinput_device : Error while opening fd\n");
      return -1;
   }

   for (int i = 0; i < NUMBER_OF_EVENTS_HANDLED; ++i) {
      int res = uinput_enable_event(fd, EVENT_TAB[i]);
      if (res == -1) {
         printf("init_uinput_device : Error while opening event %d, at position %d\n", EVENT_TAB[i], i);
         return -1;
      }
   }

   struct uinput_setup usetup;
   memset(&usetup, 0, sizeof(usetup));
   usetup.id.bustype = BUS_VIRTUAL;
   usetup.id.vendor = 0x1234;  /* sample vendor */
   usetup.id.product = 0x5678; /* sample product */
   usetup.id.version = 0x1;
   strcpy(usetup.name, "real_time_device");

   int res = uinput_create_device(fd, &usetup);
   if (res == -1) {
      printf("init_uinput_device : Error while creating device\n");
      return -1;
   }
   return fd;
}

int send_movement(int fd, enum movements mouvement_id) {
   if (uinput_emit_event(fd, EV_KEY, get_movement_value(mouvement_id), KEY_PRESSED) == -1) {
      return -1;
   }
   if (uinput_emit_syn(fd) == -1) {
      return -1;
   }
   if (uinput_emit_event(fd, EV_KEY, get_movement_value(mouvement_id), KEY_RELEASED) == -1) {
      return -1;
   }
   if (uinput_emit_syn(fd) == -1) {
      return -1;
   }
   return 0;
}

int close_uinput_device(int fd) {
   if (uinput_close(fd) == -1) {
      printf("close_uinput_device : Error while closing fd\n");
      return -1;
   }
   return 0;
}
