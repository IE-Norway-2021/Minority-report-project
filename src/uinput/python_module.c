/**
 * @file python_module.c
 * @author David González León
 * @brief Defines all functions and elements available in the python module
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
// External function/elements

/**
 * @brief Defines all movements available
 *
 */
enum movements { M_SCROLL_RIGHT, M_SCROLL_LEFT, M_SCROLL_UP, M_SCROLL_DOWN, M_ZOOM_IN, M_ZOOM_OUT };

/**
 * @brief Initiates a new uinput device. Enables all predefined events
 *
 * @return an int corresponding to the fd of the created device, or -1 if there was an error
 */
int init_uinput_device();

/**
 * @brief Sends a movement as an input
 *
 * @param fd the id of the device
 * @param mouvement_id The id of the mouvement to send
 * @return 0 if everithing went well, -1 if there was an error
 */
int send_movement(int fd, enum movements mouvement_id);

/**
 * @brief Closes the given device.
 *
 * @param fd The id of the device to close
 * @return 0 if the device was correctly closed, -1 if there was an error while closing
 */
int close_uinput_device(int fd);

// Internal functions (not available in the python module)
int get_movement_value(enum movements mouvement_id);
// TODO essayer d'extraire ce tableau dans event_codes.c en le marquant extern dans event_codes.h
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
