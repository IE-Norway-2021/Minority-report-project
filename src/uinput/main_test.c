/**
 * @file main.c
 * @author David González León, Jade Gröli
 * @brief Tests of the uinput API.
 * @version 0.1
 * @date 02-11-2021
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "uinput_api.h"
#include <fcntl.h>
#include <libevdev/libevdev.h>
#include <linux/input.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/**
 * @brief Number of errors encoutered during the execution of the tests. If >0, main will return 1
 *
 */
int error_nb = 0;

const int EVENT_TAB[] = {SCROLL_UP, SCROLL_DOWN, SCROLL_RIGHT, SCROLL_LEFT, ZOOM_IN, ZOOM_OUT};

/**
 * Prototype of test functions. Each function tests a specific feature.
 */

/**
 * @brief Tests opening and closing an fd.
 * Error detected :
 * - user did not have access to uinput, so open did not work. Fix : give user access to uinput. Check if it is definite, otherwise find another
 *    way.
 *    Error still here after compiling, so creating a group and adding user to it, and adding /dev/uinput to that group with a rule. Solution worked
 */
void test_uinput_open_close();

/**
 * @brief Tests that all events are correctly enabled (using the EVENT_TAB array to loop through each input)
 *
 */
void test_uinput_enable_event();

/**
 * @brief Tests that a device is correctly created with all needed events enabled
 *
 */
void test_uinput_create_device();

/**
 * @brief Tests that each event in EVENT_TAB is correctly emitted. Uses the libevdev library to read inputs from the kernel
 *
 */
void test_uinput_emit_event_syn();

/**
 * @brief Tests that we can emit more than one event at a time. Not used so not tested
 *
 */
void test_uinput_emit_event_combo();

int main(int argc, char const *argv[]) {
   // Test everything
   test_uinput_open_close();
   test_uinput_enable_event();
   test_uinput_create_device();
   test_uinput_emit_event_syn();
   test_uinput_emit_event_combo();

   if (error_nb > 0) { // If there was an error inform
      printf("There were %d error!\n", error_nb);
      return 1;
   }
   printf("Testing successful, no error detected\n");

   return 0;
}

// Definition of test functions

void test_uinput_open_close() {
   int fd = uinput_open();
   if (fd == -1) {
      printf("test_uinput_open_close : Error while opening fd\n");
      ++error_nb;
      return;
   }

   int res = uinput_close(fd);
   if (res == -1) {
      printf("test_uinput_open_close : Error while closing fd\n");
      ++error_nb;
   }
}

void test_uinput_enable_event() {
   int fd = uinput_open();
   if (fd == -1) {
      printf("test_uinput_enable_event : Error while opening fd\n");
      ++error_nb;
      return;
   }

   for (int i = 0; i < NUMBER_OF_EVENTS_HANDLED; ++i) {
      int res = uinput_enable_event(fd, EVENT_TAB[i]);
      if (res == -1) {
         printf("test_uinput_enable_event : Error while opening event %d, at position %d\n", EVENT_TAB[i], i);
         ++error_nb;
         return;
      }
   }

   int res = uinput_close(fd);
   if (res == -1) {
      printf("test_uinput_enable_event : Error while closing fd\n");
      ++error_nb;
   }
}

void test_uinput_create_device() {
   int fd = uinput_open();
   if (fd == -1) {
      printf("test_uinput_create_device : Error while opening fd\n");
      ++error_nb;
      return;
   }

   for (int i = 0; i < NUMBER_OF_EVENTS_HANDLED; ++i) {
      int res = uinput_enable_event(fd, EVENT_TAB[i]);
      if (res == -1) {
         printf("test_uinput_create_device : Error while opening event %d, at position %d\n", EVENT_TAB[i], i);
         ++error_nb;
         return;
      }
   }

   struct uinput_setup usetup;
   memset(&usetup, 0, sizeof(usetup));
   usetup.id.bustype = BUS_VIRTUAL;
   usetup.id.vendor = 0x1234;  /* sample vendor */
   usetup.id.product = 0x5678; /* sample product */
   usetup.id.version = 0x1;
   strcpy(usetup.name, "Example device");

   int res = uinput_create_device(fd, &usetup);
   if (res == -1) {
      printf("test_uinput_create_device : Error while creating device\n");
      ++error_nb;
      return;
   }

   res = uinput_close(fd);
   if (res == -1) {
      printf("test_uinput_enable_event : Error while closing fd\n");
      ++error_nb;
   }
}

void test_uinput_emit_event_syn() {
   int fd = uinput_open();
   if (fd == -1) {
      printf("test_uinput_emit_event_syn : Error while opening fd\n");
      ++error_nb;
      return;
   }

   for (int i = 0; i < NUMBER_OF_EVENTS_HANDLED; ++i) {
      int res = uinput_enable_event(fd, EVENT_TAB[i]);
      if (res == -1) {
         printf("test_uinput_emit_event_syn : Error while opening event %d, at position %d\n", EVENT_TAB[i], i);
         ++error_nb;
         return;
      }
   }

   int res = ioctl(fd, UI_SET_EVBIT, EV_SYN);
   if (res == -1) {
      printf("test_uinput_emit_event_syn : Error while opening ev_syn\n");
      ++error_nb;
      return;
   }

   struct uinput_setup usetup;
   memset(&usetup, 0, sizeof(usetup));
   usetup.id.bustype = BUS_USB;
   usetup.id.vendor = 0x1234;  /* sample vendor */
   usetup.id.product = 0x5678; /* sample product */
   usetup.id.version = 0x1;
   strcpy(usetup.name, "Example device");

   res = uinput_create_device(fd, &usetup);
   if (res == -1) {
      printf("test_uinput_emit_event_syn : Error while creating device\n");
      ++error_nb;
      return;
   }

   sleep(1); // To give the userspace time to create the device

   // Open input event and create wrapper to read events
   struct libevdev *dev = NULL;
   int input_fd = open("/dev/input/event0", O_RDONLY | O_NONBLOCK);
   if (input_fd == -1) {
      printf("test_uinput_emit_event_syn: error while opening the input reader");
      error_nb++;
      return;
   }
   res = libevdev_new_from_fd(input_fd, &dev);
   if (res < 0) {
      printf("test_uinput_emit_event_syn: error while opening the input reader");
      error_nb++;
      return;
   }

   // fcntl(fd, F_SETFL, FNDELAY); // Rend la lecture blocante

   struct input_event read_event;
   for (int i = 0; i < NUMBER_OF_EVENTS_HANDLED; ++i) {

      // Press key and syn
      uinput_emit_event(fd, EV_KEY, EVENT_TAB[i], KEY_PRESSED);
      uinput_emit_syn(fd);
      res = libevdev_next_event(dev, LIBEVDEV_READ_FLAG_NORMAL, &read_event);
      if (read_event.code != EVENT_TAB[i] || read_event.type != EV_KEY || read_event.value != KEY_PRESSED) {
         printf("test_uinput_emit_event_syn : Error while reading the input. event code : %d vs %d, event type : %d vs %d, value : %d vs %d\n", read_event.code,
                EVENT_TAB[i], read_event.type, EV_KEY, read_event.value, KEY_PRESSED);
         close(input_fd);
         ++error_nb;
         return;
      }
      res = libevdev_next_event(dev, LIBEVDEV_READ_FLAG_NORMAL, &read_event);
      if (read_event.code != SYN_REPORT || read_event.type != EV_SYN || read_event.value != KEY_RELEASED) {
         printf("test_uinput_emit_event_syn : Error while reading the input. event code : %d vs %d, event type : %d vs %d, value : %d vs %d\n", read_event.code,
                EVENT_TAB[i], read_event.type, EV_KEY, read_event.value, KEY_PRESSED);
         close(input_fd);
         ++error_nb;
         return;
      }

      // Release key and syn
      uinput_emit_event(fd, EV_KEY, EVENT_TAB[i], KEY_RELEASED);
      uinput_emit_syn(fd);
      res = libevdev_next_event(dev, LIBEVDEV_READ_FLAG_NORMAL, &read_event);
      if (read_event.code != EVENT_TAB[i] || read_event.type != EV_KEY || read_event.value != KEY_RELEASED) {
         printf("test_uinput_emit_event_syn : Error while reading the input. event code : %d vs %d, event type : %d vs %d, value : %d vs %d\n", read_event.code,
                EVENT_TAB[i], read_event.type, EV_KEY, read_event.value, KEY_PRESSED);
         close(input_fd);
         ++error_nb;
         return;
      }
      res = libevdev_next_event(dev, LIBEVDEV_READ_FLAG_NORMAL, &read_event);
      if (read_event.code != SYN_REPORT || read_event.type != EV_SYN || read_event.value != KEY_RELEASED) {
         printf("test_uinput_emit_event_syn : Error while reading the input. event code : %d vs %d, event type : %d vs %d, value : %d vs %d\n", read_event.code,
                EVENT_TAB[i], read_event.type, EV_KEY, read_event.value, KEY_PRESSED);
         close(input_fd);
         ++error_nb;
         return;
      }
   }

   // Free memory and close input
   free(dev);
   close(input_fd);

   res = uinput_close(fd);
   if (res == -1) {
      printf("test_uinput_enable_event : Error while closing fd\n");
      ++error_nb;
   }
}

void test_uinput_emit_event_combo() {
   // Function not used currently, so not testing
}
