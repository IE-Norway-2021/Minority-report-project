/**
 * @file main.c
 * @author David González León, Jade Gröli
 * @brief Test l'api uinput.
 * @version 0.1
 * @date 02-11-2021
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "uinput_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Nombre d'erreurs rencontrées pendant l'exécution du programme. S'il est >0, main retournera 1
 *
 */
int error_nb = 0;
// Pour tester
const int EVENT_TAB[] = {SCROLL_UP, SCROLL_DOWN, SCROLL_RIGHT, SCROLL_LEFT, ZOOM_IN, ZOOM_OUT};

void test_uinput_open_close();
void test_uinput_enable_event();
void test_uinput_create_device();
void test_uinput_emit_event();
void test_uinput_emit_event_combo();
void test_uinput_emit_syn();

int main(int argc, char const *argv[]) {
   // Test everything
   test_uinput_open_close();
   test_uinput_enable_event();
   test_uinput_create_device();
   test_uinput_emit_event_syn();
   test_uinput_emit_event_combo();

   if (error_nb > 0) {
      printf("There were %d error!\n", error_nb);
      return 1;
   }
   printf("Testing successful, no error detected\n");

   return 0;
}

/**
 * @brief
 * Error detected :
 * - user did not have access to uinput, so open did not work. Fix : give user access to uinput. Check if it is definite, otherwise find another
 *    way.
 *    Error still here after compiling, so creating a group and adding user to it, and adding /dev/uinput to that group with a rule. Solution worked
 */
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

void test_uinput_emit_event_syn() {}
void test_uinput_emit_event_combo() {}
