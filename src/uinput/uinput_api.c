/**
 * @file uinput_api.c
 * @author David González León, Jade Gröli
 * @brief
 * @version 0.1
 * @date 02-11-2021
 *
 * @copyright Copyright (c) 2021
 *
 */

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "uinput_api.h"

int uinput_open() {
   int uinput_fd;

   uinput_fd = open(UINPUT_DEFAULT_PATH, O_WRONLY | O_NONBLOCK);
   return uinput_fd;
}

int uinput_close(int fd) {
   if (ioctl(fd, UI_DEV_DESTROY) == -1) {
      close(fd);
      return -1;
   }

   return close(fd);
}

int uinput_enable_event(int uinput_fd, uint16_t event_code) {

   if (ioctl(uinput_fd, UI_SET_EVBIT, EV_KEY) == -1) {
      return -1;
   }

   return ioctl(uinput_fd, UI_SET_KEYBIT, event_code);
}

int uinput_create_device(int uinput_fd, struct uinput_setup *usetup) {
   if (ioctl(uinput_fd, UI_DEV_SETUP, &usetup) == -1) {
      return -1;
   }

   if (ioctl(uinput_fd, UI_DEV_CREATE) == -1) {
      return -1;
   }

   return 0;
}

int uinput_emit_event(int uinput_fd, uint16_t event_type, uint16_t event_code, int32_t eventvalue) {
   struct input_event event;

   memset(&event, 0, sizeof(event));
   gettimeofday(&event.time, 0);
   event.type = event_type;
   event.code = event_code;
   event.value = eventvalue;

   ssize_t bytes;
   bytes = write(uinput_fd, &event, sizeof(struct input_event));
   if (bytes != sizeof(struct input_event)) {
      return -1;
   }
   return 0;
}
int uinput_emit_event_combo(int uinput_fd, const uint16_t *key_codes, size_t length) {
   int retval = 0;
   size_t i;

   for (i = 0; i < length; ++i) {
      if (uinput_emit_event(uinput_fd, EV_KEY, key_codes[i], KEY_PRESSED) == -1) {
         retval = -1;
         break; /* The combination or the device is
                   somehow broken: there's no sense to
                   press any of the rest of the
                   keys. It's like pressing physical keys
                   one by one and then discovering that
                   one of the keys required for this
                   combination is missing or broken. */
      }
   }

   /* Try to release every pressed key, no matter what. */
   while (i--) {
      if (uinput_emit_event(uinput_fd, EV_KEY, key_codes[i], KEY_RELEASED) == -1) {
         retval = -1;
      }
   }

   return retval;
}
int uinput_emit_syn(int uinput_fd) { return uinput_emit_event(uinput_fd, EV_SYN, SYN_REPORT, KEY_RELEASED); }