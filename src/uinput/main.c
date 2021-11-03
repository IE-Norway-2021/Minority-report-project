/**
 * @file main.c
 * @author David González León, Jade Gröli
 * @brief Test l'api uinput
 * @version 0.1
 * @date 02-11-2021
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "uinput_api.h"
#include <stdio.h>

int main(int argc, char const *argv[]) {
   printf("test\n");
   int fd = uinput_open();
   uinput_close(fd);
   return 0;
}
