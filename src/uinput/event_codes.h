/**
 * @file event_codes.h
 * @author David González León, Jade Gröli
 * @brief defines all event codes used by the app
 * @version 0.1
 * @date 02-11-2021
 *
 * @copyright Copyright (c) 2021
 *
 */

// Scroll up/down/left/right == flèches directionnelles

// zoom in/out == +/-

#include <linux/input-event-codes.h>

#define KEY_PRESSED 1
#define KEY_RELEASED 0

#define ZOOM_IN KEY_KPPLUS
#define ZOOM_OUT KEY_KPMINUS
#define SCROLL_UP KEY_UP
#define SCROLL_DOWN KEY_DOWN
#define SCROLL_RIGHT KEY_RIGHT
#define SCROLL_LEFT KEY_LEFT