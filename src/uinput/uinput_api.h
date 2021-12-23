/**
 * @file uinput_api.h
 * @author David González León, Jade Gröli
 * @brief Defines all function allowing access to the uinput api of the linux kernel
 * @version 0.1
 * @date 02-11-2021
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef UINPUT_API_H_
#define UINPUT_API_H_

#include "event_codes.h"
#include <linux/uinput.h>
#include <stdint.h>

#define UINPUT_DEFAULT_PATH "/dev/uinput"

/**
 * @brief Creates an input device
 *
 * @return the id of the input device (with a value >= 0), or -1 of there was an error
 */
int uinput_open();

/**
 * @brief Closes the given input device
 *
 * @param fd the fd of the given input device
 * @return 0 if the device was correctly closed, or -1 if there was an error
 */
int uinput_close(int fd);

/**
 * @brief Enables a specific event. The event needs to be a EV_KEY event.
 *
 * @param uinput_fd the fd of the input device
 * @param event_code the code of the event to enable
 * @return 0 if the event was enabled, -1 if there was an error
 */
int uinput_enable_event(int uinput_fd, uint16_t event_code);

/**
 * @brief Creates a virtual input device
 *
 * @param uinput_fd the fd of the input device
 * @param usetup the basic information of the device
 * @return 0 if the device was correctly created, -1 if there was an error
 */
int uinput_create_device(int uinput_fd, struct uinput_setup *usetup);

/**
 * @brief Emits an event using the given device
 *
 * @param uinput_fd the fd of the given device
 * @param event_type the type of the event
 * @param event_code the code of the event
 * @param eventvalue the value of the event
 * @return 0 if the event was correctly emited, -1 if there was an error
 */
int uinput_emit_event(int uinput_fd, uint16_t event_type, uint16_t event_code, int32_t eventvalue);

/**
 * @brief Emits a series of events
 *
 * @param uinput_fd the fd of the device
 * @param key_codes the codes of the events to send
 * @param length the length of the the key codes array
 * @return 0 if all events were correctly emited, -1 if there was an error
 */
int uinput_emit_event_combo(int uinput_fd, const uint16_t *key_codes, size_t length);

/**
 * @brief Emits the EV_SYN event
 *
 * @param uinput_fd the fd of the device
 * @return 0 if the event was correctly sent, -1 if there was an error
 */
int uinput_emit_syn(int uinput_fd);

#endif