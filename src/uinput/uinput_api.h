/**
 * @file uinput_api.h
 * @author David González León, Jade Gröli
 * @brief
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

// crée un device virtuel, et le renvoie. Retourne -1 si erreur
int uinput_open();
// Ferme le device virtuel. retourne -1 si erreur
int uinput_close(int fd);

// Enable un event spécifique (uniquement un event du keyboard). Retourne -1 si erreur
int uinput_enable_event(int uinput_fd, uint16_t event_code);
// Crée un device avec le pointeur passé en param
int uinput_create_device(int uinput_fd, const struct uinput_user_dev *user_dev_p);
// Emet un event, renvoie -1 si erreur
int uinput_emit_event(int uinput_fd, uint16_t event_type, uint16_t event_code, int32_t eventvalue);
// Emet une succéssion d'events. Renvoie -1 si erreur
int uinput_emit_event_combo(int uinput_fd, const uint16_t *key_codes, size_t length);
// Emet l'event syn. Renvoie -1 si erreur
int uinput_emit_syn(int uinput_fd);

#endif