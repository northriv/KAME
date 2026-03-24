/*
 * gpib_stubs.c
 *
 * Userspace stubs for gpib_common kernel module functions referenced
 * by ni_usb_gpib.c (normally provided by gpib_common.ko on Linux).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "osx_compat.h"
#include "gpibP.h"

/* Global driver registry (replaces gpib_common.ko globals) */
struct list_head registered_drivers = { &registered_drivers, &registered_drivers };
struct gpib_board board_array[GPIB_MAX_NUM_BOARDS];

/* Saved pointers filled during module init */
struct usb_driver     *g_ni_usb_driver    = NULL;
struct gpib_interface *g_ni_gpib_interface = NULL;

int gpib_register_driver(struct gpib_interface *interface, struct module *mod)
{
    (void)mod;
    g_ni_gpib_interface = interface;
    printf("gpib: registering driver '%s'\n", interface ? interface->name : "(null)");
    return 0;
}

void gpib_unregister_driver(struct gpib_interface *interface)
{
    printf("gpib: unregistering driver '%s'\n", interface ? interface->name : "(null)");
}

int push_gpib_event(struct gpib_board *board, short event_type)
{
    (void)board;
    (void)event_type;
    return 0;
}

int pop_gpib_event(struct gpib_board *board, struct gpib_event_queue *queue,
                   short *event_type)
{
    (void)board; (void)queue; (void)event_type;
    return -1;
}

unsigned int num_gpib_events(const struct gpib_event_queue *queue)
{
    (void)queue;
    return 0;
}

int gpib_request_pseudo_irq(struct gpib_board *board,
                             irqreturn_t (*handler)(int, void * PT_REGS_ARG))
{
    (void)board; (void)handler;
    return 0;
}

void gpib_free_pseudo_irq(struct gpib_board *board)
{
    (void)board;
}

/*
 * gpib_match_device_path: return nonzero if dev matches the sysfs path string.
 * A NULL path matches everything.
 */
int gpib_match_device_path(struct device *dev, const char *device_path)
{
    (void)dev;
    return (device_path == NULL) ? 1 : 0;
}

struct pci_dev *gpib_pci_get_device(const struct gpib_board_config *config,
                                     unsigned int vendor_id, unsigned int device_id,
                                     struct pci_dev *from)
{
    (void)config; (void)vendor_id; (void)device_id; (void)from;
    return NULL;
}

struct pci_dev *gpib_pci_get_subsys(const struct gpib_board_config *config,
                                     unsigned int vendor_id, unsigned int device_id,
                                     unsigned int ss_vendor, unsigned int ss_device,
                                     struct pci_dev *from)
{
    (void)config; (void)vendor_id; (void)device_id;
    (void)ss_vendor; (void)ss_device; (void)from;
    return NULL;
}

void init_gpib_status_queue(struct gpib_status_queue *device)
{
    if (!device) return;
    INIT_LIST_HEAD(&device->list);
    INIT_LIST_HEAD(&device->status_bytes);
    device->pad = 0;
    device->sad = -1;
    device->num_status_bytes = 0;
    device->reference_count = 0;
    device->dropped_byte = 0;
}
