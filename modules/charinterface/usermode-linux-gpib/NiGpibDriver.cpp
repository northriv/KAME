/*
 * NiGpibDriver.cpp — implementation of NiGpibDriver.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 */

#include "NiGpibDriver.h"

#include <cstdio>
#include <cstring>
#include <cerrno>
#include <stdexcept>
#include <unistd.h>     /* usleep */

extern "C" {
extern int  ni_module_init(void);
extern void ni_module_exit(void);
extern struct usb_driver     *g_ni_usb_driver;
extern struct gpib_interface *g_ni_gpib_interface;
}

/* ------------------------------------------------------------------ */

static const uint16_t NI_VENDOR = 0x3923;
static const struct { uint16_t pid; const char *name; } NI_PIDS[] = {
    { 0x702a, "NI USB-B"   },
    { 0x709b, "NI USB-HS"  },
    { 0x7618, "NI USB-HS+" },
    { 0x725c, "KUSB-488A"  },
    { 0x725d, "MC USB-488" },
    { 0,      nullptr      },
};

/* ------------------------------------------------------------------ */

NiGpibDriver::NiGpibDriver(int board_pad, unsigned int timeout_usec)
    : board_pad_(board_pad), timeout_usec_(timeout_usec) {}

NiGpibDriver::~NiGpibDriver() { close(); }

bool NiGpibDriver::open()
{
    if (libusb_init(&ctx_) != 0) {
        fprintf(stderr, "libusb_init failed\n");
        return false;
    }
    libusb_set_option(ctx_, LIBUSB_OPTION_LOG_LEVEL, LIBUSB_LOG_LEVEL_NONE);

    if (!findAndOpenDevice()) {
        fprintf(stderr, "No supported NI USB-GPIB adapter found.\n");
        return false;
    }

    if (ni_module_init() != 0) {
        fprintf(stderr, "ni_module_init failed\n");
        return false;
    }
    module_inited_ = true;

    if (!g_ni_usb_driver || !g_ni_gpib_interface) {
        fprintf(stderr, "Driver did not register properly\n");
        return false;
    }

    struct usb_device_id fake_id = {
        .idVendor         = usb_dev_.descriptor.idVendor,
        .idProduct        = usb_dev_.descriptor.idProduct,
        .bInterfaceNumber = -1,
    };
    if (g_ni_usb_driver->probe(&usb_intf_, &fake_id) != 0) {
        fprintf(stderr, "USB probe failed\n");
        return false;
    }
    probed_ = true;

    memset(&board_, 0, sizeof(board_));
    board_.interface    = g_ni_gpib_interface;
    board_.pad          = board_pad_;
    board_.sad          = -1;
    board_.master       = 1;
    board_.usec_timeout = timeout_usec_;
    board_.minor        = 0;
    init_waitqueue_head(&board_.wait);
    spin_lock_init(&board_.spinlock);
    spin_lock_init(&board_.locking_pid_spinlock);
    mutex_init(&board_.user_mutex);
    mutex_init(&board_.big_gpib_mutex);
    INIT_LIST_HEAD(&board_.device_list);
    init_event_queue(&board_.event_queue);
    init_gpib_pseudo_irq(&board_.pseudo_irq);

    struct gpib_board_config cfg = {};
    if (g_ni_gpib_interface->attach(&board_, &cfg) != 0) {
        fprintf(stderr, "Driver attach failed\n");
        return false;
    }
    attached_ = true;
    printf("Driver attached successfully.\n");

    /* Assert REN so instruments enter remote mode and will accept commands. */
    g_ni_gpib_interface->remote_enable(&board_, 1);

    /* Assert IFC for ≥100 µs to reset the bus and become CIC.
     * gpib_common.ko does this by default (assert_ifc=1 in gpib_config).
     * Without it the board stays in controller-idle state and the first
     * command() / write() will time out. */
    g_ni_gpib_interface->interface_clear(&board_, 1);
    usleep(150);
    g_ni_gpib_interface->interface_clear(&board_, 0);

    return true;
}

bool NiGpibDriver::openForTest(struct gpib_interface *iface)
{
    /* Mirror the board_ setup from open(), but skip USB and driver init. */
    memset(&board_, 0, sizeof(board_));
    board_.interface    = iface;
    board_.pad          = board_pad_;
    board_.sad          = -1;
    board_.master       = 1;
    board_.usec_timeout = timeout_usec_;
    board_.minor        = 0;
    init_waitqueue_head(&board_.wait);
    spin_lock_init(&board_.spinlock);
    spin_lock_init(&board_.locking_pid_spinlock);
    mutex_init(&board_.user_mutex);
    mutex_init(&board_.big_gpib_mutex);
    INIT_LIST_HEAD(&board_.device_list);
    init_event_queue(&board_.event_queue);
    init_gpib_pseudo_irq(&board_.pseudo_irq);

    /* Point the global used by close() at the mock interface. */
    g_ni_gpib_interface = iface;
    attached_ = true;
    return true;
}

void NiGpibDriver::close()
{
    if (attached_) {
        g_ni_gpib_interface->detach(&board_);
        attached_ = false;
    }
    if (probed_ && g_ni_usb_driver && g_ni_usb_driver->disconnect) {
        g_ni_usb_driver->disconnect(&usb_intf_);
        probed_ = false;
    }
    if (module_inited_) {
        ni_module_exit();
        module_inited_ = false;
    }
    if (usb_dev_.handle) {
        libusb_release_interface(usb_dev_.handle, 0);
        libusb_close(usb_dev_.handle);
        usb_dev_.handle = nullptr;
    }
    if (ctx_) {
        libusb_exit(ctx_);
        ctx_ = nullptr;
    }
}

bool NiGpibDriver::findAndOpenDevice()
{
    libusb_device **list = nullptr;
    ssize_t n = libusb_get_device_list(ctx_, &list);
    if (n < 0) {
        fprintf(stderr, "libusb_get_device_list failed: %s\n",
                libusb_error_name((int)n));
        return false;
    }

    bool found = false;
    for (ssize_t i = 0; i < n && !found; i++) {
        struct libusb_device_descriptor desc;
        if (libusb_get_device_descriptor(list[i], &desc) != 0) continue;
        if (desc.idVendor != NI_VENDOR) continue;

        for (int j = 0; NI_PIDS[j].pid; j++) {
            if (desc.idProduct != NI_PIDS[j].pid) continue;

            libusb_device_handle *h = nullptr;
            int r = libusb_open(list[i], &h);
            if (r != 0) {
                fprintf(stderr, "libusb_open failed (%s): %s\n",
                        NI_PIDS[j].name, libusb_error_name(r));
                break;
            }
            libusb_set_auto_detach_kernel_driver(h, 1);
            int rc = libusb_claim_interface(h, 0);
            if (rc != 0) {
                fprintf(stderr, "libusb_claim_interface(0) failed: %s\n",
                        libusb_error_name(rc));
                if (rc == LIBUSB_ERROR_ACCESS)
                    fprintf(stderr,
                        "  Hint: another driver (e.g. NI-488.2) owns the interface.\n"
                        "  Try running with sudo, or unload the NI kernel extension first.\n");
            }

            printf("Opened %s (VID=%04x PID=%04x bus=%d dev=%d)\n",
                   NI_PIDS[j].name, desc.idVendor, desc.idProduct,
                   libusb_get_bus_number(list[i]),
                   libusb_get_device_address(list[i]));

            memset(&usb_dev_, 0, sizeof(usb_dev_));
            usb_dev_.handle               = h;
            usb_dev_.descriptor.idVendor  = desc.idVendor;
            usb_dev_.descriptor.idProduct = desc.idProduct;
            usb_dev_.devnum               = (int)libusb_get_device_address(list[i]);
            usb_dev_._bus_storage.busnum  = (int)libusb_get_bus_number(list[i]);
            usb_dev_.bus                  = &usb_dev_._bus_storage;
            snprintf(usb_dev_.dev.name, sizeof(usb_dev_.dev.name),
                     "usb%d-%d", usb_dev_.bus->busnum, usb_dev_.devnum);

            memset(&usb_intf_, 0, sizeof(usb_intf_));
            usb_intf_.udev = &usb_dev_;
            usb_set_intfdata(&usb_intf_, nullptr);

            found = true;
            break;
        }
    }
    libusb_free_device_list(list, 1);
    return found;
}

void NiGpibDriver::cmd(std::initializer_list<int> bytes)
{
    uint8_t buf[16];
    size_t  len = 0;
    for (int b : bytes) buf[len++] = static_cast<uint8_t>(b);
    size_t bw = 0;
    g_ni_gpib_interface->command(&board_, buf, len, &bw);
}

void NiGpibDriver::interfaceClear()
{
    g_ni_gpib_interface->interface_clear(&board_, 1);
    usleep(150);    /* IEEE 488 requires ≥ 100 µs */
    g_ni_gpib_interface->interface_clear(&board_, 0);
}

void NiGpibDriver::deviceClear(int addr)
{
    if (addr < 0) {
        cmd({ DCL });
    } else {
        cmd({ UNL, MLA((uint32_t)addr), SDC });
        cmd({ UNL });
    }
}

void NiGpibDriver::enableRemote(bool enable)
{
    g_ni_gpib_interface->remote_enable(&board_, enable ? 1 : 0);
}

void NiGpibDriver::send(int addr, const std::string &command, const char *term)
{
    /* Real linux-gpib create_send_setup(): MTA(board) → UNL → MLA(device) */
    cmd({ MTA((uint32_t)board_pad_), UNL, MLA((uint32_t)addr) });

    std::string payload = command;
    bool use_eoi = (term == nullptr || term[0] == '\0');
    if (!use_eoi)
        payload += term;

    size_t bw = 0;
    int r = g_ni_gpib_interface->write(
                &board_,
                reinterpret_cast<uint8_t *>(payload.data()),
                payload.size(), use_eoi ? 1 : 0, &bw);
    if (r != 0)
        throw std::runtime_error("GPIB write error: " + std::string(strerror(-r)));
    cmd({ UNL, UNT });
    printf("Sent %s (%zu bytes written)\n", command.c_str(), bw);
}

std::vector<uint8_t> NiGpibDriver::readRaw(int addr, size_t max_len)
{
    /* Real linux-gpib InternalReceiveSetup(): UNL → MLA(board) → MTA(device) */
    cmd({ UNL, MLA((uint32_t)board_pad_), MTA((uint32_t)addr) });

    std::vector<uint8_t> buf(max_len);
    size_t bytes_read = 0;
    int    end = 0;
    int r = g_ni_gpib_interface->read(&board_, buf.data(), max_len,
                                       &end, &bytes_read);
    if (r != 0)
        throw std::runtime_error("GPIB read error: " + std::string(strerror(-r)));
    cmd({ UNL, UNT });
    buf.resize(bytes_read);
    return buf;
}

static void stripCrLf(std::vector<uint8_t> &buf)
{
    while (!buf.empty() && (buf.back() == '\r' || buf.back() == '\n'))
        buf.pop_back();
}

std::string NiGpibDriver::read(int addr)
{
    auto buf = readRaw(addr, 512);
    stripCrLf(buf);
    return std::string(reinterpret_cast<char *>(buf.data()), buf.size());
}

std::string NiGpibDriver::read(int addr, const char *term)
{
    /* Use the last byte of term as the hardware EOS character.
     * For "\r\n" this is '\n'; for "\r" it is '\r'; etc. */
    size_t len = term ? strlen(term) : 0;
    uint8_t eos = len ? static_cast<uint8_t>(term[len - 1]) : '\n';

    g_ni_gpib_interface->enable_eos(&board_, eos, 0);
    auto buf = readRaw(addr, 512);
    g_ni_gpib_interface->disable_eos(&board_);

    stripCrLf(buf);
    return std::string(reinterpret_cast<char *>(buf.data()), buf.size());
}

std::string NiGpibDriver::read(int addr, size_t length)
{
    auto buf = readRaw(addr, length);
    return std::string(reinterpret_cast<char *>(buf.data()), buf.size());
}

std::string NiGpibDriver::readEOS(int addr, uint8_t eosChar, size_t max_len)
{
    g_ni_gpib_interface->enable_eos(&board_, eosChar, 0);
    auto buf = readRaw(addr, max_len);
    g_ni_gpib_interface->disable_eos(&board_);
    stripCrLf(buf);
    return std::string(reinterpret_cast<char *>(buf.data()), buf.size());
}

std::string NiGpibDriver::query(int addr, const std::string &command,
                                 const char *term)
{
    send(addr, command, term);
    if (term && term[0])
        return read(addr, term);
    return read(addr);
}

bool NiGpibDriver::checkSRQ()
{
    if (!g_ni_gpib_interface || !g_ni_gpib_interface->line_status)
        return false;
    int status = g_ni_gpib_interface->line_status(&board_);
    /* -EBUSY means another USB transfer is in progress — treat as no SRQ */
    if (status < 0)
        return false;
    return (status & BUS_SRQ) != 0;
}

uint8_t NiGpibDriver::serialPoll(int addr)
{
    /* Real linux-gpib sequence (gpib_os.c: setup/read/cleanup_serial_poll):
     *   UNL + MLA(board) + SPE  →  MTA(device) → read STB  →  SPD + UNL + UNT */
    cmd({ UNL, MLA((uint32_t)board_pad_), SPE });
    cmd({ MTA((uint32_t)addr) });

    uint8_t stb = 0;
    size_t  br  = 0;
    int     end = 0;
    int r = g_ni_gpib_interface->read(&board_, &stb, 1, &end, &br);

    cmd({ SPD, UNL, UNT });

    if (r != 0)
        throw std::runtime_error("Serial poll error: " + std::string(strerror(-r)));
    return stb;
}
