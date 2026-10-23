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

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <mutex>
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

/* Module init/exit refcount.  ni_usb_init_module() clears the static
 * ni_usb_driver_interfaces[] array, so calling it again after a previous
 * controller has been probed would wipe its registration.  Refcount it
 * so init/exit happen exactly once across all NiGpibDriver instances. */
static std::mutex g_module_mutex;
static int        g_module_refcount = 0;

NiGpibDriver::NiGpibDriver(int board_pad, unsigned int timeout_usec,
                            int adapter_index)
    : board_pad_(board_pad),
      timeout_usec_(timeout_usec),
      adapter_index_(adapter_index) {}

NiGpibDriver::~NiGpibDriver() { close(); }

bool NiGpibDriver::fail(const char *fmt, ...)
{
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    last_error_ = buf;
    fprintf(stderr, "%s\n", buf);
    return false;
}

bool NiGpibDriver::open()
{
    last_error_.clear();
    if (libusb_init(&ctx_) != 0)
        return fail("libusb_init failed");
    libusb_set_option(ctx_, LIBUSB_OPTION_LOG_LEVEL, LIBUSB_LOG_LEVEL_NONE);

    if (!findAndOpenDevice(adapter_index_)) {
        /* An adapter that was found but would not open has said why. */
        if (last_error_.empty())
            fail("No supported NI USB-GPIB adapter found at index %d.", adapter_index_);
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(g_module_mutex);
        if (g_module_refcount == 0) {
            if (ni_module_init() != 0)
                return fail("ni_module_init failed");
        }
        g_module_refcount++;
    }
    module_inited_ = true;

    if (!g_ni_usb_driver || !g_ni_gpib_interface)
        return fail("Driver did not register properly");

    struct usb_device_id fake_id = {
        .idVendor         = usb_dev_.descriptor.idVendor,
        .idProduct        = usb_dev_.descriptor.idProduct,
        .bInterfaceNumber = -1,
    };
    int r = g_ni_usb_driver->probe(&usb_intf_, &fake_id);
    if (r != 0)
        return fail("USB probe failed (%s)", strerror(-r));
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
    r = g_ni_gpib_interface->attach(&board_, &cfg);
    if (r != 0) {
        /* Detached, as the kernel's ibonline() does after a failed attach:
         * ni_usb_attach() returns without undoing what it got as far as --
         * ni_priv, and the interrupt URB already submitted -- and close()
         * detaches only what attached_ says was attached. */
        g_ni_gpib_interface->detach(&board_);
        return fail("Driver attach failed (%s)", strerror(-r));
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
        std::lock_guard<std::mutex> lock(g_module_mutex);
        if (--g_module_refcount == 0)
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

bool NiGpibDriver::findAndOpenDevice(int adapter_index)
{
    libusb_device **list = nullptr;
    ssize_t n = libusb_get_device_list(ctx_, &list);
    if (n < 0)
        return fail("libusb_get_device_list failed: %s", libusb_error_name((int)n));

    /* Enumerate NI USB-GPIB adapters in USB-bus order; pick the
     * adapter_index'th one.  Each NiGpibDriver instance must pass a
     * distinct adapter_index to talk to a distinct physical adapter. */
    bool found = false;
    int  match_count = 0;
    for (ssize_t i = 0; i < n && !found; i++) {
        struct libusb_device_descriptor desc;
        if (libusb_get_device_descriptor(list[i], &desc) != 0) continue;
        if (desc.idVendor != NI_VENDOR) continue;

        const char *name = nullptr;
        for (int j = 0; NI_PIDS[j].pid; j++) {
            if (desc.idProduct == NI_PIDS[j].pid) { name = NI_PIDS[j].name; break; }
        }
        if (!name) continue;

        if (match_count++ != adapter_index) continue;

        libusb_device_handle *h = nullptr;
        int r = libusb_open(list[i], &h);
        if (r != 0) {
            fail("libusb_open failed (%s): %s", name, libusb_error_name(r));
            break;
        }
        libusb_set_auto_detach_kernel_driver(h, 1);
        /* KAME: SET_CONFIGURATION with the configuration the adapter already
         * has, which is what upstream's usb_reset_configuration() in
         * ni_usb_attach() sends and what libusb documents as a lightweight
         * reset (halts cleared, data toggles reset).  That call is a no-op
         * in our shims, and has to be: libusb refuses to touch the
         * configuration while an interface is claimed, so it is done here,
         * before the claim.  Best effort -- a backend that refuses it, or a
         * device another process holds open, is covered by
         * restartEndpoints() below.  An unconfigured adapter (0) gets its
         * configuration from the claim, which is as fresh a start. */
        int config = 0;
        if (libusb_get_configuration(h, &config) == 0 && config > 0)
            libusb_set_configuration(h, config);
        int rc = libusb_claim_interface(h, 0);
        if (rc != 0) {
            fail("libusb_claim_interface(0) failed (%s): %s", name, libusb_error_name(rc));
            if (rc == LIBUSB_ERROR_ACCESS)
                fprintf(stderr,
                    "  Hint: another driver (e.g. NI-488.2) owns the interface.\n"
                    "  Try running with sudo, or unload the NI kernel extension first.\n");
            libusb_close(h);
            break;
        }
        restartEndpoints(h, list[i]);

        printf("Opened %s [#%d] (VID=%04x PID=%04x bus=%d dev=%d)\n",
               name, adapter_index, desc.idVendor, desc.idProduct,
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
    }
    libusb_free_device_list(list, 1);
    return found;
}

/* KAME: the adapter as a fresh plug-in leaves it.  In the kernel the driver
 * outlives any process using it, so a client dying never leaves the adapter
 * in the middle of a transfer.  Here the driver dies with KAME -- a crash, a
 * kill, the STM hang watchdog -- and whatever it was in the middle of stays
 * in the adapter for the next open: a reply still queued on an IN endpoint,
 * a data toggle out of step.  After a crash during a LakeShore exchange the
 * adapter would not open again until it was unplugged (2026-10-17).
 *
 * clear_halt resets the halt and the toggle at both ends, endpoint by
 * endpoint -- again if SET_CONFIGURATION already did, which costs a control
 * transfer each.  The drain is needed either way: whether a device empties
 * its FIFOs on SET_CONFIGURATION is up to its firmware.  It runs before
 * probe() and attach(), so nothing of the new session's can be thrown away. */
void NiGpibDriver::restartEndpoints(libusb_device_handle *h, libusb_device *dev)
{
    struct libusb_config_descriptor *cfg = nullptr;
    if (libusb_get_active_config_descriptor(dev, &cfg) != 0)
        return;
    if (cfg->bNumInterfaces > 0 && cfg->interface[0].num_altsetting > 0) {
        const struct libusb_interface_descriptor *alt = &cfg->interface[0].altsetting[0];
        for (int e = 0; e < alt->bNumEndpoints; e++) {
            const struct libusb_endpoint_descriptor *ep = &alt->endpoint[e];
            unsigned char addr = ep->bEndpointAddress;
            libusb_clear_halt(h, addr);
            if (!(addr & LIBUSB_ENDPOINT_IN))
                continue;
            bool intr = (ep->bmAttributes & LIBUSB_TRANSFER_TYPE_MASK)
                == LIBUSB_TRANSFER_TYPE_INTERRUPT;
            /* Bounded: a little over a second at most, should the adapter
             * still be relaying an instrument that is talking. */
            unsigned char buf[4096];
            long stale = 0;
            for (int tries = 0; tries < 64; tries++) {
                int got = 0;
                int r = intr
                    ? libusb_interrupt_transfer(h, addr, buf, sizeof(buf), &got, 20)
                    : libusb_bulk_transfer(h, addr, buf, sizeof(buf), &got, 20);
                stale += got;
                if (r != 0 || got == 0)
                    break; /* a timeout: nothing more queued */
            }
            if (stale)
                fprintf(stderr, "Discarded %ld stale byte(s) queued on endpoint 0x%02x.\n",
                        stale, addr);
        }
    }
    libusb_free_config_descriptor(cfg);
}

void NiGpibDriver::cmd(std::initializer_list<int> bytes)
{
    uint8_t buf[16];
    size_t  len = 0;
    for (int b : bytes) buf[len++] = static_cast<uint8_t>(b);
    size_t bw = 0;
    int r = g_ni_gpib_interface->command(&board_, buf, len, &bw);
    if (r != 0)
        throw std::runtime_error("GPIB command error: " + std::string(strerror(-r)));
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
        cmd({ UNL, UNT });
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
    try { cmd({ UNL, UNT }); } catch (const std::exception &e) { fprintf(stderr, "GPIB write cleanup error: %s\n", e.what()); }
    if (r != 0)
        throw std::runtime_error("GPIB write error: " + std::string(strerror(-r)));
    // printf("Sent %s (%zu bytes written)\n", command.c_str(), bw);
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
    try { cmd({ UNL, UNT }); } catch (const std::exception &e) { fprintf(stderr, "GPIB read cleanup error: %s\n", e.what()); }
    if (r != 0)
        throw std::runtime_error("GPIB read error: " + std::string(strerror(-r)));
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

    try { cmd({ SPD, UNL, UNT }); } catch (const std::exception &e) { fprintf(stderr, "GPIB serial poll cleanup error: %s\n", e.what()); }

    if (r != 0)
        throw std::runtime_error("Serial poll error: " + std::string(strerror(-r)));
    return stb;
}
