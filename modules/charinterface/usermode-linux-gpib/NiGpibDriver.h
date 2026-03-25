/*
 * NiGpibDriver.h — C++ driver class for NI USB-GPIB adapters (macOS port).
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
 *
 * Include this header in any project that needs to talk to a NI USB-GPIB
 * adapter via the linux-gpib-wrapper compatibility layer.
 *
 * Link with: ni_usb_gpib.o gpib_stubs.o -lusb-1.0 -lpthread
 * (ni_usb_gpib.c and gpib_stubs.c must be compiled as C, not C++.)
 */
#pragma once

#include <cstdint>
#include <string>
#include <initializer_list>
#include <vector>

extern "C" {
#include "compat.h"
#include "gpibP.h"
}

/* ------------------------------------------------------------------ */

class NiGpibDriver {
public:
    /* board_pad:      GPIB primary address of the controller (usually 0).
     * timeout_usec:   read/write timeout in microseconds (default 3 s). */
    explicit NiGpibDriver(int board_pad = 0,
                          unsigned int timeout_usec = 3000000);
    ~NiGpibDriver();

    NiGpibDriver(const NiGpibDriver &) = delete;
    NiGpibDriver &operator=(const NiGpibDriver &) = delete;

    /* Find the first supported USB adapter, initialise the driver, probe,
     * and attach.  Returns false (with a message to stderr) on any failure. */
    bool open();

    /* Test-only initialiser: skip all USB/hardware init and use iface
     * directly.  Sets up board_ exactly as open() does (mutexes, waitqueues,
     * etc.) so that query(), serialPoll(), and close() work against a mock
     * gpib_interface.  Never call on an already-open driver. */
    bool openForTest(struct gpib_interface *iface);

    /* Detach, disconnect, and release all resources.
     * Safe to call multiple times; also called by the destructor. */
    void close();

    /* Assert IFC for ≥ 100 µs then de-assert (resets all bus devices). */
    void interfaceClear();

    /* Device clear.
     *   addr < 0 : DCL — universal clear, affects all devices.
     *   addr >= 0 : SDC — selected device clear, addressed to one device. */
    void deviceClear(int addr = -1);

    /* Assert (enable=true) or de-assert (enable=false) the REN line. */
    void enableRemote(bool enable = true);

    /* Address addr as listener and send command.
     * term is the terminator string appended to the payload:
     *   ""     — assert EOI, no extra bytes  (default)
     *   "\r"   — append CR,   no EOI
     *   "\n"   — append LF,   no EOI
     *   "\r\n" — append CRLF, no EOI */
    void send(int addr, const std::string &command, const char *term = "");

    /* Read until the device asserts EOI.
     * Trailing CR/LF are stripped from the result. */
    std::string read(int addr);

    /* Read using hardware EOS detection on the last byte of term.
     * EOI is not required.  Trailing CR/LF are stripped from the result. */
    std::string read(int addr, const char *term);

    /* Read exactly length bytes.  EOI is not required.
     * Returns however many bytes the driver actually delivered. */
    std::string read(int addr, size_t length);

    /* Convenience: send(addr, command, term) followed by read(addr).
     * If term is non-empty, hardware EOS detection is used for the read. */
    std::string query(int addr, const std::string &command,
                      const char *term = "");

    /* Perform a serial poll of addr; returns the status byte (STB).
     * Bit 6 (0x40) is RQS — set when the device is requesting service. */
    uint8_t serialPoll(int addr);

private:
    int          board_pad_;
    unsigned int timeout_usec_;

    libusb_context      *ctx_      = nullptr;
    struct usb_device    usb_dev_  = {};
    struct usb_interface usb_intf_ = {};
    struct gpib_board    board_    = {};

    bool module_inited_ = false;
    bool probed_        = false;
    bool attached_      = false;

    bool findAndOpenDevice();

    /* Send a sequence of GPIB command bytes with ATN asserted. */
    void cmd(std::initializer_list<int> bytes);

    /* Address addr as talker / board as listener, then read up to max_len
     * bytes.  Shared implementation for all read() overloads. */
    std::vector<uint8_t> readRaw(int addr, size_t max_len);
};
