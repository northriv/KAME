# KAME interface smoke test — serial and GPIB, without any hardware.
#
# Drives XCharInterface's own `Device` / `Port` / `Control` / `Query` nodes
# against (a) a device path that cannot exist, to check that a failed open is
# reported cleanly rather than crashing, and (b) a pty pair, where this script
# plays the instrument on the master side while KAME opens the slave.  So it
# covers the real `termios` path in modules/charinterface/serial.cpp, the
# Prologix layer in gpib.cpp, and — as `Device = GPIB` on a host with neither
# linux-gpib nor NI4882 — the usermode NI USB-GPIB driver's libusb
# enumeration, which with no adapter plugged in must fail cleanly.
#
# Usage — pass it as KAME's positional argument; `runNewScript()` routes any
# non-.py/.kam file to Ruby, so the .py extension matters:
#
#     kame --moduledir <dir> tools/iftest_serial_gpib.py
#     cat /tmp/kame_iftest.txt
#
# Results go to a FILE on purpose: xpythonsupport.py redirects both stdout and
# stderr into KAME's GUI message pane, so printing here shows nothing on a
# terminal even when the script ran perfectly.
#
# Requires a driver with a character interface; KE2000 is used if present.
# Note the driver runs its own protocol on the same port, so the byte counts
# below include whatever it decides to send.

import os, time, threading, traceback

OUT = os.environ.get("KAME_IFTEST_OUT", "/tmp/kame_iftest.txt")
L = open(OUT, "w", buffering=1)

def log(s): L.write(s + "\n")

# ---------------------------------------------------------------- simulator
ID = b"KAME-SIM,MODEL,0,1.0"

def responder_plain(line):
    """Dumb instrument: answers every line with its ID, EOS-terminated."""
    return ID + b"\r\n"

def responder_prologix(line):
    """A Prologix GPIB-USB controller with one instrument behind it.
       ++spoll  -> status byte with MAV (0x10) set, terminated by \\r
       ++read   -> the instrument's answer, terminated by the ETX that
                   `++eot_enable 1 / ++eot_char 3` asked the adapter to append
       other ++ -> configuration, silent
       plain    -> a device command being forwarded, silent
       This is a stub, NOT a faithful Prologix: enough to prove the transport,
       not enough to complete a full query handshake."""
    if line.startswith(b"++spoll"):
        return b"16\r"
    if line.startswith(b"++read") and not line.startswith(b"++read_tmo"):
        return ID + b"\x03"          # ++read_tmo_ms is CONFIG, not a read
    return b""

class Pty:
    """A pty pair: KAME opens the slave, we act as the instrument on master."""
    def __init__(self, name, responder=responder_plain):
        self.master, self.slave = os.openpty()
        self.path = os.ttyname(self.slave)
        self.name, self.responder = name, responder
        self.rx = bytearray()
        self.trace = []
        self._pending = bytearray()
        self.stop = False
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
    def _run(self):
        import select
        while not self.stop:
            r, _, _ = select.select([self.master], [], [], 0.2)
            if not r: continue
            try: data = os.read(self.master, 4096)
            except OSError: break
            if not data: continue
            self.rx += data
            self._pending += data
            while True:                       # dispatch per complete line
                i = min([j for j in (self._pending.find(b"\r"),
                                     self._pending.find(b"\n")) if j >= 0],
                        default=-1)
                if i < 0: break
                line = bytes(self._pending[:i])
                del self._pending[:i + 1]
                if not line: continue
                out = self.responder(line)
                self.trace.append((time.time(), bytes(line), bytes(out)))
                if out:
                    try: os.write(self.master, out)
                    except OSError: return
    def close(self):
        self.stop = True
        for fd in (self.master, self.slave):
            try: os.close(fd)
            except OSError: pass

# ---------------------------------------------------------------- helpers
def control(itf, on):
    itf["Control"] = "1" if on else "0"

def wait_control(itf, want, secs=8.0):
    """onControlChanged runs on its own thread; poll for the settled state."""
    t0 = time.time()
    while time.time() - t0 < secs:
        if (str(itf["Control"]).lower() in ("true", "1")) == want:
            return True
        time.sleep(0.1)
    return False

def try_open(itf, device, port, label, expect_ok):
    itf["Device"] = device
    itf["Port"] = port
    log("--- %s: Device=%s Port=%s" % (label, device, port))
    control(itf, True)
    wait_control(itf, True)
    time.sleep(1.0)                       # let a failure settle Control back
    state = str(itf["Control"]).lower() in ("true", "1")
    log("    result: %s   (expected %s)  %s" %
        ("OPEN-OK" if state else "OPEN-FAILED",
         "OPEN-OK" if expect_ok else "OPEN-FAILED",
         "PASS" if state == expect_ok else "*** MISMATCH ***"))
    return state

def round_trip(itf, sim, label, secs=8.0):
    """Query sends a command, reads the reply, and writes the reply back into
    the same node — so the node's value afterwards is the instrument's answer."""
    before = len(sim.rx)
    try:
        itf["Query"] = "*IDN?"
        time.sleep(secs)
        got_back = str(itf["Query"])
    except Exception as e:
        log("    %s round-trip RAISED %r" % (label, e)); return False
    log("    %s round-trip: instrument saw %r" % (label, bytes(sim.rx[before:])))
    ok = "KAME-SIM" in got_back
    log("    %s round-trip: KAME read back %r  %s" %
        (label, got_back, "PASS" if ok else "(no reply — see notes)"))
    return ok

# ---------------------------------------------------------------- test
try:
    dc = Root()["Drivers"].dynamic_cast()
    types = dc.typenames()
    drv_type = "KE2000" if "KE2000" in types else None
    if not drv_type:
        log("no character-interface driver available; is charinterface loaded?")
        raise SystemExit
    log("driver type = %s" % drv_type)
    drv = dc.createByTypename(drv_type, "IFTEST")
    itf = drv["Interface"]
    log("interface children = %s" % [c.getName() for c in Snapshot(itf).list(itf)])

    # 1/2. A device that cannot exist — must fail cleanly on both transports.
    try_open(itf, "SERIAL", "/dev/kame_no_such_tty", "SERIAL / missing device", False)
    control(itf, False); time.sleep(0.5)
    try_open(itf, "PrologixGPIBUSB", "/dev/kame_no_such_tty", "PrologixGPIBUSB / missing device", False)
    control(itf, False); time.sleep(0.5)

    # 3. SERIAL onto a real pty: open, then a full command/response round trip.
    sim = Pty("serial")
    log("pty for SERIAL = %s" % sim.path)
    if try_open(itf, "SERIAL", sim.path, "SERIAL / real pty", True):
        time.sleep(2.0)
        round_trip(itf, sim, "SERIAL", secs=2.0)
    control(itf, False); time.sleep(1.0)
    sim.close()

    # 4. GPIB via the usermode NI USB-GPIB driver, when that is what
    #    `Device = GPIB` resolves to (no linux-gpib, no NI4882).  With no NI
    #    adapter plugged in this MUST fail cleanly at libusb enumeration --
    #    that failure is the point of the test.
    # (If this build has no usermode driver either, `GPIB` is the Prologix
    # port and an empty Port fails just the same, so the expectation holds.)
    try_open(itf, "GPIB", "", "GPIB / usermode NI USB (no adapter)", False)
    control(itf, False); time.sleep(0.5)

    # 5. GPIB (Prologix) onto a real pty.  The adapter initialisation that
    #    XPrologixInternalSerialPort::open() writes is the proof that the GPIB
    #    layer ran and not merely the serial one.
    sim2 = Pty("gpib", responder_prologix)
    log("pty for GPIB = %s" % sim2.path)
    try_open(itf, "PrologixGPIBUSB", sim2.path, "PrologixGPIBUSB / real pty", True)
    time.sleep(2.0)
    got = bytes(sim2.rx)
    log("    adapter init seen: %d bytes %r" % (len(got), got[:200]))
    log("    Prologix '++' commands seen: %s" % ("YES" if b"++" in got else "NO"))
    round_trip(itf, sim2, "GPIB")
    for (t, ln, out) in sim2.trace:
        log("      in=%-22r out=%r" % (ln, out))
    control(itf, False); time.sleep(1.0)
    sim2.close()

    dc.release(drv)
    log("DONE")
except Exception as e:
    log("EXCEPTION %r\n%s" % (e, traceback.format_exc()))
    log("DONE")
L.close()
