# KAME — K's Adaptive Measurement Engine

KAME is an open-source scientific instrument control and data-acquisition framework built on Qt.
It provides a transactional node tree, a scriptable UI, and a rich set of hardware drivers for laboratory equipment.

**License:** GPL v2 or later
**Authors:** Kentaro Kitagawa, Shota Suetsugu

---

## Features

- Transactional, thread-safe node/data model
- Ruby and Python scripting (pybind11)
- 2-D / 1-D graph display with math tools (sum, average, …)
- Save / restore full measurement state to `.kam` files
- Modular driver plug-in architecture

### Supported instrument categories

| Module | Examples |
|---|---|
| Digital storage oscilloscopes | `dso` |
| Signal generators / function synthesisers | `sg`, `funcsynth` |
| Lock-in amplifiers | `lia` |
| Digital multimeters | `dmm` |
| DC sources | `dcsource` |
| Network analysers | `networkanalyzer` |
| NMR / pulse programmers | `nmr` |
| Temperature controllers | `tempcontrol` |
| Magnet power supplies | `magnetps` |
| Motors / positioners | `motor` |
| Flow controllers | `flowcontroller` |
| Optical / camera (ODMR imaging) | `optics` |
| Quantum Design PPMS | `qd` |
| NI DAQmx / counter / GPIB | `nidaq`, `counter`, `charinterface` |
| Monte Carlo simulation | `montecarlo` |

---

## Dependencies

| Library | Notes |
|---|---|
| **Qt** ≥ 5.7 or Qt 6 | Qt 5 compatibility module required for Qt 6 |
| **boost** | |
| **Ruby** | scripting |
| **pybind11** | Python scripting |
| **GSL** | |
| **FFTW 3** | |
| **Eigen 3** | |
| LAPACK / ATLAS / BLAS *(optional)* | |
| **libusb** | USB instrument interfaces |
| linux-gpib or NI 488.2 *(optional)* | GPIB interfaces |
| NI DAQmx *(optional)* | NI data-acquisition hardware |

A C++ compiler supporting C++17 or later is required (GCC ≥ 10, Clang ≥ 2.1 with appropriate dialect flags).

---

## Building

### Generic (Linux / Unix)

```sh
qmake [options] /path/to/kame/source
make
sudo make install
```

Or with CMake (KDE4 build):

```sh
mkdir build && cd build
cmake /path/to/kame/source
make
make install DESTDIR=/path/to/install
```

---

### macOS

> Open `kame.pro` in **Qt Creator** (use the genuine open-source Qt, **not** the MacPorts Qt).

Install dependencies via MacPorts:

```sh
sudo port install gsl fftw-3 libtool-ltdl libusb eigen3 pybind11
```

Optionally, for a universal (arm64 + x86_64) binary, build fftw-3 with:

```sh
sudo port install fftw-3 +universal +clang13 -gfortran
```

Additional notes:

- Add `/opt/local/bin` to PATH in the Qt Creator build-environment pane if needed.
- In Qt Creator's **executable environment** pane, **deactivate** "Add build library search path to DYLD_LIBRARY_PATH …", otherwise KAME crashes on launch.
- If `ruby.h` is not found, reinstall Xcode command-line tools: `xcode-select --install`.
- Qt 6: the **Qt5 compatibility module** must be selected during Qt installation.
- NI 488.2 is not supported on Apple Silicon.

---

### Windows (x86-64, MSYS2 / MinGW)

> Requires **Qt ≥ 6.10** with the llvm-mingw64 toolchain.
> Open `kame.pro` in **Qt Creator**.

Install dependencies via MSYS2:

```sh
pacman -S make \
    mingw-w64-x86_64-zlib \
    mingw-w64-x86_64-fftw \
    mingw-w64-x86_64-gsl \
    mingw-w64-x86_64-eigen3 \
    mingw-w64-x86_64-pybind11 \
    mingw-w64-x86_64-libusb \
    mingw-w64-x86_64-python-numpy \
    mingw-w64-x86_64-ruby
```

NI 488.2 or DAQmx drivers are optional.

**Before running KAME**, copy the following DLLs from `C:\msys64\mingw64\bin` alongside the KAME executable:

```
libfftw3-3.dll  libgsl.dll  libgslcblas-0.dll
zlib1.dll  libgmp-10.dll  libusb-1.0.dll
x64-msvcrt-ruby3**.dll
```

Also copy `kame/script/rubylineshell.rb` and `kame/script/pythonlineshell.py` to `./Resources`.

**Launch scripts:**

| Script | Purpose |
|---|---|
| `kame.bat` | Standard launch (system Python) |
| `kame-msyspython.bat` | Launch with MSYS2 Python (numpy, etc.) |

To launch from Qt Creator, add to **Projects → Environment**:

```
PATH=C:\msys64\usr\bin;C:\msys64\mingw64\bin;C:\msys64\mingw64\lib
PYTHONHOME=C:\msys64\mingw64
```

---

## Scripting

KAME exposes its node tree to Ruby and Python. Scripts can be run from the **Script** tab in the UI or loaded from `.kam` files. A `.kam` file is a Ruby script that recreates the full measurement state when executed.

---

## Contributing

Bug reports and pull requests are welcome on [GitHub](https://github.com/northriv/KAME).
