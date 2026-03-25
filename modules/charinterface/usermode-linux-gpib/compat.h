/*
 * compat.h — platform dispatcher for the userspace GPIB port.
 *
 * Copyright (C) 2026 Kentaro Kitagawa
 *
 * Selects the appropriate kernel-API shim based on the target OS:
 *   _WIN32   → win_compat.h  (Windows, MinGW-w64 / Clang, x86_64 + arm64)
 *   otherwise → osx_compat.h  (macOS and Linux, pthreads + POSIX)
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
#ifndef _COMPAT_H_
#define _COMPAT_H_

#ifdef _WIN32
#  include "win_compat.h"
#else
#  include "osx_compat.h"
#endif

#endif /* _COMPAT_H_ */
