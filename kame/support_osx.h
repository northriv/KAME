#ifndef SUPPORT_OSX_H
#define SUPPORT_OSX_H

#if !defined(__cplusplus)
    #define C_API extern
#else
    #define C_API extern "C"
#endif

C_API void suspendLazySleeps();
C_API void resumeLazySleeps();

//! 0 = follow the desktop, 1 = light, 2 = dark.
//!
//! AppKit directly, rather than only QStyleHints::setColorScheme(): going back
//! to "follow the desktop" means handing NSApplication a nil appearance, and
//! whether Qt's unset path does that is not something to take on trust.
//! Setting the two together keeps Qt's palette and the native chrome saying
//! the same thing.
C_API void setAppAppearance(int mode);

C_API void *autoReleasePoolInit();
C_API void autoReleasePoolRelease(void *);

#endif // SUPPORT_OSX_H
