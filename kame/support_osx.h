#ifndef SUPPORT_OSX_H
#define SUPPORT_OSX_H

#if !defined(__cplusplus)
    #define C_API extern
#else
    #define C_API extern "C"
#endif

C_API void suspendLazySleeps();
C_API void resumeLazySleeps();

#endif // SUPPORT_OSX_H
