#include "support_osx.h"

#include <Foundation/Foundation.h>
#include <AppKit/AppKit.h>

static id activity;

void suspendLazySleeps() {
    // | NSActivityLatencyCritical
    NSActivityOptions options =
        NSActivityUserInitiated | NSActivityAutomaticTerminationDisabled | NSActivityIdleSystemSleepDisabled
        | NSActivityBackground | NSActivityLatencyCritical;
    activity = [[NSProcessInfo processInfo] beginActivityWithOptions:options reason:@"realtime measurements"];
}

void resumeLazySleeps() {
    [[NSProcessInfo processInfo] endActivity:activity];
}

void setAppAppearance(int mode) {
    NSAppearance *appearance = nil;   //!< nil is "whatever the desktop says"
    if(mode == 1)
        appearance = [NSAppearance appearanceNamed:NSAppearanceNameAqua];
    else if(mode == 2)
        appearance = [NSAppearance appearanceNamed:NSAppearanceNameDarkAqua];
    [NSApplication sharedApplication].appearance = appearance;
}

void *autoReleasePoolInit() {
    NSAutoreleasePool* p = [[NSAutoreleasePool alloc] init];
    return p;
}
void autoReleasePoolRelease(void *pool) {
    auto p = (NSAutoreleasePool*)pool;
    //        NSLog(@"%@", [NSAutoreleasePool showPools]);
    [p release];
}
