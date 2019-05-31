#include "support_osx.h"

#include <Foundation/Foundation.h>

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

void *autoReleasePoolInit() {
    NSAutoreleasePool* p = [[NSAutoreleasePool alloc] init];
    return p;
}
void autoReleasePoolRelease(void *pool) {
    auto p = (NSAutoreleasePool*)pool;
    //        NSLog(@"%@", [NSAutoreleasePool showPools]);
    [p release];
}
