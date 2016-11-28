#include "support_osx.h"

#include <Foundation/Foundation.h>

static id activity;

void suspendLazySleeps() {
    NSActivityOptions options = NSActivityUserInitiated | NSActivityLatencyCritical | NSActivityAutomaticTerminationDisabled | NSActivityIdleSystemSleepDisabled;
    activity = [[NSProcessInfo processInfo] beginActivityWithOptions:options reason:@"realtime measurements"];
}

void resumeLazySleeps() {
    [[NSProcessInfo processInfo] endActivity:activity];
}
