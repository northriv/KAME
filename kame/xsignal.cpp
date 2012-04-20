/***************************************************************************
		Copyright (C) 2002-2012 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "xsignal.h"
#include "atomic_queue.h"
#include "xscheduler.h"

threadid_t g_main_thread_id = threadID();

bool isMainThread() {
    return pthread_equal(threadID(), g_main_thread_id);
}

XListener::XListener(FLAGS flags) :
	m_flags(flags) {
    if(flags & FLAG_AVOID_DUP) {
        assert(flags & FLAG_MAIN_THREAD_CALL);
    }
    if((flags & FLAG_DELAY_SHORT) || (flags & FLAG_DELAY_ADAPTIVE)) {
        assert(flags & FLAG_AVOID_DUP);
    }
}
XListener::~XListener() {}

unsigned int
XListener::delay_ms() const {
	unsigned int delay = std::min(20u, g_adaptiveDelay);
	if(m_flags & FLAG_DELAY_ADAPTIVE)
		delay = g_adaptiveDelay;
	if(m_flags & FLAG_DELAY_SHORT)
		delay /= 4;
	return delay;
}

