/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "xrubythread.h"

XRubyThread::XRubyThread(const char *name, bool runtime, const QString &filename)
	: XNode(name, runtime),
	  m_filename(create<XStringNode>("Filename", true)),
	  m_status(create<XStringNode>("Status", true)),
	  m_action(create<XStringNode>("Action", true)),
	  m_threadID(create<XIntNode>("ThreadID", true))
 
{
    m_threadID->value(-1);
    m_filename->value(filename);
    m_action->value(RUBY_THREAD_ACTION_STARTING);
    m_status->value(RUBY_THREAD_STATUS_STARTING);
}
 
bool
XRubyThread::isRunning() const
{
    return (std::string(*m_status) == RUBY_THREAD_STATUS_RUN);
}
bool
XRubyThread::isAlive() const
{
    return (std::string(*m_status) != RUBY_THREAD_STATUS_N_A);
}
void
XRubyThread::kill()
{
    m_action->value(RUBY_THREAD_ACTION_KILL);
}
void
XRubyThread::resume()
{
    m_action->value(RUBY_THREAD_ACTION_WAKEUP);
}
