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
    return (*m_status == RUBY_THREAD_STATUS_RUN);
}
bool
XRubyThread::isAlive() const
{
    return (*m_status != RUBY_THREAD_STATUS_N_A);
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
