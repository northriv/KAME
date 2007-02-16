/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitagawa@scphys.kyoto-u.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
 ***************************************************************************/
#ifndef xrubythreadH
#define xrubythreadH
//---------------------------------------------------------------------------
#include "xnode.h"

extern "C" {
#include <ruby.h>
}

#define RUBY_THREAD_ACTION_KILL "kill"
#define RUBY_THREAD_ACTION_WAKEUP "wakeup"
#define RUBY_THREAD_ACTION_FAILURE "failure"
#define RUBY_THREAD_ACTION_STARTING "starting"

#define RUBY_THREAD_STATUS_RUN "run"
#define RUBY_THREAD_STATUS_SLEEP "sleep"
#define RUBY_THREAD_STATUS_ABORTING "aborting"
#define RUBY_THREAD_STATUS_STARTING "starting"
#define RUBY_THREAD_STATUS_N_A ""
//---------------------------------------------------------------------------
class XRubyThread : public XNode
{
 XNODE_OBJECT
 protected:
  XRubyThread(const char *name, bool runtime, const QString &filename);
 public:
  virtual ~XRubyThread() {}

  bool isRunning() const;
  bool isAlive() const;
  void kill();
  void resume();
  
   XTalker<shared_ptr<std::string> > &onMessageOut() {return m_tlkOnMessageOut;}
  const shared_ptr<XStringNode> &status() const {return m_status;}
  const shared_ptr<XStringNode> &filename() const {return m_filename;}
//  shared_ptr<XStringNode> &action() const {return m_action;}
  const shared_ptr<XIntNode> &threadID() const {return m_threadID;}
 private:
  XTalker<shared_ptr<std::string> > m_tlkOnMessageOut;
  const shared_ptr<XStringNode> m_filename;
  shared_ptr<XStringNode> m_status;
  shared_ptr<XStringNode> m_action;
  shared_ptr<XIntNode> m_threadID;
};

//---------------------------------------------------------------------------
#endif
