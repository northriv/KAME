#ifndef XRUBYTHREADCONNECTOR_H_
#define XRUBYTHREADCONNECTOR_H_

#include "xnodeconnector.h"

class XRubyThread;
class XRuby;
class FrmRubyThread;

class XRubyThreadConnector : public XQConnector
{
 Q_OBJECT
 XQCON_OBJECT
 protected:
    XRubyThreadConnector(const shared_ptr<XRubyThread> &rbthread, FrmRubyThread *form,
            const shared_ptr<XRuby> &rbsupport);
 public: 
    virtual ~XRubyThreadConnector();
    
  const   shared_ptr<XBoolNode> &resume() const {return m_resume;}
  const   shared_ptr<XNode> &kill() const {return m_kill;}
 private:
    shared_ptr<XBoolNode> m_resume;
    shared_ptr<XNode> m_kill;
    shared_ptr<XListener> m_lsnOnResumeTouched;
    shared_ptr<XListener> m_lsnOnKillTouched;
    shared_ptr<XListener> m_lsnOnDefout;
    shared_ptr<XListener> m_lsnOnStatusChanged;
    void onResumeTouched(const shared_ptr<XNode> &node);
    void onKillTouched(const shared_ptr<XNode> &node);
    void onDefout(const shared_ptr<std::string> &str);
    void onStatusChanged(const shared_ptr<XValueNodeBase> &node);
    FrmRubyThread *m_pForm;
    shared_ptr<XRubyThread> m_rubyThread;
    shared_ptr<XRuby> m_rubySupport;
    xqcon_ptr m_conFilename, m_conStatus, m_conResume, m_conKill;
};

#endif /*XRUBYTHREADCONNECTOR_H_*/
