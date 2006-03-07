#include "xrubythreadconnector.h"
#include "xrubysupport.h"
#include "xrubythread.h"
#include <qpushbutton.h>
#include <qlabel.h>
#include <qtextbrowser.h>
#include "forms/rubythreadtool.h"
#include "icons/icon.h"
#include <kapplication.h>
#include <qdeepcopy.h>
#include <kiconloader.h>

XRubyThreadConnector::XRubyThreadConnector(
    const shared_ptr<XRubyThread> &rbthread, FrmRubyThread *form,
    const shared_ptr<XRuby> &rbsupport) :
    XQConnector(rbthread, form),
    m_resume(createOrphan<XBoolNode>("Resume", true)),
    m_kill(createOrphan<XNode>("Kill", true)),
    m_pForm(form),
    m_rubyThread(rbthread),
    m_rubySupport(rbsupport),
    m_conFilename(xqcon_create<XQLabelConnector>(
        rbthread->filename(), form->m_plblFilename)),
    m_conStatus(xqcon_create<XQLabelConnector>(
        rbthread->status(), form->m_plblStatus)),
    m_conResume(xqcon_create<XQButtonConnector>(
        m_resume, (form->m_pbtnResume))),
    m_conKill(xqcon_create<XQButtonConnector>(
        m_kill, (form->m_pbtnKill)))
{
    form->m_pbtnResume->setIconSet(
             KApplication::kApplication()->iconLoader()->loadIconSet("exec", 
            KIcon::Toolbar, KIcon::SizeSmall, true ) );  
    form->m_pbtnKill->setIconSet(
             KApplication::kApplication()->iconLoader()->loadIconSet("stop", 
            KIcon::Toolbar, KIcon::SizeSmall, true ) );  
            
    m_pForm->m_ptxtDefout->setMaxLogLines(10000);
    m_pForm->m_ptxtDefout->setTextFormat(Qt::LogText);    
    
    m_lsnOnResumeTouched = m_resume->onTouch().connectWeak(
        false, shared_from_this(), &XRubyThreadConnector::onResumeTouched);
    m_lsnOnKillTouched = m_kill->onTouch().connectWeak(
        false, shared_from_this(), &XRubyThreadConnector::onKillTouched);
    m_lsnOnDefout = rbthread->onMessageOut().connectWeak(
        true, shared_from_this(), &XRubyThreadConnector::onDefout, false);
    m_lsnOnStatusChanged = rbthread->status()->onValueChanged().connectWeak(
        false, shared_from_this(), &XRubyThreadConnector::onStatusChanged);
        
    form->setIcon(*g_pIconScript);
    form->setCaption(rbthread->getName());
    
    onStatusChanged(rbthread->status());    
}
XRubyThreadConnector::~XRubyThreadConnector()
{
    if(isItemAlive()) {
        m_pForm->m_ptxtDefout->clear();
    }
//    m_rubyThread->kill();
    m_rubySupport->releaseChild(m_rubyThread);
}
void
XRubyThreadConnector::onStatusChanged(const shared_ptr<XValueNodeBase> &) {
    bool alive = m_rubyThread->isAlive();
    m_kill->setUIEnabled(alive);
    bool running = m_rubyThread->isRunning();
    m_resume->setUIEnabled(alive && !running);
}
void
XRubyThreadConnector::onResumeTouched(const shared_ptr<XNode> &) {
    m_rubyThread->resume();
}
void
XRubyThreadConnector::onKillTouched(const shared_ptr<XNode> &) {
    m_rubyThread->kill();
}
void
XRubyThreadConnector::onDefout(const shared_ptr<QString> &str) {
    QString s = QDeepCopy<QString>(*str);
    m_pForm->m_ptxtDefout->append(s);
}
