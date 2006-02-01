#ifndef driverlistconnectorH
#define driverlistconnectorH

#include "driver.h"
#include "xnodeconnector.h"
//---------------------------------------------------------------------------

class FrmDriver;
class QTable;
class QLabel;

class XDriverListConnector : public XListQConnector
{
  Q_OBJECT
  XQCON_OBJECT
 protected:
  XDriverListConnector
  (const shared_ptr<XDriverList> &node, FrmDriver *item);
 public:
  virtual ~XDriverListConnector();
 protected:
  virtual void onListChanged(const shared_ptr<XListNodeBase> &) {}
  virtual void onCatch(const shared_ptr<XNode> &node);
  virtual void onRelease(const shared_ptr<XNode> &node);
 protected slots:
    void clicked ( int row, int col, int button, const QPoint& );
 private:
  shared_ptr<XDriverList> m_list; 
  QTable *m_pItem;
  
  shared_ptr<XNode> m_create, m_release;
  
  struct tcons {
    struct tlisttext {
        QLabel *label;
        shared_ptr<QString> str;
    };
    QLabel *label;
    shared_ptr<XDriver> driver;
    shared_ptr<XTalker<tlisttext> > tlkOnRecordRedirected;
    shared_ptr<XListener> lsnOnRecordRedirected;
    void onRecordRedirected(const tlisttext &);
  };
  typedef std::deque<shared_ptr<tcons> > tconslist;
  tconslist m_cons;
  
  shared_ptr<XListener> m_lsnOnRecord;
  shared_ptr<XListener> m_lsnOnCreateTouched, m_lsnOnReleaseTouched;
  
  xqcon_ptr m_conCreate, m_conRelease;
  void onRecord(const shared_ptr<XDriver> &driver);
  void onCreateTouched(const shared_ptr<XNode> &);
  void onReleaseTouched(const shared_ptr<XNode> &);
};


#endif
