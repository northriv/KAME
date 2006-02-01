#ifndef entrylistconnectorH
#define entrylistconnectorH

#include "xnodeconnector.h"
//---------------------------------------------------------------------------

class QTable;

class XScalarEntry;
class XChartList;
class XScalarEntryList;
class XDriver;

class XEntryListConnector : public XListQConnector
{
  Q_OBJECT
  XQCON_OBJECT
 protected:
  XEntryListConnector
  (const shared_ptr<XScalarEntryList> &node, QTable *item, const shared_ptr<XChartList> &chartlist);
 public:
  virtual ~XEntryListConnector();
 protected:
  virtual void onListChanged(const shared_ptr<XListNodeBase> &) {}
  virtual void onCatch(const shared_ptr<XNode> &node);
  virtual void onRelease(const shared_ptr<XNode> &node);
 protected slots:
    void clicked ( int row, int col, int button, const QPoint& );
 private:
  QTable *m_pItem;
  shared_ptr<XChartList> m_chartList;

  struct tcons {
      struct tlisttext {
        QLabel *label;
        shared_ptr<QString> str;
      };
    int row;
    xqcon_ptr constore, condelta;
    QLabel *label;
    shared_ptr<XScalarEntry> entry;
    shared_ptr<XDriver> driver;
    shared_ptr<XTalker<tlisttext> > tlkOnRecordRedirected;
    shared_ptr<XListener> lsnOnRecordRedirected;
    void onRecordRedirected(const tlisttext &);
  };
  typedef std::deque<shared_ptr<tcons> > tconslist;
  tconslist m_cons;
  shared_ptr<XListener> m_lsnOnRecord;
  void onRecord(const shared_ptr<XDriver> &driver);
};

#endif
