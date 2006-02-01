#ifndef INTERFACELISTCONNECTOR_H_
#define INTERFACELISTCONNECTOR_H_

#include "interface.h"
#include "xnodeconnector.h"

class QTable;
class QPushButton;

class XInterfaceListConnector : public XListQConnector
{
  Q_OBJECT
  XQCON_OBJECT
 protected:
  XInterfaceListConnector(const shared_ptr<XInterfaceList> &node, QTable *item);
 public:
  virtual ~XInterfaceListConnector();
 protected:
  virtual void onListChanged(const shared_ptr<XListNodeBase> &) {}
  virtual void onCatch(const shared_ptr<XNode> &node);
  virtual void onRelease(const shared_ptr<XNode> &node);
 protected slots:
    void clicked ( int row, int col, int button, const QPoint& );
 private:
  struct tcons {
    xqcon_ptr condev, concontrol, conport, conaddr;
    shared_ptr<XInterface> interface;
    QPushButton *btn;
    shared_ptr<XNode> control;
    shared_ptr<XListener> lsnOnOpenedChanged;
    shared_ptr<XListener> lsnOnControlTouched;
  };
  typedef std::deque<tcons> tconslist;
  tconslist m_cons;

  shared_ptr<XInterfaceList> m_interfaceList;
  QTable *m_pItem;
  void onOpenedChanged(const shared_ptr<XValueNodeBase> &);
  void onControlTouched(const shared_ptr<XNode> &);
};

#endif /*INTERFACELISTCONNECTOR_H_*/
