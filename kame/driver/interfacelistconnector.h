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
  virtual ~XInterfaceListConnector() {}
 protected:
  virtual void onCatch(const shared_ptr<XNode> &node);
  virtual void onRelease(const shared_ptr<XNode> &node);
 protected slots:
    void clicked ( int row, int col, int button, const QPoint& );
 private:
  struct tcons {
    xqcon_ptr condev, concontrol, conport, conaddr;
    shared_ptr<XInterface> interface;
    QPushButton *btn;
    shared_ptr<XListener> lsnOnControlChanged;
  };
  typedef std::deque<tcons> tconslist;
  tconslist m_cons;

  shared_ptr<XInterfaceList> m_interfaceList;
  void onControlChanged(const shared_ptr<XValueNodeBase> &);
};

#endif /*INTERFACELISTCONNECTOR_H_*/
