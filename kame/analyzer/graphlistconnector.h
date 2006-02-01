//---------------------------------------------------------------------------

#ifndef graphlistconnectorH
#define graphlistconnectorH

#include "xnodeconnector.h"
//---------------------------------------------------------------------------

class QPushButton;
class QTable;
class XGraphList;

class XGraphListConnector : public XListQConnector
{
 Q_OBJECT
 XQCON_OBJECT
 protected:
  XGraphListConnector(const shared_ptr<XGraphList> &node, QTable *item,
     QPushButton *btnnew, QPushButton *btndelete);
 public:
  virtual ~XGraphListConnector();
 protected:
  virtual void onListChanged(const shared_ptr<XListNodeBase> &) {}
  virtual void onCatch(const shared_ptr<XNode> &node);
  virtual void onRelease(const shared_ptr<XNode> &node);
 protected slots:
  void clicked ( int row, int col, int button, const QPoint& );
 private:
  shared_ptr<XGraphList> m_graphlist;
  QTable *m_pItem;
  
  shared_ptr<XNode> m_newGraph;
  shared_ptr<XNode> m_deleteGraph;
  struct tcons {
    xqcon_ptr conx, cony1, conz;
    shared_ptr<XNode> node;
    QWidget *widget;
  };
  typedef std::deque<tcons> tconslist;
  tconslist m_cons;
    
  
  xqcon_ptr m_conNewGraph, m_conDeleteGraph;
  shared_ptr<XListener> m_lsnNewGraph, m_lsnDeleteGraph;
  
  void onNewGraph (const shared_ptr<XNode> &);
  void onDeleteGraph (const shared_ptr<XNode> &);
};

#endif
