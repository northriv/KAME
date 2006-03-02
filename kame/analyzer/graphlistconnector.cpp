//---------------------------------------------------------------------------

#include "graphlistconnector.h"

#include <qtable.h>
#include <qcombobox.h>
#include <qpushbutton.h>
#include <kiconloader.h>
#include <kapplication.h>
#include <klocale.h>

#include "recorder.h"
#include "analyzer.h"

//---------------------------------------------------------------------------

XGraphListConnector::XGraphListConnector(const shared_ptr<XGraphList> &node, QTable *item,
     QPushButton *btnnew, QPushButton *btndelete) :
    XListQConnector(node, item),
    m_graphlist(node),
    m_pItem(item),
    m_newGraph(createOrphan<XNode>("NewGraph", true)),
    m_deleteGraph(createOrphan<XNode>("DeleteGraph", true)),
    m_conNewGraph(xqcon_create<XQButtonConnector>(m_newGraph, btnnew)),
    m_conDeleteGraph(xqcon_create<XQButtonConnector>(m_deleteGraph, btndelete))
{
  btnnew->setIconSet( KApplication::kApplication()->iconLoader()->loadIconSet("filenew", 
            KIcon::Toolbar, KIcon::SizeSmall, true ) );  
  btndelete->setIconSet( KApplication::kApplication()->iconLoader()->loadIconSet("fileclose", 
            KIcon::Toolbar, KIcon::SizeSmall, true ) ); 
               
  connect(item, SIGNAL( clicked( int, int, int, const QPoint& )),
	  this, SLOT(clicked( int, int, int, const QPoint& )) );
  m_pItem->setNumCols(4);
  const double def = 50;
  m_pItem->setColumnWidth(0, (int)(def * 2.0));
  m_pItem->setColumnWidth(1, (int)(def * 2.0));
  m_pItem->setColumnWidth(2, (int)(def * 2.0));
  m_pItem->setColumnWidth(3, (int)(def * 2.0));
  QStringList labels;
  labels += i18n("Name");
  labels += i18n("Axis X");
  labels += i18n("Axis Y");
  labels += i18n("Axis Z");
  m_pItem->setColumnLabels(labels);
  { XScopedReadLock<XRecursiveRWLock> lock(node->childMutex());
      for(unsigned int i = 0; i < node->count(); i++)
        onCatch((*node)[i]);
  }
  
  m_lsnNewGraph = m_newGraph->onTouch().connectWeak(
        true, shared_from_this(), &XGraphListConnector::onNewGraph);
  m_lsnDeleteGraph = m_deleteGraph->onTouch().connectWeak(
        true, shared_from_this(), &XGraphListConnector::onDeleteGraph);
}
XGraphListConnector::~XGraphListConnector()
{
    if(isItemAlive()) {
      disconnect(m_pItem, NULL, this, NULL );
      m_pItem->setNumRows(0);
    }
}
void
XGraphListConnector::onNewGraph (const shared_ptr<XNode> &) {
static int graphidx = 1;
    m_graphlist->createByTypename("", std::string(QString().sprintf("Graph-%d", graphidx++).utf8()));
}
void
XGraphListConnector::onDeleteGraph (const shared_ptr<XNode> &) {
      int n = m_pItem->currentRow();
      XScopedReadLock<XRecursiveRWLock> lock(m_graphlist->childMutex());
      if((n >= 0) && (n < (int)m_graphlist->count())) {
          shared_ptr<XNode> node = (*m_graphlist)[n];
          m_graphlist->releaseChild(node);
      }
}
void
XGraphListConnector::clicked ( int row, int col, int, const QPoint& ) {
      XScopedReadLock<XRecursiveRWLock> lock(m_graphlist->childMutex());
      switch(col) {
      case 0:
        if((row >= 0) && (row < (int)m_graphlist->count()))
             (*m_graphlist)[row]->showGraph();
        break;
      default:
        break;
      }
}
void
XGraphListConnector::onRelease(const shared_ptr<XNode> &node)
{
  for(tconslist::iterator it = m_cons.begin(); it != m_cons.end();)
    {
      if(it->node == node)
	{
          for(int i = 0; i < m_pItem->numRows(); i++)
	    {
              if(m_pItem->cellWidget(i, 1) == it->widget) m_pItem->removeRow(i);
	    }
          it = m_cons.erase(it);
	}
      else
	{
          it++;
	}
    }
}
void
XGraphListConnector::onCatch(const shared_ptr<XNode> &node)
{
  shared_ptr<XValGraph> graph = dynamic_pointer_cast<XValGraph>(node);
  int i = m_pItem->numRows();
  m_pItem->insertRows(i);
  m_pItem->setText(i, 0, graph->getName());

  struct tcons con;
  con.node = node;
  QComboBox *cmbX = new QComboBox(m_pItem);
  con.conx = xqcon_create<XQComboBoxConnector>(graph->axisX(), cmbX);
  m_pItem->setCellWidget(i, 1, cmbX);
  QComboBox *cmbY1 = new QComboBox(m_pItem);
  con.cony1 = xqcon_create<XQComboBoxConnector>(graph->axisY1(), cmbY1);
  m_pItem->setCellWidget(i, 2, cmbY1);
  QComboBox *cmbZ = new QComboBox(m_pItem);
  con.conz = xqcon_create<XQComboBoxConnector>(graph->axisZ(), cmbZ);
  m_pItem->setCellWidget(i, 3, cmbZ);

  con.widget = m_pItem->cellWidget(i, 1);
  m_cons.push_back(con);
}
