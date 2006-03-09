//---------------------------------------------------------------------------
#include "interfacelistconnector.h"
#include "driver.h"

#include <qtable.h>
#include <qlineedit.h>
#include <qcombobox.h>
#include <qpushbutton.h>
#include <qspinbox.h>
#include <kiconloader.h>
#include <kapplication.h>
#include <klocale.h>

XInterfaceListConnector::XInterfaceListConnector(
    const shared_ptr<XInterfaceList> &node, QTable *item)
  : XListQConnector(node, item), m_interfaceList(node)
{
  connect(m_pItem, SIGNAL( clicked( int, int, int, const QPoint& )),
      this, SLOT(clicked( int, int, int, const QPoint& )) );
  item->setNumCols(5);
  double def = 50;
  item->setColumnWidth(0, (int)(def * 1.5));
  item->setColumnWidth(1, (int)(def * 1.2));
  item->setColumnWidth(2, (int)(def * 2));
  item->setColumnWidth(3, (int)(def * 1));
  item->setColumnWidth(4, (int)(def * 1));
  QStringList labels;
  labels += i18n("Driver");
  labels += i18n("Control");
  labels += i18n("Device");
  labels += i18n("Port");
  labels += i18n("Addr");
  item->setColumnLabels(labels);

  atomic_shared_ptr<const XNode::NodeList> list(node->children());
  if(list) { 
      for(XNode::NodeList::const_iterator it = list->begin(); it != list->end(); it++)
        onCatch(*it);
  }
}

void
XInterfaceListConnector::onControlTouched(const shared_ptr<XNode> &node)
{
  for(tconslist::iterator it = m_cons.begin(); it != m_cons.end(); it++)
    {
      if(it->control == node)
        {
            if(!it->interface->isOpened()) {
//                it->control->isUIEnabled(false);
                it->interface->driver()->startMeas();
            }
            else {
//                it->control->isUIEnabled(false);
                it->interface->driver()->stopMeas();
            }
        }
    }
}
void
XInterfaceListConnector::onOpenedChanged(const shared_ptr<XValueNodeBase> &node)
{
  for(tconslist::iterator it = m_cons.begin(); it != m_cons.end(); it++)
    {
      if(it->interface->opened() == node)
        {
            it->control->setUIEnabled(true);
            KApplication *app = KApplication::kApplication();
            if(*it->interface->opened()) {
                it->btn->setIconSet( app->iconLoader()->loadIconSet("stop", 
                    KIcon::Toolbar, KIcon::SizeSmall, true ) );
                it->btn->setText(i18n("&STOP"));
             }
             else {
                it->btn->setIconSet( app->iconLoader()->loadIconSet("run", 
                    KIcon::Toolbar, KIcon::SizeSmall, true ) );
                it->btn->setText(i18n("&RUN"));
            }
        }
    }
}
void
XInterfaceListConnector::onCatch(const shared_ptr<XNode> &node) {
  shared_ptr<XInterface> interface = dynamic_pointer_cast<XInterface>(node);
  int i = m_pItem->numRows();
  m_pItem->insertRows(i);
  m_pItem->setText(i, 0, interface->driver()->getLabel());
  struct tcons con;
  con.interface = interface;
  con.control = createOrphan<XNode>("Control", true);
  con.btn = new QPushButton(m_pItem);
  con.btn->setToggleButton(false);
  con.btn->setAutoDefault(false);
  con.btn->setFlat(true);
  con.concontrol = xqcon_create<XQButtonConnector>(con.control, con.btn);    
  m_pItem->setCellWidget(i, 1, con.btn);
  QComboBox *cmbdev(new QComboBox(m_pItem) );
  con.condev = xqcon_create<XQComboBoxConnector>(interface->device(), cmbdev);
  m_pItem->setCellWidget(i, 2, cmbdev);
  QLineEdit *edPort(new QLineEdit(m_pItem) );
  con.conport = xqcon_create<XQLineEditConnector>(interface->port(), edPort, false);
  m_pItem->setCellWidget(i, 3, edPort);
  QSpinBox *numAddr(new QSpinBox(0, 32, 1, m_pItem) );
  con.conaddr = xqcon_create<XQSpinBoxConnector>(interface->address(), numAddr);
  m_pItem->setCellWidget(i, 4, numAddr);
  con.lsnOnControlTouched = con.control->onTouch().connectWeak(
        true, shared_from_this(), &XInterfaceListConnector::onControlTouched, true);
  con.lsnOnOpenedChanged = interface->opened()->onValueChanged().connectWeak(
        true, shared_from_this(), &XInterfaceListConnector::onOpenedChanged, true);
  m_cons.push_back(con);
  onOpenedChanged(interface->opened());
}
void
XInterfaceListConnector::onRelease(const shared_ptr<XNode> &node) {
  for(tconslist::iterator it = m_cons.begin(); it != m_cons.end();)
    {
      if(it->interface == node)
        {
            for(int i = 0; i < m_pItem->numRows(); i++)
            {
                  if(m_pItem->cellWidget(i, 1) == it->btn) m_pItem->removeRow(i);
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
XInterfaceListConnector::clicked ( int row, int , int , const QPoint& ) {
  for(tconslist::iterator it = m_cons.begin(); it != m_cons.end(); it++)
    {
          if(m_pItem->cellWidget(row, 1) == it->btn)
                it->interface->driver()->showForms();
    }
}
