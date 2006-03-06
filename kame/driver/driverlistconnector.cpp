//---------------------------------------------------------------------------
#include "driverlistconnector.h"
#include "driver.h"
#include "drivercreate.h"
#include "drivertool.h"
#include "measure.h"
#include <qdeepcopy.h>
#include <qlineedit.h>
#include <qpushbutton.h>
#include <qtable.h>
#include <qlistbox.h>
#include <qlabel.h>
#include <kiconloader.h>
#include <kapplication.h>
#include <klocale.h>

XDriverListConnector::XDriverListConnector
  (const shared_ptr<XDriverList> &node, FrmDriver *item)
   : XListQConnector(node, item->m_tblDrivers),
   m_create(createOrphan<XNode>("Create", true)),
   m_release(createOrphan<XNode>("Release", true)),
   m_conCreate(xqcon_create<XQButtonConnector>(m_create, item->m_btnNew)),
   m_conRelease(xqcon_create<XQButtonConnector>(m_release, item->m_btnDelete))  
{
  item->m_btnNew->setIconSet( KApplication::kApplication()->iconLoader()->loadIconSet("filenew", 
            KIcon::Toolbar, KIcon::SizeSmall, true ) );  
  item->m_btnDelete->setIconSet( KApplication::kApplication()->iconLoader()->loadIconSet("fileclose", 
            KIcon::Toolbar, KIcon::SizeSmall, true ) );  
    
  connect(m_pItem, SIGNAL( clicked( int, int, int, const QPoint& )),
      this, SLOT(clicked( int, int, int, const QPoint& )) );
  
  m_pItem->setNumCols(3);
  double def = 50;
  m_pItem->setColumnWidth(0, (int)(def * 1.5));
  m_pItem->setColumnWidth(1, (int)(def * 1.0));
  m_pItem->setColumnWidth(2, (int)(def * 4.5));
  QStringList labels;
  labels += i18n("Driver");
  labels += i18n("Type");
  labels += i18n("Recorded Time");
  m_pItem->setColumnLabels(labels);

  atomic_shared_ptr<const XNode::NodeList> list(node->children());
  for(XNode::NodeList::const_iterator it = list->begin(); it != list->end(); it++)
    onCatch(*it);

  m_lsnOnCreateTouched = m_create->onTouch().connectWeak(true, shared_from_this(),
    &XDriverListConnector::onCreateTouched);
  m_lsnOnReleaseTouched = m_release->onTouch().connectWeak(true, shared_from_this(),
    &XDriverListConnector::onReleaseTouched);
}

void
XDriverListConnector::onCatch(const shared_ptr<XNode> &node) {
  shared_ptr<XDriver> driver(dynamic_pointer_cast<XDriver>(node));
  if(m_lsnOnRecord)
    driver->onRecord().connect(m_lsnOnRecord);
  else
    m_lsnOnRecord = driver->onRecord().connectWeak(
        false, shared_from_this(), &XDriverListConnector::onRecord);
  
  int i = m_pItem->numRows();
  m_pItem->insertRows(i);
  m_pItem->setText(i, 0, driver->getName());
  // typename is not set at this moment
  m_pItem->setText(i, 1, driver->getTypename().c_str());

  m_cons.push_back(shared_ptr<tcons>(new tcons));
  m_cons.back()->label = new QLabel(m_pItem);
  m_pItem->setCellWidget(i, 2, m_cons.back()->label);
  m_cons.back()->driver = driver;
  m_cons.back()->tlkOnRecordRedirected.reset(new XTalker<tcons::tlisttext>);
  m_cons.back()->lsnOnRecordRedirected = m_cons.back()->tlkOnRecordRedirected->connectWeak(
        true, m_cons.back(), &XDriverListConnector::tcons::onRecordRedirected, true, 30);

  ASSERT(m_pItem->numRows() == (int)m_cons.size());
}
void
XDriverListConnector::onRelease(const shared_ptr<XNode> &node) {
  for(tconslist::iterator it = m_cons.begin(); it != m_cons.end();)
    {
      ASSERT(m_pItem->numRows() == (int)m_cons.size());
      if((*it)->driver == node)
        {
              (*it)->driver->onRecord().disconnect(m_lsnOnRecord);
              for(int i = 0; i < m_pItem->numRows(); i++)
                {
                  if(m_pItem->cellWidget(i, 2) == (*it)->label)
                    m_pItem->removeRow(i);
                }
              it = m_cons.erase(it);
        }
      else
          it++;
    }
}
void
XDriverListConnector::clicked ( int row, int col, int , const QPoint& ) {
 for(tconslist::iterator it = m_cons.begin(); it != m_cons.end(); it++)
    {
         if(m_pItem->cellWidget(row, 2) == (*it)->label)
        {
          if(col < 3) (*it)->driver->showForms();
        }
    }
}

void
XDriverListConnector::onRecord(const shared_ptr<XDriver> &driver)
{
 for(tconslist::iterator it = m_cons.begin(); it != m_cons.end(); it++)
    {
      if((*it)->driver == driver)
        {
            tcons::tlisttext text;
            text.label = (*it)->label;
            text.str.reset(new QString(QDeepCopy<QString>((*it)->driver->time().getTimeStr())));
            (*it)->tlkOnRecordRedirected->talk(text);
        }
    }
}
void
XDriverListConnector::tcons::onRecordRedirected(const tlisttext &text)
{
    text.label->setText(QDeepCopy<QString>(*text.str));
}
void
XDriverListConnector::onCreateTouched(const shared_ptr<XNode> &)
{
    //! modal dialog
   qshared_ptr<DlgCreateDriver> dlg(new DlgCreateDriver(g_pFrmMain, "Create Driver", true));
   static int num = 0;
   num++;
   dlg->m_edName->setText(QString("NewDriver%1").arg(num));
   
   dlg->m_lstType->clear();
   for(unsigned int i = 0; i < XDriverList::typenames().size(); i++) {
        dlg->m_lstType->insertItem(XDriverList::typenames()[i].c_str());
   }
   
   dlg->m_lstType->setCurrentItem(-1);
   if(dlg->exec() == QDialog::Rejected) {
       return;
   }
   int idx = dlg->m_lstType->currentItem();
   shared_ptr<XNode> driver;
   if((idx >= 0) && (idx < (int)XDriverList::typenames().size()))
        driver = m_list->createByTypename(XDriverList::typenames()[idx],
                 dlg->m_edName->text());
   if(!driver)
        gErrPrint(i18n("Driver creation failed"));
}
void
XDriverListConnector::onReleaseTouched(const shared_ptr<XNode> &)
{
    shared_ptr<XDriver> driver;
     for(tconslist::iterator it = m_cons.begin(); it != m_cons.end(); it++)
        {
          if((*it)->label == m_pItem->cellWidget(m_pItem->currentRow(), 2))
            {
                driver = (*it)->driver;
            }
        }    
    if(driver) m_list->releaseChild(driver);
}