#include "entrylistconnector.h"
#include "analyzer.h"
#include "driver.h"
#include <qdeepcopy.h>
#include <qtable.h>
#include <qlabel.h>
#include <qpushbutton.h>
#include <qcheckbox.h>
#include <knuminput.h>
#include <klocale.h>

//---------------------------------------------------------------------------

XEntryListConnector::XEntryListConnector
  (const shared_ptr<XScalarEntryList> &node, QTable *item, const shared_ptr<XChartList> &chartlist)
  : XListQConnector(node, item),
  m_pItem(item),
  m_chartList(chartlist)
{
  connect(item, SIGNAL( clicked( int, int, int, const QPoint& )),
	  this, SLOT(clicked( int, int, int, const QPoint& )) );
  m_pItem->setNumCols(4);
  const double def = 50;
  m_pItem->setColumnWidth(0, (int)(def * 2.5));
  m_pItem->setColumnWidth(1, (int)(def * 2.0));
  m_pItem->setColumnWidth(2, (int)(def * 0.8));
  m_pItem->setColumnWidth(3, (int)(def * 2.5));
  QStringList labels;
  labels += i18n("Entry");
  labels += i18n("Value");
  labels += i18n("Store");
  labels += i18n("Delta");
  m_pItem->setColumnLabels(labels);
  node->childLock();
  for(unsigned int i = 0; i < node->count(); i++)
    onCatch((*node)[i]);
  node->childUnlock();
}
XEntryListConnector::~XEntryListConnector()
{
    if(isItemAlive()) {
      disconnect(m_pItem, NULL, this, NULL );
      m_pItem->setNumRows(0);
    }
}

void
XEntryListConnector::onRecord(const shared_ptr<XDriver> &driver)
{
  for(tconslist::iterator it = m_cons.begin(); it != m_cons.end(); it++)
    {
      if((*it)->entry->driver() == driver)
	   {
        	   tcons::tlisttext text;
            text.label = (*it)->label;
            text.str.reset(new QString(QDeepCopy<QString>((*it)->entry->value()->to_str())));
            (*it)->tlkOnRecordRedirected->talk(text);
    	   }
    }
}
void
XEntryListConnector::tcons::onRecordRedirected(const tlisttext &text)
{
    text.label->setText(*text.str);
}

void
XEntryListConnector::clicked ( int row, int col, int, const QPoint& ) {
    m_chartList->childLock();
      switch(col) {
      case 0:
      case 1:
        if((row >= 0) && (row < (int)m_chartList->count()))
            (*m_chartList)[row]->showChart();
        break;
      default:
        break;
      }
    m_chartList->childUnlock();
}
void
XEntryListConnector::onRelease(const shared_ptr<XNode> &node)
{
  for(tconslist::iterator it = m_cons.begin(); it != m_cons.end();)
    {
      ASSERT(m_pItem->numRows() == (int)m_cons.size());
      if((*it)->entry == node)
        	{
          (*it)->driver->onRecord().disconnect(m_lsnOnRecord);
          for(int i = 0; i < m_pItem->numRows(); i++)
	       {
              if(m_pItem->cellWidget(i, 1) == (*it)->label) m_pItem->removeRow(i);
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
XEntryListConnector::onCatch(const shared_ptr<XNode> &node)
{
  shared_ptr<XScalarEntry> entry = dynamic_pointer_cast<XScalarEntry>(node);
  int i = m_pItem->numRows();
  m_pItem->insertRows(i);
  m_pItem->setText(i, 0, entry->getEntryTitle());

  shared_ptr<XDriver> driver = entry->driver();
  if(m_lsnOnRecord)
    driver->onRecord().connect(m_lsnOnRecord);
  else
    m_lsnOnRecord = driver->onRecord().connectWeak(
        false, shared_from_this(), &XEntryListConnector::onRecord);

  m_cons.push_back(shared_ptr<tcons>(new tcons));
  m_cons.back()->row = i;
  m_cons.back()->entry = entry;
  m_cons.back()->label = new QLabel(m_pItem);
  m_pItem->setCellWidget(i, 1, m_cons.back()->label);
  QCheckBox *ckbStore = new QCheckBox(m_pItem);
  m_cons.back()->constore = xqcon_create<XQToggleButtonConnector>(entry->store(), ckbStore);
  m_pItem->setCellWidget(i, 2, ckbStore);
  KDoubleSpinBox *numDelta = new KDoubleSpinBox(-1, 1e4, 1, 0, 5, m_pItem);
  m_cons.back()->condelta = xqcon_create<XKDoubleSpinBoxConnector>(entry->delta(), numDelta);
  m_pItem->setCellWidget(i, 3, numDelta);
  m_cons.back()->driver = driver;
  m_cons.back()->tlkOnRecordRedirected.reset(new XTalker<tcons::tlisttext>);
  m_cons.back()->lsnOnRecordRedirected = m_cons.back()->tlkOnRecordRedirected->connectWeak(
        true, m_cons.back(), &XEntryListConnector::tcons::onRecordRedirected, true, 30);
  

  ASSERT(m_pItem->numRows() == (int)m_cons.size());
}

