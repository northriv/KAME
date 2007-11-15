/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "xnodeconnector.h"
#include <deque>
#include <kapp.h>
#include <qbutton.h>
#include <qlineedit.h>
#include <kurlrequester.h>
#include <qcheckbox.h>
#include <qlistbox.h>
#include <qcombobox.h>
#include <kcolorbutton.h>
#include <kcolorcombo.h>
#include <qlabel.h>
#include <qtable.h>
#include <kled.h>
#include <knuminput.h>
#include <qspinbox.h>
#include <qlcdnumber.h>
#include <qtextbrowser.h>
#include <qtooltip.h>
#include <qstatusbar.h>
#include <kpassivepopup.h>
#include <qmainwindow.h>
#include <kfiledialog.h>
#include <klocale.h>
#include <map>
#include "measure.h"

#include "icons/icon.h"

static std::deque<shared_ptr<XQConnector> > s_conCreating;
static std::deque<shared_ptr<XStatusPrinter> > s_statusPrinterCreating;
static std::map<const QWidget*, weak_ptr<XNode> > s_widgetMap;

//ms
#define UI_DISP_DELAY 10

void _sharedPtrQDeleter(QObject *obj) {
    if(isMainThread())
        delete obj;
    else
        obj->deleteLater();
}

_XQConnectorHolder::_XQConnectorHolder(XQConnector *con) :
    QObject(0L, "connector_holder") {
    m_connector = s_conCreating.back();
    s_conCreating.pop_back();
    connect(con->m_pWidget, SIGNAL( destroyed() ), this, SLOT( destroyed() ) );
    ASSERT(con->shared_from_this());
}
_XQConnectorHolder::~_XQConnectorHolder() {
    if(m_connector)
        disconnect(m_connector->m_pWidget, SIGNAL( destroyed() ), this, SLOT( destroyed() ) );
}
bool
_XQConnectorHolder::isAlive() const {
    return m_connector;
}

void
_XQConnectorHolder::destroyed () {
	disconnect(m_connector->m_pWidget, SIGNAL( destroyed() ), this, SLOT( destroyed() ) );
	std::map<const QWidget*, weak_ptr<XNode> >::iterator it = s_widgetMap.find(m_connector->m_pWidget);
	ASSERT(it != s_widgetMap.end());
	s_widgetMap.erase(it);
	m_connector->m_pWidget = 0L;
	m_connector.reset();
}


XQConnector::XQConnector(const shared_ptr<XNode> &node, QWidget *item)
	: QObject(0L, QString("connector node:%1 widget:%2").arg(node->getName()).arg(item->name())), 
	  m_pWidget(item) 
{
    ASSERT(node);
    ASSERT(item);
    s_conCreating.push_back(shared_ptr<XQConnector>(this));
    m_lsnUIEnabled = node->onUIEnabled().connectWeak
        (shared_from_this(), &XQConnector::onUIEnabled,
		 XListener::FLAG_MAIN_THREAD_CALL | XListener::FLAG_AVOID_DUP);
    onUIEnabled(node);
    dbgPrint(QString("connector %1 created., addr=0x%2, size=0x%3")
			 .arg(name())
			 .arg((unsigned int)this, 0, 16)
			 .arg((unsigned int)sizeof(XQConnector), 0, 16));
    
    std::pair<std::map<const QWidget*, weak_ptr<XNode> >::iterator, bool> ret = 
    		s_widgetMap.insert(std::pair<const QWidget*, weak_ptr<XNode> >(item, node));
    if(!ret.second)
    	gErrPrint("Connection to Widget Duplicated!");
#ifdef HAVE_LIBGCCPP
    GC_gcollect();
#endif
}
XQConnector::~XQConnector() {
    if(isItemAlive()) {
        m_pWidget->setEnabled(false);
        dbgPrint(QString("connector %1 released., addr=0x%2").arg(name()).arg((unsigned int)this, 0, 16));
    	std::map<const QWidget*, weak_ptr<XNode> >::iterator it = s_widgetMap.find(m_pWidget);
    	ASSERT(it != s_widgetMap.end());
    	s_widgetMap.erase(it);
    }
    else {
        dbgPrint(QString("connector %1 & widget released., addr=0x%2").arg(name()).arg((unsigned int)this, 0, 16));
    }

#ifdef HAVE_LIBGCCPP
    GC_gcollect();
#endif
}
shared_ptr<XNode>
XQConnector::connectedNode(const QWidget *item) {
	std::map<const QWidget*, weak_ptr<XNode> >::iterator it = s_widgetMap.find(item);
	if(it == s_widgetMap.end())
		return shared_ptr<XNode>();
	return it->second.lock();
}

void
XQConnector::onUIEnabled(const shared_ptr<XNode> &node) {
    m_pWidget->setEnabled(node->isUIEnabled());
}

XQButtonConnector::XQButtonConnector(const shared_ptr<XNode> &node, QButton *item)
	: XQConnector(node, item),
	  m_node(node), m_pItem(item) 
{
    connect(item, SIGNAL( clicked() ), this, SLOT( onClick() ) );
    m_lsnTouch = node->onTouch().connectWeak
        (shared_from_this(), &XQButtonConnector::onTouch, XListener::FLAG_MAIN_THREAD_CALL);
}
XQButtonConnector::~XQButtonConnector() {
}
void
XQButtonConnector::onClick() {
	m_node->touch();
}
void
XQButtonConnector::onTouch(const shared_ptr<XNode> &) {
}

XValueQConnector::XValueQConnector(const shared_ptr<XValueNodeBase> &node, QWidget *item)
	: XQConnector(node, item) {
    m_lsnBeforeValueChanged = node->beforeValueChanged().connectWeak(
        shared_from_this(), &XValueQConnector::beforeValueChanged,
        XListener::FLAG_MAIN_THREAD_CALL | XListener::FLAG_AVOID_DUP);
    m_lsnValueChanged = node->onValueChanged().connectWeak(
        shared_from_this(), &XValueQConnector::onValueChanged,
        XListener::FLAG_MAIN_THREAD_CALL | XListener::FLAG_AVOID_DUP);
}
XValueQConnector::~XValueQConnector() {
}

XQLineEditConnector::XQLineEditConnector(
    const shared_ptr<XValueNodeBase> &node, QLineEdit *item, bool forcereturn)
	: XValueQConnector(node, item),
	  m_node(node), m_pItem(item) {
    connect(item, SIGNAL( returnPressed() ), this, SLOT( onReturnPressed() ) );
    connect(item, SIGNAL( lostFocus() ), this, SLOT( onExit() ) );
    if(forcereturn) {
		connect(item, SIGNAL( textChanged( const QString &) ),
				this, SLOT( onTextChanged(const QString &) ) );
    }
    else {
		connect(item, SIGNAL( textChanged( const QString &) ),
				this, SLOT( onTextChanged2(const QString &) ) );
    }
    onValueChanged(node);
}
void
XQLineEditConnector::onTextChanged(const QString &text) {
	m_pItem->setPaletteForegroundColor(Qt::blue);
}
void
XQLineEditConnector::onTextChanged2(const QString &text) {
	m_pItem->setPaletteForegroundColor(Qt::blue);
    m_lsnValueChanged->mask();
    try {
		m_node->str(text);
    }
    catch (XKameError &e) {
        e.print();
    }
    m_lsnValueChanged->unmask();
}
void
XQLineEditConnector::onReturnPressed() {
	m_pItem->setPaletteForegroundColor(Qt::black);
    try {
		m_node->str(m_pItem->text());
    }
    catch (XKameError &e) {
        e.print();
    }
    m_pItem->blockSignals(true);
    m_pItem->setText(m_node->to_str());
    m_pItem->blockSignals(false);
}
void
XQLineEditConnector::onExit() {
	m_pItem->setPaletteForegroundColor(Qt::black);
	if(m_node->to_str() != m_pItem->text()) {
	    m_pItem->blockSignals(true);
	    m_pItem->setText(m_node->to_str());
	    m_pItem->blockSignals(false);
		shared_ptr<XStatusPrinter> statusprinter = g_statusPrinter;
		if(statusprinter) statusprinter->printMessage(i18n("Input cancelled."));
	}
}
void
XQLineEditConnector::onValueChanged(const shared_ptr<XValueNodeBase> &node) {
	m_pItem->blockSignals(true);
	m_pItem->setText(node->to_str());
	m_pItem->blockSignals(false);
}
  
XQSpinBoxConnector::XQSpinBoxConnector(const shared_ptr<XIntNode> &node, QSpinBox *item)
	: XValueQConnector(node, item),
	  m_iNode(node),
	  m_uINode(),
	  m_pItem(item)
{
    connect(item, SIGNAL( valueChanged(int) ), this, SLOT( onChange(int) ) );
    onValueChanged(node);
}
XQSpinBoxConnector::XQSpinBoxConnector(const shared_ptr<XUIntNode> &node, QSpinBox *item)
	: XValueQConnector(node, item),
	  m_iNode(),
	  m_uINode(node),
	  m_pItem(item)
{
    connect(item, SIGNAL( valueChanged(int) ), this, SLOT( onChange(int) ) );
    onValueChanged(node);
}
void
XQSpinBoxConnector::onChange(int val) {
    m_lsnValueChanged->mask();
    if(m_iNode) {
		m_iNode->value(val);
    }
    if(m_uINode) {
		m_uINode->value(val);
    }
    m_lsnValueChanged->unmask();
}
void
XQSpinBoxConnector::onValueChanged(const shared_ptr<XValueNodeBase> &node) {
	m_pItem->blockSignals(true);
	m_pItem->setValue(QString(node->to_str()).toInt());
	m_pItem->blockSignals(false);
}
    
XKIntNumInputConnector::XKIntNumInputConnector(const shared_ptr<XIntNode> &node, KIntNumInput *item)
	: XValueQConnector(node, item),
	  m_iNode(node),
	  m_uINode(),
	  m_pItem(item)
{
    connect(item, SIGNAL( valueChanged(int) ), this, SLOT( onChange(int) ) );
    onValueChanged(node);
}
XKIntNumInputConnector::XKIntNumInputConnector(const shared_ptr<XUIntNode> &node, KIntNumInput *item)
	: XValueQConnector(node, item),
	  m_iNode(),
	  m_uINode(node),
	  m_pItem(item)
{
    connect(item, SIGNAL( valueChanged(int) ), this, SLOT( onChange(int) ) );
    onValueChanged(node);
}
void
XKIntNumInputConnector::onChange(int val) {
    m_lsnValueChanged->mask();
    if(m_iNode) {
		m_iNode->value(val);
    }
    if(m_uINode) {
		m_uINode->value(val);
    }
    m_lsnValueChanged->unmask();
}
void
XKIntNumInputConnector::onValueChanged(const shared_ptr<XValueNodeBase> &node) {
	m_pItem->blockSignals(true);
	m_pItem->setValue(QString(node->to_str()).toInt());
	m_pItem->blockSignals(false);
}

XKDoubleNumInputConnector::XKDoubleNumInputConnector(const shared_ptr<XDoubleNode> &node, KDoubleNumInput *item)
	: XValueQConnector(node, item),
	  m_node(node),
	  m_pItem(item)
{
    connect(item, SIGNAL( valueChanged(double) ), this, SLOT( onChange(double) ) );
    onValueChanged(node);
}
void
XKDoubleNumInputConnector::onChange(double val) {
    m_lsnValueChanged->mask();
	m_node->value(val);
    m_lsnValueChanged->unmask();
}
void
XKDoubleNumInputConnector::onValueChanged(const shared_ptr<XValueNodeBase> &) {
	m_pItem->blockSignals(true);
	m_pItem->setValue((double)*m_node);
	m_pItem->blockSignals(false);
}


XKDoubleSpinBoxConnector::XKDoubleSpinBoxConnector(const shared_ptr<XDoubleNode> &node, KDoubleSpinBox *item)
	: XValueQConnector(node, item),
	  m_node(node),
	  m_pItem(item)
{
    connect(item, SIGNAL( valueChanged(double) ), this, SLOT( onChange(double) ) );
    onValueChanged(node);
}
void
XKDoubleSpinBoxConnector::onChange(double val) {
    m_lsnValueChanged->mask();
	m_node->value(val);
    m_lsnValueChanged->unmask();
}
void
XKDoubleSpinBoxConnector::onValueChanged(const shared_ptr<XValueNodeBase> &) {
	m_pItem->blockSignals(true);
	m_pItem->setValue((double)*m_node);
	m_pItem->blockSignals(false);
}
      
XKURLReqConnector::XKURLReqConnector(const shared_ptr<XStringNode> &node,
									 KURLRequester *item, const char *filter, bool saving)
	: XValueQConnector(node, item),
	  m_node(node),
	  m_pItem(item)
{
    connect(item, SIGNAL( urlSelected ( const QString& ) ),
			this, SLOT( onSelect( const QString& ) ) );
    m_pItem->button()->setAutoDefault(false);
    m_pItem->setFilter(filter);
    if(saving) m_pItem->fileDialog()->setOperationMode( KFileDialog::Saving );
    onValueChanged(node);
}
void
XKURLReqConnector::onSelect( const QString& str) {
    try {
		m_node->str(str);
    }
    catch (XKameError &e) {
        e.print();
    }
}

void
XKURLReqConnector::onValueChanged(const shared_ptr<XValueNodeBase> &node) {
	m_pItem->setURL(node->to_str());
}

XQLabelConnector::XQLabelConnector(const shared_ptr<XValueNodeBase> &node, QLabel *item)
	: XValueQConnector(node, item),
	  m_node(node), m_pItem(item)
{
    onValueChanged(node);
}

void
XQLabelConnector::onValueChanged(const shared_ptr<XValueNodeBase> &node) {
	m_pItem->setText(node->to_str());
}

XQTextBrowserConnector::XQTextBrowserConnector(const shared_ptr<XValueNodeBase> &node, QTextBrowser *item)
	: XValueQConnector(node, item),
	  m_node(node), m_pItem(item) {
    onValueChanged(node);
}
void
XQTextBrowserConnector::onValueChanged(const shared_ptr<XValueNodeBase> &node) {
	m_pItem->setText(node->to_str());
}
  
XQLCDNumberConnector::XQLCDNumberConnector(const shared_ptr<XDoubleNode> &node, QLCDNumber *item)
	: XValueQConnector(node, item),
	  m_node(node), m_pItem(item) {
    onValueChanged(node);
}

void
XQLCDNumberConnector::onValueChanged(const shared_ptr<XValueNodeBase> &) {
    QString buf(m_node->to_str());
    if((int)buf.length() > m_pItem->numDigits())
        m_pItem->setNumDigits(buf.length());
    m_pItem->display(buf);
}
  
XKLedConnector::XKLedConnector(const shared_ptr<XBoolNode> &node, KLed *item)
	: XValueQConnector(node, item),
	  m_node(node), m_pItem(item) {
    onValueChanged(node);
}

void
XKLedConnector::onValueChanged(const shared_ptr<XValueNodeBase> &) {
	if(*m_node) m_pItem->on(); else m_pItem->off();
}

XQToggleButtonConnector::XQToggleButtonConnector(const shared_ptr<XBoolNode> &node, QButton *item)
	: XValueQConnector(node, item),
	  m_node(node), m_pItem(item) {
    connect(item, SIGNAL( clicked() ), this, SLOT( onClick() ) );
    onValueChanged(node);
}

void
XQToggleButtonConnector::onClick() {
    m_node->value(m_pItem->isOn());
}

void
XQToggleButtonConnector::onValueChanged(const shared_ptr<XValueNodeBase> &) {
	if(((bool)*m_node) ^ m_pItem->isOn()) m_pItem->toggle();
}

XListQConnector::XListQConnector(const shared_ptr<XListNodeBase> &node, QTable *item)
	: XQConnector(node, item),
	  m_pItem(item), m_list(node) {
    m_lsnMove = node->onMove().connectWeak
        (shared_from_this(),
         &XListQConnector::onMove, XListener::FLAG_MAIN_THREAD_CALL);
    m_lsnCatch = node->onCatch().connectWeak
		(shared_from_this(), &XListQConnector::onCatch, XListener::FLAG_MAIN_THREAD_CALL);
    m_lsnRelease = node->onRelease().connectWeak
		(shared_from_this(), &XListQConnector::onRelease, XListener::FLAG_MAIN_THREAD_CALL);
    m_pItem->setReadOnly(true);

    m_pItem->setSelectionMode(QTable::SingleRow);

    m_pItem->setRowMovingEnabled(true);
    QHeader *header = m_pItem->verticalHeader();
    header->setResizeEnabled(false);
    header->setMovingEnabled(true);
    connect(header, SIGNAL( indexChange(int, int, int)),
			this, SLOT( indexChange(int, int, int)));    
    QToolTip::add(header, KAME::i18n("Use drag-n-drop with ctrl pressed to reorder."));
}
XListQConnector::~XListQConnector() {
    if(isItemAlive()) {
		disconnect(m_pItem, NULL, this, NULL );
		m_pItem->setNumRows(0);
    }
}
void
XListQConnector::indexChange ( int section, int fromIndex, int toIndex )
{
    unsigned int src = fromIndex;
    unsigned int dst = toIndex;
    
    atomic_shared_ptr<const XNode::NodeList> list(m_list->children());
    if(!list || src > list->size() || (dst > list->size())) {
        throw XKameError(KAME::i18n("Invalid range of selections."), __FILE__, __LINE__);
    }
    m_lsnMove->mask();
    m_list->move(src, dst);
    m_lsnMove->unmask();
}
void
XListQConnector::onMove(const XListNodeBase::MoveEvent &e)
{
    int dir = (e.src_idx - e.dst_idx) ? 1 : -1;
    for(unsigned int idx = e.dst_idx; idx != e.src_idx; idx += dir) {
        m_pItem->swapRows(idx, idx + dir);
    }
    m_pItem->updateContents();
}

XItemQConnector::XItemQConnector(const shared_ptr<XItemNodeBase> &node, QWidget *item)
	: XValueQConnector(node, item) {
    m_lsnListChanged = node->onListChanged().connectWeak
        (shared_from_this(), &XItemQConnector::onListChanged,
		 XListener::FLAG_MAIN_THREAD_CALL | XListener::FLAG_AVOID_DUP);
}
XItemQConnector::~XItemQConnector() {
}

XQComboBoxConnector::XQComboBoxConnector(const shared_ptr<XItemNodeBase> &node,
										 QComboBox *item)
	: XItemQConnector(node, item),
	  m_node(node), m_pItem(item) {
    connect(item, SIGNAL( activated(int) ), this, SLOT( onSelect(int) ) );
    onListChanged(node);
}
void
XQComboBoxConnector::onSelect(int idx) {
    try {
        if(!m_itemStrings || (idx >= m_itemStrings->size()) || (idx < 0))
            m_node->str(std::string());
        else
            m_node->str(m_itemStrings->at(idx).name);
    }
    catch (XKameError &e) {
        e.print();
    }
}

int
XQComboBoxConnector::findItem(const QString &text) {
	for(int i = 0; i < m_pItem->count(); i++) {
        if(text == m_pItem->text(i)) {
            return i;
        }
	}
	return -1;
}

void
XQComboBoxConnector::onValueChanged(const shared_ptr<XValueNodeBase> &) {
	m_pItem->blockSignals(true);
	QString str = m_node->to_str();
	int idx = -1;
	int i = 0;
	for(std::deque<XItemNodeBase::Item>::const_iterator it = m_itemStrings->begin();
		it != m_itemStrings->end(); it++) {
        if(QString(it->name) == str) {
            idx = i;
        }
        i++;
	}
	if(idx >= 0) {
		m_pItem->setCurrentItem(idx);
		if(m_node->autoSetAny()) {
			int idx1 = findItem(KAME::i18n("(UNSEL)"));
			if(idx1 >= 0) {
	            m_pItem->removeItem(idx1);
			}
		}
	}
	else {
		if(m_node->autoSetAny()) {
			int idx = findItem(KAME::i18n("(UNSEL)"));
			if(idx < 0) {
	            m_pItem->insertItem(KAME::i18n("(UNSEL)"));
			}
		}
		int idx = findItem(KAME::i18n("(UNSEL)"));
		ASSERT(idx >= 0);
		m_pItem->setCurrentItem(idx);
	}
	m_pItem->blockSignals(false);
}
void
XQComboBoxConnector::onListChanged(const shared_ptr<XItemNodeBase> &)
{
	m_itemStrings = m_node->itemStrings();
	m_pItem->clear();
	bool exist = false;
	for(std::deque<XItemNodeBase::Item>::const_iterator it = m_itemStrings->begin(); 
		it != m_itemStrings->end(); it++) {
        if(it->label.empty()) {
            m_pItem->insertItem(KAME::i18n("(NO NAME)"));
        }
        else {
            m_pItem->insertItem(QString(it->label));
            exist = true;
        }
	}
	if(!m_node->autoSetAny())
		m_pItem->insertItem(KAME::i18n("(UNSEL)"));
	onValueChanged(m_node);
}

XQListBoxConnector::XQListBoxConnector(const shared_ptr<XItemNodeBase> &node, QListBox *item)
	: XItemQConnector(node, item),
	  m_node(node), m_pItem(item) {
    connect(item, SIGNAL(highlighted(int) ), this, SLOT( onSelect(int) ) );
    connect(item, SIGNAL(selected(int) ), this, SLOT( onSelect(int) ) );
    onListChanged(node);
}
void
XQListBoxConnector::onSelect(int idx) {
    try {
        if(!m_itemStrings || (idx >= m_itemStrings->size()) || (idx < 0))
            m_node->str(std::string());
        else
            m_node->str(m_itemStrings->at(idx).name);
    }
    catch (XKameError &e) {
        e.print();
    }
}
void
XQListBoxConnector::onValueChanged(const shared_ptr<XValueNodeBase> &) {
    QString str = m_node->to_str();
	m_pItem->blockSignals(true);
	unsigned int i = 0;
	for(std::deque<XItemNodeBase::Item>::const_iterator it = m_itemStrings->begin(); 
		it != m_itemStrings->end(); it++) {
        if(str == QString(it->name))
            m_pItem->setCurrentItem(i);
        i++;
	}
	m_pItem->blockSignals(false);
}
void
XQListBoxConnector::onListChanged(const shared_ptr<XItemNodeBase> &)
{
	m_itemStrings = m_node->itemStrings();
	m_pItem->clear();
	for(std::deque<XItemNodeBase::Item>::const_iterator it = m_itemStrings->begin(); 
		it != m_itemStrings->end(); it++) {
		m_pItem->insertItem(it->label);
	}
	onValueChanged(m_node);
}

XKColorButtonConnector::XKColorButtonConnector(const shared_ptr<XHexNode> &node, KColorButton *item)
	: XValueQConnector(node, item),
	  m_node(node), m_pItem(item) {
    connect(item, SIGNAL( changed(const QColor &) ), this, SLOT( onClick(const QColor &) ) );
    onValueChanged(node);
}
void
XKColorButtonConnector::onClick(const QColor &newColor) {
	m_node->value(newColor.rgb());
}
void
XKColorButtonConnector::onValueChanged(const shared_ptr<XValueNodeBase> &) {
	m_pItem->setColor(QColor((QRgb)(unsigned int)*m_node));
}
  
XKColorComboConnector::XKColorComboConnector(const shared_ptr<XHexNode> &node, KColorCombo *item)
	: XValueQConnector(node, item),
	  m_node(node), m_pItem(item) {
    connect(item, SIGNAL( activated(const QColor &) ), this, SLOT( onClick(const QColor &) ) );
    onValueChanged(node);
}
void
XKColorComboConnector::onClick(const QColor &newColor) {
	m_node->value(newColor.rgb());
}
void
XKColorComboConnector::onValueChanged(const shared_ptr<XValueNodeBase> &) {
	m_pItem->setColor(QColor((QRgb)(unsigned int)*m_node));
}

XStatusPrinter::XStatusPrinter(QMainWindow *window) 
{
    if(!window) window = dynamic_cast<QMainWindow*>(g_pFrmMain);
    m_pWindow = (window);
    m_pBar = (window->statusBar());
    m_pPopup  = (new KPassivePopup( window ));
    s_statusPrinterCreating.push_back(shared_ptr<XStatusPrinter>(this));
	m_pBar->hide();
	m_lsn = m_tlkTalker.connectWeak(
        shared_from_this(), &XStatusPrinter::print,
        XListener::FLAG_MAIN_THREAD_CALL | XListener::FLAG_AVOID_DUP);
}
XStatusPrinter::~XStatusPrinter() {
}
shared_ptr<XStatusPrinter>
XStatusPrinter::create(QMainWindow *window)
{
    new XStatusPrinter(window);
    shared_ptr<XStatusPrinter> ptr = s_statusPrinterCreating.back();
    s_statusPrinterCreating.pop_back();
    return ptr;
}
void
XStatusPrinter::printMessage(const QString &str, bool popup) {
	tstatus status;
	status.ms = 3000;
	status.str = str.utf8();
	status.popup = popup;
	status.type = tstatus::Normal;
	m_tlkTalker.talk(status);
}
void
XStatusPrinter::printWarning(const QString &str, bool popup) {
	tstatus status;
	status.ms = 3000;
	status.str = (KAME::i18n("Warning: ") + str).utf8();
	status.popup = popup;
	status.type = tstatus::Warning;
    m_tlkTalker.talk(status);
}
void
XStatusPrinter::printError(const QString &str, bool popup) {
	tstatus status;
	status.ms = 5000;
	status.str = (KAME::i18n("Error: ") + str).utf8();
	status.popup = popup;
	status.type = tstatus::Error;
    m_tlkTalker.talk(status);
}
void
XStatusPrinter::clear(void) {
	tstatus status;
	status.ms = 0;
	status.str = "";
    m_tlkTalker.talk(status);
}

void
XStatusPrinter::print(const tstatus &status) {
	bool popup = status.popup;
	QString str = status.str;
	if(status.ms) {
		m_pBar->show();
		m_pBar->message(str, status.ms);
	}
	else {
		m_pBar->hide();
		m_pBar->clear();
	}
	if(status.ms && popup) {
		m_pPopup->hide();
		m_pPopup->setTimeout(status.ms);
		QPixmap *icon;
		switch(status.type) {
		case tstatus::Normal:
			icon = g_pIconInfo;
			break;
		case tstatus::Warning:
			icon = g_pIconWarn;
			break;
		case tstatus::Error:
			icon = g_pIconError;
			break;
		}
		m_pPopup->setView(m_pWindow->caption(), str, *icon );
		m_pPopup->show();
	}
	else {
		m_pPopup->hide();
	}
}
