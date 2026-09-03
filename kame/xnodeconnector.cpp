/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "xnodeconnector.h"
#include "driver/driver.h"
#include "kamesettings.h"
#include <deque>
#include <QPushButton>
#include <QLineEdit>
#include <QCheckBox>
#include <QListWidget>
#include <QComboBox>
#include <QLabel>
#include <QTableWidget>
#include <QHeaderView>
#include <QDoubleSpinBox>
#include <QSlider>
#include <QLCDNumber>
#include <QTextBrowser>
#include <QToolTip>
#include <QStatusBar>
#include <QSlider>
#include <QToolButton>
#include <QFileDialog>
#include <QFileInfo>
#include <QDir>
#include <QColorDialog>
#include <QPainter>
#include <QMainWindow>
#include <QApplication>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QCheckBox>

#include <map>
#include "measure.h"
#include "icons/icon.h"
#include "messagebox.h"
#include <type_traits>

//ms
#define UI_DISP_DELAY 10

static std::deque<shared_ptr<XStatusPrinter> > s_statusPrinterCreating;
static std::deque<shared_ptr<XQConnector> > s_conCreating;
static std::map<const QWidget*, weak_ptr<XNode> > s_widgetMap;

// See the doc block in xnodeconnector.h.  Depth-counted so nested connector
// creation (forms building forms) stays exempt until the outermost scope ends.
static thread_local int stl_starvationExemptDepth = 0;
XQConnector_StarvationExempt::XQConnector_StarvationExempt() {
    ++stl_starvationExemptDepth;
}
XQConnector_StarvationExempt::~XQConnector_StarvationExempt() {
    --stl_starvationExemptDepth;
}
bool xqcon_starvationExempted() {
    return stl_starvationExemptDepth > 0;
}

void sharedPtrQDeleter_(QObject *obj) {
    if(isMainThread())
        delete obj;
    else
        obj->deleteLater();
}

XQConnectorHolder_::XQConnectorHolder_(XQConnector *con) :
    QObject(0L) {
    // Pairing check: the entry XQConnector's constructor pushed must be for
    // THIS object.  A constructor that throws after the push (the STM
    // starvation throw made that reachable, and xqcon_create's exemption
    // covers only that one source) leaves its entry behind, and the next
    // holder would adopt the dead one -- a use-after-free that the 2026-07-30
    // investigation found by reading, not by crashing.  Cheap and exact:
    // compare against the raw pointer we were handed.
    if(s_conCreating.empty() || (s_conCreating.back().get() != con)) {
        // Stale entries are BOTH dangling and owning — a constructor that
        // threw after its push had its memory freed by `new T` while the
        // entry kept a refcount — so popping them would double-free.
        // Neutralise by leaking the control block (a few dozen bytes on an
        // error path) and then fail loudly rather than adopting a stranger.
        while( !s_conCreating.empty() && (s_conCreating.back().get() != con)) {
            new shared_ptr<XQConnector>(std::move(s_conCreating.back()));
            s_conCreating.pop_back();
        }
        if(s_conCreating.empty())
            throw std::runtime_error(
                "XQConnectorHolder_: connector construction was abandoned "
                "(an exception escaped a connector constructor).");
    }
    m_connector = s_conCreating.back();
    s_conCreating.pop_back();
    connect(con->m_pWidget, SIGNAL( destroyed() ), this, SLOT( destroyed() ) );
    assert(con->shared_from_this());
}
XQConnectorHolder_::~XQConnectorHolder_() {
    if(m_connector)
        disconnect(m_connector->m_pWidget, SIGNAL( destroyed() ), this, SLOT( destroyed() ) );
}
bool
XQConnectorHolder_::isAlive() const {
    return !!m_connector;
}

void
XQConnectorHolder_::destroyed () {
	disconnect(m_connector->m_pWidget, SIGNAL( destroyed() ), this, SLOT( destroyed() ) );
	std::map<const QWidget*, weak_ptr<XNode> >::iterator it = s_widgetMap.find(m_connector->m_pWidget);
	assert(it != s_widgetMap.end());
	s_widgetMap.erase(it);
	m_connector->m_pWidget = 0L;
	m_connector.reset();
}

XQConnector::XQConnector(const shared_ptr<XNode> &node, QWidget *item)
	: QObject(),
	  m_pWidget(item)  {

    if( !node || !item)
        throw std::runtime_error("No valid node or item.");

    s_conCreating.push_back(shared_ptr<XQConnector>(this));

    node->iterate_commit([=](Transaction &tr){
    	m_lsnUIEnabled = tr[ *node].onUIFlagsChanged().connectWeakly(shared_from_this(), &XQConnector::onUIFlagsChanged,
            Listener::FLAG_MAIN_THREAD_CALL | Listener::FLAG_AVOID_DUP);
    });
    XQConnector::onUIFlagsChanged(Snapshot(*node), node.get());
    dbgPrint(QString("connector %1 created., addr=0x%2, size=0x%3")
			 .arg(node->getLabel())
			 .arg((uintptr_t)this, 0, 16)
			 .arg((uintptr_t)sizeof(XQConnector), 0, 16));
    
    auto ret = s_widgetMap.insert(std::pair<const QWidget*, weak_ptr<XNode> >(item, node));
    if( !ret.second)
    	gErrPrint("Connection to Widget Duplicated!");
}
XQConnector::~XQConnector() {
    if(isItemAlive()) {
        m_pWidget->setEnabled(false);
        dbgPrint(QString("connector %1 released., addr=0x%2").arg(objectName()).arg((uintptr_t)this, 0, 16));
    	auto it = s_widgetMap.find(m_pWidget);
    	assert(it != s_widgetMap.end());
    	s_widgetMap.erase(it);
    }
    else {
        dbgPrint(QString("connector %1 & widget released., addr=0x%2").arg(objectName()).arg((uintptr_t)this, 0, 16));
    }
}
//! Here rather than in driver.cpp, which includes no Qt at all: what a
//! driver's form IS is a question only the connector registry can answer.
void
XDriver::showForm(QWidget *w) {
    if(w) {
        w->showNormal();
        w->raise();
    }
}
void
XDriver::showForms() {
    showForm(XQConnector::windowOf( *this));
}
QWidget *
XQConnector::windowOf(const XNode &owner) {
    //isUpperOf, not a lookup that can throw: an O(1) containment predicate is
    //exactly the question here, and two drivers of the same type are told
    //apart by it, each form's widgets being bound to their own driver's nodes.
    Snapshot shot(owner);
    for(auto &&x: s_widgetMap) {
        shared_ptr<XNode> node = x.second.lock();
        if(node && shot.isUpperOf( *node))
            return const_cast<QWidget *>(x.first)->window();
    }
    return nullptr;
}
shared_ptr<XNode>
XQConnector::connectedNode(const QWidget *item) {
	auto it = s_widgetMap.find(item);
	if(it == s_widgetMap.end())
		return shared_ptr<XNode>();
	return it->second.lock();
}

void
XQConnector::onUIFlagsChanged(const Snapshot &shot, XNode *node) {
    m_pWidget->setEnabled(shot[ *node].isUIEnabled());
}

XQButtonConnector::XQButtonConnector(const shared_ptr<XTouchableNode> &node,
	QAbstractButton *item)
	: XQConnector(node, item),
	  m_node(node), m_pItem(item) {

    connect(item, SIGNAL( clicked() ), this, SLOT( onClick() ) );
    node->iterate_commit([=](Transaction &tr){
		m_lsnTouch = tr[ *node].onTouch().connectWeakly
            (shared_from_this(), &XQButtonConnector::onTouch, Listener::FLAG_MAIN_THREAD_CALL);
    });
}
XQButtonConnector::~XQButtonConnector() {
}
void
XQButtonConnector::onClick() {
    m_node->iterate_commit([=](Transaction &tr){
		tr[ *m_node].touch();
		tr.unmark(m_lsnTouch);
    });
}
void
XQButtonConnector::onTouch(const Snapshot &shot, XTouchableNode *node) {
}

XValueQConnector::XValueQConnector(const shared_ptr<XValueNodeBase> &node, QWidget *item)
	: XQConnector(node, item) {
    node->iterate_commit([=](Transaction &tr){
		m_lsnValueChanged = tr[ *node].onValueChanged().connectWeakly(
			shared_from_this(), &XValueQConnector::onValueChanged,
            Listener::FLAG_MAIN_THREAD_CALL | Listener::FLAG_AVOID_DUP);
    });
}
XValueQConnector::~XValueQConnector() {
}

XQLineEditConnector::XQLineEditConnector(
    const shared_ptr<XValueNodeBase> &node, QLineEdit *item, bool forcereturn)
	: XValueQConnector(node, item),
      m_node(node), m_pItem(item), m_editing(false) {
    connect(item, SIGNAL( returnPressed() ), this, SLOT( onReturnPressed() ) );
    if(forcereturn) {
        connect(item, SIGNAL( editingFinished() ), this, SLOT( onExit() ) );
        connect(item, SIGNAL( textEdited( const QString &) ),
                this, SLOT( onTextChanged(const QString &) ) );
    }
    else {
        connect(item, SIGNAL( textEdited( const QString &) ),
                this, SLOT( onTextChanged2(const QString &) ) );
    }
    onValueChanged(Snapshot( *node), node.get());
}
void
XQLineEditConnector::onTextChanged(const QString &text) {
	QPalette palette(m_pItem->palette());
    palette.setColor(QPalette::Text, QColor(95, 186, 200));
	m_pItem->setPalette(palette);
    m_editing = true;
}
void
XQLineEditConnector::onTextChanged2(const QString &text) {
	QPalette palette(m_pItem->palette());
    try {
        m_node->iterate_commit([=](Transaction &tr){
    		tr[ *m_node].str(text);
    		tr.unmark(m_lsnValueChanged);
        });
        palette.setColor(QPalette::Text, QApplication::palette(m_pItem).color(QPalette::Active, QPalette::Text));
    }
    catch (XKameError &e) {
    	palette.setColor(QPalette::Text, Qt::red);
    }
	m_pItem->setPalette(palette);
}
void
XQLineEditConnector::onReturnPressed() {
	QPalette palette(m_pItem->palette());
    try {
        m_node->iterate_commit([=](Transaction &tr){
			tr[ *m_node].str(m_pItem->text());
			tr.unmark(m_lsnValueChanged);
        });
        palette.setColor(QPalette::Text, QApplication::palette(m_pItem).color(QPalette::Active, QPalette::Text));
    }
    catch (XKameError &e) {
        e.print();
    	palette.setColor(QPalette::Text, Qt::red);
    }
	m_pItem->setPalette(palette);
    m_editing = false;
}
void
XQLineEditConnector::onExit() {
    if( !m_editing) return;
	QPalette palette(m_pItem->palette());
    palette.setColor(QPalette::Text, QApplication::palette(m_pItem).color(QPalette::Active, QPalette::Text));
    m_pItem->setPalette(palette);
	Snapshot shot( *m_node);
	if(QString(shot[ *m_node].to_str()) != m_pItem->text()) {
	    m_pItem->blockSignals(true);
	    m_pItem->setText(shot[ *m_node].to_str());
	    m_pItem->blockSignals(false);
		shared_ptr<XStatusPrinter> statusprinter = g_statusPrinter;
        if(statusprinter) statusprinter->printMessage(i18n("Input canceled."), true, 0, 0, true);
	}
}
void
XQLineEditConnector::onValueChanged(const Snapshot &shot, XValueNodeBase *node) {
	m_pItem->blockSignals(true);
	m_pItem->setText(shot[ *node].to_str());
	m_pItem->blockSignals(false);
}
  
template <class QN, class XN, class X>
XQSpinBoxConnectorTMPL<QN,XN,X>::XQSpinBoxConnectorTMPL(const shared_ptr<XN> &node,
    QN *item, QSlider *slider)
	: XValueQConnector(node, item),
      m_node(node),
	  m_pItem(item),
	  m_pSlider(slider) {
    if(slider) {
        if(std::is_integral<X>::value) {
            slider->setRange(item->minimum(), item->maximum());
            slider->setSingleStep(item->singleStep());
        }
        else {
            slider->setRange(0, 100);
            slider->setSingleStep(5);
        }
    }
}
XQSpinBoxConnector::XQSpinBoxConnector(const shared_ptr<XIntNode> &node,
    QSpinBox *item, QSlider *slider)
    : XQSpinBoxConnectorTMPL<QSpinBox, XIntNode, int>(node, item, slider) {
    connect(item, SIGNAL( valueChanged(int) ), this, SLOT( onChange(int) ) );
    if(slider) {
        connect(slider, SIGNAL( valueChanged(int) ), this, SLOT( onSliderChange(int) ) );
    }
    onValueChanged(Snapshot( *node), node.get());
}
XQSpinBoxUnsignedConnector::XQSpinBoxUnsignedConnector(const shared_ptr<XUIntNode> &node,
    QSpinBox *item, QSlider *slider)
    : XQSpinBoxConnectorTMPL<QSpinBox, XUIntNode, int>(node, item, slider) {
    connect(item, SIGNAL( valueChanged(int) ), this, SLOT( onChange(int) ) );
    if(slider) {
        connect(slider, SIGNAL( valueChanged(int) ), this, SLOT( onSliderChange(int) ) );
    }
    onValueChanged(Snapshot( *node), node.get());
}
XQDoubleSpinBoxConnector::XQDoubleSpinBoxConnector(const shared_ptr<XDoubleNode> &node,
    QDoubleSpinBox *item, QSlider *slider)
    : XQSpinBoxConnectorTMPL<QDoubleSpinBox, XDoubleNode, double>(node, item, slider) {
    connect(item, SIGNAL( valueChanged(double) ), this, SLOT( onChange(double) ) );
    if(slider) {
        connect(slider, SIGNAL( valueChanged(int) ), this, SLOT( onSliderChange(int) ) );
    }
    onValueChanged(Snapshot( *node), node.get());
}
template <class QN, class XN, class X>
void
XQSpinBoxConnectorTMPL<QN,XN,X>::onUIFlagsChanged(const Snapshot &shot, XNode *node) {
    if(m_pSlider)
        m_pSlider->setEnabled(shot[ *node].isUIEnabled());
    XQConnector::onUIFlagsChanged(shot, node);
}

template <class QN, class XN, class X>
void
XQSpinBoxConnectorTMPL<QN,XN,X>::onChangeTMPL(X val) {
    int var = val;
    if( !std::is_integral<X>::value) {
        double max = m_pItem->maximum();
        double min = m_pItem->minimum();
        var = lrint((val - min) / (max - min) * 100);
    }
    if(m_pSlider) {
        m_pSlider->blockSignals(true);
        m_pSlider->setValue(var);
        m_pSlider->blockSignals(false);
    }
    m_node->iterate_commit([=](Transaction &tr){
        tr[ *m_node] = val;
        tr.unmark(m_lsnValueChanged);
    });
}
template <class QN, class XN, class X>
void
XQSpinBoxConnectorTMPL<QN,XN,X>::onSliderChangeTMPL(int val) {
    X var = val;
    if( !std::is_integral<X>::value) {
        double max = m_pItem->maximum();
        double min = m_pItem->minimum();
        var = val / 100.0 * (max - min) + min;
    }
    m_pItem->blockSignals(true);
    m_pItem->setValue(var);
    m_pItem->blockSignals(false);
    m_node->iterate_commit([=](Transaction &tr){
        tr[ *m_node] = var;
        tr.unmark(m_lsnValueChanged);
    });
}
template <class QN, class XN, class X>
void
XQSpinBoxConnectorTMPL<QN,XN,X>::onValueChangedTMPL(const Snapshot &shot, XValueNodeBase *node) {
    X var = std::is_integral<X>::value ?
        QString(shot[ *node].to_str()).toInt() :
        QString(shot[ *node].to_str()).toDouble();
	m_pItem->blockSignals(true);
    m_pItem->setValue(var);
    m_pItem->blockSignals(false);
	if(m_pSlider) {
        int svar = var;
        if( !std::is_integral<X>::value) {
            double max = m_pItem->maximum();
            double min = m_pItem->minimum();
            svar = lrint((var - min) / (max - min) * 100);
        }
		m_pSlider->blockSignals(true);
        m_pSlider->setValue(svar);
        m_pSlider->blockSignals(false);
	}
}

template class XQSpinBoxConnectorTMPL<QSpinBox, XIntNode, int>;
template class XQSpinBoxConnectorTMPL<QSpinBox, XUIntNode, int>;
template class XQSpinBoxConnectorTMPL<QDoubleSpinBox, XDoubleNode, double>;

XFilePathConnector::XFilePathConnector(const shared_ptr<XStringNode> &node,
    QLineEdit *edit, QAbstractButton *btn, const char *filter, bool saving)
    : XQLineEditConnector(node, edit),
      m_pBtn(btn), m_filter(filter), m_saving(saving) {
    connect(btn, SIGNAL( clicked ( ) ), this, SLOT( onClick( ) ) );
}
void
XFilePathConnector::onClick() {
#if QT_VERSION < QT_VERSION_CHECK(5,0,0)
    XString str =
    m_saving ? QFileDialog::
        getSaveFileName(m_pItem, QString(), m_pItem->text(), m_filter)
     : QFileDialog::
        getOpenFileName(m_pItem, QString(), m_pItem->text(), m_filter);
#else
    //old qt cannot make native dialog in this mode.
    QFileDialog dialog(m_pItem);
    dialog.setViewMode(QFileDialog::Detail);
//    dialog.setConfirmOverwrite(false);
    // m_filter is a ";;"-separated *list* ("Data files (*.dat);;All files (*.*)").
    // setNameFilter() is singular and does NOT split on ";;": it took the whole
    // string as one filter, so the type combo showed the raw ";;" text as a single
    // garbled entry and only the trailing glob ("*.*") was ever applied.  The
    // native macOS/Windows helpers do not show the combo, which is why this
    // survived unnoticed until Qt's own widget dialog was used on Linux.
    QStringList filters;
    for(const QString &s: QString(m_filter).split(";;")) {
        QString t = s.trimmed();
        if(t.length())
            filters.push_back(t);
    }
    if( !filters.isEmpty())
        dialog.setNameFilters(filters);
    // Default suffix comes from the first extension of the *first* filter,
    // e.g. "Data files (*.dat)" -> "dat".  Skip it when that filter is a
    // catch-all ("*.*"), which has no meaningful suffix.
    if( !filters.isEmpty()) {
        const QString &head = filters.front();
        QString suf;
        for(int i = head.indexOf('.') + 1; (i > 0) && (i < head.length()); ++i) {
            if( !head.at(i).isLetterOrNumber())
                break;
            suf += head.at(i);
        }
        if(suf.length())
            dialog.setDefaultSuffix(suf);
    }
    // The line edit holds a *file* path, not a directory.  Handing it straight
    // to setDirectory() makes Qt's widget dialog root itself at a non-directory
    // and list nothing at all; the native helpers silently treated it as a
    // preselected file instead.  Do that split explicitly.
    QString curpath = m_pItem->text();
    //Nothing typed yet: start where a file was last browsed for rather than
    //in the process's working directory, which on a lab machine is wherever
    //KAME happened to be launched from.
    if( !curpath.length())
        dialog.setDirectory(kameLastDir("data"));
    if(curpath.length()) {
        QFileInfo fi(curpath);
        if(fi.isDir())
            dialog.setDirectory(fi.absoluteFilePath());
        else {
            if(QDir(fi.absolutePath()).exists())
                dialog.setDirectory(fi.absolutePath());
            if(fi.fileName().length())
                dialog.selectFile(fi.fileName());
        }
    }
    dialog.setAcceptMode(m_saving ? QFileDialog::AcceptSave: QFileDialog::AcceptOpen);
    if( !dialog.exec())
        return;
    if(dialog.selectedFiles().isEmpty())
        return;
    QString str = dialog.selectedFiles().at(0);
#endif
    if(str.length()) {
        kameStoreLastDir("data", str);
	    m_pItem->blockSignals(true);
        m_pItem->setText(str);
	    m_pItem->blockSignals(false);
	    try {
            m_node->iterate_commit([=](Transaction &tr){
            tr[ *m_node].str(str);
				tr.unmark(m_lsnValueChanged);
            });
	    }
	    catch (XKameError &e) {
		e.print();
	    }
     }
}

void
XFilePathConnector::onValueChanged(const Snapshot &shot, XValueNodeBase *node) {
    XQLineEditConnector::onValueChanged(shot, node);
}

XQLabelConnector::XQLabelConnector(const shared_ptr<XValueNodeBase> &node, QLabel *item)
	: XValueQConnector(node, item),
	  m_node(node), m_pItem(item) {
    onValueChanged(Snapshot( *node), node.get());
}

void
XQLabelConnector::onValueChanged(const Snapshot &shot, XValueNodeBase *node) {
    m_pItem->setText(shot[ *node].to_str());
}

XQTextEditConnector::XQTextEditConnector(const shared_ptr<XValueNodeBase> &node, QTextEdit *item)
	: XValueQConnector(node, item),
	  m_node(node), m_pItem(item) {
    onValueChanged(Snapshot( *node), node.get());
    if( !item->isReadOnly())
        connect(item, SIGNAL( textChanged() ),
                this, SLOT( onTextChanged() ) );
}
void
XQTextEditConnector::onValueChanged(const Snapshot &shot, XValueNodeBase *node) {
	m_pItem->setText(shot[ *node].to_str());
}
void
XQTextEditConnector::onTextChanged() {
    try {
        m_node->iterate_commit([=](Transaction &tr){
            tr[ *m_node].str(m_pItem->toPlainText());
            tr.unmark(m_lsnValueChanged);
        });
    }
    catch (XKameError &e) {
        e.print();
    }
}
  
XQLCDNumberConnector::XQLCDNumberConnector(const shared_ptr<XDoubleNode> &node, QLCDNumber *item)
	: XValueQConnector(node, item),
	  m_node(node), m_pItem(item) {
    //A readout, made to look like one: dark panel, bright digits, the same in
    //either theme.  A measured value should not change colour because the
    //window did.
    //
    //Flat, not Qt's default Outline: Outline draws the segments as edges in
    //QPalette::Light, washed out on a light window and all but invisible on a
    //dark one.  Flat fills them.
    //
    //Button and ButtonText, NOT Window/WindowText, and this is the whole
    //reason a first attempt did nothing: QLCDNumber's backgroundRole() is
    //Button and its foregroundRole() ButtonText -- measured by setting each
    //role in turn and counting the pixels of every colour the widget painted.
    //Left alone, the digits come out in whatever ButtonText is, which on a
    //dark theme is a thin grey line on the form's own background, with a
    //frame in Dark/Light that is invisible there as well: the reading a
    //driver form exists to show, hard to read in the theme KAME now starts in
    //(user, on TempControl, whose LCDs were already Flat in their .ui and so
    //were untouched by the style alone).
    item->setSegmentStyle(QLCDNumber::Flat);
    item->setAutoFillBackground(true);
    QPalette pal(item->palette());
    pal.setColor(QPalette::Button, QColor(0x0b, 0x10, 0x13));
    pal.setColor(QPalette::ButtonText, QColor(0x7c, 0xe4, 0xff));
    item->setPalette(pal);
    onValueChanged(Snapshot( *node), node.get());
}

void
XQLCDNumberConnector::onValueChanged(const Snapshot &shot, XValueNodeBase *node) {
    QString buf(shot[ *node].to_str());
    if((int)buf.length() > m_pItem->digitCount())
        m_pItem->setDigitCount(buf.length());
    m_pItem->display(buf);
    m_pItem->update(); //is this necessary?
}
  
//KAME_LED_BEGIN -- tools extract this verbatim; keep it self-contained.
//! An instrument LED, drawn rather than loaded.
//!
//! The two bitmaps it replaces were made in 2016 for a light window: a pale
//! blue disc lit and a pale grey one dark.  On a dark window both read as
//! bright smudges and they barely differ from each other, which is the whole
//! of "the LEDs are hard to see" -- and being bitmaps they were soft on every
//! display since.
//!
//! Lit is luminous: a white core inside the lamp's own colour, with a halo
//! outside it, so the eye reads light coming OUT of the thing rather than a
//! coloured circle.  Unlit is a lamp, not a hole -- a disc a shade away from
//! the window colour, whichever direction that has to be, with a rim to give
//! it an edge and a small highlight so the glass is still glass.  The lit
//! colour is deliberately not from the palette: it carries the meaning, and
//! it must not change with the theme.  Blue-cyan, as the old artwork was,
//! because these nodes are plain booleans -- green or red would promise a
//! good/bad reading that "Slipping" or "PCSHeater" does not have.
static QPixmap kameLedPixmap(bool on, const QPalette &pal, qreal dpr) {
    const qreal S = 16.0;
    QPixmap pm(QSize(int(S * dpr), int(S * dpr)));
    pm.setDevicePixelRatio(dpr);
    pm.fill(Qt::transparent);
    QPainter p( &pm);
    p.setRenderHint(QPainter::Antialiasing);
    QRectF lamp(2.0, 2.0, S - 4.0, S - 4.0);
    QPointF spec = lamp.center() - QPointF(lamp.width() * 0.20, lamp.height() * 0.24);
    QColor win = pal.color(QPalette::Window);
    bool darkui = (win.lightness() < 128);
    if(on) {
        QColor lit(0x2f, 0xb8, 0xff);
        QRadialGradient halo(lamp.center(), S / 2.0);
        halo.setColorAt(0.0, QColor(lit.red(), lit.green(), lit.blue(), darkui ? 170 : 120));
        halo.setColorAt(0.60, QColor(lit.red(), lit.green(), lit.blue(), darkui ? 70 : 45));
        halo.setColorAt(1.0, QColor(lit.red(), lit.green(), lit.blue(), 0));
        p.setPen(Qt::NoPen);
        p.setBrush(halo);
        p.drawEllipse(QRectF(0.0, 0.0, S, S));
        QRadialGradient body(spec, lamp.width() * 1.05, spec);
        body.setColorAt(0.0, QColor(255, 255, 255, 235));
        body.setColorAt(0.30, lit.lighter(125));
        body.setColorAt(1.0, lit.darker(150));
        p.setBrush(body);
        p.setPen(QPen(lit.darker(darkui ? 260 : 190), 1.0));
        p.drawEllipse(lamp);
    }
    else {
        //Away from the window colour in whichever direction is visible, so the
        //unlit lamp is never the background with a line around it.  Rendered
        //and looked at rather than reasoned about: at the 16 px this actually
        //ships at, the first attempt (lighter(190) on #1e1e1e) was a grey dot
        //that read as nothing being there, which for a panel is worse than
        //wrong -- an indicator has to be visibly PRESENT while it is off.
        QColor body = darkui ? win.lighter(260) : win.darker(112);
        QRadialGradient g(spec, lamp.width() * 1.1, spec);
        g.setColorAt(0.0, body.lighter(darkui ? 135 : 112));
        g.setColorAt(1.0, body.darker(darkui ? 120 : 130));
        p.setBrush(g);
        QColor rim = pal.color(QPalette::Mid);
        p.setPen(QPen(darkui ? rim.lighter(150) : rim, 1.0));
        p.drawEllipse(lamp);
    }
    //The glass, lit or not.
    QRectF hi(lamp.left() + lamp.width() * 0.22, lamp.top() + lamp.height() * 0.16,
        lamp.width() * 0.36, lamp.height() * 0.26);
    p.setPen(Qt::NoPen);
    p.setBrush(QColor(255, 255, 255, on ? 150 : (darkui ? 60 : 90)));
    p.drawEllipse(hi);
    return pm;
}
//KAME_LED_END

XQLedConnector::XQLedConnector(const shared_ptr<XBoolNode> &node, QPushButton *item)
	: XValueQConnector(node, item),
	  m_node(node), m_pItem(item) {
    item->setCheckable(false);
    item->setAutoDefault(false);
    item->setFlat(true);
    item->setFocusPolicy(Qt::NoFocus);
    item->setIconSize(QSize(16, 16));
    item->installEventFilter(this);
    onValueChanged(Snapshot( *node), node.get());
}

void
XQLedConnector::onValueChanged(const Snapshot &shot, XValueNodeBase *node) {
    m_lit = shot[ *m_node];
    updateIcon();
}
void
XQLedConnector::updateIcon() {
    m_pItem->setIcon(QIcon(kameLedPixmap(m_lit, m_pItem->palette(),
        m_pItem->devicePixelRatioF())));
}
bool
XQLedConnector::eventFilter(QObject *obj, QEvent *event) {
    //Half of the drawing comes from the palette, so it is redrawn when the
    //palette moves -- an appearance change reaches every widget as this.
    if((event->type() == QEvent::PaletteChange)
        || (event->type() == QEvent::ScreenChangeInternal))
        updateIcon();
    return XValueQConnector::eventFilter(obj, event);
}

XQToggleButtonConnector::XQToggleButtonConnector(const shared_ptr<XBoolNode> &node, QAbstractButton *item)
	: XValueQConnector(node, item),
	  m_node(node), m_pItem(item) {
    connect(item, SIGNAL( clicked() ), this, SLOT( onClick() ) );
    onValueChanged(Snapshot( *node), node.get());
    item->installEventFilter(this);
}

bool
XQToggleButtonConnector::eventFilter(QObject *obj, QEvent *event) {
    QCheckBox *cb = qobject_cast<QCheckBox *>(obj);
    if(cb) {
        //filtering event in inactive window.
        static QPoint s_pressPos;
        static XTime s_pressTime;
        if(event->type() == QEvent::WindowActivate) {
            s_pressPos = QCursor::pos();
            s_pressTime = XTime::now();
        }
        if(event->type() == QEvent::MouseButtonRelease) {
            if(XTime::now().diff_sec(s_pressTime) < 0.3) {
                QPoint releasePos = QCursor::pos();
                int distance = (releasePos - s_pressPos).manhattanLength();
                if(distance > QApplication::startDragDistance()) {
                    cb->window()->activateWindow();
                    cb->window()->raise();
                    gWarnPrint(i18n("Ignoring accidental touch on a checkbox."));
                    return true; //blocks.
                }
            }
        }
    }
    return QObject::eventFilter(obj, event);
}

void
XQToggleButtonConnector::onClick() {
    m_node->iterate_commit([=](Transaction &tr){
		tr[ *m_node] = m_pItem->isChecked();
		tr.unmark(m_lsnValueChanged);
    });
}

void
XQToggleButtonConnector::onValueChanged(const Snapshot &shot, XValueNodeBase *node) {
	if((shot[ *m_node]) ^ m_pItem->isChecked())
		m_pItem->toggle();
}

XListQConnector::XListQConnector(const shared_ptr<XListNodeBase> &node, QTableWidget *item)
	: XQConnector(node, item),
	  m_pItem(item), m_list(node) {

    node->iterate_commit([=](Transaction &tr){
	    m_lsnMove = tr[ *node].onMove().connectWeakly(shared_from_this(),
             &XListQConnector::onMove, Listener::FLAG_MAIN_THREAD_CALL);
	    m_lsnCatch = tr[ *node].onCatch().connectWeakly(shared_from_this(),
            &XListQConnector::onCatch, Listener::FLAG_MAIN_THREAD_CALL);
	    m_lsnRelease = tr[ *node].onRelease().connectWeakly(shared_from_this(),
            &XListQConnector::onRelease, Listener::FLAG_MAIN_THREAD_CALL);
    });
    QHeaderView *header = m_pItem->verticalHeader();
#if QT_VERSION  < QT_VERSION_CHECK(5,0,0)
    header->setMovable(true);
    header->setResizeMode(QHeaderView::ResizeToContents); //QHeaderView::Fixed
#else
    header->setSectionsMovable(true);
    header->setSectionResizeMode(QHeaderView::ResizeToContents);
#endif
    connect(header, SIGNAL( sectionMoved(int, int, int)),
            this, SLOT( OnSectionMoved(int, int, int)));
    header->setToolTip(i18n("Use drag-n-drop to reorder."));
}
XListQConnector::~XListQConnector() {
    if(isItemAlive()) {
        QHeaderView *header = m_pItem->verticalHeader();
        disconnect(header, nullptr, this, nullptr);
        m_pItem->clearSpans();
    }
}
void
XListQConnector::OnSectionMoved(int logicalIndex, int oldVisualIndex, int newVisualIndex) {
    int fromIndex = oldVisualIndex;
    int toIndex = newVisualIndex;
    if(toIndex == fromIndex)
        toIndex++;
    m_list->iterate_commit([=](Transaction &tr){
        unsigned int src = fromIndex;
        unsigned int dst = toIndex;
		if( !tr.size() || src >= tr.size() || (dst >= tr.size())) {
//			gErrPrint(i18n("Invalid range of selections."));
			return;
		}
        for(int i = src; i != dst;) {
            int next = i + ((src < dst) ? 1: -1);
            m_list->swap(tr, tr.list()->at(i), tr.list()->at(next));
            i = next;
        }
		tr.unmark(m_lsnMove);
    });
}
void
XListQConnector::onMove(const Snapshot &shot, const XListNodeBase::Payload::MoveEvent &e) {
    QHeaderView *header = m_pItem->verticalHeader();
    int dir = (e.src_idx - e.dst_idx > 0) ? 1 : -1;
    for(unsigned int idx = e.dst_idx; idx != e.src_idx; idx += dir) {
        header->swapSections(idx, idx + dir);
    }
}

XItemQConnector::XItemQConnector(const shared_ptr<XItemNodeBase> &node, QWidget *item)
	: XValueQConnector(node, item) {
    node->iterate_commit([=](Transaction &tr){
	    m_lsnListChanged = tr[ *node].onListChanged().connectWeakly(shared_from_this(),
	    	&XItemQConnector::onListChanged,
            Listener::FLAG_MAIN_THREAD_CALL | Listener::FLAG_AVOID_DUP);
    });
}
XItemQConnector::~XItemQConnector() {
}

XQComboBoxConnector::XQComboBoxConnector(const shared_ptr<XItemNodeBase> &node,
										 QComboBox *item, const Snapshot &shot_of_list)
	: XItemQConnector(node, item),
	  m_node(node), m_pItem(item) {
    connect(item, SIGNAL( activated(int) ), this, SLOT( onSelect(int) ) );
    onListChanged(Snapshot( *node),
        XItemNodeBase::Payload::ListChangeEvent({shot_of_list, node.get()}));
    item->installEventFilter(this);
}
bool
XQComboBoxConnector::eventFilter(QObject *obj, QEvent *event) {
    //QComboBox does not feature aboutToShow(), instead, filtering events.
    if(obj == m_pItem) {
        bool popup = false;
        switch (event->type()) {
        case QEvent::KeyPress: {
            auto *keyEvent = static_cast<QKeyEvent *>(event);
            if(keyEvent->key() == Qt::Key_Space ||
                (keyEvent->key() == Qt::Key_Down && keyEvent->modifiers() & Qt::AltModifier))
                popup = true;
            }
            break;
        case QEvent::MouseButtonPress:
            popup = true;
            break;
        default:
            break;
        }
        if(popup) {
            Snapshot shot( *m_node);
            shot.talk(shot[ *m_node].onItemRefreshRequested(), m_node.get());
        }
    }
    return QObject::eventFilter(obj, event);
}
void
XQComboBoxConnector::onSelect(int idx) {
    try {
        m_node->iterate_commit([=](Transaction &tr){
            if( (idx >= m_itemStrings.size()) || (idx < 0))
	            tr[ *m_node].str(XString());
	        else
                tr[ *m_node].str(m_itemStrings.at(idx).label);
			tr.unmark(m_lsnValueChanged);
        });
    }
    catch (XKameError &e) {
        e.print();
    }
}

int
XQComboBoxConnector::findItem(const QString &text) {
	for(int i = 0; i < m_pItem->count(); i++) {
        if(text == m_pItem->itemText(i)) {
            return i;
        }
	}
	return -1;
}

void
XQComboBoxConnector::onValueChanged(const Snapshot &shot, XValueNodeBase *node) {
	m_pItem->blockSignals(true);
	QString str = shot[ *node].to_str();
	int idx = -1;
	int i = 0;
    for(auto it = m_itemStrings.begin(); it != m_itemStrings.end(); it++) {
        if(QString(it->label) == str) {
            idx = i;
        }
        i++;
	}
	if(idx >= 0) {
		m_pItem->setCurrentIndex(idx);
		if(m_node->autoSetAny()) {
			int idx1 = findItem(i18n("(UNSEL)"));
			if(idx1 >= 0) {
	            m_pItem->removeItem(idx1);
			}
		}
	}
	else {
		if(m_node->autoSetAny()) {
			int idx = findItem(i18n("(UNSEL)"));
			if(idx < 0) {
	            m_pItem->addItem(i18n("(UNSEL)"));
			}
		}
		int idx = findItem(i18n("(UNSEL)"));
		assert(idx >= 0);
		m_pItem->setCurrentIndex(idx);
	}
	m_pItem->blockSignals(false);
}
void
XQComboBoxConnector::onListChanged(const Snapshot &shot, const XItemNodeBase::Payload::ListChangeEvent &e) {
	m_itemStrings = m_node->itemStrings(e.shot_of_list);
	m_pItem->clear();
	bool exist = false;
    for(auto it = m_itemStrings.begin(); it != m_itemStrings.end(); it++) {
        if(it->label.empty()) {
            m_pItem->addItem(i18n("(NO NAME)"));
        }
        else {
            m_pItem->addItem(QString(it->label));
            exist = true;
        }
	}
	if( !m_node->autoSetAny())
		m_pItem->addItem(i18n("(UNSEL)"));
    onValueChanged(shot, e.emitter);
}

XQListWidgetConnector::XQListWidgetConnector(const shared_ptr<XItemNodeBase> &node,
    QListWidget *item, const Snapshot &shot_of_list)
	: XItemQConnector(node, item),
	  m_node(node), m_pItem(item) {
    connect(item, SIGNAL( itemSelectionChanged() ),
            this, SLOT( OnItemSelectionChanged() ) );
    item->setMovement(QListView::Static);
    item->setSelectionBehavior(QAbstractItemView::SelectRows);
    item->setSelectionMode(QAbstractItemView::SingleSelection);
    onListChanged(Snapshot( *node),
        XItemNodeBase::Payload::ListChangeEvent({shot_of_list, node.get()}));
}
XQListWidgetConnector::~XQListWidgetConnector() {
    if(isItemAlive()) {
        m_pItem->clear();
    }
}
void
XQListWidgetConnector::OnItemSelectionChanged() {
    int idx = m_pItem->currentRow();
    try {
        m_node->iterate_commit([=](Transaction &tr){
            if((idx >= m_itemStrings.size()) || (idx < 0))
	            tr[ *m_node].str(XString());
	        else
                tr[ *m_node].str(m_itemStrings.at(idx).label);
			tr.unmark(m_lsnValueChanged);
        });
    }
    catch (XKameError &e) {
        e.print();
    }
}
void
XQListWidgetConnector::onValueChanged(const Snapshot &shot, XValueNodeBase *node) {
    QString str = shot[ *node].to_str();
	m_pItem->blockSignals(true);
	unsigned int i = 0;
    for(auto it = m_itemStrings.begin(); it != m_itemStrings.end(); it++) {
        if(str == QString(it->label))
            m_pItem->setCurrentRow(i);
        i++;
	}
	m_pItem->blockSignals(false);
}
void
XQListWidgetConnector::onListChanged(const Snapshot &shot, const XItemNodeBase::Payload::ListChangeEvent &e) {
	m_itemStrings = m_node->itemStrings(e.shot_of_list);
	m_pItem->clear();
    for(auto it = m_itemStrings.begin(); it != m_itemStrings.end(); it++) {
        new QListWidgetItem(it->label, m_pItem);
	}
    onValueChanged(shot, e.emitter);
}

XColorConnector::XColorConnector(const shared_ptr<XHexNode> &node, QPushButton *item)
	: XValueQConnector(node, item),
	  m_node(node), m_pItem(item) {
    connect(item, SIGNAL( clicked() ), this, SLOT( onClick() ) );
    item->setAutoDefault(false);
    item->setDefault(false);
    onValueChanged(Snapshot( *node), node.get());
}
void
XColorConnector::onClick() {
    auto dialog = m_dialog;
    if( !dialog) {
        dialog.reset(new QColorDialog(m_pItem));
        m_dialog = dialog;
    }
    disconnect( &*dialog, SIGNAL( colorSelected(const QColor &) ), this, SLOT( OnColorSelected(const QColor &) ) );
    connect( &*dialog, SIGNAL( colorSelected(const QColor &) ), this, SLOT( OnColorSelected(const QColor &) ) );
    Snapshot shot( *m_node);
    dialog->setCurrentColor(QColor((QRgb)(unsigned int)shot[ *m_node]));
    dialog->open();
}
void
XColorConnector::OnColorSelected(const QColor & color) {
    m_node->iterate_commit([=](Transaction &tr){
        tr[ *m_node] = color.rgb();
    });
}
void
XColorConnector::onValueChanged(const Snapshot &shot, XValueNodeBase *) {
    QColor color((QRgb)(unsigned int)shot[ *m_node]);
    auto dialog = m_dialog;
    if(dialog)
        dialog->setCurrentColor(color);
    QPixmap pixmap(m_pItem->size());
    pixmap.fill(color);
    m_pItem->setIcon(pixmap);
}

XStatusPrinter::XStatusPrinter(QMainWindow *window) {
    if( !window)
        window = dynamic_cast<QMainWindow*>(g_pFrmMain);
    m_pWindow = window;
    m_pBar = window ? (window->statusBar()) : nullptr;
    s_statusPrinterCreating.push_back(shared_ptr<XStatusPrinter>(this));
    if(m_pBar) m_pBar->hide();
    m_lsn = m_tlkTalker.connectWeakly(
        shared_from_this(), &XStatusPrinter::print,
        Listener::FLAG_MAIN_THREAD_CALL);
}
XStatusPrinter::~XStatusPrinter() {
}
shared_ptr<XStatusPrinter>
XStatusPrinter::create(QMainWindow *window) {
    // Same pairing hazard as XQConnectorHolder_ above: the constructor
    // pushes shared_ptr(this) and this pops it, so a throw in between would
    // hand the next caller someone else's entry.  Pop by identity.
    XStatusPrinter *raw = new XStatusPrinter(window);
    while( !s_statusPrinterCreating.empty()
            && (s_statusPrinterCreating.back().get() != raw)) {
        //Dangling-and-owning; leak rather than double-free.  See above.
        new shared_ptr<XStatusPrinter>(std::move(s_statusPrinterCreating.back()));
        s_statusPrinterCreating.pop_back();
    }
    if(s_statusPrinterCreating.empty())
        throw std::runtime_error(
            "XStatusPrinter::create: construction was abandoned.");
    shared_ptr<XStatusPrinter> ptr = s_statusPrinterCreating.back();
    s_statusPrinterCreating.pop_back();
    return ptr;
}
void
XStatusPrinter::printMessage(const XString &str, bool popup, const char *file, int line, bool beep) {
	tstatus status;
	status.ms = 3000;
	status.str = str;
    if(file) status.tooltip = i18n_noncontext("Message emitted at %1:%2").arg(file).arg(line);
    status.popup = popup;
    status.beep = beep;
    status.type = tstatus::Normal;
    m_tlkTalker.talk(std::move(status));
}
void
XStatusPrinter::printWarning(const XString &str, bool popup, const char *file, int line, bool beep) {
	tstatus status;
	status.ms = 3000;
    status.str = XString(i18n_noncontext("Warning: ")) + str;
    if(file) status.tooltip = i18n_noncontext("Warning emitted at %1:%2").arg(file).arg(line);
    status.popup = popup;
    status.beep = beep;
    status.type = tstatus::Warning;
    m_tlkTalker.talk(std::move(status));
}
void
XStatusPrinter::printError(const XString &str, bool popup, const char *file, int line, bool beep) {
	tstatus status;
	status.ms = 5000;
    status.str = XString(i18n_noncontext("Error: ")) + str;
    if(file) status.tooltip = i18n_noncontext("Error emitted at %1:%2").arg(file).arg(line);
    status.popup = popup;
    status.beep = beep;
    status.type = tstatus::Error;
    m_tlkTalker.talk(std::move(status));
}
void
XStatusPrinter::clear(void) {
	tstatus status;
	status.ms = 0;
	status.str = "";
    m_tlkTalker.talk(std::move(status));
}

void
XStatusPrinter::print(const tstatus &status) {
	bool popup = status.popup;
    QString str = std::move(status.str);
	if(status.ms) {
        if(m_pBar) {
            m_pBar->show();
            m_pBar->showMessage(str, status.ms);
        }
        if(status.beep)
            QApplication::beep();

        QPixmap *icon;
        switch(status.type) {
        case tstatus::Normal:
        default:
            icon = g_pIconInfo;
            break;
        case tstatus::Warning:
            icon = g_pIconWarn;
            break;
        case tstatus::Error:
            icon = g_pIconError;
            break;
        }
        if(popup || !m_pBar)
            XMessageBox::post(str, QIcon( *icon), popup, status.ms, status.tooltip);
    }
    else if(m_pBar) {
		m_pBar->hide();
		m_pBar->clearMessage();
	}
}
