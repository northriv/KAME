/***************************************************************************
		Copyright (C) 2002-2010 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
 ***************************************************************************/
//---------------------------------------------------------------------------

#ifndef xnodeconnectorH
#define xnodeconnectorH

#include <support.h>
#include <QObject>

#include <xsignal.h>

class QWidget;

void _sharedPtrQDeleter(QObject *);

template <class T>
class qshared_ptr : public shared_ptr<T> {
public:
    qshared_ptr() : shared_ptr<T>() {}
    template <class Y>
    qshared_ptr(const qshared_ptr<Y> &p)
        : shared_ptr<T>(static_cast<const shared_ptr<Y> &>(p) ) {}
    template <class Y>
    explicit qshared_ptr(Y * p)
        : shared_ptr<T>(p, _sharedPtrQDeleter) {
		ASSERT(isMainThread());
	}
    template <class Y>
    qshared_ptr<T> &operator=(const qshared_ptr<Y> &p) {
        shared_ptr<T>::operator=(p);
        return *this;
    }
};

class XQConnector;

class _XQConnectorHolder : public QObject {
	Q_OBJECT
public:
	_XQConnectorHolder(XQConnector *con);
	~_XQConnectorHolder();
	bool isAlive() const;
private slots:
		protected slots:
void destroyed ();
protected:
	shared_ptr<XQConnector> m_connector;
public:
};

typedef qshared_ptr<_XQConnectorHolder> xqcon_ptr;

#include <xnodeconnector_prv.h>

#include <xnode.h>
#include <xlistnode.h>
#include <xitemnode.h>

#include <fstream>

#include <QColor>
#include <QPoint>
#include <QTimer>

//! Needed for making new forms.
extern QWidget *g_pFrmMain;

//! Providing an easy access to make a new form with UIs designed by Qt designer.
template <class FRM, class UI>
struct QForm : public FRM, public UI {
	QForm() : FRM(), UI() {this->setupUi(this);}
	template <typename A>
	explicit QForm(A a) : FRM(a), UI() {this->setupUi(this);}
	template <typename A, typename B>
	QForm(A a, B b) : FRM(a, b), UI() {this->setupUi(this);}
	template <typename A, typename B, typename C>
	QForm(A a, B b, C c) : FRM(a, b, c), UI() {this->setupUi(this);}
};

//! Associate QWidget to XNode.
//! use connectWeak() to make XListener.
//! use xqcon_create<T>() to make instances.
//! \sa xqcon_create()
class XQConnector : public QObject,
public enable_shared_from_this<XQConnector> {
	//! Don't forget this macro for XQConnector objects.
	Q_OBJECT
public:
	//! Don't use this stuff directly, use xqcon_create() instead
	//! \sa xqcon_create()
	XQConnector(const shared_ptr<XNode> &node, QWidget *item);
	//! Disconnect all signals & slots
	virtual ~XQConnector();

	static shared_ptr<XNode> connectedNode(const QWidget *item);
private slots:
protected slots:
protected:
	friend class _XQConnectorHolder;
	bool isItemAlive() const {return m_pWidget;}
	shared_ptr<XListener> m_lsnUIEnabled;
	void onUIFlagsChanged(const Snapshot &shot, XNode *node);
	QWidget *m_pWidget;
};

class QAbstractButton;

class XQButtonConnector : public XQConnector {
	Q_OBJECT
public:
	XQButtonConnector(const shared_ptr<XNode> &node, QAbstractButton *item);
	virtual ~XQButtonConnector();
private slots:
protected slots:
virtual void onClick();
protected:
	virtual void onTouch(const shared_ptr<XNode> &node);
	shared_ptr<XListener> m_lsnTouch;
	const shared_ptr<XNode> m_node;
	QAbstractButton *const m_pItem;
};

class XValueQConnector : public XQConnector {
	Q_OBJECT
public:
	XValueQConnector(const shared_ptr<XValueNodeBase> &node, QWidget *item);
	virtual ~XValueQConnector();
private slots:
protected:
	shared_ptr<XListener> m_lsnBeforeValueChanged;
	shared_ptr<XListener> m_lsnValueChanged;
	virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &node) = 0;
	virtual void onValueChanged(const shared_ptr<XValueNodeBase> &node) = 0;
};

class QLineEdit;

class XQLineEditConnector : public XValueQConnector {
	Q_OBJECT
public:
	XQLineEditConnector(const shared_ptr<XValueNodeBase> &node,
		QLineEdit *item, bool forcereturn = true);
	virtual ~XQLineEditConnector() {}
protected slots:
void onTextChanged(const QString &);
void onTextChanged2(const QString &);
void onReturnPressed();
void onExit();
protected:
	virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
	virtual void onValueChanged(const shared_ptr<XValueNodeBase> &);
	const shared_ptr<XValueNodeBase> m_node;
	QLineEdit *const m_pItem;
};

class Q3TextBrowser;

class XQTextBrowserConnector : public XValueQConnector {
	Q_OBJECT
public:
	XQTextBrowserConnector(const shared_ptr<XValueNodeBase> &node,
		Q3TextBrowser *item);
	virtual ~XQTextBrowserConnector() {}
protected slots:
protected:
	virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
	virtual void onValueChanged(const shared_ptr<XValueNodeBase> &);
	const shared_ptr<XValueNodeBase> m_node;
	Q3TextBrowser *const m_pItem;
};

class QSpinBox;
class QSlider;

class XQSpinBoxConnector : public XValueQConnector {
	Q_OBJECT
public:
	XQSpinBoxConnector(const shared_ptr<XIntNode> &node,
		QSpinBox *item, QSlider *slider = 0L);
	XQSpinBoxConnector(const shared_ptr<XUIntNode> &node,
		QSpinBox *item, QSlider *slider = 0L);
	virtual ~XQSpinBoxConnector() {}
protected slots:
void onChange(int val);
protected:
	virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
	virtual void onValueChanged(const shared_ptr<XValueNodeBase> &node);
	const shared_ptr<XIntNode> m_iNode;
	const shared_ptr<XUIntNode> m_uINode;
	QSpinBox *const m_pItem;
	QSlider *const m_pSlider;
};

class KDoubleNumInput;

class XKDoubleNumInputConnector : public XValueQConnector {
	Q_OBJECT
public:
	XKDoubleNumInputConnector(const shared_ptr<XDoubleNode> &node, 
		KDoubleNumInput *item);
	virtual ~XKDoubleNumInputConnector() {}
protected slots:
void onChange(double val);
protected:
	virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
	virtual void onValueChanged(const shared_ptr<XValueNodeBase> &node);
	const shared_ptr<XDoubleNode> m_node;
	KDoubleNumInput *const m_pItem;
};

class QDoubleSpinBox;

class XQDoubleSpinBoxConnector : public XValueQConnector {
	Q_OBJECT
public:
	XQDoubleSpinBoxConnector(const shared_ptr<XDoubleNode> &node,
		QDoubleSpinBox *item);
	virtual ~XQDoubleSpinBoxConnector() {}
protected slots:
void onChange(double val);
protected:
	virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
	virtual void onValueChanged(const shared_ptr<XValueNodeBase> &node);
	const shared_ptr<XDoubleNode> m_node;
	QDoubleSpinBox *const m_pItem;
};

class KUrlRequester;
class KUrl;
class XKURLReqConnector : public XValueQConnector {
	Q_OBJECT
public:
	XKURLReqConnector(const shared_ptr<XStringNode> &node, 
		KUrlRequester *item, const char *filter, bool saving);
	virtual ~XKURLReqConnector() {}
protected slots:
void onSelect( const KUrl& );
protected:
	virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
	virtual void onValueChanged(const shared_ptr<XValueNodeBase> &node);
	const shared_ptr<XStringNode> m_node;
	KUrlRequester *const m_pItem;
};

class QLabel;

class XQLabelConnector : public XValueQConnector {
	Q_OBJECT
public:
	XQLabelConnector(const shared_ptr<XValueNodeBase> &node, 
		QLabel *item);
	virtual ~XQLabelConnector() {}
protected slots:
protected:
	virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
	virtual void onValueChanged(const shared_ptr<XValueNodeBase> &node);
	const shared_ptr<XValueNodeBase> m_node;
	QLabel *const m_pItem;
};

class KLed;

class XKLedConnector : public XValueQConnector {
	Q_OBJECT
public:
	XKLedConnector(const shared_ptr<XBoolNode> &node,
		KLed *item);
	virtual ~XKLedConnector() {}
protected slots:
protected:
	virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
	virtual void onValueChanged(const shared_ptr<XValueNodeBase> &node);
	const shared_ptr<XBoolNode> m_node;
	KLed *const m_pItem;
};

class QLCDNumber;

class XQLCDNumberConnector : public XValueQConnector {
	Q_OBJECT
public:
	XQLCDNumberConnector(const shared_ptr<XDoubleNode> &node,
		QLCDNumber *item);
	virtual ~XQLCDNumberConnector() {}
protected:
	virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
	virtual void onValueChanged(const shared_ptr<XValueNodeBase> &node);
	const shared_ptr<XDoubleNode> m_node;
	QLCDNumber *const m_pItem;
};

class XQToggleButtonConnector : public XValueQConnector {
	Q_OBJECT
public:
	XQToggleButtonConnector(const shared_ptr<XBoolNode> &node,
		QAbstractButton *item);
	virtual ~XQToggleButtonConnector() {}
protected slots:
void onClick();
protected:
	virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
	virtual void onValueChanged(const shared_ptr<XValueNodeBase> &node);
	const shared_ptr<XBoolNode> m_node;
	QAbstractButton *const m_pItem;
};

class Q3Table;
class XListQConnector : public XQConnector {
	Q_OBJECT
public:
	XListQConnector(const shared_ptr<XListNodeBase> &node, Q3Table *item);
	virtual ~XListQConnector();
private slots:
protected slots:
void indexChange(int section, int fromIndex, int toIndex);
protected:
	shared_ptr<XListener> m_lsnMove;
	virtual void onMove(const Snapshot &shot, const XListNodeBase::Payload::MoveEvent &e);
	shared_ptr<XListener> m_lsnCatch;
	shared_ptr<XListener> m_lsnRelease;
	virtual void onCatch(const Snapshot &shot, const XListNodeBase::Payload::CatchEvent &e) = 0;
	virtual void onRelease(const Snapshot &shot, const XListNodeBase::Payload::ReleaseEvent &e) = 0;
	Q3Table *const m_pItem;
	const shared_ptr<XListNodeBase> m_list;
};

class XItemQConnector : public XValueQConnector {
	Q_OBJECT
public:
	XItemQConnector(const shared_ptr<XItemNodeBase> &node,
		QWidget *item);
	virtual ~XItemQConnector();
private slots:
protected slots:
protected:
	shared_ptr<XListener>  m_lsnListChanged;
	virtual void onListChanged(const Snapshot &shot, XItemNodeBase *) = 0;
	shared_ptr<const std::deque<XItemNodeBase::Item> > m_itemStrings;
};

class QComboBox;

class XQComboBoxConnector : public XItemQConnector {
	Q_OBJECT
public:
	XQComboBoxConnector(const shared_ptr<XItemNodeBase> &node,
		QComboBox *item, const Snapshot &shot_of_list);
	virtual ~XQComboBoxConnector() {}
protected slots:
virtual void onSelect(int index);
protected:
	virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
	virtual void onValueChanged(const shared_ptr<XValueNodeBase> &node);
	virtual void onListChanged(const Snapshot &shot, XItemNodeBase *);
	const shared_ptr<XItemNodeBase> m_node;
	QComboBox *const m_pItem;
	int findItem(const QString &);
};

class Q3ListBox;

class XQListBoxConnector : public XItemQConnector {
	Q_OBJECT
public:
	XQListBoxConnector(const shared_ptr<XItemNodeBase> &node,
		Q3ListBox *item, const Snapshot &shot_of_list);
	virtual ~XQListBoxConnector() {}
protected slots:
virtual void onSelect(int index);
protected:
	virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
	virtual void onValueChanged(const shared_ptr<XValueNodeBase> &node);
	virtual void onListChanged(const Snapshot &shot, XItemNodeBase *);
	const shared_ptr<XItemNodeBase> m_node;
	Q3ListBox *const m_pItem;
};

class KColorButton;

class XKColorButtonConnector : public XValueQConnector {
	Q_OBJECT
public:
	XKColorButtonConnector(const shared_ptr<XHexNode> &node,
		KColorButton *item);
	virtual ~XKColorButtonConnector() {}
protected slots:
void onClick(const QColor &newColor);
protected:
	virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
	virtual void onValueChanged(const shared_ptr<XValueNodeBase> &);
	const shared_ptr<XHexNode> m_node;
	KColorButton *const m_pItem;
};

class KColorCombo;

class XKColorComboConnector : public XValueQConnector {
	Q_OBJECT
public:
	XKColorComboConnector(const shared_ptr<XHexNode> &node, 
		KColorCombo *item);
	virtual ~XKColorComboConnector() {}
protected slots:
void onClick(const QColor &newColor);
protected:
	virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
	virtual void onValueChanged(const shared_ptr<XValueNodeBase> &);
	const shared_ptr<XHexNode> m_node;
	KColorCombo *const m_pItem;
};

//! Show status
class QMainWindow;
class QStatusBar;
class KPassivePopup;
class XStatusPrinter : public enable_shared_from_this<XStatusPrinter> {
protected:
	explicit XStatusPrinter(QMainWindow *window = NULL);
public:
	static shared_ptr<XStatusPrinter> create(QMainWindow *window = NULL);
	~XStatusPrinter();
	void printMessage(const XString &str, bool popup = true);
	void printWarning(const XString &str, bool popup = false);
	void printError(const XString &str, bool popup = true);
	void clear();
private:
	struct tstatus {XString str; int ms; bool popup; enum {Normal, Warning, Error} type;};
	XTalker<tstatus> m_tlkTalker;
	shared_ptr<XListener> m_lsn;
	QMainWindow *m_pWindow;
	QStatusBar *m_pBar;
	KPassivePopup *m_pPopup;
	void print(const tstatus &status);
};

//---------------------------------------------------------------------------
#endif
