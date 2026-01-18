/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
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

#include "support.h"
#include <QObject>
#include "transaction_signal.h"
using Transactional::isMainThread;

class QWidget;

DECLSPEC_KAME void sharedPtrQDeleter_(QObject *);

template <class T>
class qshared_ptr : public shared_ptr<T> {
public:
    qshared_ptr() : shared_ptr<T>() {}
    template <class Y>
    qshared_ptr(const qshared_ptr<Y> &p)
        : shared_ptr<T>(static_cast<const shared_ptr<Y> &>(p) ) {}
    template <class Y>
    explicit qshared_ptr(Y * p)
        : shared_ptr<T>(p, sharedPtrQDeleter_) {
		assert(isMainThread());
	}
    template <class Y>
    qshared_ptr<T> &operator=(const qshared_ptr<Y> &p) {
        shared_ptr<T>::operator=(p);
        return *this;
    }
};

class XQConnector;

class DECLSPEC_KAME XQConnectorHolder_ : public QObject {
	Q_OBJECT
public:
	XQConnectorHolder_(XQConnector *con);
	~XQConnectorHolder_();
	bool isAlive() const;
private slots:
		protected slots:
void destroyed ();
protected:
	shared_ptr<XQConnector> m_connector;
public:
};

using xqcon_ptr = qshared_ptr<XQConnectorHolder_>;

//! function for creating XQConnector instances
template <class T, class A, class B, typename...Args>
xqcon_ptr xqcon_create(const shared_ptr<A> &a, B *b, Args&&...args) {
    return xqcon_ptr(new XQConnectorHolder_( new T(a, b, std::forward<Args>(args)...)));
}

#include "xnode.h"
#include "xlistnode.h"
#include "xitemnode.h"

#include <fstream>

#include <QColor>
#include <QPoint>
#include <QTimer>

//! Needed for making new forms.
extern DECLSPEC_KAME QWidget *g_pFrmMain;

//! Providing an easy access to make a new form with UIs designed by Qt designer.
template <class FRM, class UI>
struct QForm : public FRM, public UI {
    template <typename...Args>
    QForm(Args&&...args) : FRM(std::forward<Args>(args)...), UI() {this->setupUi(this);}
};

//! Associate QWidget to XNode.
//! use connectWeak() to make Listener.
//! use xqcon_create<T>() to make instances.
//! \sa xqcon_create()
class DECLSPEC_KAME XQConnector : public QObject,
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
	friend class XQConnectorHolder_;
	bool isItemAlive() const {return m_pWidget;}
    shared_ptr<Listener> m_lsnUIEnabled;
    virtual void onUIFlagsChanged(const Snapshot &shot, XNode *node);
	QWidget *m_pWidget;
};

class QAbstractButton;

class DECLSPEC_KAME XQButtonConnector : public XQConnector {
	Q_OBJECT
public:
	XQButtonConnector(const shared_ptr<XTouchableNode> &node, QAbstractButton *item);
	virtual ~XQButtonConnector();
private slots:
protected slots:
virtual void onClick();
protected:
	virtual void onTouch(const Snapshot &shot, XTouchableNode *node);
    shared_ptr<Listener> m_lsnTouch;
	const shared_ptr<XTouchableNode> m_node;
	QAbstractButton *const m_pItem;
};

class DECLSPEC_KAME XValueQConnector : public XQConnector {
	Q_OBJECT
public:
	XValueQConnector(const shared_ptr<XValueNodeBase> &node, QWidget *item);
	virtual ~XValueQConnector();
private slots:
protected:
    shared_ptr<Listener> m_lsnValueChanged;
	virtual void onValueChanged(const Snapshot &shot, XValueNodeBase *node) = 0;
};

class QLineEdit;

class DECLSPEC_KAME XQLineEditConnector : public XValueQConnector {
	Q_OBJECT
public:
	XQLineEditConnector(const shared_ptr<XValueNodeBase> &node,
		QLineEdit *item, bool forcereturn = true);
protected slots:
void onTextChanged(const QString &);
void onTextChanged2(const QString &);
void onReturnPressed();
void onExit();
protected:
    virtual void onValueChanged(const Snapshot &shot, XValueNodeBase *node) override;
	const shared_ptr<XValueNodeBase> m_node;
	QLineEdit *const m_pItem;
    bool m_editing;
};

class QTextEdit;

class DECLSPEC_KAME XQTextEditConnector : public XValueQConnector {
	Q_OBJECT
public:
    XQTextEditConnector(const shared_ptr<XValueNodeBase> &node,
        QTextEdit *item);
protected slots:
    void onTextChanged();
protected:
    virtual void onValueChanged(const Snapshot &shot, XValueNodeBase *node) override;
	const shared_ptr<XValueNodeBase> m_node;
    QTextEdit *const m_pItem;
};
using XQTextBrowserConnector = XQTextEditConnector;

class QSlider;

template <class QN, class XN, class X>
class XQSpinBoxConnectorTMPL : public XValueQConnector {
public:
    XQSpinBoxConnectorTMPL(const shared_ptr<XN> &node,
        QN *item, QSlider *slider);
protected:
    virtual void onUIFlagsChanged(const Snapshot &shot, XNode *node) override;
    void onChangeTMPL(X val);
    void onSliderChangeTMPL(int val);
    void onValueChangedTMPL(const Snapshot &shot, XValueNodeBase *node);
    const shared_ptr<XN> m_node;
    QN *const m_pItem;
    QSlider *const m_pSlider;
};

class QSpinBox;

class DECLSPEC_KAME XQSpinBoxConnector : public XQSpinBoxConnectorTMPL<QSpinBox, XIntNode, int> {
    Q_OBJECT
public:
    XQSpinBoxConnector(const shared_ptr<XIntNode> &node,
        QSpinBox *item, QSlider *slider = 0L);
protected slots:
void onChange(int val) {onChangeTMPL(val);}
void onSliderChange(int val) {onSliderChangeTMPL(val);}
protected:
    virtual void onValueChanged(const Snapshot &shot, XValueNodeBase *node) override {
        onValueChangedTMPL(shot, node); }
};
class DECLSPEC_KAME XQSpinBoxUnsignedConnector : public XQSpinBoxConnectorTMPL<QSpinBox, XUIntNode, int> {
    Q_OBJECT
public:
    XQSpinBoxUnsignedConnector(const shared_ptr<XUIntNode> &node,
        QSpinBox *item, QSlider *slider = 0L);
protected slots:
    void onChange(int val) {onChangeTMPL(val);}
    void onSliderChange(int val) {onSliderChangeTMPL(val);}
protected:
    virtual void onValueChanged(const Snapshot &shot, XValueNodeBase *node) override {
        onValueChangedTMPL(shot, node); }
};

class QDoubleSpinBox;
class DECLSPEC_KAME XQDoubleSpinBoxConnector : public XQSpinBoxConnectorTMPL<QDoubleSpinBox, XDoubleNode, double> {
    Q_OBJECT
public:
    XQDoubleSpinBoxConnector(const shared_ptr<XDoubleNode> &node,
        QDoubleSpinBox *item, QSlider *slider = 0L);
protected slots:
    void onChange(double val) {onChangeTMPL(val);}
    void onSliderChange(int val) {onSliderChangeTMPL(val);}
protected:
    virtual void onValueChanged(const Snapshot &shot, XValueNodeBase *node) override {
        onValueChangedTMPL(shot, node); }
};

class QLabel;

class DECLSPEC_KAME XQLabelConnector : public XValueQConnector {
	Q_OBJECT
public:
	XQLabelConnector(const shared_ptr<XValueNodeBase> &node, 
		QLabel *item);
protected slots:
protected:
    virtual void onValueChanged(const Snapshot &shot, XValueNodeBase *node) override;
	const shared_ptr<XValueNodeBase> m_node;
	QLabel *const m_pItem;
};

class QLCDNumber;

class DECLSPEC_KAME XQLCDNumberConnector : public XValueQConnector {
	Q_OBJECT
public:
	XQLCDNumberConnector(const shared_ptr<XDoubleNode> &node,
		QLCDNumber *item);
protected:
    virtual void onValueChanged(const Snapshot &shot, XValueNodeBase *node) override;
	const shared_ptr<XDoubleNode> m_node;
	QLCDNumber *const m_pItem;
};

class QIcon;
class QPushButton;
class DECLSPEC_KAME XQLedConnector : public XValueQConnector {
    Q_OBJECT
public:
    XQLedConnector(const shared_ptr<XBoolNode> &node,
        QPushButton *item);
protected slots:
protected:
    virtual void onValueChanged(const Snapshot &shot, XValueNodeBase *node) override;
    const shared_ptr<XBoolNode> m_node;
    QPushButton *const m_pItem;
    QIcon *m_pIconOn;
    QIcon *m_pIconOff;
};

class DECLSPEC_KAME XQToggleButtonConnector : public XValueQConnector {
	Q_OBJECT
public:
	XQToggleButtonConnector(const shared_ptr<XBoolNode> &node,
		QAbstractButton *item);
protected slots:
void onClick();
protected:
    virtual void onValueChanged(const Snapshot &shot, XValueNodeBase *node) override;
	const shared_ptr<XBoolNode> m_node;
	QAbstractButton *const m_pItem;
};

class QToolButton;
class DECLSPEC_KAME XFilePathConnector : public XQLineEditConnector {
    Q_OBJECT
public:
    XFilePathConnector(const shared_ptr<XStringNode> &node,
        QLineEdit *edit, QAbstractButton *btn, const char *filter, bool saving);
protected slots:
void onClick();
protected:
    virtual void onValueChanged(const Snapshot &shot, XValueNodeBase *node) override;
    QAbstractButton *const m_pBtn;
    bool m_saving;
    XString m_filter;
};

class QTableWidget;
class DECLSPEC_KAME XListQConnector : public XQConnector {
	Q_OBJECT
public:
    XListQConnector(const shared_ptr<XListNodeBase> &node, QTableWidget *item);
	virtual ~XListQConnector();
private slots:
protected slots:
void OnSectionMoved(int logicalIndex, int oldVisualIndex, int newVisualIndex);
protected:
    shared_ptr<Listener> m_lsnMove;
	virtual void onMove(const Snapshot &shot, const XListNodeBase::Payload::MoveEvent &e);
    shared_ptr<Listener> m_lsnCatch;
    shared_ptr<Listener> m_lsnRelease;
	virtual void onCatch(const Snapshot &shot, const XListNodeBase::Payload::CatchEvent &e) = 0;
	virtual void onRelease(const Snapshot &shot, const XListNodeBase::Payload::ReleaseEvent &e) = 0;
    QTableWidget *const m_pItem;
	const shared_ptr<XListNodeBase> m_list;
};

class DECLSPEC_KAME XItemQConnector : public XValueQConnector {
	Q_OBJECT
public:
	XItemQConnector(const shared_ptr<XItemNodeBase> &node,
		QWidget *item);
	virtual ~XItemQConnector();
private slots:
protected slots:
protected:
    shared_ptr<Listener>  m_lsnListChanged;
	virtual void onListChanged(const Snapshot &shot, const XItemNodeBase::Payload::ListChangeEvent &e) = 0;
    std::vector<XItemNodeBase::Item> m_itemStrings;
};

class QComboBox;

class DECLSPEC_KAME XQComboBoxConnector : public XItemQConnector {
	Q_OBJECT
public:
	XQComboBoxConnector(const shared_ptr<XItemNodeBase> &node,
		QComboBox *item, const Snapshot &shot_of_list);
protected slots:
    virtual void onSelect(int index);
protected:
    bool eventFilter(QObject *obj, QEvent *event) override;
    virtual void onValueChanged(const Snapshot &shot, XValueNodeBase *node) override;
    virtual void onListChanged(const Snapshot &shot, const XItemNodeBase::Payload::ListChangeEvent &e) override;
	const shared_ptr<XItemNodeBase> m_node;
	QComboBox *const m_pItem;
	int findItem(const QString &);
};

class QListWidget;
class QListWidgetItem;

class DECLSPEC_KAME XQListWidgetConnector : public XItemQConnector {
	Q_OBJECT
public:
    XQListWidgetConnector(const shared_ptr<XItemNodeBase> &node,
        QListWidget *item, const Snapshot &shot_of_list);
    virtual ~XQListWidgetConnector();
protected slots:
virtual void OnItemSelectionChanged();
protected:
    virtual void onValueChanged(const Snapshot &shot, XValueNodeBase *node) override;
    virtual void onListChanged(const Snapshot &shot, const XItemNodeBase::Payload::ListChangeEvent &e) override;
	const shared_ptr<XItemNodeBase> m_node;
    QListWidget *const m_pItem;
};

class QColorDialog;
class QPushButton;
class DECLSPEC_KAME XColorConnector : public XValueQConnector {
	Q_OBJECT
public:
    XColorConnector(const shared_ptr<XHexNode> &node, QPushButton *item);
protected slots:
void onClick();
void OnColorSelected(const QColor & color);
protected:
    virtual void onValueChanged(const Snapshot &shot, XValueNodeBase *node) override;
	const shared_ptr<XHexNode> m_node;
    QPushButton *const m_pItem;
    qshared_ptr<QColorDialog> m_dialog;
};

//! Show status
class QMainWindow;
class QStatusBar;
class DECLSPEC_KAME XStatusPrinter : public enable_shared_from_this<XStatusPrinter> {
protected:
	explicit XStatusPrinter(QMainWindow *window = NULL);
public:
	static shared_ptr<XStatusPrinter> create(QMainWindow *window = NULL);
	~XStatusPrinter();
    void printMessage(const XString &str, bool popup = true, const char *file = 0L, int line = 0, bool beep = false);
    void printWarning(const XString &str, bool popup = false, const char *file = 0L, int line = 0, bool beep = false);
    void printError(const XString &str, bool popup = true, const char *file = 0L, int line = 0, bool beep = false);
	void clear();
private:
    struct tstatus {XString str; XString tooltip; int ms; bool popup; bool beep;
        enum : int {Normal, Warning, Error} type;};
    Transactional::Talker<tstatus> m_tlkTalker;
    shared_ptr<Listener> m_lsn;
	QMainWindow *m_pWindow;
	QStatusBar *m_pBar;
    void print(const tstatus &status);
};

//---------------------------------------------------------------------------
#endif
