//---------------------------------------------------------------------------

#ifndef xnodeconnectorH
#define xnodeconnectorH

#include "support.h"
#include "xnode.h"
#include "xlistnode.h"
#include "xitemnode.h"
#include "xnodeconnector_prv.h"

//! need for making new forms
extern QWidget *g_pFrmMain;

//! Associate QWidget to XNode
//! use connectWeak() to make XListener.
//! use xqcon_create<T>() to make instances
class XQConnector : public QObject,
 public enable_shared_from_this<XQConnector>
{
  //! don't forget this
  Q_OBJECT
  //! needed for XQConnector and cousins
  XQCON_OBJECT
 protected:
  //! don't use this, use xqcon_create() instead
  //! \sa xqcon_create()
  XQConnector(const shared_ptr<XNode> &node, QWidget *item);
 public:
  //! Here, disconnect all signals & slots
  virtual ~XQConnector();
 private slots:
 protected slots:
 protected:
  bool isItemAlive() const {return m_pWidget;}
  shared_ptr<XListener> m_lsnUIEnabled;
  void onUIEnabled(const shared_ptr<XNode> &node);
  QWidget *m_pWidget;
  static std::deque<shared_ptr<XQConnector> > s_thisCreating;
};

class QButton;

class XQButtonConnector : public XQConnector
{
  Q_OBJECT
  XQCON_OBJECT
 protected:
  XQButtonConnector(const shared_ptr<XNode> &node, QButton *item);
 public:
  virtual ~XQButtonConnector();
 private slots:
 protected slots:
  virtual void onClick();
 protected:
  virtual void onTouch(const shared_ptr<XNode> &node);
  shared_ptr<XListener> m_lsnTouch;
  shared_ptr<XNode> m_node;
  QButton *m_pItem;
};

class XValueQConnector : public XQConnector
{
  Q_OBJECT
  XQCON_OBJECT
 protected:
  XValueQConnector(const shared_ptr<XValueNodeBase> &node, QWidget *item);
 public:
  virtual ~XValueQConnector();
 private slots:
 protected:
  shared_ptr<XListener> m_lsnBeforeValueChanged;
  shared_ptr<XListener> m_lsnValueChanged;
  virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &node) = 0;
  virtual void onValueChanged(const shared_ptr<XValueNodeBase> &node) = 0;
};

class QLineEdit;

class XQLineEditConnector : public XValueQConnector
{
  Q_OBJECT
  XQCON_OBJECT
 protected:
  XQLineEditConnector(const shared_ptr<XValueNodeBase> &node,
     QLineEdit *item, bool forcereturn = true);
 public:
  virtual ~XQLineEditConnector() {}
 protected slots:
  void onTextChanged(const QString &);
  void onTextChanged2(const QString &);
  void onReturnPressed();
  void onExit();
 protected:
  virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
  virtual void onValueChanged(const shared_ptr<XValueNodeBase> &);
  shared_ptr<XValueNodeBase> m_node;
  QLineEdit *m_pItem;
};

class QTextBrowser;

class XQTextBrowserConnector : public XValueQConnector
{
  Q_OBJECT
  XQCON_OBJECT
 protected:
  XQTextBrowserConnector(const shared_ptr<XValueNodeBase> &node,
     QTextBrowser *item);
 public:
  virtual ~XQTextBrowserConnector() {}
 protected slots:
 protected:
  virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
  virtual void onValueChanged(const shared_ptr<XValueNodeBase> &);
  shared_ptr<XValueNodeBase> m_node;
  QTextBrowser *m_pItem;
};

class KIntNumInput;

class XKIntNumInputConnector : public XValueQConnector
{
  Q_OBJECT
  XQCON_OBJECT
 protected:
  XKIntNumInputConnector(const shared_ptr<XIntNode> &node,
     KIntNumInput *item);
  XKIntNumInputConnector(const shared_ptr<XUIntNode> &node,
     KIntNumInput *item);
 public:
  virtual ~XKIntNumInputConnector() {}
 protected slots:
  void onChange(int val);
 protected:
  virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
  virtual void onValueChanged(const shared_ptr<XValueNodeBase> &node);
  shared_ptr<XIntNode> m_iNode;
  shared_ptr<XUIntNode> m_uINode;
  KIntNumInput *m_pItem;
};

class QSpinBox;

class XQSpinBoxConnector : public XValueQConnector
{
  Q_OBJECT
  XQCON_OBJECT
 protected:
  XQSpinBoxConnector(const shared_ptr<XIntNode> &node,
    QSpinBox *item);
  XQSpinBoxConnector(const shared_ptr<XUIntNode> &node,
    QSpinBox *item);
 public:
  virtual ~XQSpinBoxConnector() {}
 protected slots:
  void onChange(int val);
 protected:
  virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
  virtual void onValueChanged(const shared_ptr<XValueNodeBase> &node);
  shared_ptr<XIntNode> m_iNode;
  shared_ptr<XUIntNode> m_uINode;
  QSpinBox *m_pItem;
};

class KDoubleNumInput;

class XKDoubleNumInputConnector : public XValueQConnector
{
  Q_OBJECT
  XQCON_OBJECT
 protected:
  XKDoubleNumInputConnector(const shared_ptr<XDoubleNode> &node, 
    KDoubleNumInput *item);
 public:
  virtual ~XKDoubleNumInputConnector() {}
 protected slots:
  void onChange(double val);
 protected:
  virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
  virtual void onValueChanged(const shared_ptr<XValueNodeBase> &node);
  shared_ptr<XDoubleNode> m_node;
  KDoubleNumInput *m_pItem;
};

class KDoubleSpinBox;

class XKDoubleSpinBoxConnector : public XValueQConnector
{
  Q_OBJECT
  XQCON_OBJECT
 protected:
  XKDoubleSpinBoxConnector(const shared_ptr<XDoubleNode> &node,
    KDoubleSpinBox *item);
 public:
  virtual ~XKDoubleSpinBoxConnector() {}
 protected slots:
  void onChange(double val);
 protected:
  virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
  virtual void onValueChanged(const shared_ptr<XValueNodeBase> &node);
  shared_ptr<XDoubleNode> m_node;
  KDoubleSpinBox *m_pItem;
};

class KURLRequester;

class XKURLReqConnector : public XValueQConnector
{
  Q_OBJECT
  XQCON_OBJECT
 protected:
  XKURLReqConnector(const shared_ptr<XStringNode> &node, 
    KURLRequester *item, const char *filter, bool saving);
 public:
  virtual ~XKURLReqConnector() {}
 protected slots:
  void onSelect( const QString& );
 protected:
  virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
  virtual void onValueChanged(const shared_ptr<XValueNodeBase> &node);
  shared_ptr<XStringNode> m_node;
  KURLRequester *m_pItem;
};

class QLabel;

class XQLabelConnector : public XValueQConnector
{
  Q_OBJECT
  XQCON_OBJECT
 protected:
  XQLabelConnector(const shared_ptr<XValueNodeBase> &node, 
    QLabel *item);
 public:
  virtual ~XQLabelConnector() {}
 protected slots:
 protected:
  virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
  virtual void onValueChanged(const shared_ptr<XValueNodeBase> &node);
  shared_ptr<XValueNodeBase> m_node;
  QLabel *m_pItem;
};

class KLed;

class XKLedConnector : public XValueQConnector
{
  Q_OBJECT
  XQCON_OBJECT
 protected:
  XKLedConnector(const shared_ptr<XBoolNode> &node,
    KLed *item);
 public:
  virtual ~XKLedConnector() {}
 protected slots:
 protected:
  virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
  virtual void onValueChanged(const shared_ptr<XValueNodeBase> &node);
  shared_ptr<XBoolNode> m_node;
  KLed *m_pItem;
};

class QLCDNumber;

class XQLCDNumberConnector : public XValueQConnector
{
  Q_OBJECT
  XQCON_OBJECT
 protected:
  XQLCDNumberConnector(const shared_ptr<XDoubleNode> &node,
    QLCDNumber *item);
 public:
  virtual ~XQLCDNumberConnector() {}
 protected:
  virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
  virtual void onValueChanged(const shared_ptr<XValueNodeBase> &node);
  shared_ptr<XDoubleNode> m_node;
  QLCDNumber *m_pItem;
};

class XQToggleButtonConnector : public XValueQConnector
{
  Q_OBJECT
  XQCON_OBJECT
 protected:
  XQToggleButtonConnector(const shared_ptr<XBoolNode> &node,
    QButton *item);
 public:
  virtual ~XQToggleButtonConnector() {}
 protected slots:
  void onClick();
 protected:
  virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
  virtual void onValueChanged(const shared_ptr<XValueNodeBase> &node);
  shared_ptr<XBoolNode> m_node;
  QButton *m_pItem;
};

class QTable;
class XListQConnector : public XQConnector
{
  Q_OBJECT
  XQCON_OBJECT
 protected:
  XListQConnector(const shared_ptr<XListNodeBase> &node, QTable *item);
 public:
  virtual ~XListQConnector();
 private slots:
 protected slots:
  void indexChange(int section, int fromIndex, int toIndex);
 protected:
  shared_ptr<XListener> m_lsnMove;
  virtual void onMove(const XListNodeBase::MoveEvent &node);
  shared_ptr<XListener> m_lsnCatch;
  shared_ptr<XListener> m_lsnRelease;
  virtual void onCatch(const shared_ptr<XNode> &node) = 0;
  virtual void onRelease(const shared_ptr<XNode> &node) = 0;
  QTable *m_pItem;
  shared_ptr<XListNodeBase> m_list;
};

class XItemQConnector : public XValueQConnector
{
  Q_OBJECT
  XQCON_OBJECT
 protected:
  XItemQConnector(const shared_ptr<XItemNodeBase> &node,
    QWidget *item);
 public:
  virtual ~XItemQConnector();
 private slots:
 protected slots:
 protected:
  shared_ptr<XListener>  m_lsnListChanged;
  virtual void onListChanged(const shared_ptr<XItemNodeBase> &) = 0;
  shared_ptr<const std::deque<XItemNodeBase::Item> > m_itemStrings;
};

class QComboBox;

class XQComboBoxConnector : public XItemQConnector
{
  Q_OBJECT
  XQCON_OBJECT
 protected:
  XQComboBoxConnector(const shared_ptr<XItemNodeBase> &node,
    QComboBox *item);
 public:
  virtual ~XQComboBoxConnector() {}
 protected slots:
  virtual void onSelect(int index);
 protected:
  virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
  virtual void onValueChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onListChanged(const shared_ptr<XItemNodeBase> &);
  shared_ptr<XItemNodeBase> m_node;
  QComboBox *m_pItem;
  int findItem(const QString &);
};

class QListBox;

class XQListBoxConnector : public XItemQConnector
{
  Q_OBJECT
  XQCON_OBJECT
 protected:
  XQListBoxConnector(const shared_ptr<XItemNodeBase> &node,
    QListBox *item);
 public:
  virtual ~XQListBoxConnector() {}
 protected slots:
  virtual void onSelect(int index);
 protected:
  virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
  virtual void onValueChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onListChanged(const shared_ptr<XItemNodeBase> &);
  shared_ptr<XItemNodeBase> m_node;
  QListBox *m_pItem;
};

class KColorButton;

class XKColorButtonConnector : public XValueQConnector
{
  Q_OBJECT
  XQCON_OBJECT
 protected:
  XKColorButtonConnector(const shared_ptr<XHexNode> &node,
    KColorButton *item);
 public:
  virtual ~XKColorButtonConnector() {}
 protected slots:
  void onClick(const QColor &newColor);
 protected:
  virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
  virtual void onValueChanged(const shared_ptr<XValueNodeBase> &);
  shared_ptr<XHexNode> m_node;
  KColorButton *m_pItem;
};

class KColorCombo;

class XKColorComboConnector : public XValueQConnector
{
  Q_OBJECT
  XQCON_OBJECT
 protected:
  XKColorComboConnector(const shared_ptr<XHexNode> &node, 
    KColorCombo *item);
 public:
  virtual ~XKColorComboConnector() {}
 protected slots:
  void onClick(const QColor &newColor);
 protected:
  virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
  virtual void onValueChanged(const shared_ptr<XValueNodeBase> &);
  shared_ptr<XHexNode> m_node;
  KColorCombo *m_pItem;
};

//! Show status
class QMainWindow;
class QStatusBar;
class KPassivePopup;
class XStatusPrinter : public enable_shared_from_this<XStatusPrinter>
{
protected:
    explicit XStatusPrinter(QMainWindow *window = NULL);
public:
    static shared_ptr<XStatusPrinter> create(QMainWindow *window = NULL);
	~XStatusPrinter();
	void printMessage(const QString &str, bool popup = true);
	void printWarning(const QString &str, bool popup = false);
	void printError(const QString &str, bool popup = true);
	void clear();
private:
struct tstatus {std::string str; int ms; bool popup; enum {Normal, Warning, Error} type;};
	XTalker<tstatus> m_tlkTalker;
	shared_ptr<XListener> m_lsn;
	QMainWindow *m_pWindow;
	QStatusBar *m_pBar;
	KPassivePopup *m_pPopup;
	void print(const tstatus &status);
    static std::deque<shared_ptr<XStatusPrinter> > s_thisCreating;
};

//---------------------------------------------------------------------------
#endif
