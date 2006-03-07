#include "xnode.h"
#include <klocale.h>
#include <typeinfo>
#include <qdeepcopy.h>

XThreadLocal<std::deque<shared_ptr<XNode> > > XNode::stl_thisCreating;

//---------------------------------------------------------------------------
XNode::XNode(const char *name, bool runtime)
 : m_children()
{
     // temporaly shared_ptr to be able to use shared_from_this() in constructors
      XNode::stl_thisCreating->push_back(shared_ptr<XNode>(this));
      ASSERT(shared_from_this());
      m_bUIEnabled = true;
      m_bRunTime = runtime;
      if(name) m_name = name;
      dbgPrint(QString("xnode %1 is created., addr=0x%2, size=0x%3")
            .arg(getName())
            .arg((unsigned int)this, 0, 16)
            .arg((unsigned int)sizeof(XNode), 0, 16));
}
XNode::~XNode() {
      dbgPrint(QString("xnode %1 is being deleted., addr=0x%2").arg(getName()).arg((unsigned int)this, 0, 16));
}
QString
XNode::getName() const {
    return QString::fromUtf8(m_name.c_str());
}
std::string
XNode::getTypename() const {
    std::string name = typeid(*this).name();
    unsigned int i = name.find('X');
    ASSERT(i != std::string::npos);
    ASSERT(i + 1 < name.length());
    return name.substr(i + 1);
}
void
XNode::insert(const shared_ptr<XNode> &ptr)
{
    ASSERT(ptr);
    for(;;) {
        atomic_shared_ptr<NodeList> old_list(m_children);
        atomic_shared_ptr<NodeList> new_list(old_list ? (new NodeList(*old_list)) : (new NodeList));        
        new_list->push_back(ptr);
        if(new_list.compareAndSwap(old_list, m_children)) break;
    }
}

void
XNode::setUIEnabled(bool v) {
    m_bUIEnabled = v;
    onUIEnabled().talk(shared_from_this());
}
void
XNode::touch() {
    onTouch().talk(shared_from_this());
}

void
XNode::clearChildren()
{
    m_children.reset();
}
int
XNode::releaseChild(const shared_ptr<XNode> &node)
{
    for(;;) {
        atomic_shared_ptr<NodeList> old_list(m_children);
        if(!old_list) return -1;
        atomic_shared_ptr<NodeList> new_list(new NodeList(*old_list));
        NodeList::iterator it = find(new_list->begin(), new_list->end(), node);
        if(it == new_list->end()) return -1;
        new_list->erase(it);
        if(new_list->empty())
            new_list.reset();
        if(new_list.compareAndSwap(old_list, m_children)) break;
    }
    return 0;
}

shared_ptr<XNode>
XNode::getChild(const std::string &var) const
{
  QString str(QString::fromUtf8(var.c_str()));
  shared_ptr<XNode> node;
  atomic_shared_ptr<const XNode::NodeList> list(children());
  if(list) { 
      for(XNode::NodeList::const_iterator it = list->begin(); it != list->end(); it++) {
          if((*it)->getName() == str) {
                node = *it;
                break;
          }
      }
  }
  return node;
}

XValueNodeBase::XValueNodeBase(const char *name, bool runtime) : 
    XNode(name, runtime), m_validator(0L)
{
}
void
XValueNodeBase::str(const QString &s) throw (XKameError &) {
    QString sc(s);
    if(m_validator)
            (*m_validator)(sc);
    _str(sc);
}
void
XValueNodeBase::setValidator(Validator v) {
    m_validator = v;
}

template <typename T, int base>
void
XValueNode<T, base>::value(const T &t) {
    if(m_tlkBeforeValueChanged.empty() && m_tlkOnValueChanged.empty()) {
        m_var = t;
    }
    else {
        XScopedLock<XRecursiveMutex> lock(m_valuemutex);
        shared_ptr<XValueNodeBase> ptr = 
            dynamic_pointer_cast<XValueNodeBase>(shared_from_this());
        m_tlkBeforeValueChanged.talk(ptr);
        m_var = t;
        m_tlkOnValueChanged.talk(ptr); //, 1, &statusmutex);
    }
}

template <>
void
XValueNode<int, 10>::_str(const QString &str) throw (XKameError &) {
    bool ok;
    int var = str.toInt(&ok, 10);
    if(!ok)
         throw XKameError(i18n("Ill string conversion to integer."), __FILE__, __LINE__);
    value(var);
}
template <>
void
XValueNode<unsigned int, 10>::_str(const QString &str) throw (XKameError &) {
    bool ok;
    unsigned int var = str.toUInt(&ok);
    if(!ok)
         throw XKameError(i18n("Ill string conversion to unsigned integer."), __FILE__, __LINE__);
    value(var);
}
template <>
void
XValueNode<unsigned int, 16>::_str(const QString &str) throw (XKameError &) {
    bool ok;
    unsigned int var = str.toUInt(&ok, 16);
    if(!ok)
         throw XKameError(i18n("Ill string conversion to hex."), __FILE__, __LINE__);
    value(var);
}
template <>
void
XValueNode<bool, 10>::_str(const QString &str) throw (XKameError &) {
  bool ok;
  bool x = str.toInt(&ok);
    if(ok) {
      value( x ? true : false );
      return;
    }
   if(str.stripWhiteSpace().lower() == "true") {
        value(true); return;
   }
   if(str.stripWhiteSpace().lower() == "false") {
        value(false); return;
   }
   throw XKameError(i18n("Ill string conversion to boolean."), __FILE__, __LINE__);
}

template <typename T, int base>
QString
XValueNode<T, base>::to_str() const {
    return QString::number(m_var, base);
}
template <>
QString
XValueNode<bool, 10>::to_str() const {
    return m_var ? "true" : "false";
}

template class XValueNode<int, 10>;
template class XValueNode<unsigned int, 10>;
template class XValueNode<unsigned int, 16>;
template class XValueNode<bool, 10>;

XStringNode::XStringNode(const char *name, bool runtime)
   : XValueNodeBase(name, runtime), m_var(new QString()) {}

QString
XStringNode::to_str() const
{
    atomic_shared_ptr<QString> buf = m_var;
    return QDeepCopy<QString>(*buf);
}
void
XStringNode::operator=(const QString &var)
{
    value(var);
}
void
XStringNode::_str(const QString &var) throw (XKameError &)
{
    value(var);
}

XStringNode::operator QString() const {
    return to_str();
}
void
XStringNode::value(const QString &t) {
    atomic_shared_ptr<QString> var(new QString(QDeepCopy<QString>(t)));
    if(beforeValueChanged().empty() && onValueChanged().empty()) {
        m_var = var;
    }
    else {
        XScopedLock<XRecursiveMutex> lock(m_valuemutex);
        beforeValueChanged().talk(dynamic_pointer_cast<XValueNodeBase>(shared_from_this()));
        m_var = var;
        onValueChanged().talk(dynamic_pointer_cast<XValueNodeBase>(shared_from_this()));
    }
}

XDoubleNode::XDoubleNode(const char *name, bool runtime, const char *format)
 : XValueNodeBase(name, runtime), m_var(new double(0.0))
{
  if(format)
     setFormat(format);
  else
     setFormat("");
}
QString
XDoubleNode::to_str() const
{
    return formatDouble(m_format.c_str(), operator double());
}
void
XDoubleNode::_str(const QString &str) throw (XKameError &)
{
bool ok;
    double var = str.toDouble(&ok);
    if(!ok) 
         throw XKameError(i18n("Ill string conversion to double float."), __FILE__, __LINE__);
    value(var);
}
void
XDoubleNode::value(const double &t) {
    if(beforeValueChanged().empty() && onValueChanged().empty()) {
        m_var.reset(new double(t));
    }
    else {
        XScopedLock<XRecursiveMutex> lock(m_valuemutex);
        beforeValueChanged().talk(dynamic_pointer_cast<XValueNodeBase>(shared_from_this()));
        m_var.reset(new double(t));
        onValueChanged().talk(dynamic_pointer_cast<XValueNodeBase>(shared_from_this()));
    }
}
XDoubleNode::operator double() const
{
    atomic_shared_ptr<double> x = m_var;
    return *x;
}

const char *
XDoubleNode::format() const {
    return m_format.c_str();
}
void
XDoubleNode::setFormat(const char* format) {
    QString fmt;
    if(format) fmt = QString::fromUtf8(format);
    try {
        formatDoubleValidator(fmt);
        m_format = fmt.utf8();
    }
    catch (XKameError &e) {
        e.print();
    }
}

