//---------------------------------------------------------------------------

#ifndef thermometerH
#define thermometerH

#include "xnode.h"
#include "xlistnode.h"

class XThermometer : public XNode
{
 XNODE_OBJECT
protected:
  XThermometer(const char *name, bool runtime);
 public:
  virtual ~XThermometer() {}

  virtual double getTemp(double res) const = 0;
  virtual double getRawValue(double temp) const = 0;

  const shared_ptr<XDoubleNode> &tempMin() const {return m_tempMin;} 
  const shared_ptr<XDoubleNode> &tempMax() const {return m_tempMax;} 
 private:
  shared_ptr<XDoubleNode> m_tempMin, m_tempMax;
};

class XThermometerList : public XCustomTypeListNode<XThermometer>
{
 XNODE_OBJECT
protected:
  XThermometerList(const char *name, bool runtime);
 public:
  virtual ~XThermometerList() {}

  DEFINE_TYPE_HOLDER
 protected:
  virtual shared_ptr<XNode> createByTypename(
        const std::string &type, const std::string &name) {
    shared_ptr<XNode> ptr = (*creator(type))(name.c_str(), false);
    if(ptr) insert(ptr);
    return ptr;
  }
};

class XRawThermometer : public XThermometer
{
 XNODE_OBJECT
protected:
  XRawThermometer(const char *name, bool runtime) : XThermometer(name, runtime) {}
 public:
  virtual ~XRawThermometer() {}

  virtual double getTemp(double res) const {return res;}
  virtual double getRawValue(double temp) const {return temp;}
};

//chebichev polynominal
class XLakeShore : public XThermometer
{
 XNODE_OBJECT
protected:
  XLakeShore(const char *name, bool runtime);
 public:
  virtual ~XLakeShore() {}
  
  double getTemp(double res) const;
  double getRawValue(double temp) const;
    
  const shared_ptr<XDoubleNode> &resMin() const {return m_resMin;}
  const shared_ptr<XDoubleNode> &resMax() const {return m_resMax;}
  typedef XListNode<XDoubleNode> XDoubleListNode;
  const shared_ptr<XDoubleListNode> &zu() const {return m_zu;}
  const shared_ptr<XDoubleListNode> &zl() const {return m_zl;}
  typedef XListNode<XDoubleListNode> XDouble2DNode;
  const shared_ptr<XDouble2DNode> &ai() const {return m_ai;}
 private:
  shared_ptr<XDoubleNode> m_resMin, m_resMax;
  shared_ptr<XDoubleListNode> m_zu, m_zl;
  shared_ptr<XDouble2DNode> m_ai;

};

class XCryoConcept : public XThermometer
{
 XNODE_OBJECT
protected:
  XCryoConcept(const char *name, bool runtime);
 public:
  virtual ~XCryoConcept() {}

  double getTemp(double res) const;
  double getRawValue(double temp) const;

  const shared_ptr<XDoubleNode> &resMin() const {return m_resMin;}
  const shared_ptr<XDoubleNode> &resMax() const {return m_resMax;}
  typedef XListNode<XDoubleNode> XDoubleListNode;
  const shared_ptr<XDoubleListNode> &ai() const {return m_ai;}
  const shared_ptr<XDoubleNode> &a() const {return m_a;}
  const shared_ptr<XDoubleNode> &t0() const {return m_t0;}
  const shared_ptr<XDoubleNode> &r0() const {return m_r0;}
  const shared_ptr<XDoubleNode> &rCrossover() const {return m_rCrossover;}
 private:
  shared_ptr<XDoubleNode> m_resMin, m_resMax;
  shared_ptr<XDoubleListNode> m_ai;
  shared_ptr<XDoubleNode> m_a, m_t0, m_r0, m_rCrossover;    
};

class XScientificInstruments : public XThermometer
{
 public:
  XScientificInstruments(const char *name, bool runtime);
  virtual ~XScientificInstruments() {}

  double getTemp(double res) const;
  double getRawValue(double temp) const;

  const shared_ptr<XDoubleNode> &resMin() const {return m_resMin;}
  const shared_ptr<XDoubleNode> &resMax() const {return m_resMax;}
  typedef XListNode<XDoubleNode> XDoubleListNode;
  const shared_ptr<XDoubleListNode> &abcde() const {return m_abcde;}
  const shared_ptr<XDoubleListNode> &abc() const {return m_abc;}
  const shared_ptr<XDoubleNode> &rCrossover() const {return m_rCrossover;}
 private:
  shared_ptr<XDoubleNode> m_resMin, m_resMax;
  shared_ptr<XDoubleListNode> m_abcde, m_abc;
  shared_ptr<XDoubleNode> m_rCrossover;    
};

//---------------------------------------------------------------------------
#endif
