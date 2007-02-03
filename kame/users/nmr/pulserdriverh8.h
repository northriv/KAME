#include "pulserdriver.h"
#include "chardevicedriver.h"
#include <vector>

//! My pulser driver
class XH8Pulser : public XCharDeviceDriver<XPulser>
{
 XNODE_OBJECT
 protected:
  XH8Pulser(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers);
 public:
  virtual ~XH8Pulser() {}

 protected:
  //! Be called just after opening interface. Call start() inside this routine appropriately.
  virtual void open() throw (XInterface::XInterfaceError &);
    //! send patterns to pulser or turn-off
    virtual void changeOutput(bool output);
    //! convert RelPatList to native patterns
    virtual void createNativePatterns();
    //! time resolution [ms]
    virtual double resolution();
    //! create RelPatList
    virtual void rawToRelPat() throw (XRecordError&);
 private:
    //A pettern at absolute time
    class tpat {
          public:
          tpat(double npos, unsigned short newpat, unsigned short nmask) {
              pos = npos; pat = newpat; mask = nmask;
          }
          tpat(const tpat &x) {
              pos = x.pos; pat = x.pat; mask = x.mask;
          }
          double pos;
          //this pattern bits will be outputted at 'pos'
          unsigned short pat;
          //mask bits
          unsigned short mask;
          
  
        bool operator< (const tpat &y) const {
              return pos < y.pos;
        }          
    }; 
 
  //! Add 1 pulse pattern
  //! \param msec a period to next pattern
  //! \param pattern a pattern for digital, to appear
  int pulseAdd(double msec, unsigned short pattern);

  struct h8ushort {unsigned char msb; unsigned char lsb;};
  std::vector<h8ushort> m_zippedPatterns;
};
