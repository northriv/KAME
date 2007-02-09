#include "pulserdriver.h"

#include "nidaqmxdriver.h"

#ifdef HAVE_NI_DAQMX

#include <vector>

class XNIDAQmxPulser : public XNIDAQmxDriver<XPulser>
{
 XNODE_OBJECT
 protected:
  XNIDAQmxPulser(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers);
 public:
  virtual ~XNIDAQmxPulser();

 protected:
	virtual void open() throw (XInterface::XInterfaceError &);
  //! Be called during stopping driver. Call interface()->stop() inside this routine.
 	 virtual void close() throw (XInterface::XInterfaceError &);
 
    //! send patterns to pulser or turn-off
    virtual void changeOutput(bool output);
    //! convert RelPatList to native patterns
    virtual void createNativePatterns();
    //! time resolution [ms]
    virtual double resolution();
    //! create RelPatList
    virtual void rawToRelPat() throw (XRecordError&);
    
  const shared_ptr<XNIDAQmxInterface> &intfDO() const {return interface();}
  const shared_ptr<XNIDAQmxInterface> &intfAO() const {return m_ao_interface;}    
 private:
	shared_ptr<XNIDAQmxInterface> m_ao_interface;
	TaskHandle m_taskAO, m_taskDO;
	shared_ptr<XListener> m_lsnOnOpenAO, m_lsnOnCloseAO;
	void onOpenAO(const shared_ptr<XInterface> &);
	void onCloseAO(const shared_ptr<XInterface> &);
	
	void startPulseGen() throw (XInterface::XInterfaceError &);
	void stopPulseGen() throw (XInterface::XInterfaceError &);

    //A pettern at absolute time
    class tpat {
          public:
          tpat(double npos, uint32_t newpat, uint32_t nmask) {
              pos = npos; pat = newpat; mask = nmask;
          }
          tpat(const tpat &x) {
              pos = x.pos; pat = x.pat; mask = x.mask;
          }
          double pos;
          //this pattern bits will be outputted at 'pos'
          uint32_t pat;
          //mask bits
          uint32_t mask;
          
  
        bool operator< (const tpat &y) const {
              return pos < y.pos;
        }          
    }; 

	typedef int16 tRawAO;
	typedef uInt16 tRawDO;
	  struct GenPattern {
	      GenPattern(uint32_t pat, long long int toapp) :
	        pattern(pat), toappear(toapp) {}
	      uint32_t pattern;
	      long long int toappear; // in samps for DO.
	  };
	std::deque<GenPattern> m_genPatternList;
	typedef std::deque<GenPattern>::iterator GenPatternIterator;
	uint32_t m_genLastPattern;
	GenPatternIterator m_genLastPatIt;
	long long int m_genRestSamps;
	unsigned int m_genAOIndex;

#define NUM_AO_CH 2

	std::vector<tRawDO> m_genBufDO;
	std::vector<tRawAO> m_genBufAO;
	std::vector<tRawAO> m_genPulseWaveAO[NUM_AO_CH][32];
	float64 m_coeffAO[NUM_AO_CH][6];
	float64 m_upperLimAO[NUM_AO_CH];
	float64 m_lowerLimAO[NUM_AO_CH];
	tRawAO aoVoltToRaw(int ch, float64 volt);
	void genPulseBuffer(uInt32 num_samps);
	static int32 _genCallBack(TaskHandle task, int32 /*type*/, uInt32 num_samps, void *data);
	int32 genCallBack(TaskHandle task, uInt32 num_samps);
    
  int makeWaveForm(int num, double pw, tpulsefunc func, double dB, double freq = 0.0, double phase = 0.0);
  
};
#endif //HAVE_NI_DAQMX
