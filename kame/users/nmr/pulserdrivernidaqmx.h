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
	shared_ptr<XListener> m_lsnOnOpenAO, m_lsnOnCloseAO;
	void openDO() throw (XInterface::XInterfaceError &);
	void onOpenAO(const shared_ptr<XInterface> &);
	void onCloseAO(const shared_ptr<XInterface> &);
	
	void startPulseGen() throw (XInterface::XInterfaceError &);
	void stopPulseGen();

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
	      GenPattern(uint32_t pat, long long int next) :
	        pattern(pat), tonext(next) {}
	      uint32_t pattern;
	      long long int tonext; // in samps for DO.
	  };

	std::deque<GenPattern> m_genPatternList;
	typedef std::deque<GenPattern>::iterator GenPatternIterator;

	GenPatternIterator m_genLastPatItAODO;
	long long int m_genRestSampsAODO;
	unsigned int m_genAOIndex;
	unsigned int finiteAOSamps(unsigned int finiteaosamps);
	unsigned int m_genFiniteAOSamps;
	unsigned int m_genFiniteAORestSamps;

	TaskHandle m_taskAO, m_taskDO,
		 m_taskDOCtr, m_taskGateCtr,
		 m_taskAOCtr;
enum { NUM_AO_CH = 2};
enum { NUM_BUF_BANK = 6};
	std::vector<tRawDO> m_genBufDO[NUM_BUF_BANK];
	std::vector<tRawAO> m_genBufAO[NUM_BUF_BANK];
	atomic<unsigned int> m_genBankWriting;
	atomic<unsigned int> m_genBankDO;
	atomic<unsigned int> m_genBankAO;
	std::vector<tRawAO> m_genPulseWaveAO[NUM_AO_CH][32];
enum { CAL_POLY_ORDER = 4};
	float64 m_coeffAO[NUM_AO_CH][CAL_POLY_ORDER];
	float64 m_upperLimAO[NUM_AO_CH];
	float64 m_lowerLimAO[NUM_AO_CH];
	
	inline tRawAO aoVoltToRaw(int ch, float64 volt);
	void genBankAODO();
	shared_ptr<XThread<XNIDAQmxPulser> > m_threadWriteAO;
	shared_ptr<XThread<XNIDAQmxPulser> > m_threadWriteDO;
	void writeBankAO(const atomic<bool> &terminated);
	void writeBankDO(const atomic<bool> &terminated);
	void *executeWriteAO(const atomic<bool> &);
	void *executeWriteDO(const atomic<bool> &);
	
  int makeWaveForm(int num, double pw, tpulsefunc func, double dB, double freq = 0.0, double phase = 0.0);
  XRecursiveMutex m_totalLock;
};

#endif //HAVE_NI_DAQMX
