#define SIZEOF_INT 4
#define SIZEOF_LONG 4
#define SIZEOF_VOID_P 4
#define SIZEOF_SHORT 2
#define SIZEOF_FLOAT 4
#define SIZEOF_DOUBLE 8
#define msecsleep(x) (x)

#include "support.h"

#include <stdint.h>


void my_assert(char const*s, int d) {
        fprintf(stderr, "Err:%s:%d\n", s, d);
        abort();
}

#include <NIDAQmx.h>

using std::max;
using std::min;

static const unsigned int CB_TRANSFER_SIZE = 16;
#define NUM_CO_CH 1

uInt32 m_genBufCOLowTicks[CB_TRANSFER_SIZE * NUM_CO_CH];
uInt32 m_genBufCOHighTicks[CB_TRANSFER_SIZE * NUM_CO_CH];

TaskHandle m_taskCO;
void _CHECK_DAQMX_RET(int ret, const char *file, int line) {
	if(ret != 0) {
	char str[2048];
		DAQmxGetExtendedErrorInfo(str, sizeof(str));
		fprintf(stderr, "%s\n@ %s: %d\n", str, file, line);
	}
	if( ret < 0 )
		throw int(ret);
}
#define CHECK_DAQMX_RET(ret) _CHECK_DAQMX_RET(ret, __FILE__, __LINE__)

unsigned int cnt = 0;

int32
_genCallBackCO(TaskHandle task, int32 signalID, void *data)
{
//	CHECK_DAQMX_RET(DAQmxWriteCtrTicksScalar(m_taskCO, false, 0, 
//	m_genBufCOHighTicks[cnt],
//	m_genBufCOLowTicks[cnt],
//	NULL));
	CHECK_DAQMX_RET(DAQmxSetCOPulseLowTicks(m_taskCO, "Dev1/ctr0", 
	m_genBufCOLowTicks[cnt]
	));
	cnt++;
	cnt = cnt % CB_TRANSFER_SIZE;
	return 0;
}
int
main(int argc, char **argv)
{
		for(int i = 0; i < CB_TRANSFER_SIZE; i++) {
			m_genBufCOLowTicks[i] = i * 10000 + 200;
			m_genBufCOHighTicks[i] = 10000;
		}
	try {
		
	    CHECK_DAQMX_RET(DAQmxCreateTask("", &m_taskCO));
	
		CHECK_DAQMX_RET(DAQmxCreateCOPulseChanTicks(m_taskCO, 
	    	"Dev1/ctr0", "", "20MHzTimebase", DAQmx_Val_Low, 100,
	    	5000, 1000));
			
		CHECK_DAQMX_RET(DAQmxCfgImplicitTiming(m_taskCO,
			DAQmx_Val_ContSamps,
			100));
/*		CHECK_DAQMX_RET(DAQmxCfgSampClkTiming(m_taskCO,
			"/Dev2/20MHzTimebase", 20e6, DAQmx_Val_Rising,
			DAQmx_Val_HWTimedSinglePoint,
			BUF_SIZE_HINT));
*/
		CHECK_DAQMX_RET(DAQmxRegisterSignalEvent(m_taskCO,
			DAQmx_Val_CounterOutputEvent, 0,
			_genCallBackCO, NULL));

	    CHECK_DAQMX_RET(DAQmxStartTask(m_taskCO));
			
		for(int cnt = 0; cnt < CB_TRANSFER_SIZE; cnt++) {
/*			int32 state;
			do {
				CHECK_DAQMX_RET(DAQmxGetCOOutputState(m_taskCO, "Dev1/ctr0",  &state));
			} while (state == DAQmx_Val_High);
*/		}
	}
	catch (...) {
	}
    getchar();
    DAQmxStopTask(m_taskCO);
	DAQmxClearTask(m_taskCO);
}
