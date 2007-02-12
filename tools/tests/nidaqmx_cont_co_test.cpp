#include "support.h"

#include <stdint.h>


void my_assert(char const*s, int d) {
        fprintf(stderr, "Err:%s:%d\n", s, d);
        abort();
}

#include <NIDAQmx.h>

using std::max;
using std::min;

static const unsigned int BUF_SIZE_HINT = 8192;
static const unsigned int CB_TRANSFER_SIZE = (BUF_SIZE_HINT/2);
#define NUM_CO_CH 1

uInt32 m_genBufCOLowTicks[CB_TRANSFER_SIZE * NUM_CO_CH];
uInt32 m_genBufCOHighTicks[CB_TRANSFER_SIZE * NUM_CO_CH];

TaskHandle m_taskCO;
void _CHECK_DAQMX_RET(int ret, const char *file, int line) {
	if(ret ! = 0) {
	char str[2048];
		DAQmxGetExtendedErrorInfo(str, sizeof(str));
		fprintf(stderr, "%s\n@ %s: %d\n", str, file, line);
	}
	if( ret < 0 )
		throw int(ret);
}
#define CHECK_DAQMX_RET(ret) _CHECK_DAQMX_RET(ret, __FILE__, __LINE__)

int32
_genCallBackCO(TaskHandle task, int32 /*type*/, uInt32 num_samps, void *data)
{
	try {
	 	#define NUM_CB_DIV 1
		for(int cnt = 0; cnt < NUM_CB_DIV; cnt++) {
			uInt32 num_samps = transfer_size / NUM_CB_DIV;
				
			int32 samps;


				CHECK_DAQMX_RET(DAQmxWriteCtrTicks(m_taskCO, num_samps, false, 0.3, 
					DAQmx_Val_GroupByScanNumber, 
					&m_genBufCOHighTicks[cnt * num_samps * NUM_CO_CH],
					&m_genBufCOLowTicks[cnt * num_samps * NUM_CO_CH],
					&samps, NULL));
		}
	}
	catch (...) {
	}
}

int
main(int argc, char **argv)
{
		for(int i = 0; i < CB_TRANSFER_SIZE; i++) {
			m_genBufCOLowTicks[i] = 1000;
			m_genBufCOHighTicks[i] = i * 10;
		}
	try {
		
	    CHECK_DAQMX_RET(DAQmxCreateTask("", &m_taskCO));
	
		CHECK_DAQMX_RET(DAQmxCreateCOPulseChanTicks(m_taskCO, 
	    	"Dev1/ctr0", "", DAQmx_Val_Hz, DAQmx_Val_Low, 100,
	    	0, 10));
			
		CHECK_DAQMX_RET(DAQmxCfgSampClkTiming(m_taskCO, "",
			1e6, DAQmx_Val_Rising, DAQmx_Val_ContSamps,
			BUF_SIZE_HINT));

		CHECK_DAQMX_RET(DAQmxCfgOutputBuffer(m_taskCO, BUF_SIZE_HINT ));
		uInt32 bufsize;
		CHECK_DAQMX_RET(DAQmxGetBufOutputBufSize(m_taskCO, &bufsize));
		printf("Using bufsize = %d\n", (int)bufsize);
		CHECK_DAQMX_RET(DAQmxGetBufOutputOnbrdBufSize(m_taskCO, &bufsize));
		printf("On-board bufsize = %d\n", bufsize);
		CHECK_DAQMX_RET(DAQmxSetWriteRegenMode(m_taskCO, DAQmx_Val_DoNotAllowRegen));

		CHECK_DAQMX_RET(DAQmxRegisterEveryNSamplesEvent(m_taskCO,
			DAQmx_Val_Transferred_From_Buffer, CB_TRANSFER_SIZE, 0,
			_genCallBackCO, this));
	    CHECK_DAQMX_RET(DAQmxStartTask(m_taskCO));

		getchar();
	}
	catch (...) {
	}
    DAQmxStopTask(m_taskCO);
	DAQmxClearTask(m_taskCO);
}
