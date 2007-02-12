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
#define NUM_AO_CH 2

int16 m_genBufAO[CB_TRANSFER_SIZE * NUM_AO_CH];

TaskHandle m_taskAO;
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
_genCallBackAO(TaskHandle task, int32 /*type*/, uInt32 num_samps, void *data)
{
	try {
	 	#define NUM_CB_DIV 2
		for(int cnt = 0; cnt < NUM_CB_DIV; cnt++) {
			uInt32 num_samps = transfer_size / NUM_CB_DIV;
				
			int32 samps;


/*				CHECK_DAQMX_RET(DAQmxWriteBinaryI16(m_taskAO, num_samps, false, 0.3, 
					DAQmx_Val_GroupByScanNumber, &m_genBufAO[m_genBankAO][cnt * num_samps * SAMPS_AO_PER_DO * NUM_AO_CH],
					 &samps, NULL));
*/				CHECK_DAQMX_RET(DAQmxWriteRaw(m_taskAO, num_samps, false, 0.3, 
					&m_genBufAO[cnt * num_samps * NUM_AO_CH],
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
			m_genBufAO[2 * i] = lrint(cos(i * 2 * PI / 10) * 5000u);
			m_genBufAO[2 * i + 1] = lrint(sin(i * 2 * PI / 10) * 20000u);
		}
	try {
	    CHECK_DAQMX_RET(DAQmxCreateTask("", &m_taskAO));
	
		CHECK_DAQMX_RET(DAQmxCreateAOVoltageChan(m_taskAO, "Dev1/ao0:1", "",
	    	-1.0, 1.0, DAQmx_Val_Volts, NULL));
			
		//DMA is slower than interrupts!
//		CHECK_DAQMX_RET(DAQmxSetAODataXferMech(m_taskAO, 
//	    	formatString("%s/ao0:1", intfAO()->devName()).c_str(),
//			DAQmx_Val_Interrupts));

		CHECK_DAQMX_RET(DAQmxCfgSampClkTiming(m_taskAO, "",
			1e6, DAQmx_Val_Rising, DAQmx_Val_ContSamps,
			BUF_SIZE_HINT * OVERSAMP_AO));

		//Buffer setup.
		CHECK_DAQMX_RET(DAQmxSetAODataXferReqCond(m_taskAO, 
	    	"Dev1/ao0:1",
//			DAQmx_Val_OnBrdMemNotFull));
			DAQmx_Val_OnBrdMemHalfFullOrLess));
		CHECK_DAQMX_RET(DAQmxCfgOutputBuffer(m_taskAO, BUF_SIZE_HINT ));
		uInt32 bufsize;
		CHECK_DAQMX_RET(DAQmxGetBufOutputBufSize(m_taskAO, &bufsize));
		printf("Using bufsize = %d\n", (int)bufsize);
		CHECK_DAQMX_RET(DAQmxGetBufOutputOnbrdBufSize(m_taskAO, &bufsize));
		printf("On-board bufsize = %d\n", bufsize);
		CHECK_DAQMX_RET(DAQmxSetWriteRegenMode(m_taskAO, DAQmx_Val_DoNotAllowRegen));

		CHECK_DAQMX_RET(DAQmxRegisterEveryNSamplesEvent(m_taskAO,
			DAQmx_Val_Transferred_From_Buffer, CB_TRANSFER_SIZE, 0,
			_genCallBackAO, this));
	    CHECK_DAQMX_RET(DAQmxStartTask(m_taskAO));
	
		getchar();
	}
	catch (...) {
	}
    DAQmxStopTask(m_taskAO);
    DAQmxClearTask(m_taskAO);


}
