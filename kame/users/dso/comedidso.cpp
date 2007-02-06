#include "comedidso.h"
#ifdef HAVE_COMEDI

XComediDSO::XComediDSO(const char *name, bool runtime, 
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers) :
    XPrimaryDriver(name, runtime, scalarentries, interfaces, thermometers, drivers),
	m_analogInput(create<XComediInterface>("AnalogInputInterface", false,
            dynamic_pointer_cast<XDriver>(shared_from_this())), COMEDI_SUBD_AI),
	m_counter(create<XComediInterface>("Counter", false,
            dynamic_pointer_cast<XDriver>(shared_from_this())), COMEDI_SUBD_COUNTER)
{
    interfaces->insert(m_analogInput);
    interfaces->insert(m_counter);
    m_lsnOnOpen = analogInput()->onOpen().connectWeak(false,
    	this->shared_from_this(), &XComediDSO::onOpen);
    m_lsnOnOpen = counter()->onOpen().connectWeak(false,
    	this->shared_from_this(), &XComediDSO::onOpen);
    m_lsnOnClose = analogInput()->onClose().connectWeak(false,
    	this->shared_from_this(), &XComediDSO::onClose);
    m_lsnOnClose = counter()->onClose().connectWeak(false,
    	this->shared_from_this(), &XComediDSO::onClose);
}

void
XComediDSO::afterStop()
{
	analogInput()->stop();
	counter()->stop();
}
void
XComediDSO::onOpen(const shared_ptr<XInterface> &interface)
{
	try {
		if(interface == counter()) {
			
		}
		if(interface == analogInput()) {
			for(unsigned int i = 0; i < analogInput()->numChannels(); i++) {
				QString ch = QString("ai%1").arg(i);
				trace1()->add(ch);
				trace2()->add(ch);
				trigSource()->add(ch);
			}
			int n_ran = comedi_get_n_ranges(analogInput()->comedi_dev(), 
				analogInput()->comedi_subdev(), 0);
			for(unsigned int i = 0; i < n_ran; i++) {
				comedi_range *ran = comedi_get_range(analogInput()->comedi_dev(), 
					analogInput()->comedi_subdev(), 0, i);
				QString ran = QString("%1").arg(ran->max);
				vFullScale1()->add(ran);
				vFullScale2()->add(ran);
			}
			vFullScale1()->value(n_ran - 1);
			vFullScale2()->value(n_ran - 1);
			for(unsigned int i = 0; i < 8; i++) {
				QString ch = QString("di%1").arg(i);
				trigSource()->add(ch);
			}
			start();
		}
	}
	catch (XInterface::XInterfaceError& e) {
		e.print(getLabel() + KAME::i18n(": Starting driver failed, because"));
	}
}
void
XComediDSO::onClose(const shared_ptr<XInterface> &)
{
	try {
		stop();
	}
	catch (XInterface::XInterfaceError& e) {
		e.print(getLabel() + KAME::i18n(": Stopping driver failed, because"));
	}
	trace1()->clear();
	trace2()->clear();
	vFullScale1()->clear();
	vFullScale2()->clear();
	trigSource()->clear();
}

void 
XComediDSO::onAverageChanged(const shared_ptr<XValueNodeBase> &) {
}

void
XComediDSO::onSingleChanged(const shared_ptr<XValueNodeBase> &)
{
}
void
XComediDSO::onTrigSourceChanged(const shared_ptr<XValueNodeBase> &)
{
	setupCommand();
}
void
XComediDSO::onTrigPosChanged(const shared_ptr<XValueNodeBase> &)
{
	setupCommand();
}
void
XComediDSO::onTrigLevelChanged(const shared_ptr<XValueNodeBase> &)
{
	setupCommand();
}
void
XComediDSO::onTrigFallingChanged(const shared_ptr<XValueNodeBase> &)
{
	setupCommand();
}
void
XComediDSO::onTimeWidthChanged(const shared_ptr<XValueNodeBase> &)
{
	setupCommand();
}
void
XComediDSO::onVFullScale1Changed(const shared_ptr<XValueNodeBase> &)
{
	setupCommand();
}
void
XComediDSO::onVFullScale2Changed(const shared_ptr<XValueNodeBase> &)
{
	setupCommand();
}
void
XComediDSO::onVOffset1Changed(const shared_ptr<XValueNodeBase> &)
{
	setupCommand();
}
void
XComediDSO::onVOffset2Changed(const shared_ptr<XValueNodeBase> &)
{
	setupCommand();
}
void
XComediDSO::onRecordLengthChanged(const shared_ptr<XValueNodeBase> &)
{
	setupCommand();
}
void
XComediDSO::onForceTriggerTouched(const shared_ptr<XNode> &)
{
}

void
XComediDSO::startSequence()
{                      
}

void
XComediDSO::setupCommand() {
	atomic_shared_ptr<Command> cmd;
	memset(cmd.ptr(), 0, sizeof(Command));
	
	bool pretrig = (triggerPos() < 0);
	if(pretrig && !counter()->isOpened()) {
	}
	
	unsigned int trig_src = TRIG_NOW;
	unsigned int trig_arg = 0;
	int trig_ai;
	if(sscanf(trigSource()->to_str(), "ai%d", &trig_ai) == 1) {
		trig_src = TRIG_OTHER;
		comedi_insn insn;
		insn.subdev = analogInput()->comedi_subdev();
		insn.insn = INSN_CONFIG;
		insn.chanspec = 0;
		insn.data = &cmd->configlist[cmd->configlist.size()];
		cmd->configlist.push_back(INSN_CONFIG_ANALOG_TRIG);
		cmd->configlist.push_back(COMEDI_EV_SCAN_BEGIN | (*fallingEdge() ? 0x09 : 0x06));
		cmd->configlist.push_back(trig_ai);
		cmd->configlist.push_back(lrintl(*trigLevel() / 100.0 * 255.0));
		cmd->configlist.push_back(0);
		insn.n = 5;
		cmd->insnlist.push_back(insn);
	}
	int trig_di;
	if(sscanf(trigSource()->to_str(), "di%d", &trig_di) == 1) {
		trig_src = TRIG_EXT;
		trig_arg = trig_di;
	}
			
	cmd->ai.subdev = analogInput()->comedi_subdev();
	cmd->cntr.subdev = counter()->isOpened() ? counter()->comedi_subdev() : 0;
	cmd->ai.flags = 0;

	unsigned int n_chanlist = 0;
	if(*trace1())
		cmd->chanlist[n_chanlist++] = CR_PACK((int)*trace1(), (int)*vFullScale1(), AREF_GROUND);
	if(*trace2())
		cmd->chanlist[n_chanlist++] = CR_PACK((int)*trace2(), (int)*vFullScale2(), AREF_GROUND);
	if(n_chanlist == 0) {
	}
	cmd->ai.chanlist = cmd->chanlist;
	cmd->ai.chanlist_len = n_chanlist;

	if(pretrig) {
		cmd->ai.start_src = TRIG_INT;
		cmd->ai.start_arg = 0;
		cmd->ai.stop_src = TRIG_NONE;
		cmd->ai.stop_arg = 0;

		comedi_insn insn;
		insn.subdev = counter()->comedi_subdev();
		insn.insn = INSN_CONFIG;
		insn.chanspec = CR_PACK(0,0,0);

		insn.data = &cmd->configlist[cmd->configlist.size()];
		cmd->configlist.push_back(GPCT_RESET);
		insn.n = 1;
		cmd->insnlist.push_back(insn);

		insn.data = &cmd->configlist[cmd->configlist.size()];
		cmd->configlist.push_back(GPCT_SET_OPERATION);
		cmd->configlist.push_back(GPCT_SINGLE_PERIOD); //GPCT_SINGLE_PW
		insn.n = 7;
		cmd->insnlist.push_back(insn);

		insn.data = &cmd->configlist[cmd->configlist.size()];
		cmd->configlist.push_back(0);
		insn.n = 1;
		cmd->insnlist.push_back(insn);
	}
	else {
		cmd->ai.start_src = trig_src;
		cmd->ai.start_arg = trig_arg;
		cmd->ai.stop_src = TRIG_COUNT;
		cmd->ai.stop_arg = *recordLength();
	}
	cmd->ai.scan_begin_src = TRIG_TIMER;
	cmd->ai.scan_begin_arg = lrintl(1e9 * (double)timeWidth() / recordLength()); //nsec
	cmd->cntr.scan_begin_src = TRIG_OTHER;
	cmd->cntr.scan_begin_arg = 0;
	cmd->ai.convert_src = TRIG_NOW;
	cmd->ai.convert_arg = 0;
	cmd->cntr.convert_src = TRIG_NOW;
	cmd->cntr.convert_arg = 0;
	cmd->ai.scan_end_src = TRIG_COUNT;
	cmd->ai.scan_end_arg = n_chanlist;
	cmd->cntr.scan_end_src = trig_src;
	cmd->cntr.scan_end_arg = trig_arg;
	analogInput()->comedi_command_test(&cmd->ai);
	if(counter()->isOpened()) counter()->comedi_command_test(&cmd->cntr);
}

int
XComediDSO::acqCount(bool *seq_busy)
{
	
	ret=comedi_command_test(dev,cmd);
	ret=comedi_command(dev,cmd);
	
	FD_ZERO(&rdset);
	FD_SET(comedi_fileno(device),&rdset);
	timeout.tv_sec = 0;
	timeout.tv_usec = 50000;
	ret = select(comedi_fileno(dev)+1,&rdset,NULL,NULL,&timeout);	
	if(ret<0){
		perror("select");
	}else if(ret==0){
		/* hit timeout */
		printf("timeout\n");
	}else if(FD_ISSET(comedi_fileno(device),&rdset)){
		/* comedi file descriptor became ready */
		printf("comedi file descriptor ready\n");
		ret=read(comedi_fileno(dev),buf,sizeof(buf));
		printf("read returned %d\n",ret);
		if(ret<0){
			if(errno==EAGAIN){
				go = 0;
				perror("read");
			}
		}else if(ret==0){
			go = 0;
		}else{
			int i;
			total+=ret;
			//printf("read %d %d\n",ret,total);
			for(i=0;i<ret/sizeof(sampl_t);i++){
				printf("%d\n",buf[i]);
			}
		}
}

double
XComediDSO::getTimeInterval()
{
}

void
XComediDSO::getWave(std::deque<std::string> &channels)
{
	XScopedLock<XComediInterface> lock(*analogInput());
	comedi_
	
}
void
XComediDSO::convertRaw() throw (XRecordError&) {
  int size = rawData().size();
  char *buf = &rawData()[0];
      if(cp >= &buf[size]) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
  setRecordDim(ch_cnt, xoff, xin, width);
}


#endif //HAVE_COMEDI