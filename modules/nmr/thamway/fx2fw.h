/*** FX2FW Ver1.01 by OPTIMIZE ***/
#define TFIFO		0		//OUT2
#define	CPIPE		1		//OUT8
#define RFIFO		2		//IN6


#define CMD_MODE    0x80    //mode(bit2-0) 000:PIO 001:FIFO 010:GPIF
                            //data_bus(bit3) 0:8bit 1:16bit, use_addr(bit4) 0:No 1:Yes
                            //flow_ctrl(bit5) 0:No 1:Yes, state_debug(bit6) 0:No 1:Yes

#define	MODE_PIO	0x00
#define	MODE_FIFO	0x01
#define	MODE_GPIF	0x02
#define	MODE_8BIT	0x00
#define	MODE_16BIT	0x08
#define	MODE_ADDR	0x10
#define	MODE_NOADDR	0x00
#define	MODE_FLOW	0x20
#define	MODE_NOFLOW	0x00
#define	MODE_DEBG	0x40
#define	MODE_NODEBG	0x00

#define CMD_GPIF    0x81    //IFCONFIG(1)+RDYCTL(7)=8byte
#define CMD_WAVE    0x82    //WAVE(32*4)=128byte
#define CMD_FLOW    0x83    //FLOW(9*4)=36byte
#define CMD_WAVE0   0x84    //WAVE 32byte
#define CMD_WAVE1   0x85    //WAVE 32byte
#define CMD_WAVE2   0x86    //WAVE 32byte
#define CMD_WAVE3   0x87    //WAVE 32byte
#define CMD_WFSEL   0x88    //bit1-0:FIFORD, bit3-2:FIFOWR, bit5-4:SINGLERD, bit7-6:SINGLEWR
#define CMD_ADSET   0x89    //byte0:GPIFADRL, byte1:GPIFADRH
#define CMD_CPUCS   0x8a    //SPEED(bit4-3) 00:12MHz 01:24MHz 10:48MHz , CLKOE(bit1) 0:no 1:out
#define CMD_USBCS   0x8b    //return 1byte(bit7 1:HI-Speed(480M) 0:Full-Speed(12M))
#define CMD_SREAD   0x90    //return (8bit mode:1byte data, 16bit mode:2byte data(H -> L))
#define CMD_SWRITE  0x91    //8bit mode:1byte data, 16bit mode:2byte data(H -> L)
#define CMD_BREAD   0x92    //byte0:words(Low), byte1:words(Hi)
#define CMD_BWRITE  0x93    //byte0:words(Low), byte1:Words(Hi)
#define CMD_OEA     0x00    //bitx 0:input 1:output
#define CMD_OEB     0x01    //bitx 0:input 1:output
#define CMD_OEC     0x02    //bitx 0:input 1:output
#define CMD_OED     0x03    //bitx 0:input 1:output
#define CMD_OEE     0x04    //bitx 0:input 1:output
#define CMD_INA     0x05    //return port
#define CMD_INB     0x06    //return port
#define CMD_INC     0x07    //return port
#define CMD_IND     0x08    //return port
#define CMD_INE     0x09    //return port
#define CMD_OUTA    0x0a    //output port
#define CMD_OUTB    0x0b    //output port
#define CMD_OUTC    0x0c    //output port
#define CMD_OUTD    0x0d    //output port
#define CMD_OUTE    0x0e    //output port
#define CMD_WAIT    0x0f	//wait (byte0)ms
#define CMD_NOP     0x10	//wait (byte0)cycles
