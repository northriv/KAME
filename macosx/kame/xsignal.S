	.section __TEXT,__text,regular,pure_instructions
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.stabs	"/Users/northriv/Documents/Projects/KAME/2.0/macosx/kame/",100,0,0,Ltext0
	.stabs	"../../kame/xsignal.cpp",100,0,0,Ltext0
.section __TEXT,__text,regular,pure_instructions
	.align 2
Ltext0:
	.stabs	"gcc2_compiled.",60,0,0,0
	.globl _g_signalBuffer
.data
.zerofill __DATA, __common, _g_signalBuffer, 8, 2
	.globl _g_main_thread_id
.data
.zerofill __DATA, __common, _g_main_thread_id, 4, 2
.section __TEXT,__text,regular,pure_instructions
	.align 2
	.align 2
	.globl __Z12isMainThreadv
.section __TEXT,__text,regular,pure_instructions
	.align 2
__Z12isMainThreadv:
	.stabd	68,0,8
LFB1518:
	mflr r0
	bcl 20,31,"L00000000001$pb"
"L00000000001$pb":
	stw r0,8(r1)
LCFI0:
	stw r31,-4(r1)
LCFI1:
	mflr r31
	stwu r1,-80(r1)
LCFI2:
	.stabd	68,0,9
LBB2:
	bl L_pthread_self$stub
	addis r4,r31,ha16(_g_main_thread_id-"L00000000001$pb")
	lwz r4,lo16(_g_main_thread_id-"L00000000001$pb")(r4)
	bl L_pthread_equal$stub
	.stabd	68,0,10
LBE2:
	addic r2,r3,-1
	subfe r0,r2,r3
	mr r3,r0
	lwz r0,88(r1)
	addi r1,r1,80
	mtlr r0
	lwz r31,-4(r1)
	blr
LFE1518:
	.stabs	"_Z12isMainThreadv:F(0,1)=@s32;-16;",36,0,8,__Z12isMainThreadv
	.stabs	"bool:t(0,1)",128,0,0,0
.section __TEXT,__text,regular,pure_instructions
	.align 2
Lscope0:
	.stabs	"",36,0,0,Lscope0-__Z12isMainThreadv
	.align 2
	.globl __ZN9XListenerC2Ebbj
.section __TEXT,__text,regular,pure_instructions
	.align 2
__ZN9XListenerC2Ebbj:
	.stabd	68,0,14
LFB1521:
	b __ZN9XListenerC4Ebbj
LFE1521:
	.stabs	"_ZN9XListenerC2Ebbj:F(0,2)=(0,2)",36,0,14,__ZN9XListenerC2Ebbj
	.stabs	"void:t(0,2)",128,0,0,0
	.stabs	"this:P(0,3)=k(0,4)=*(0,5)=xsXListener:",64,0,14,3
	.stabs	"mainthreadcall:P(0,1)",64,0,14,4
	.stabs	"avoid_dup:P(0,1)",64,0,14,5
	.stabs	"delay_ms:P(0,6)=r(0,6);0000000000000;0037777777777;",64,0,14,6
	.stabs	"XListener:Tt(0,5)=s20_vptr$XListener:(0,7)=*(0,8)=*(0,9)=f(0,10)=r(0,10);-2147483648;2147483647;,0,32;m_bMainThreadCall:/1(0,1),32,32;m_bAvoidDup:/1(0,1),64,32;m_delay_ms:/1(0,6),96,32;m_bMasked:/1(0,11)=xsatomic<bool>:,128,32;operator=::(0,12)=#(0,5),(0,13)=&(0,5),(0,4),(0,14)=&(0,15)=k(0,5),(0,2);:_ZN9XListeneraSERKS_;2A.;__base_ctor::(0,16)=#(0,5),(0,2),(0,4),(0,14),(0,2);:_ZN9XListenerC2ERKS_;2A.;__comp_ctor::(0,16):_ZN9XListenerC1ERKS_;2A.;__base_dtor::(0,17)=#(0,5),(0,2),(0,4),(0,2);:_ZN9XListenerD2Ev;2A.;__comp_dtor::(0,17):_ZN9XListenerD1Ev;2A*0;(0,5);;__deleting_dtor::(0,17):_ZN9XListenerD0Ev;2A*1;(0,5);;mask::(0,17):_ZN9XListener4maskEv;2A.;unmask::(0,17):_ZN9XListener6unmaskEv;2A.;delay_ms::(0,18)=#(0,5),(0,6),(0,19)=*(0,15),(0,2);:_ZNK9XListener8delay_msEv;2B.;XListener::(0,20)=#(0,5),(0,2),(0,4),(0,1),(0,1),(0,6),(0,2);:_ZN9XListenerC4Ebbj;1A.;__base_ctor::(0,20):_ZN9XListenerC2Ebbj;1A.;__comp_ctor::(0,20):_ZN9XListenerC1Ebbj;1A.;;~%(0,5);",128,0,24,0
	.stabs	"unsigned int:t(0,6)",128,0,0,0
	.stabs	"__vtbl_ptr_type:t(0,8)",128,0,0,0
	.stabs	"int:t(0,10)",128,0,0,0
	.stabs	"atomic<bool>:Tt(0,11)=s4m_var:/0(0,1),0,32;operator=::(0,21)=#(0,11),(0,22)=&(0,11),(0,23)=*(0,11),(0,24)=&(0,25)=k(0,11),(0,2);:_ZN6atomicIbEaSERKS0_;2A.;__base_ctor::(0,26)=#(0,11),(0,2),(0,23),(0,2);:_ZN6atomicIbEC2Ev;2A.;__comp_ctor::(0,26):_ZN6atomicIbEC1Ev;2A.;__base_ctor::(0,27)=#(0,11),(0,2),(0,23),(0,1),(0,2);:_ZN6atomicIbEC2Eb;2A.;__comp_ctor::(0,27):_ZN6atomicIbEC1Eb;2A.;__base_ctor::(0,28)=#(0,11),(0,2),(0,23),(0,24),(0,2);:_ZN6atomicIbEC2ERKS0_;2A.;__comp_ctor::(0,28):_ZN6atomicIbEC1ERKS0_;2A.;__base_dtor::(0,26):_ZN6atomicIbED2Ev;2A.;__comp_dtor::(0,26):_ZN6atomicIbED1Ev;2A.;operator b::(0,29)=#(0,11),(0,1),(0,30)=*(0,25),(0,2);:_ZNK6atomicIbEcvbEv;2B.;operator=::(0,31)=#(0,11),(0,22),(0,23),(0,1),(0,2);:_ZN6atomicIbEaSEb;2A.;operator++::(0,32)=#(0,11),(0,22),(0,23),(0,2);:_ZN6atomicIbEppEv;2A.;operator--::(0,32):_ZN6atomicIbEmmEv;2A.;operator+=::(0,31):_ZN6atomicIbEpLEb;2A.;operator-=::(0,31):_ZN6atomicIbEmIEb;2A.;swap::(0,33)=f(0,1):_ZN6atomicIbE4swapEbRS0_;2A?;decAndTest::(0,34)=#(0,11),(0,1),(0,23),(0,2);:_ZN6atomicIbE10decAndTestEv;2A.;addAndTest::(0,35)=#(0,11),(0,1),(0,23),(0,1),(0,2);:_ZN6atomicIbE10addAndTestEb;2A.;;",128,0,209,0
.section __TEXT,__text,regular,pure_instructions
	.align 2
Lscope1:
	.stabs	"",36,0,0,Lscope1-__ZN9XListenerC2Ebbj
	.align 2
	.globl __ZN9XListenerC1Ebbj
.section __TEXT,__text,regular,pure_instructions
	.align 2
__ZN9XListenerC1Ebbj:
	.stabd	68,0,14
LFB1523:
	b __ZN9XListenerC4Ebbj
LFE1523:
	.stabs	"_ZN9XListenerC1Ebbj:F(0,2)",36,0,14,__ZN9XListenerC1Ebbj
	.stabs	"this:P(0,3)",64,0,14,3
	.stabs	"mainthreadcall:P(0,1)",64,0,14,4
	.stabs	"avoid_dup:P(0,1)",64,0,14,5
	.stabs	"delay_ms:P(0,6)",64,0,14,6
.section __TEXT,__text,regular,pure_instructions
	.align 2
Lscope2:
	.stabs	"",36,0,0,Lscope2-__ZN9XListenerC1Ebbj
.data
.cstring
	.align 2
LC0:
	.ascii "../../kame/xsignal.cpp\0"
.section __TEXT,__text,regular,pure_instructions
	.align 2
	.align 2
	.globl __ZN9XListenerC4Ebbj
.section __TEXT,__text,regular,pure_instructions
	.align 2
__ZN9XListenerC4Ebbj:
	.stabd	68,0,14
LFB1530:
	mflr r0
	mfcr r2
	bcl 20,31,"L00000000002$pb"
"L00000000002$pb":
	stmw r30,-8(r1)
LCFI3:
	mflr r31
	stw r2,4(r1)
LCFI4:
	.stabd	68,0,18
	addis r2,r31,ha16(L__ZTV9XListener$non_lazy_ptr-"L00000000002$pb")
	.stabd	68,0,14
	stw r0,8(r1)
LCFI5:
	.stabd	68,0,18
	lwz r2,lo16(L__ZTV9XListener$non_lazy_ptr-"L00000000002$pb")(r2)
	.stabs	"../../kame/atomic.h",132,0,0,0
	.stabd	68,0,212
LBB3:
LBB4:
	li r0,0
	.stabs	"../../kame/xsignal.cpp",132,0,0,0
	.stabd	68,0,14
	stwu r1,-80(r1)
LCFI6:
	mr r30,r6
	.stabd	68,0,18
	addi r2,r2,8
	.stabs	"../../kame/atomic.h",132,0,0,0
	.stabd	68,0,212
LBB5:
LBB6:
	stw r0,16(r3)
	.stabs	"../../kame/xsignal.cpp",132,0,0,0
	.stabd	68,0,18
LBE6:
LBE5:
LBE4:
	stw r2,0(r3)
	stw r4,4(r3)
	stw r5,8(r3)
	stw r6,12(r3)
	.stabs	"../../kame/atomic.h",132,0,0,0
	.stabd	68,0,123
LBB7:
LBB8:
LBB9:
LBB10:
LBB11:
	sync
	.stabs	"../../kame/xsignal.cpp",132,0,0,0
	.stabd	68,0,19
LBE11:
LBE10:
LBE9:
LBE8:
LBE7:
	cmpwi cr4,r5,0
	beq- cr4,L8
	.stabs	"../../kame/support.h",132,0,0,0
	.stabd	68,0,85
LBB12:
LBB13:
	cmpwi cr7,r4,0
	.stabd	68,0,84
LBE13:
	addis r3,r31,ha16(LC0-"L00000000002$pb")
	la r3,lo16(LC0-"L00000000002$pb")(r3)
	.stabd	68,0,85
LBB14:
	bne+ cr7,L8
	li r4,20
LEHB0:
	bl L__Z9my_assertPKci$stub
L8:
	.stabs	"../../kame/xsignal.cpp",132,0,0,0
	.stabd	68,0,22
LBE14:
LBE12:
	cmpwi cr7,r30,0
	beq- cr7,L4
	.stabs	"../../kame/support.h",132,0,0,0
	.stabd	68,0,84
LBB15:
	addis r3,r31,ha16(LC0-"L00000000002$pb")
	la r3,lo16(LC0-"L00000000002$pb")(r3)
	.stabd	68,0,85
LBB16:
	bne+ cr4,L4
	li r4,23
	bl L__Z9my_assertPKci$stub
LEHE0:
LBE16:
LBE15:
	b L4
L22:
L14:
	.stabs	"../../kame/atomic.h",132,0,0,0
	.stabd	68,0,214
L20:
LEHB1:
	bl L__Unwind_Resume$stub
LEHE1:
	.stabs	"../../kame/xsignal.cpp",132,0,0,0
	.stabd	68,0,25
L4:
	lwz r0,88(r1)
	addi r1,r1,80
	lwz r11,4(r1)
	lmw r30,-8(r1)
	mtlr r0
	mtcrf 8,r11
	blr
LBE3:
LFE1530:
.data
.section __DATA,__gcc_except_tab
LLSDA1530:
	.byte	0xff
	.byte	0xff
	.byte	0x3
	.byte	0x1a
	.set L$set$0,LEHB0-LFB1530
	.long L$set$0
	.set L$set$1,LEHE0-LEHB0
	.long L$set$1
	.set L$set$2,L22-LFB1530
	.long L$set$2
	.byte	0x0
	.set L$set$3,LEHB1-LFB1530
	.long L$set$3
	.set L$set$4,LEHE1-LEHB1
	.long L$set$4
	.long	0x0
	.byte	0x0
.section __TEXT,__text,regular,pure_instructions
	.align 2
	.stabs	"_ZN9XListenerC4Ebbj:F(0,2)",36,0,14,__ZN9XListenerC4Ebbj
	.stabs	"this:P(0,3)",64,0,14,3
	.stabs	"mainthreadcall:P(0,1)",64,0,14,4
	.stabs	"avoid_dup:P(0,1)",64,0,14,5
	.stabs	"delay_ms:P(0,6)",64,0,14,30
	.stabs	"t:r(0,1)",64,0,212,0
	.stabn	192,0,0,LBB4
	.stabn	224,0,0,LBE4
	.stabs	"t:r(0,1)",64,0,212,0
	.stabn	192,0,0,LBB7
	.stabn	224,0,0,LBE7
	.stabs	"file:r(0,36)=*(0,37)=k(0,38)=r(0,38);0;127;",64,0,84,3
	.stabs	"char:t(0,38)",128,0,0,0
	.stabn	192,0,0,LBB12
	.stabn	224,0,0,LBE12
	.stabs	"file:r(0,36)",64,0,84,3
	.stabn	192,0,0,LBB15
	.stabn	224,0,0,LBE15
.section __TEXT,__text,regular,pure_instructions
	.align 2
Lscope3:
	.stabs	"",36,0,0,Lscope3-__ZN9XListenerC4Ebbj
	.align 2
	.globl __ZN9XListenerD2Ev
.section __TEXT,__text,regular,pure_instructions
	.align 2
__ZN9XListenerD2Ev:
	.stabd	68,0,26
LFB1533:
	li r4,0
	b __ZN9XListenerD4Ev
LFE1533:
	.stabs	"_ZN9XListenerD2Ev:F(0,2)",36,0,26,__ZN9XListenerD2Ev
	.stabs	"this:P(0,3)",64,0,26,3
.section __TEXT,__text,regular,pure_instructions
	.align 2
Lscope4:
	.stabs	"",36,0,0,Lscope4-__ZN9XListenerD2Ev
	.align 2
	.globl __ZN9XListenerD1Ev
.section __TEXT,__text,regular,pure_instructions
	.align 2
__ZN9XListenerD1Ev:
	.stabd	68,0,26
LFB1535:
	li r4,2
	b __ZN9XListenerD4Ev
LFE1535:
	.stabs	"_ZN9XListenerD1Ev:F(0,2)",36,0,26,__ZN9XListenerD1Ev
	.stabs	"this:P(0,3)",64,0,26,3
.section __TEXT,__text,regular,pure_instructions
	.align 2
Lscope5:
	.stabs	"",36,0,0,Lscope5-__ZN9XListenerD1Ev
	.align 2
	.globl __ZN9XListenerD0Ev
.section __TEXT,__text,regular,pure_instructions
	.align 2
__ZN9XListenerD0Ev:
	.stabd	68,0,26
LFB1537:
	li r4,3
	b __ZN9XListenerD4Ev
LFE1537:
	.stabs	"_ZN9XListenerD0Ev:F(0,2)",36,0,26,__ZN9XListenerD0Ev
	.stabs	"this:P(0,3)",64,0,26,3
.section __TEXT,__text,regular,pure_instructions
	.align 2
Lscope6:
	.stabs	"",36,0,0,Lscope6-__ZN9XListenerD0Ev
	.align 2
	.globl __ZN9XListenerD4Ev
.section __TEXT,__text,regular,pure_instructions
	.align 2
__ZN9XListenerD4Ev:
	.stabd	68,0,26
LFB1538:
L28:
L34:
	mflr r0
	bcl 20,31,"L00000000003$pb"
"L00000000003$pb":
	mflr r10
	mtlr r0
	addis r2,r10,ha16(L__ZTV9XListener$non_lazy_ptr-"L00000000003$pb")
	andi. r0,r4,1
	lwz r2,lo16(L__ZTV9XListener$non_lazy_ptr-"L00000000003$pb")(r2)
LBB17:
	addi r2,r2,8
	stw r2,0(r3)
LBE17:
	beqlr- cr0
	b L__ZdlPv$stub
LFE1538:
	.stabs	"_ZN9XListenerD4Ev:F(0,2)",36,0,26,__ZN9XListenerD4Ev
	.stabs	"this:P(0,3)",64,0,26,3
	.stabs	"__in_chrg:P(0,10)",64,0,26,4
.section __TEXT,__text,regular,pure_instructions
	.align 2
Lscope7:
	.stabs	"",36,0,0,Lscope7-__ZN9XListenerD4Ev
	.align 2
	.globl __ZN9XListener4maskEv
.section __TEXT,__text,regular,pure_instructions
	.align 2
__ZN9XListener4maskEv:
	.stabd	68,0,28
LFB1542:
	mflr r0
	bcl 20,31,"L00000000004$pb"
"L00000000004$pb":
	stmw r30,-8(r1)
LCFI7:
	stw r0,8(r1)
LCFI8:
	mflr r31
	stwu r1,-80(r1)
LCFI9:
	mr r30,r3
	.stabs	"../../kame/atomic.h",132,0,0,0
	.stabd	68,0,120
LBB18:
LBB19:
LBB20:
LBB21:
LBB22:
LBB23:
	isync
	.stabd	68,0,215
LBE23:
LBE22:
	lwz r0,16(r3)
	.stabs	"../../kame/support.h",132,0,0,0
	.stabd	68,0,84
LBE21:
LBE20:
	addis r3,r31,ha16(LC0-"L00000000004$pb")
	.stabd	68,0,85
LBB24:
	cmpwi cr7,r0,1
	.stabd	68,0,84
LBE24:
	la r3,lo16(LC0-"L00000000004$pb")(r3)
	.stabd	68,0,85
LBB25:
	bne+ cr7,L46
	li r4,29
	bl L__Z9my_assertPKci$stub
L46:
	.stabs	"../../kame/atomic.h",132,0,0,0
	.stabd	68,0,216
LBE25:
LBE19:
LBB26:
	li r0,1
LBB27:
	stw r0,16(r30)
	.stabd	68,0,123
LBB28:
LBB29:
	sync
	.stabd	68,0,216
	lwz r0,88(r1)
	addi r1,r1,80
	mtlr r0
	lmw r30,-8(r1)
	blr
LBE29:
LBE28:
LBE27:
LBE26:
LBE18:
LFE1542:
	.stabs	"_ZN9XListener4maskEv:F(0,2)",36,0,28,__ZN9XListener4maskEv
	.stabs	"this:P(0,3)",64,0,28,30
	.stabs	"file:r(0,36)",64,0,84,3
	.stabn	192,0,0,LBB19
	.stabn	224,0,0,LBE19
	.stabs	"t:r(0,1)",64,0,216,0
	.stabn	192,0,0,LBB26
	.stabn	224,0,0,LBE26
.section __TEXT,__text,regular,pure_instructions
	.align 2
Lscope8:
	.stabs	"",36,0,0,Lscope8-__ZN9XListener4maskEv
	.align 2
	.globl __ZN9XListener6unmaskEv
.section __TEXT,__text,regular,pure_instructions
	.align 2
__ZN9XListener6unmaskEv:
	.stabs	"../../kame/xsignal.cpp",132,0,0,0
	.stabd	68,0,33
LFB1544:
	mflr r0
	bcl 20,31,"L00000000005$pb"
"L00000000005$pb":
	stmw r30,-8(r1)
LCFI10:
	stw r0,8(r1)
LCFI11:
	mflr r31
	stwu r1,-80(r1)
LCFI12:
	mr r30,r3
	.stabs	"../../kame/atomic.h",132,0,0,0
	.stabd	68,0,120
LBB30:
LBB31:
LBB32:
LBB33:
LBB34:
LBB35:
	isync
	.stabd	68,0,215
LBE35:
LBE34:
	lwz r0,16(r3)
	.stabs	"../../kame/support.h",132,0,0,0
	.stabd	68,0,84
LBE33:
LBE32:
	addis r3,r31,ha16(LC0-"L00000000005$pb")
	.stabd	68,0,85
LBB36:
	cmpwi cr7,r0,0
	.stabd	68,0,84
LBE36:
	la r3,lo16(LC0-"L00000000005$pb")(r3)
	.stabd	68,0,85
LBB37:
	bne+ cr7,L53
	li r4,34
	bl L__Z9my_assertPKci$stub
L53:
	.stabs	"../../kame/atomic.h",132,0,0,0
	.stabd	68,0,216
LBE37:
LBE31:
LBB38:
	li r0,0
LBB39:
	stw r0,16(r30)
	.stabd	68,0,123
LBB40:
LBB41:
	sync
	.stabd	68,0,216
	lwz r0,88(r1)
	addi r1,r1,80
	mtlr r0
	lmw r30,-8(r1)
	blr
LBE41:
LBE40:
LBE39:
LBE38:
LBE30:
LFE1544:
	.stabs	"_ZN9XListener6unmaskEv:F(0,2)",36,0,33,__ZN9XListener6unmaskEv
	.stabs	"this:P(0,3)",64,0,33,30
	.stabs	"file:r(0,36)",64,0,84,3
	.stabn	192,0,0,LBB31
	.stabn	224,0,0,LBE31
	.stabs	"t:r(0,1)",64,0,216,0
	.stabn	192,0,0,LBB38
	.stabn	224,0,0,LBE38
.section __TEXT,__text,regular,pure_instructions
	.align 2
Lscope9:
	.stabs	"",36,0,0,Lscope9-__ZN9XListener6unmaskEv
	.align 2
	.globl __ZN12_XTalkerBaseC2Ev
.section __TEXT,__text,regular,pure_instructions
	.align 2
__ZN12_XTalkerBaseC2Ev:
	.stabs	"../../kame/xsignal.cpp",132,0,0,0
	.stabd	68,0,38
LFB1547:
	b __ZN12_XTalkerBaseC4Ev
LFE1547:
	.stabs	"_ZN12_XTalkerBaseC2Ev:F(0,2)",36,0,38,__ZN12_XTalkerBaseC2Ev
	.stabs	"this:P(0,39)=k(0,40)=*(0,41)=xs_XTalkerBase:",64,0,38,3
	.stabs	"_XTalkerBase:Tt(0,41)=s8_vptr$_XTalkerBase:(0,7),0,32;m_bMasked:/1(0,11),32,32;operator=::(0,42)=#(0,41),(0,43)=&(0,41),(0,40),(0,44)=&(0,45)=k(0,41),(0,2);:_ZN12_XTalkerBaseaSERKS_;2A.;__base_ctor::(0,46)=#(0,41),(0,2),(0,40),(0,44),(0,2);:_ZN12_XTalkerBaseC2ERKS_;2A.;__comp_ctor::(0,46):_ZN12_XTalkerBaseC1ERKS_;2A.;_XTalkerBase::(0,47)=#(0,41),(0,2),(0,40),(0,2);:_ZN12_XTalkerBaseC4Ev;1A.;__base_ctor::(0,47):_ZN12_XTalkerBaseC2Ev;1A.;__comp_ctor::(0,47):_ZN12_XTalkerBaseC1Ev;1A.;__base_dtor::(0,47):_ZN12_XTalkerBaseD2Ev;2A.;__comp_dtor::(0,47):_ZN12_XTalkerBaseD1Ev;2A*0;(0,41);;__deleting_dtor::(0,47):_ZN12_XTalkerBaseD0Ev;2A*1;(0,41);;mask::(0,47):_ZN12_XTalkerBase4maskEv;2A.;unmask::(0,47):_ZN12_XTalkerBase6unmaskEv;2A.;;~%(0,41);",128,0,50,0
.section __TEXT,__text,regular,pure_instructions
	.align 2
Lscope10:
	.stabs	"",36,0,0,Lscope10-__ZN12_XTalkerBaseC2Ev
	.align 2
	.globl __ZN12_XTalkerBaseC1Ev
.section __TEXT,__text,regular,pure_instructions
	.align 2
__ZN12_XTalkerBaseC1Ev:
	.stabd	68,0,38
LFB1549:
	b __ZN12_XTalkerBaseC4Ev
LFE1549:
	.stabs	"_ZN12_XTalkerBaseC1Ev:F(0,2)",36,0,38,__ZN12_XTalkerBaseC1Ev
	.stabs	"this:P(0,39)",64,0,38,3
.section __TEXT,__text,regular,pure_instructions
	.align 2
Lscope11:
	.stabs	"",36,0,0,Lscope11-__ZN12_XTalkerBaseC1Ev
	.align 2
	.globl __ZN12_XTalkerBaseC4Ev
.section __TEXT,__text,regular,pure_instructions
	.align 2
__ZN12_XTalkerBaseC4Ev:
	.stabd	68,0,38
LFB1550:
	mflr r0
	bcl 20,31,"L00000000006$pb"
"L00000000006$pb":
	mflr r10
	mtlr r0
	addis r2,r10,ha16(L__ZTV12_XTalkerBase$non_lazy_ptr-"L00000000006$pb")
	.stabs	"../../kame/atomic.h",132,0,0,0
	.stabd	68,0,212
LBB42:
LBB43:
	li r0,0
	.stabs	"../../kame/xsignal.cpp",132,0,0,0
	.stabd	68,0,38
	lwz r2,lo16(L__ZTV12_XTalkerBase$non_lazy_ptr-"L00000000006$pb")(r2)
	.stabs	"../../kame/atomic.h",132,0,0,0
	.stabd	68,0,212
LBB44:
LBB45:
	stw r0,4(r3)
	.stabs	"../../kame/xsignal.cpp",132,0,0,0
	.stabd	68,0,38
	addi r2,r2,8
LBE45:
LBE44:
LBE43:
	stw r2,0(r3)
	.stabs	"../../kame/atomic.h",132,0,0,0
	.stabd	68,0,123
LBB46:
LBB47:
LBB48:
LBB49:
LBB50:
	sync
	.stabd	68,0,214
	blr
L62:
L68:
LBE50:
LBE49:
LBE48:
LBE47:
LBE46:
LBE42:
LFE1550:
	.stabs	"_ZN12_XTalkerBaseC4Ev:F(0,2)",36,0,38,__ZN12_XTalkerBaseC4Ev
	.stabs	"this:P(0,39)",64,0,38,3
	.stabs	"t:r(0,1)",64,0,212,0
	.stabn	192,0,0,LBB43
	.stabn	224,0,0,LBE43
	.stabs	"t:r(0,1)",64,0,212,0
	.stabn	192,0,0,LBB46
	.stabn	224,0,0,LBE46
.section __TEXT,__text,regular,pure_instructions
	.align 2
Lscope12:
	.stabs	"",36,0,0,Lscope12-__ZN12_XTalkerBaseC4Ev
	.align 2
	.globl __ZN12_XTalkerBaseD2Ev
.section __TEXT,__text,regular,pure_instructions
	.align 2
__ZN12_XTalkerBaseD2Ev:
	.stabs	"../../kame/xsignal.cpp",132,0,0,0
	.stabd	68,0,39
LFB1553:
	li r4,0
	b __ZN12_XTalkerBaseD4Ev
LFE1553:
	.stabs	"_ZN12_XTalkerBaseD2Ev:F(0,2)",36,0,39,__ZN12_XTalkerBaseD2Ev
	.stabs	"this:P(0,39)",64,0,39,3
.section __TEXT,__text,regular,pure_instructions
	.align 2
Lscope13:
	.stabs	"",36,0,0,Lscope13-__ZN12_XTalkerBaseD2Ev
	.align 2
	.globl __ZN12_XTalkerBaseD1Ev
.section __TEXT,__text,regular,pure_instructions
	.align 2
__ZN12_XTalkerBaseD1Ev:
	.stabd	68,0,39
LFB1555:
	li r4,2
	b __ZN12_XTalkerBaseD4Ev
LFE1555:
	.stabs	"_ZN12_XTalkerBaseD1Ev:F(0,2)",36,0,39,__ZN12_XTalkerBaseD1Ev
	.stabs	"this:P(0,39)",64,0,39,3
.section __TEXT,__text,regular,pure_instructions
	.align 2
Lscope14:
	.stabs	"",36,0,0,Lscope14-__ZN12_XTalkerBaseD1Ev
	.align 2
	.globl __ZN12_XTalkerBaseD0Ev
.section __TEXT,__text,regular,pure_instructions
	.align 2
__ZN12_XTalkerBaseD0Ev:
	.stabd	68,0,39
LFB1557:
	li r4,3
	b __ZN12_XTalkerBaseD4Ev
LFE1557:
	.stabs	"_ZN12_XTalkerBaseD0Ev:F(0,2)",36,0,39,__ZN12_XTalkerBaseD0Ev
	.stabs	"this:P(0,39)",64,0,39,3
.section __TEXT,__text,regular,pure_instructions
	.align 2
Lscope15:
	.stabs	"",36,0,0,Lscope15-__ZN12_XTalkerBaseD0Ev
	.align 2
	.globl __ZN12_XTalkerBaseD4Ev
.section __TEXT,__text,regular,pure_instructions
	.align 2
__ZN12_XTalkerBaseD4Ev:
	.stabd	68,0,39
LFB1558:
L75:
L81:
	mflr r0
	bcl 20,31,"L00000000007$pb"
"L00000000007$pb":
	mflr r10
	mtlr r0
	addis r2,r10,ha16(L__ZTV12_XTalkerBase$non_lazy_ptr-"L00000000007$pb")
	andi. r0,r4,1
	lwz r2,lo16(L__ZTV12_XTalkerBase$non_lazy_ptr-"L00000000007$pb")(r2)
LBB51:
	addi r2,r2,8
	stw r2,0(r3)
LBE51:
	beqlr- cr0
	b L__ZdlPv$stub
LFE1558:
	.stabs	"_ZN12_XTalkerBaseD4Ev:F(0,2)",36,0,39,__ZN12_XTalkerBaseD4Ev
	.stabs	"this:P(0,39)",64,0,39,3
	.stabs	"__in_chrg:P(0,10)",64,0,39,4
.section __TEXT,__text,regular,pure_instructions
	.align 2
Lscope16:
	.stabs	"",36,0,0,Lscope16-__ZN12_XTalkerBaseD4Ev
	.align 2
	.globl __ZN12_XTalkerBase4maskEv
.section __TEXT,__text,regular,pure_instructions
	.align 2
__ZN12_XTalkerBase4maskEv:
	.stabd	68,0,41
LFB1560:
	mflr r0
	bcl 20,31,"L00000000008$pb"
"L00000000008$pb":
	stmw r30,-8(r1)
LCFI13:
	stw r0,8(r1)
LCFI14:
	mflr r31
	stwu r1,-80(r1)
LCFI15:
	mr r30,r3
	.stabs	"../../kame/atomic.h",132,0,0,0
	.stabd	68,0,120
LBB52:
LBB53:
LBB54:
LBB55:
LBB56:
LBB57:
	isync
	.stabd	68,0,215
LBE57:
LBE56:
	lwz r0,4(r3)
	.stabs	"../../kame/support.h",132,0,0,0
	.stabd	68,0,84
LBE55:
LBE54:
	addis r3,r31,ha16(LC0-"L00000000008$pb")
	.stabd	68,0,85
LBB58:
	cmpwi cr7,r0,1
	.stabd	68,0,84
LBE58:
	la r3,lo16(LC0-"L00000000008$pb")(r3)
	.stabd	68,0,85
LBB59:
	bne+ cr7,L93
	li r4,42
	bl L__Z9my_assertPKci$stub
L93:
	.stabs	"../../kame/atomic.h",132,0,0,0
	.stabd	68,0,216
LBE59:
LBE53:
LBB60:
	li r0,1
LBB61:
	stw r0,4(r30)
	.stabd	68,0,123
LBB62:
LBB63:
	sync
	.stabd	68,0,216
	lwz r0,88(r1)
	addi r1,r1,80
	mtlr r0
	lmw r30,-8(r1)
	blr
LBE63:
LBE62:
LBE61:
LBE60:
LBE52:
LFE1560:
	.stabs	"_ZN12_XTalkerBase4maskEv:F(0,2)",36,0,41,__ZN12_XTalkerBase4maskEv
	.stabs	"this:P(0,39)",64,0,41,30
	.stabs	"file:r(0,36)",64,0,84,3
	.stabn	192,0,0,LBB53
	.stabn	224,0,0,LBE53
	.stabs	"t:r(0,1)",64,0,216,0
	.stabn	192,0,0,LBB60
	.stabn	224,0,0,LBE60
.section __TEXT,__text,regular,pure_instructions
	.align 2
Lscope17:
	.stabs	"",36,0,0,Lscope17-__ZN12_XTalkerBase4maskEv
	.align 2
	.globl __ZN12_XTalkerBase6unmaskEv
.section __TEXT,__text,regular,pure_instructions
	.align 2
__ZN12_XTalkerBase6unmaskEv:
	.stabs	"../../kame/xsignal.cpp",132,0,0,0
	.stabd	68,0,46
LFB1562:
	mflr r0
	bcl 20,31,"L00000000009$pb"
"L00000000009$pb":
	stmw r30,-8(r1)
LCFI16:
	stw r0,8(r1)
LCFI17:
	mflr r31
	stwu r1,-80(r1)
LCFI18:
	mr r30,r3
	.stabs	"../../kame/atomic.h",132,0,0,0
	.stabd	68,0,120
LBB64:
LBB65:
LBB66:
LBB67:
LBB68:
LBB69:
	isync
	.stabd	68,0,215
LBE69:
LBE68:
	lwz r0,4(r3)
	.stabs	"../../kame/support.h",132,0,0,0
	.stabd	68,0,84
LBE67:
LBE66:
	addis r3,r31,ha16(LC0-"L00000000009$pb")
	.stabd	68,0,85
LBB70:
	cmpwi cr7,r0,0
	.stabd	68,0,84
LBE70:
	la r3,lo16(LC0-"L00000000009$pb")(r3)
	.stabd	68,0,85
LBB71:
	bne+ cr7,L100
	li r4,47
	bl L__Z9my_assertPKci$stub
L100:
	.stabs	"../../kame/atomic.h",132,0,0,0
	.stabd	68,0,216
LBE71:
LBE65:
LBB72:
	li r0,0
LBB73:
	stw r0,4(r30)
	.stabd	68,0,123
LBB74:
LBB75:
	sync
	.stabd	68,0,216
	lwz r0,88(r1)
	addi r1,r1,80
	mtlr r0
	lmw r30,-8(r1)
	blr
LBE75:
LBE74:
LBE73:
LBE72:
LBE64:
LFE1562:
	.stabs	"_ZN12_XTalkerBase6unmaskEv:F(0,2)",36,0,46,__ZN12_XTalkerBase6unmaskEv
	.stabs	"this:P(0,39)",64,0,46,30
	.stabs	"file:r(0,36)",64,0,84,3
	.stabn	192,0,0,LBB65
	.stabn	224,0,0,LBE65
	.stabs	"t:r(0,1)",64,0,216,0
	.stabn	192,0,0,LBB72
	.stabn	224,0,0,LBE72
.section __TEXT,__text,regular,pure_instructions
	.align 2
Lscope18:
	.stabs	"",36,0,0,Lscope18-__ZN12_XTalkerBase6unmaskEv
	.align 2
	.globl __ZN13XSignalBufferC2Ev
.section __TEXT,__text,regular,pure_instructions
	.align 2
__ZN13XSignalBufferC2Ev:
	.stabs	"../../kame/xsignal.cpp",132,0,0,0
	.stabd	68,0,52
LFB1565:
	b __ZN13XSignalBufferC4Ev
LFE1565:
	.stabs	"_ZN13XSignalBufferC2Ev:F(0,2)",36,0,52,__ZN13XSignalBufferC2Ev
	.stabs	"this:P(0,48)=k(0,49)=*(0,50)=xsXSignalBuffer:",64,0,52,3
	.stabs	"XSignalBuffer:Tt(0,50)=s92m_queue:/0(0,51)=xsdeque<_XTransaction*,std::allocator<_XTransaction*> >:,0,320;m_queue_mutex:/0(0,52)=xsXMutex:,320,352;m_queue_oldest_time_hint:/0(0,53)=xsatomic<long unsigned int>:,672,32;m_queue_size_hint:/0(0,54)=xsatomic<unsigned int>:,704,32;operator=::(0,55)=#(0,50),(0,56)=&(0,50),(0,49),(0,57)=&(0,58)=k(0,50),(0,2);:_ZN13XSignalBufferaSERKS_;2A.;__base_ctor::(0,59)=#(0,50),(0,2),(0,49),(0,57),(0,2);:_ZN13XSignalBufferC2ERKS_;2A.;__comp_ctor::(0,59):_ZN13XSignalBufferC1ERKS_;2A.;XSignalBuffer::(0,60)=#(0,50),(0,2),(0,49),(0,2);:_ZN13XSignalBufferC4Ev;2A.;__base_ctor::(0,60):_ZN13XSignalBufferC2Ev;2A.;__comp_ctor::(0,60):_ZN13XSignalBufferC1Ev;2A.;__base_dtor::(0,60):_ZN13XSignalBufferD2Ev;2A.;__comp_dtor::(0,60):_ZN13XSignalBufferD1Ev;2A.;registerTransactionList::(0,61)=#(0,50),(0,2),(0,49),(0,62)=*(0,63)=xs_XTransaction:,(0,2);:_ZN13XSignalBuffer23registerTransactionListEP13_XTransaction;2A.;synchronize::(0,64)=#(0,50),(0,1),(0,49),(0,2);:_ZN13XSignalBuffer11synchronizeEv;2A.;;",128,0,206,0
	.stabs	"deque<_XTransaction*,std::allocator<_XTransaction*> >:Tt(0,51)=s40!1,000,(0,65)=xs_Deque_base<_XTransaction*,std::allocator<_XTransaction*> >:;_S_buffer_size::(0,66)=f(0,67)=(0,68)=(0,69)=r(0,69);0000000000000;0037777777777;:_ZNSt5dequeIP13_XTransactionSaIS1_EE14_S_buffer_sizeEv;1A?;__base_ctor::(0,70)=#(0,51),(0,2),(0,71)=*(0,51),(0,72)=&(0,73)=k(0,74)=xsallocator<_XTransaction*>:,(0,2);:_ZNSt5dequeIP13_XTransactionSaIS1_EEC2ERKS2_;2A.;__comp_ctor::(0,70):_ZNSt5dequeIP13_XTransactionSaIS1_EEC1ERKS2_;2A.;__base_ctor::(0,75)=#(0,51),(0,2),(0,71),(0,69),(0,76)=&(0,77)=k(0,62),(0,72),(0,2);:_ZNSt5dequeIP13_XTransactionSaIS1_EEC2EmRKS1_RKS2_;2A.;__comp_ctor::(0,75):_ZNSt5dequeIP13_XTransactionSaIS1_EEC1EmRKS1_RKS2_;2A.;__base_ctor::(0,78)=#(0,51),(0,2),(0,71),(0,69),(0,2);:_ZNSt5dequeIP13_XTransactionSaIS1_EEC2Em;2A.;__comp_ctor::(0,78):_ZNSt5dequeIP13_XTransactionSaIS1_EEC1Em;2A.;__base_ctor::(0,79)=#(0,51),(0,2),(0,71),(0,80)=&(0,81)=k(0,51),(0,2);:_ZNSt5dequeIP13_XTransactionSaIS1_EEC2ERKS3_;2A.;__comp_ctor::(0,79):_ZNSt5dequeIP13_XTransactionSaIS1_EEC1ERKS3_;2A.;__base_dtor::(0,82)=#(0,51),(0,2),(0,71),(0,2);:_ZNSt5dequeIP13_XTransactionSaIS1_EED2Ev;2A.;__comp_dtor::(0,82):_ZNSt5dequeIP13_XTransactionSaIS1_EED1Ev;2A.;operator=::(0,83)=#(0,51),(0,84)=&(0,51),(0,71),(0,80),(0,2);:_ZNSt5dequeIP13_XTransactionSaIS1_EEaSERKS3_;2A.;assign::(0,85)=#(0,51),(0,2),(0,71),(0,69),(0,76),(0,2);:_ZNSt5dequeIP13_XTransactionSaIS1_EE6assignEmRKS1_;2A.;get_allocator::(0,86)=#(0,51),(0,74),(0,87)=*(0,81),(0,2);:_ZNKSt5dequeIP13_XTransactionSaIS1_EE13get_allocatorEv;2B.;begin::(0,88)=#(0,51),(0,89)=xs_Deque_iterator<_XTransaction*,_XTransaction*&,_XTransaction**>:,(0,71),(0,2);:_ZNSt5dequeIP13_XTransactionSaIS1_EE5beginEv;2A.(0,90)=#(0,51),(0,91)=xs_Deque_iterator<_XTransaction*,_XTransaction* const&,_XTransaction* const*>:,(0,87),(0,2);:_ZNKSt5dequeIP13_XTransactionSaIS1_EE5beginEv;2B.;end::(0,88):_ZNSt5dequeIP13_XTransactionSaIS1_EE3endEv;2A.(0,90):_ZNKSt5dequeIP13_XTransactionSaIS1_EE3endEv;2B.;rbegin::(0,92)=#(0,51),(0,93)=xsreverse_iterator<std::_Deque_iterator<_XTransaction*, _XTransaction*&, _XTransaction**> >:,(0,71),(0,2);:_ZNSt5dequeIP13_XTransactionSaIS1_EE6rbeginEv;2A.(0,94)=#(0,51),(0,95)=xsreverse_iterator<std::_Deque_iterator<_XTransaction*, _XTransaction* const&, _XTransaction* const*> >:,(0,87),(0,2);:_ZNKSt5dequeIP13_XTransactionSaIS1_EE6rbeginEv;2B.;rend::(0,92):_ZNSt5dequeIP13_XTransactionSaIS1_EE4rendEv;2A.(0,94):_ZNKSt5dequeIP13_XTransactionSaIS1_EE4rendEv;2B.;size::(0,96)=#(0,51),(0,67),(0,87),(0,2);:_ZNKSt5dequeIP13_XTransactionSaIS1_EE4sizeEv;2B.;max_size::(0,96):_ZNKSt5dequeIP13_XTransactionSaIS1_EE8max_sizeEv;2B.;resize::(0,85):_ZNSt5dequeIP13_XTransactionSaIS1_EE6resizeEmRKS1_;2A.(0,78):_ZNSt5dequeIP13_XTransactionSaIS1_EE6resizeEm;2A.;empty::(0,97)=#(0,51),(0,1),(0,87),(0,2);:_ZNKSt5dequeIP13_XTransactionSaIS1_EE5emptyEv;2B.;operator[]::(0,98)=#(0,51),(0,99)=&(0,62),(0,71),(0,69),(0,2);:_ZNSt5dequeIP13_XTransactionSaIS1_EEixEm;2A.(0,100)=#(0,51),(0,76),(0,87),(0,69),(0,2);:_ZNKSt5dequeIP13_XTransactionSaIS1_EEixEm;2B.;_M_range_check::(0,101)=#(0,51),(0,2),(0,87),(0,69),(0,2);:_ZNKSt5dequeIP13_XTransactionSaIS1_EE14_M_range_checkEm;1B.;at::(0,98):_ZNSt5dequeIP13_XTransactionSaIS1_EE2atEm;2A.(0,100):_ZNKSt5dequeIP13_XTransactionSaIS1_EE2atEm;2B.;front::(0,102)=#(0,51),(0,99),(0,71),(0,2);:_ZNSt5dequeIP13_XTransactionSaIS1_EE5frontEv;2A.(0,103)=#(0,51),(0,76),(0,87),(0,2);:_ZNKSt5dequeIP13_XTransactionSaIS1_EE5frontEv;2B.;back::(0,102):_ZNSt5dequeIP13_XTransactionSaIS1_EE4backEv;2A.(0,103):_ZNKSt5dequeIP13_XTransactionSaIS1_EE4backEv;2B.;push_front::(0,104)=#(0,51),(0,2),(0,71),(0,76),(0,2);:_ZNSt5dequeIP13_XTransactionSaIS1_EE10push_frontERKS1_;2A.;push_back::(0,104):_ZNSt5dequeIP13_XTransactionSaIS1_EE9push_backERKS1_;2A.;pop_front::(0,82):_ZNSt5dequeIP13_XTransactionSaIS1_EE9pop_frontEv;2A.;pop_back::(0,82):_ZNSt5dequeIP13_XTransactionSaIS1_EE8pop_backEv;2A.;insert::(0,105)=#(0,51),(0,89),(0,71),(0,89),(0,76),(0,2);:_ZNSt5dequeIP13_XTransactionSaIS1_EE6insertESt15_Deque_iteratorIS1_RS1_PS1_ERKS1_;2A.(0,106)=#(0,51),(0,2),(0,71),(0,89),(0,69),(0,76),(0,2);:_ZNSt5dequeIP13_XTransactionSaIS1_EE6insertESt15_Deque_iteratorIS1_RS1_PS1_EmRKS1_;2A.;erase::(0,107)=#(0,51),(0,89),(0,71),(0,89),(0,2);:_ZNSt5dequeIP13_XTransactionSaIS1_EE5eraseESt15_Deque_iteratorIS1_RS1_PS1_E;2A.(0,108)=#(0,51),(0,89),(0,71),(0,89),(0,89),(0,2);:_ZNSt5dequeIP13_XTransactionSaIS1_EE5eraseESt15_Deque_iteratorIS1_RS1_PS1_ES7_;2A.;swap::(0,109)=#(0,51),(0,2),(0,71),(0,84),(0,2);:_ZNSt5dequeIP13_XTransactionSaIS1_EE4swapERS3_;2A.;clear::(0,82):_ZNSt5dequeIP13_XTransactionSaIS1_EE5clearEv;2A.;_M_fill_initialize::(0,104):_ZNSt5dequeIP13_XTransactionSaIS1_EE18_M_fill_initializeERKS1_;1A.;_M_fill_assign::(0,85):_ZNSt5dequeIP13_XTransactionSaIS1_EE14_M_fill_assignEmRKS1_;1A.;_M_push_back_aux::(0,104):_ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_push_back_auxERKS1_;1A.;_M_push_front_aux::(0,104):_ZNSt5dequeIP13_XTransactionSaIS1_EE17_M_push_front_auxERKS1_;1A.;_M_pop_back_aux::(0,82):_ZNSt5dequeIP13_XTransactionSaIS1_EE15_M_pop_back_auxEv;1A.;_M_pop_front_aux::(0,82):_ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_pop_front_auxEv;1A.;_M_fill_insert::(0,106):_ZNSt5dequeIP13_XTransactionSaIS1_EE14_M_fill_insertESt15_Deque_iteratorIS1_RS1_PS1_EmRKS1_;1A.;_M_insert_aux::(0,105):_ZNSt5dequeIP13_XTransactionSaIS1_EE13_M_insert_auxESt15_Deque_iteratorIS1_RS1_PS1_ERKS1_;1A.(0,106):_ZNSt5dequeIP13_XTransactionSaIS1_EE13_M_insert_auxESt15_Deque_iteratorIS1_RS1_PS1_EmRKS1_;1A.;_M_reserve_elements_at_front::(0,110)=#(0,51),(0,89),(0,71),(0,69),(0,2);:_ZNSt5dequeIP13_XTransactionSaIS1_EE28_M_reserve_elements_at_frontEm;1A.;_M_reserve_elements_at_back::(0,110):_ZNSt5dequeIP13_XTransactionSaIS1_EE27_M_reserve_elements_at_backEm;1A.;_M_new_elements_at_front::(0,78):_ZNSt5dequeIP13_XTransactionSaIS1_EE24_M_new_elements_at_frontEm;1A.;_M_new_elements_at_back::(0,78):_ZNSt5dequeIP13_XTransactionSaIS1_EE23_M_new_elements_at_backEm;1A.;_M_reserve_map_at_back::(0,111)=#(0,51),(0,2),(0,71),(0,69),(0,2);:_ZNSt5dequeIP13_XTransactionSaIS1_EE22_M_reserve_map_at_backEm;1A.;_M_reserve_map_at_front::(0,111):_ZNSt5dequeIP13_XTransactionSaIS1_EE23_M_reserve_map_at_frontEm;1A.;_M_reallocate_map::(0,112)=#(0,51),(0,2),(0,71),(0,69),(0,1),(0,2);:_ZNSt5dequeIP13_XTransactionSaIS1_EE17_M_reallocate_mapEmb;1A.;;",128,0,648,0
	.stabs	"XMutex:Tt(0,52)=s44m_mutex:/1(0,113)=(0,114)=(0,115)=xs_opaque_pthread_mutex_t:,0,352;operator=::(0,116)=#(0,52),(0,117)=&(0,52),(0,118)=*(0,52),(0,119)=&(0,120)=k(0,52),(0,2);:_ZN6XMutexaSERKS_;2A.;__base_ctor::(0,121)=#(0,52),(0,2),(0,118),(0,119),(0,2);:_ZN6XMutexC2ERKS_;2A.;__comp_ctor::(0,121):_ZN6XMutexC1ERKS_;2A.;__base_ctor::(0,122)=#(0,52),(0,2),(0,118),(0,2);:_ZN6XMutexC2Ev;2A.;__comp_ctor::(0,122):_ZN6XMutexC1Ev;2A.;__base_dtor::(0,122):_ZN6XMutexD2Ev;2A.;__comp_dtor::(0,122):_ZN6XMutexD1Ev;2A.;lock::(0,122):_ZN6XMutex4lockEv;2A.;unlock::(0,122):_ZN6XMutex6unlockEv;2A.;trylock::(0,123)=#(0,52),(0,1),(0,118),(0,2);:_ZN6XMutex7trylockEv;2A.;;",128,0,56,0
	.stabs	"atomic<long unsigned int>:Tt(0,53)=s4m_var:/0(0,69),0,32;operator=::(0,124)=#(0,53),(0,125)=&(0,53),(0,126)=*(0,53),(0,127)=&(0,128)=k(0,53),(0,2);:_ZN6atomicImEaSERKS0_;2A.;__base_ctor::(0,129)=#(0,53),(0,2),(0,126),(0,2);:_ZN6atomicImEC2Ev;2A.;__comp_ctor::(0,129):_ZN6atomicImEC1Ev;2A.;__base_ctor::(0,130)=#(0,53),(0,2),(0,126),(0,69),(0,2);:_ZN6atomicImEC2Em;2A.;__comp_ctor::(0,130):_ZN6atomicImEC1Em;2A.;__base_ctor::(0,131)=#(0,53),(0,2),(0,126),(0,127),(0,2);:_ZN6atomicImEC2ERKS0_;2A.;__comp_ctor::(0,131):_ZN6atomicImEC1ERKS0_;2A.;__base_dtor::(0,129):_ZN6atomicImED2Ev;2A.;__comp_dtor::(0,129):_ZN6atomicImED1Ev;2A.;operator m::(0,132)=#(0,53),(0,69),(0,133)=*(0,128),(0,2);:_ZNK6atomicImEcvmEv;2B.;operator=::(0,134)=#(0,53),(0,125),(0,126),(0,69),(0,2);:_ZN6atomicImEaSEm;2A.;operator++::(0,135)=#(0,53),(0,125),(0,126),(0,2);:_ZN6atomicImEppEv;2A.;operator--::(0,135):_ZN6atomicImEmmEv;2A.;operator+=::(0,134):_ZN6atomicImEpLEm;2A.;operator-=::(0,134):_ZN6atomicImEmIEm;2A.;swap::(0,136)=f(0,69):_ZN6atomicImE4swapEmRS0_;2A?;decAndTest::(0,137)=#(0,53),(0,1),(0,126),(0,2);:_ZN6atomicImE10decAndTestEv;2A.;addAndTest::(0,138)=#(0,53),(0,1),(0,126),(0,69),(0,2);:_ZN6atomicImE10addAndTestEm;2A.;;",128,0,209,0
	.stabs	"atomic<unsigned int>:Tt(0,54)=s4m_var:/0(0,6),0,32;operator=::(0,139)=#(0,54),(0,140)=&(0,54),(0,141)=*(0,54),(0,142)=&(0,143)=k(0,54),(0,2);:_ZN6atomicIjEaSERKS0_;2A.;__base_ctor::(0,144)=#(0,54),(0,2),(0,141),(0,2);:_ZN6atomicIjEC2Ev;2A.;__comp_ctor::(0,144):_ZN6atomicIjEC1Ev;2A.;__base_ctor::(0,145)=#(0,54),(0,2),(0,141),(0,6),(0,2);:_ZN6atomicIjEC2Ej;2A.;__comp_ctor::(0,145):_ZN6atomicIjEC1Ej;2A.;__base_ctor::(0,146)=#(0,54),(0,2),(0,141),(0,142),(0,2);:_ZN6atomicIjEC2ERKS0_;2A.;__comp_ctor::(0,146):_ZN6atomicIjEC1ERKS0_;2A.;__base_dtor::(0,144):_ZN6atomicIjED2Ev;2A.;__comp_dtor::(0,144):_ZN6atomicIjED1Ev;2A.;operator j::(0,147)=#(0,54),(0,6),(0,148)=*(0,143),(0,2);:_ZNK6atomicIjEcvjEv;2B.;operator=::(0,149)=#(0,54),(0,140),(0,141),(0,6),(0,2);:_ZN6atomicIjEaSEj;2A.;operator++::(0,150)=#(0,54),(0,140),(0,141),(0,2);:_ZN6atomicIjEppEv;2A.;operator--::(0,150):_ZN6atomicIjEmmEv;2A.;operator+=::(0,149):_ZN6atomicIjEpLEj;2A.;operator-=::(0,149):_ZN6atomicIjEmIEj;2A.;swap::(0,151)=f(0,6):_ZN6atomicIjE4swapEjRS0_;2A?;decAndTest::(0,152)=#(0,54),(0,1),(0,141),(0,2);:_ZN6atomicIjE10decAndTestEv;2A.;addAndTest::(0,153)=#(0,54),(0,1),(0,141),(0,6),(0,2);:_ZN6atomicIjE10addAndTestEj;2A.;;",128,0,209,0
	.stabs	"_XTransaction:Tt(0,63)=s8_vptr$_XTransaction:(0,7),0,32;registered_time:(0,69),32,32;operator=::(0,154)=#(0,63),(0,155)=&(0,63),(0,62),(0,156)=&(0,157)=k(0,63),(0,2);:_ZN13_XTransactionaSERKS_;2A.;__base_ctor::(0,158)=#(0,63),(0,2),(0,62),(0,156),(0,2);:_ZN13_XTransactionC2ERKS_;2A.;__comp_ctor::(0,158):_ZN13_XTransactionC1ERKS_;2A.;_XTransaction::(0,159)=#(0,63),(0,2),(0,62),(0,2);:_ZN13_XTransactionC4Ev;2A.;__base_ctor::(0,159):_ZN13_XTransactionC2Ev;2A.;__comp_ctor::(0,159):_ZN13_XTransactionC1Ev;2A.;__base_dtor::(0,159):_ZN13_XTransactionD2Ev;2A.;__comp_dtor::(0,159):_ZN13_XTransactionD1Ev;2A*0;(0,63);;__deleting_dtor::(0,159):_ZN13_XTransactionD0Ev;2A*1;(0,63);;talkBuffered::(0,160)=#(0,63),(0,1),(0,62),(0,2);:_ZN13_XTransaction12talkBufferedEv;2A*2;(0,63);;;~%(0,63);",128,0,70,0
	.stabs	"_Deque_base<_XTransaction*,std::allocator<_XTransaction*> >:Tt(0,65)=s40!1,020,(0,161)=xs_Deque_alloc_base<_XTransaction*,std::allocator<_XTransaction*>,true>:;_M_start:/1(0,89),64,128;_M_finish:/1(0,89),192,128;operator=::(0,162)=#(0,65),(0,163)=&(0,65),(0,164)=*(0,65),(0,165)=&(0,166)=k(0,65),(0,2);:_ZNSt11_Deque_baseIP13_XTransactionSaIS1_EEaSERKS3_;2A.;__base_ctor::(0,167)=#(0,65),(0,2),(0,164),(0,165),(0,2);:_ZNSt11_Deque_baseIP13_XTransactionSaIS1_EEC2ERKS3_;2A.;__comp_ctor::(0,167):_ZNSt11_Deque_baseIP13_XTransactionSaIS1_EEC1ERKS3_;2A.;__base_ctor::(0,168)=#(0,65),(0,2),(0,164),(0,72),(0,69),(0,2);:_ZNSt11_Deque_baseIP13_XTransactionSaIS1_EEC2ERKS2_m;2A.;__comp_ctor::(0,168):_ZNSt11_Deque_baseIP13_XTransactionSaIS1_EEC1ERKS2_m;2A.;__base_ctor::(0,169)=#(0,65),(0,2),(0,164),(0,72),(0,2);:_ZNSt11_Deque_baseIP13_XTransactionSaIS1_EEC2ERKS2_;2A.;__comp_ctor::(0,169):_ZNSt11_Deque_baseIP13_XTransactionSaIS1_EEC1ERKS2_;2A.;__base_dtor::(0,170)=#(0,65),(0,2),(0,164),(0,2);:_ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED2Ev;2A.;__comp_dtor::(0,170):_ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED1Ev;2A.;_M_initialize_map::(0,171)=#(0,65),(0,2),(0,164),(0,69),(0,2);:_ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE17_M_initialize_mapEm;1A.;_M_create_nodes::(0,172)=#(0,65),(0,2),(0,164),(0,173)=*(0,174)=*(0,62),(0,173),(0,2);:_ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE15_M_create_nodesEPPS1_S5_;1A.;_M_destroy_nodes::(0,172):_ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE16_M_destroy_nodesEPPS1_S5_;1A.;;",128,0,454,0
	.stabs	"size_t:t(0,67)",128,0,79,0
	.stabs	"__darwin_size_t:t(0,68)",128,0,84,0
	.stabs	"long unsigned int:t(0,69)",128,0,0,0
	.stabs	"allocator<_XTransaction*>:Tt(0,74)=s1operator=::(0,175)=#(0,74),(0,176)=&(0,74),(0,177)=*(0,74),(0,72),(0,2);:_ZNSaIP13_XTransactionEaSERKS1_;2A.;__base_ctor::(0,178)=#(0,74),(0,2),(0,177),(0,2);:_ZNSaIP13_XTransactionEC2Ev;2A.;__comp_ctor::(0,178):_ZNSaIP13_XTransactionEC1Ev;2A.;__base_ctor::(0,179)=#(0,74),(0,2),(0,177),(0,72),(0,2);:_ZNSaIP13_XTransactionEC2ERKS1_;2A.;__comp_ctor::(0,179):_ZNSaIP13_XTransactionEC1ERKS1_;2A.;__base_dtor::(0,178):_ZNSaIP13_XTransactionED2Ev;2A.;__comp_dtor::(0,178):_ZNSaIP13_XTransactionED1Ev;2A.;address::(0,180)=#(0,74),(0,174),(0,181)=*(0,73),(0,99),(0,2);:_ZNKSaIP13_XTransactionE7addressERS0_;2B.(0,182)=#(0,74),(0,183)=*(0,77),(0,181),(0,76),(0,2);:_ZNKSaIP13_XTransactionE7addressERKS0_;2B.;allocate::(0,184)=#(0,74),(0,174),(0,177),(0,69),(0,185)=*(0,186)=k(0,2),(0,2);:_ZNSaIP13_XTransactionE8allocateEmPKv;2A.;deallocate::(0,187)=#(0,74),(0,2),(0,177),(0,174),(0,69),(0,2);:_ZNSaIP13_XTransactionE10deallocateEPS0_m;2A.;max_size::(0,188)=#(0,74),(0,67),(0,181),(0,2);:_ZNKSaIP13_XTransactionE8max_sizeEv;2B.;construct::(0,189)=#(0,74),(0,2),(0,177),(0,174),(0,76),(0,2);:_ZNSaIP13_XTransactionE9constructEPS0_RKS0_;2A.;destroy::(0,190)=#(0,74),(0,2),(0,177),(0,174),(0,2);:_ZNSaIP13_XTransactionE7destroyEPS0_;2A.;;",128,0,637,0
	.stabs	"_Deque_iterator<_XTransaction*,_XTransaction*&,_XTransaction**>:Tt(0,89)=s16_M_cur:(0,174),0,32;_M_first:(0,174),32,32;_M_last:(0,174),64,32;_M_node:(0,173),96,32;operator=::(0,191)=#(0,89),(0,192)=&(0,89),(0,193)=*(0,89),(0,194)=&(0,195)=k(0,89),(0,2);:_ZNSt15_Deque_iteratorIP13_XTransactionRS1_PS1_EaSERKS4_;2A.;_S_buffer_size::(0,66):_ZNSt15_Deque_iteratorIP13_XTransactionRS1_PS1_E14_S_buffer_sizeEv;2A?;__base_ctor::(0,196)=#(0,89),(0,2),(0,193),(0,174),(0,173),(0,2);:_ZNSt15_Deque_iteratorIP13_XTransactionRS1_PS1_EC2ES3_PS3_;2A.;__comp_ctor::(0,196):_ZNSt15_Deque_iteratorIP13_XTransactionRS1_PS1_EC1ES3_PS3_;2A.;__base_ctor::(0,197)=#(0,89),(0,2),(0,193),(0,2);:_ZNSt15_Deque_iteratorIP13_XTransactionRS1_PS1_EC2Ev;2A.;__comp_ctor::(0,197):_ZNSt15_Deque_iteratorIP13_XTransactionRS1_PS1_EC1Ev;2A.;__base_ctor::(0,198)=#(0,89),(0,2),(0,193),(0,194),(0,2);:_ZNSt15_Deque_iteratorIP13_XTransactionRS1_PS1_EC2ERKS4_;2A.;__comp_ctor::(0,198):_ZNSt15_Deque_iteratorIP13_XTransactionRS1_PS1_EC1ERKS4_;2A.;operator*::(0,199)=#(0,89),(0,99),(0,200)=*(0,195),(0,2);:_ZNKSt15_Deque_iteratorIP13_XTransactionRS1_PS1_EdeEv;2B.;operator->::(0,201)=#(0,89),(0,174),(0,200),(0,2);:_ZNKSt15_Deque_iteratorIP13_XTransactionRS1_PS1_EptEv;2B.;operator++::(0,202)=#(0,89),(0,192),(0,193),(0,2);:_ZNSt15_Deque_iteratorIP13_XTransactionRS1_PS1_EppEv;2A.(0,203)=#(0,89),(0,89),(0,193),(0,10),(0,2);:_ZNSt15_Deque_iteratorIP13_XTransactionRS1_PS1_EppEi;2A.;operator--::(0,202):_ZNSt15_Deque_iteratorIP13_XTransactionRS1_PS1_EmmEv;2A.(0,203):_ZNSt15_Deque_iteratorIP13_XTransactionRS1_PS1_EmmEi;2A.;operator+=::(0,204)=#(0,89),(0,192),(0,193),(0,10),(0,2);:_ZNSt15_Deque_iteratorIP13_XTransactionRS1_PS1_EpLEi;2A.;operator+::(0,205)=#(0,89),(0,89),(0,200),(0,10),(0,2);:_ZNKSt15_Deque_iteratorIP13_XTransactionRS1_PS1_EplEi;2B.;operator-=::(0,204):_ZNSt15_Deque_iteratorIP13_XTransactionRS1_PS1_EmIEi;2A.;operator-::(0,205):_ZNKSt15_Deque_iteratorIP13_XTransactionRS1_PS1_EmiEi;2B.;operator[]::(0,206)=#(0,89),(0,99),(0,200),(0,10),(0,2);:_ZNKSt15_Deque_iteratorIP13_XTransactionRS1_PS1_EixEi;2B.;_M_set_node::(0,207)=#(0,89),(0,2),(0,193),(0,173),(0,2);:_ZNSt15_Deque_iteratorIP13_XTransactionRS1_PS1_E11_M_set_nodeEPS3_;2A.;;",128,0,109,0
	.stabs	"_opaque_pthread_mutex_t:Tt(0,115)=s44__sig:(0,208)=r(0,208);-2147483648;2147483647;,0,32;__opaque:(0,209)=ar(0,210)=r(0,210);0000000000000;0037777777777;;0;39;(0,38),32,320;operator=::(0,211)=#(0,115),(0,212)=&(0,115),(0,213)=*(0,115),(0,214)=&(0,215)=k(0,115),(0,2);:_ZN23_opaque_pthread_mutex_taSERKS_;2A.;__base_ctor::(0,216)=#(0,115),(0,2),(0,213),(0,214),(0,2);:_ZN23_opaque_pthread_mutex_tC2ERKS_;2A.;__comp_ctor::(0,216):_ZN23_opaque_pthread_mutex_tC1ERKS_;2A.;__base_ctor::(0,217)=#(0,115),(0,2),(0,213),(0,2);:_ZN23_opaque_pthread_mutex_tC2Ev;2A.;__comp_ctor::(0,217):_ZN23_opaque_pthread_mutex_tC1Ev;2A.;;",128,0,69,0
	.stabs	"_Deque_alloc_base<_XTransaction*,std::allocator<_XTransaction*>,true>:Tt(0,161)=s8_M_map:/1(0,173),0,32;_M_map_size:/1(0,67),32,32;operator=::(0,218)=#(0,161),(0,219)=&(0,161),(0,220)=*(0,161),(0,221)=&(0,222)=k(0,161),(0,2);:_ZNSt17_Deque_alloc_baseIP13_XTransactionSaIS1_ELb1EEaSERKS3_;2A.;__base_ctor::(0,223)=#(0,161),(0,2),(0,220),(0,221),(0,2);:_ZNSt17_Deque_alloc_baseIP13_XTransactionSaIS1_ELb1EEC2ERKS3_;2A.;__comp_ctor::(0,223):_ZNSt17_Deque_alloc_baseIP13_XTransactionSaIS1_ELb1EEC1ERKS3_;2A.;get_allocator::(0,224)=#(0,161),(0,74),(0,225)=*(0,222),(0,2);:_ZNKSt17_Deque_alloc_baseIP13_XTransactionSaIS1_ELb1EE13get_allocatorEv;2B.;__base_ctor::(0,226)=#(0,161),(0,2),(0,220),(0,72),(0,2);:_ZNSt17_Deque_alloc_baseIP13_XTransactionSaIS1_ELb1EEC2ERKS2_;2A.;__comp_ctor::(0,226):_ZNSt17_Deque_alloc_baseIP13_XTransactionSaIS1_ELb1EEC1ERKS2_;2A.;_M_allocate_node::(0,227)=#(0,161),(0,174),(0,220),(0,2);:_ZNSt17_Deque_alloc_baseIP13_XTransactionSaIS1_ELb1EE16_M_allocate_nodeEv;1A.;_M_deallocate_node::(0,228)=#(0,161),(0,2),(0,220),(0,174),(0,2);:_ZNSt17_Deque_alloc_baseIP13_XTransactionSaIS1_ELb1EE18_M_deallocate_nodeEPS1_;1A.;_M_allocate_map::(0,229)=#(0,161),(0,173),(0,220),(0,69),(0,2);:_ZNSt17_Deque_alloc_baseIP13_XTransactionSaIS1_ELb1EE15_M_allocate_mapEm;1A.;_M_deallocate_map::(0,230)=#(0,161),(0,2),(0,220),(0,173),(0,69),(0,2);:_ZNSt17_Deque_alloc_baseIP13_XTransactionSaIS1_ELb1EE17_M_deallocate_mapEPPS1_m;1A.;;",128,0,358,0
	.stabs	"long int:t(0,208)",128,0,0,0
.section __TEXT,__text,regular,pure_instructions
	.align 2
Lscope19:
	.stabs	"",36,0,0,Lscope19-__ZN13XSignalBufferC2Ev
	.align 2
	.globl __ZN13XSignalBufferC1Ev
.section __TEXT,__text,regular,pure_instructions
	.align 2
__ZN13XSignalBufferC1Ev:
	.stabd	68,0,52
LFB1567:
	b __ZN13XSignalBufferC4Ev
LFE1567:
	.stabs	"_ZN13XSignalBufferC1Ev:F(0,2)",36,0,52,__ZN13XSignalBufferC1Ev
	.stabs	"this:P(0,48)",64,0,52,3
.section __TEXT,__text,regular,pure_instructions
	.align 2
Lscope20:
	.stabs	"",36,0,0,Lscope20-__ZN13XSignalBufferC1Ev
	.align 2
	.globl __ZN13XSignalBufferC4Ev
.section __TEXT,__text,regular,pure_instructions
	.align 2
__ZN13XSignalBufferC4Ev:
	.stabd	68,0,52
LFB1606:
L106:
L108:
	mflr r0
	stmw r26,-24(r1)
LCFI19:
	stw r0,8(r1)
LCFI20:
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_deque.h",132,0,0,0
	.stabd	68,0,131
LBB76:
LBB77:
LBB78:
LBB79:
LBB80:
LBB81:
LBB82:
LBB83:
	addi r30,r3,8
	.stabd	68,0,464
LBE83:
LBE82:
LBE81:
	li r0,0
	.stabd	68,0,131
LBB84:
LBB85:
LBB86:
	addi r28,r3,24
	.stabs	"../../kame/xsignal.cpp",132,0,0,0
	.stabd	68,0,52
	stwu r1,-176(r1)
LCFI21:
	mr r29,r3
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_deque.h",132,0,0,0
	.stabd	68,0,131
LBB87:
LBB88:
	stw r0,12(r28)
	.stabd	68,0,465
LBE88:
LBE87:
LBE86:
	li r4,0
	.stabd	68,0,408
LBB89:
LBB90:
LBB91:
	stw r0,0(r3)
	stw r0,4(r3)
	.stabd	68,0,131
LBE91:
LBE90:
LBE89:
LBB92:
LBB93:
LBB94:
	stw r0,8(r3)
	stw r0,4(r30)
	stw r0,8(r30)
	stw r0,12(r30)
LBE94:
LBE93:
LBE92:
LBB95:
LBB96:
LBB97:
	stw r0,24(r3)
	stw r0,4(r28)
	stw r0,8(r28)
	.stabd	68,0,465
LEHB2:
LBE97:
LBE96:
LBE95:
	bl L__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE17_M_initialize_mapEm$stub
LEHE2:
L120:
L122:
LBE85:
LBE84:
LBE80:
LBE79:
LBE78:
LBE77:
	b L127
L196:
	b L197
L126:
L128:
L132:
L136:
L127:
L138:
L142:
	.stabs	"../../kame/xsignal.cpp",132,0,0,0
	.stabd	68,0,54
	addi r27,r29,40
	mr r3,r27
LEHB3:
	bl L__ZN6XMutexC1Ev$stub
LEHE3:
	.stabs	"../../kame/atomic.h",132,0,0,0
	.stabd	68,0,212
LEHB4:
LBB98:
	bl L__Z9timeStampv$stub
LEHE4:
L152:
L158:
L160:
L166:
LBB99:
LBB100:
	stw r3,84(r29)
	.stabd	68,0,123
LBB101:
LBB102:
	sync
	.stabd	68,0,212
LBE102:
LBE101:
LBE100:
LBE99:
LBE98:
LBB103:
	li r0,0
LBB104:
LBB105:
	stw r0,88(r29)
	.stabd	68,0,123
LBB106:
LBB107:
	sync
	.stabd	68,0,214
	lwz r0,184(r1)
	addi r1,r1,176
	mtlr r0
	lmw r26,-24(r1)
	blr
L194:
L168:
L170:
	mr r26,r3
LBE107:
LBE106:
LBE105:
LBE104:
LBE103:
	mr r3,r27
	bl L__ZN6XMutexD1Ev$stub
	b L172
L195:
	mr r26,r3
L172:
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_deque.h",132,0,0,0
	.stabd	68,0,134
L185:
L187:
L192:
LBB108:
LBB109:
LBB110:
LBB111:
LBB112:
LBB113:
LBB114:
	lwz r7,12(r30)
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_construct.h",132,0,0,0
	.stabd	68,0,115
LBE114:
LBE113:
LBE112:
LBE111:
	mr r3,r29
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_deque.h",132,0,0,0
	.stabd	68,0,134
LBB115:
LBB116:
LBB117:
LBB118:
	lwz r6,12(r28)
LBE118:
LBE117:
LBE116:
LBB119:
LBB120:
LBB121:
	lwz r0,8(r29)
	lwz r2,4(r30)
	lwz r9,8(r30)
LBE121:
LBE120:
LBE119:
LBB122:
LBB123:
LBB124:
	lwz r11,24(r29)
	lwz r10,4(r28)
	lwz r8,8(r28)
LBE124:
LBE123:
LBE122:
LBB125:
LBB126:
LBB127:
LBB128:
LBB129:
	stw r0,112(r1)
	stw r2,116(r1)
	stw r9,120(r1)
	stw r7,124(r1)
LBE129:
LBE128:
LBE127:
LBB130:
LBB131:
LBB132:
	stw r11,128(r1)
	stw r10,132(r1)
	stw r8,136(r1)
	stw r6,140(r1)
LBE132:
LBE131:
LBE130:
LBE126:
LBE125:
LBB133:
LBB134:
LBB135:
	stw r0,80(r1)
	stw r2,84(r1)
	stw r9,88(r1)
	stw r7,92(r1)
LBE135:
LBE134:
LBE133:
LBB136:
LBB137:
LBB138:
	stw r11,96(r1)
	stw r10,100(r1)
	stw r8,104(r1)
	stw r6,108(r1)
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_construct.h",132,0,0,0
	.stabd	68,0,115
LBE138:
LBE137:
LBE136:
LBE115:
	bl L__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED2Ev$stub
	mr r3,r26
L197:
LEHB5:
	bl L__Unwind_Resume$stub
LEHE5:
LBE110:
LBE109:
LBE108:
LBE76:
LFE1606:
.data
.section __DATA,__gcc_except_tab
LLSDA1606:
	.byte	0xff
	.byte	0xff
	.byte	0x3
	.byte	0x34
	.set L$set$5,LEHB2-LFB1606
	.long L$set$5
	.set L$set$6,LEHE2-LEHB2
	.long L$set$6
	.set L$set$7,L196-LFB1606
	.long L$set$7
	.byte	0x0
	.set L$set$8,LEHB3-LFB1606
	.long L$set$8
	.set L$set$9,LEHE3-LEHB3
	.long L$set$9
	.set L$set$10,L195-LFB1606
	.long L$set$10
	.byte	0x0
	.set L$set$11,LEHB4-LFB1606
	.long L$set$11
	.set L$set$12,LEHE4-LEHB4
	.long L$set$12
	.set L$set$13,L194-LFB1606
	.long L$set$13
	.byte	0x0
	.set L$set$14,LEHB5-LFB1606
	.long L$set$14
	.set L$set$15,LEHE5-LEHB5
	.long L$set$15
	.long	0x0
	.byte	0x0
.section __TEXT,__text,regular,pure_instructions
	.align 2
	.stabs	"_ZN13XSignalBufferC4Ev:F(0,2)",36,0,52,__ZN13XSignalBufferC4Ev
	.stabs	"this:P(0,48)",64,0,52,29
	.stabs	"__num_elements:r(0,67)",64,0,464,0
	.stabn	192,0,0,LBB80
	.stabs	"this:r(0,231)=k(0,193)",64,0,131,30
	.stabn	192,0,0,LBB83
	.stabn	224,0,0,LBE83
	.stabs	"this:r(0,231)",64,0,131,28
	.stabn	192,0,0,LBB86
	.stabn	224,0,0,LBE86
	.stabs	"this:r(0,231)",64,0,131,30
	.stabn	192,0,0,LBB92
	.stabn	224,0,0,LBE92
	.stabs	"this:r(0,231)",64,0,131,28
	.stabn	192,0,0,LBB95
	.stabn	224,0,0,LBE95
	.stabn	224,0,0,LBE80
	.stabs	"t:r(0,6)",64,0,212,0
	.stabn	192,0,0,LBB103
	.stabn	224,0,0,LBE103
.section __TEXT,__text,regular,pure_instructions
	.align 2
Lscope21:
	.stabs	"",36,0,0,Lscope21-__ZN13XSignalBufferC4Ev
	.align 2
	.globl __ZN13XSignalBufferD2Ev
.section __TEXT,__text,regular,pure_instructions
	.align 2
__ZN13XSignalBufferD2Ev:
	.stabs	"../../kame/xsignal.cpp",132,0,0,0
	.stabd	68,0,57
LFB1609:
	li r4,0
	b __ZN13XSignalBufferD4Ev
LFE1609:
	.stabs	"_ZN13XSignalBufferD2Ev:F(0,2)",36,0,57,__ZN13XSignalBufferD2Ev
	.stabs	"this:P(0,48)",64,0,57,3
.section __TEXT,__text,regular,pure_instructions
	.align 2
Lscope22:
	.stabs	"",36,0,0,Lscope22-__ZN13XSignalBufferD2Ev
	.align 2
	.globl __ZN13XSignalBufferD1Ev
.section __TEXT,__text,regular,pure_instructions
	.align 2
__ZN13XSignalBufferD1Ev:
	.stabd	68,0,57
LFB1611:
	li r4,2
	b __ZN13XSignalBufferD4Ev
LFE1611:
	.stabs	"_ZN13XSignalBufferD1Ev:F(0,2)",36,0,57,__ZN13XSignalBufferD1Ev
	.stabs	"this:P(0,48)",64,0,57,3
.section __TEXT,__text,regular,pure_instructions
	.align 2
Lscope23:
	.stabs	"",36,0,0,Lscope23-__ZN13XSignalBufferD1Ev
	.align 2
	.globl __ZN13XSignalBufferD4Ev
.section __TEXT,__text,regular,pure_instructions
	.align 2
__ZN13XSignalBufferD4Ev:
	.stabd	68,0,57
LFB1612:
L202:
L208:
L214:
L220:
L226:
L228:
	mflr r0
	stmw r29,-12(r1)
LCFI22:
	stw r0,8(r1)
LCFI23:
	mr r30,r3
	stwu r1,-144(r1)
LCFI24:
	.stabs	"../../kame/atomic.h",132,0,0,0
	.stabd	68,0,214
LBB139:
	addi r3,r3,40
LEHB6:
	bl L__ZN6XMutexD1Ev$stub
LEHE6:
	b L231
L271:
L230:
L243:
L245:
L250:
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_deque.h",132,0,0,0
	.stabd	68,0,133
LBB140:
LBB141:
LBB142:
LBB143:
LBB144:
	addi r2,r30,8
LBE144:
LBB145:
	addi r9,r30,24
	.stabd	68,0,134
LBE145:
LBB146:
LBB147:
LBB148:
	lwz r6,12(r2)
	.stabs	"../../kame/atomic.h",132,0,0,0
	.stabd	68,0,214
	mr r29,r3
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_deque.h",132,0,0,0
	.stabd	68,0,134
	lwz r11,4(r2)
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_construct.h",132,0,0,0
	.stabd	68,0,115
LBE148:
LBE147:
LBE146:
LBE143:
	mr r3,r30
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_deque.h",132,0,0,0
	.stabd	68,0,134
LBB149:
LBB150:
LBB151:
LBB152:
	lwz r10,8(r2)
LBE152:
LBE151:
LBE150:
LBB153:
LBB154:
LBB155:
	lwz r5,12(r9)
LBE155:
LBE154:
LBE153:
LBB156:
LBB157:
LBB158:
	lwz r0,8(r30)
LBE158:
LBE157:
LBE156:
LBB159:
LBB160:
LBB161:
	lwz r8,24(r30)
	lwz r2,4(r9)
	lwz r7,8(r9)
LBE161:
LBE160:
LBE159:
LBB162:
LBB163:
LBB164:
LBB165:
LBB166:
	stw r0,96(r1)
	stw r11,100(r1)
	stw r10,104(r1)
	stw r6,108(r1)
LBE166:
LBE165:
LBE164:
LBB167:
LBB168:
LBB169:
	stw r8,112(r1)
	stw r2,116(r1)
	stw r7,120(r1)
	stw r5,124(r1)
LBE169:
LBE168:
LBE167:
LBE163:
LBE162:
LBB170:
LBB171:
LBB172:
	stw r0,64(r1)
	stw r11,68(r1)
	stw r10,72(r1)
	stw r6,76(r1)
LBE172:
LBE171:
LBE170:
LBB173:
LBB174:
LBB175:
	stw r8,80(r1)
	stw r2,84(r1)
	stw r7,88(r1)
	stw r5,92(r1)
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_construct.h",132,0,0,0
	.stabd	68,0,115
LBE175:
LBE174:
LBE173:
LBE149:
	bl L__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED2Ev$stub
	mr r3,r29
LEHB7:
	bl L__Unwind_Resume$stub
L231:
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_deque.h",132,0,0,0
	.stabd	68,0,133
L263:
L265:
LBE142:
LBE141:
LBE140:
	addi r2,r30,8
	addi r9,r30,24
	.stabd	68,0,134
	lwz r0,8(r30)
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_construct.h",132,0,0,0
	.stabd	68,0,115
	mr r3,r30
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_deque.h",132,0,0,0
	.stabd	68,0,134
	lwz r6,12(r2)
	lwz r11,4(r2)
	lwz r10,8(r2)
	lwz r8,24(r30)
	lwz r5,12(r9)
	lwz r2,4(r9)
	lwz r7,8(r9)
	stw r0,96(r1)
	stw r0,64(r1)
	stw r11,100(r1)
	stw r10,104(r1)
	stw r6,108(r1)
	stw r8,112(r1)
	stw r2,116(r1)
	stw r7,120(r1)
	stw r5,124(r1)
	stw r11,68(r1)
	stw r10,72(r1)
	stw r6,76(r1)
	stw r8,80(r1)
	stw r2,84(r1)
	stw r7,88(r1)
	stw r5,92(r1)
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_construct.h",132,0,0,0
	.stabd	68,0,115
	bl L__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED2Ev$stub
LEHE7:
	lwz r0,152(r1)
	addi r1,r1,144
	lmw r29,-12(r1)
	mtlr r0
	blr
LBE139:
LFE1612:
.data
.section __DATA,__gcc_except_tab
LLSDA1612:
	.byte	0xff
	.byte	0xff
	.byte	0x3
	.byte	0x1a
	.set L$set$16,LEHB6-LFB1612
	.long L$set$16
	.set L$set$17,LEHE6-LEHB6
	.long L$set$17
	.set L$set$18,L271-LFB1612
	.long L$set$18
	.byte	0x0
	.set L$set$19,LEHB7-LFB1612
	.long L$set$19
	.set L$set$20,LEHE7-LEHB7
	.long L$set$20
	.long	0x0
	.byte	0x0
.section __TEXT,__text,regular,pure_instructions
	.align 2
	.stabs	"_ZN13XSignalBufferD4Ev:F(0,2)",36,0,57,__ZN13XSignalBufferD4Ev
	.stabs	"this:P(0,48)",64,0,57,30
	.stabs	"__in_chrg:P(0,10)",64,0,57,4
	.stabs	"__x:r(0,194)",64,0,133,2
	.stabn	192,0,0,LBB144
	.stabn	224,0,0,LBE144
	.stabs	"__x:r(0,194)",64,0,133,9
	.stabn	192,0,0,LBB145
	.stabn	224,0,0,LBE145
	.stabs	"__x:r(0,194)",64,0,133,2
	.stabn	192,0,0,LBB146
	.stabn	224,0,0,LBE146
	.stabs	"__x:r(0,194)",64,0,133,2
	.stabn	192,0,0,LBB150
	.stabn	224,0,0,LBE150
	.stabs	"__x:r(0,194)",64,0,133,9
	.stabn	192,0,0,LBB153
	.stabn	224,0,0,LBE153
	.stabs	"__x:r(0,194)",64,0,133,2
	.stabn	192,0,0,LBB156
	.stabn	224,0,0,LBE156
	.stabs	"__x:r(0,194)",64,0,133,9
	.stabn	192,0,0,LBB159
	.stabn	224,0,0,LBE159
	.stabs	"__x:r(0,194)",64,0,133,2
	.stabn	192,0,0,LBB170
	.stabn	224,0,0,LBE170
	.stabs	"__x:r(0,194)",64,0,133,9
	.stabn	192,0,0,LBB173
	.stabn	224,0,0,LBE173
.section __TEXT,__text,regular,pure_instructions
	.align 2
Lscope24:
	.stabs	"",36,0,0,Lscope24-__ZN13XSignalBufferD4Ev
.data
.cstring
	.align 2
LC1:
	.ascii "Talking cost exceeds 10, ../../kame/xsignal.cpp:%d\12\0"
.section __TEXT,__text,regular,pure_instructions
	.align 2
	.align 2
	.globl __ZN13XSignalBuffer23registerTransactionListEP13_XTransaction
.section __TEXT,__text,regular,pure_instructions
	.align 2
__ZN13XSignalBuffer23registerTransactionListEP13_XTransaction:
	.stabs	"../../kame/xsignal.cpp",132,0,0,0
	.stabd	68,0,61
LFB1628:
	mflr r0
	stmw r21,-44(r1)
LCFI25:
	stw r0,8(r1)
LCFI26:
	bcl 20,31,"L00000000010$pb"
"L00000000010$pb":
	stwu r1,-144(r1)
LCFI27:
	mflr r31
	stw r4,172(r1)
	mr r27,r3
	.stabd	68,0,62
LBB176:
	lwz r28,4(r4)
	.stabs	"../../kame/support.h",132,0,0,0
	.stabd	68,0,84
LEHB8:
LBB177:
	bl __Z12isMainThreadv
	.stabd	68,0,85
LBB178:
	cmpwi cr7,r3,1
	.stabd	68,0,84
LBE178:
	addis r3,r31,ha16(LC0-"L00000000010$pb")
	la r3,lo16(LC0-"L00000000010$pb")(r3)
	.stabd	68,0,85
LBB179:
	bne+ cr7,L274
	li r4,63
	bl L__Z9my_assertPKci$stub
L274:
	.stabs	"../../kame/xsignal.cpp",132,0,0,0
	.stabd	68,0,64
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000010$pb")
	lis r23,0x1062
	lis r24,0x98
	lis r25,0x431b
	lwz r21,lo16(L___sF$non_lazy_ptr-"L00000000010$pb")(r2)
	ori r23,r23,19923
	ori r24,r24,38528
	ori r25,r25,56963
LBE179:
LBE177:
LBB180:
	li r30,0
	addi r22,r1,80
	li r26,10
L293:
	.stabd	68,0,65
LBB181:
LBB182:
	li r29,0
	.stabs	"../../kame/atomic.h",132,0,0,0
	.stabd	68,0,120
LBB183:
LBB184:
LBB185:
LBB186:
	isync
	.stabd	68,0,215
LBE186:
LBE185:
	lwz r0,88(r27)
LBE184:
LBE183:
	cmpwi cr7,r0,0
	beq- cr7,L279
	.stabd	68,0,120
LBB187:
LBB188:
LBB189:
LBB190:
	isync
	.stabd	68,0,215
LBE190:
LBE189:
	lwz r0,84(r27)
LBE188:
LBE187:
	subf r29,r0,r28
L279:
	.stabd	68,0,120
LBB191:
LBB192:
LBB193:
LBB194:
	isync
	.stabd	68,0,215
LBE194:
LBE193:
	lwz r0,88(r27)
LBE192:
LBE191:
	cmplwi cr7,r0,1999
	ble- cr7,L284
	.stabd	68,0,120
LBB195:
LBB196:
LBB197:
LBB198:
	isync
	.stabd	68,0,215
LBE198:
LBE197:
	lwz r0,88(r27)
LBE196:
LBE195:
	mulhwu r0,r0,r23
	srwi r0,r0,7
	mullw r29,r29,r0
L284:
	.stabs	"../../kame/xsignal.cpp",132,0,0,0
	.stabd	68,0,72
	cmplwi cr7,r29,49999
	ble- cr7,L276
	.stabd	68,0,73
	cmplw cr7,r29,r24
	ble+ cr7,L290
	.stabd	68,0,74
	addis r4,r31,ha16(LC1-"L00000000010$pb")
	addi r3,r21,176
	la r4,lo16(LC1-"L00000000010$pb")(r4)
	li r5,74
	bl L_fprintf$stub
L290:
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_algobase.h",132,0,0,0
	.stabd	68,0,152
LBB199:
	mullw r0,r29,r30
	stw r26,84(r1)
	.stabd	68,0,156
LBB200:
	addi r3,r1,84
	.stabd	68,0,152
LBE200:
	mulhwu r0,r0,r25
	srwi r0,r0,18
	.stabd	68,0,156
LBB201:
	cmplw cr7,r26,r0
	.stabd	68,0,152
LBE201:
	stw r0,80(r1)
	.stabd	68,0,156
LBB202:
	blt- cr7,L292
	mr r3,r22
L292:
	.stabd	68,0,152
LBE202:
LBE199:
	lwz r3,0(r3)
	.stabs	"../../kame/xsignal.cpp",132,0,0,0
	.stabd	68,0,64
LBE182:
LBE181:
	addi r30,r30,1
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_algobase.h",132,0,0,0
	.stabd	68,0,152
LBB203:
LBB204:
	bl L__Z9msecsleepj$stub
	.stabs	"../../kame/xsignal.cpp",132,0,0,0
	.stabd	68,0,76
	bl L__Z9timeStampv$stub
	.stabd	68,0,64
LBE204:
LBE203:
	cmplwi cr7,r30,19
	.stabd	68,0,76
LBB205:
LBB206:
	mr r28,r3
	.stabd	68,0,64
LBE206:
LBE205:
	ble+ cr7,L293
L276:
	.stabs	"../../kame/atomic.h",132,0,0,0
	.stabd	68,0,120
LBE180:
LBB207:
LBB208:
LBB209:
LBB210:
	isync
	.stabd	68,0,215
LBE210:
LBE209:
	lwz r0,88(r27)
LBE208:
LBE207:
	cmpwi cr7,r0,0
	bne- cr7,L294
	.stabd	68,0,216
LBB211:
LBB212:
	stw r28,84(r27)
	.stabd	68,0,123
LBB213:
LBB214:
	sync
L294:
	.stabs	"../../kame/thread.h",132,0,0,0
	.stabd	68,0,20
LBE214:
LBE213:
LBE212:
LBE211:
LBB215:
LBB216:
	addi r0,r27,40
	.stabd	68,0,21
LBB217:
LBB218:
	mr r3,r0
	.stabd	68,0,20
	stw r0,64(r1)
	.stabd	68,0,21
	bl L__ZN6XMutex4lockEv$stub
LEHE8:
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_deque.h",132,0,0,0
	.stabd	68,0,1074
LBE218:
LBE217:
LBE216:
LBB219:
LBB220:
	lwz r2,32(r27)
	lwz r9,24(r27)
	addi r2,r2,-4
	cmpw cr7,r9,r2
	beq- cr7,L301
	.stabs	"/usr/include/gcc/darwin/3.3/c++/new",132,0,0,0
	.stabd	68,0,92
L303:
LBB221:
LBB222:
	cmpwi cr7,r9,0
	beq- cr7,L318
	lwz r0,172(r1)
	stw r0,0(r9)
	lwz r9,24(r27)
L307:
	.stabd	68,0,96
L313:
L316:
L318:
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_deque.h",132,0,0,0
	.stabd	68,0,1076
LBE222:
LBE221:
	addi r0,r9,4
	stw r0,24(r27)
	b L320
L301:
	.stabd	68,0,1079
	addi r4,r1,172
	mr r3,r27
LEHB9:
	bl L__ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_push_back_auxERKS1_$stub
LEHE9:
	b L320
L335:
L323:
L329:
	mr r29,r3
	.stabs	"../../kame/thread.h",132,0,0,0
	.stabd	68,0,24
LBE220:
LBE219:
LBB223:
LBB224:
LBB225:
	lwz r3,64(r1)
	bl L__ZN6XMutex6unlockEv$stub
	mr r3,r29
LEHB10:
	bl L__Unwind_Resume$stub
L320:
	.stabs	"../../kame/atomic.h",132,0,0,0
	.stabd	68,0,217
LBE225:
LBE224:
LBE223:
LBB226:
	addi r0,r27,88
	.stabd	68,0,162
LBB227:
LBB228:
LBB229:
	1: 
lwarx r2, 0, r0 
addi r2, r2, 1 
stwcx. r2, 0, r0 
bne- 1b
	.stabs	"../../kame/thread.h",132,0,0,0
	.stabd	68,0,24
LBE229:
LBE228:
LBE227:
LBE226:
	lwz r3,64(r1)
	bl L__ZN6XMutex6unlockEv$stub
LEHE10:
	lwz r0,152(r1)
	addi r1,r1,144
	mtlr r0
	lmw r21,-44(r1)
	blr
LBE215:
LBE176:
LFE1628:
.data
.section __DATA,__gcc_except_tab
LLSDA1628:
	.byte	0xff
	.byte	0xff
	.byte	0x3
	.byte	0x27
	.set L$set$21,LEHB8-LFB1628
	.long L$set$21
	.set L$set$22,LEHE8-LEHB8
	.long L$set$22
	.long	0x0
	.byte	0x0
	.set L$set$23,LEHB9-LFB1628
	.long L$set$23
	.set L$set$24,LEHE9-LEHB9
	.long L$set$24
	.set L$set$25,L335-LFB1628
	.long L$set$25
	.byte	0x0
	.set L$set$26,LEHB10-LFB1628
	.long L$set$26
	.set L$set$27,LEHE10-LEHB10
	.long L$set$27
	.long	0x0
	.byte	0x0
.section __TEXT,__text,regular,pure_instructions
	.align 2
	.stabs	"_ZN13XSignalBuffer23registerTransactionListEP13_XTransaction:F(0,2)",36,0,61,__ZN13XSignalBuffer23registerTransactionListEP13_XTransaction
	.stabs	"this:P(0,48)",64,0,61,27
	.stabs	"transaction:p(0,62)",160,0,61,172
	.stabs	"time:r(0,69)",64,0,62,28
	.stabn	192,0,0,LBB176
	.stabs	"file:r(0,36)",64,0,84,3
	.stabn	192,0,0,LBB177
	.stabn	224,0,0,LBE177
	.stabs	"i:r(0,6)",64,0,64,30
	.stabn	192,0,0,LBB180
	.stabs	"cost:r(0,69)",64,0,65,29
	.stabn	192,0,0,LBB182
	.stabn	224,0,0,LBE182
	.stabs	"cost:r(0,69)",64,0,65,29
	.stabn	192,0,0,LBB204
	.stabn	224,0,0,LBE204
	.stabs	"cost:r(0,69)",64,0,65,29
	.stabn	192,0,0,LBB206
	.stabn	224,0,0,LBE206
	.stabn	224,0,0,LBE180
	.stabs	"lock:(0,232)=xsXScopedLock<XMutex>:",128,0,79,64
	.stabs	"XScopedLock<XMutex>:Tt(0,232)=s4m_mutex:/0(0,117),0,32;operator=::(0,233)=#(0,232),(0,234)=&(0,232),(0,235)=*(0,232),(0,236)=&(0,237)=k(0,232),(0,2);:_ZN11XScopedLockI6XMutexEaSERKS1_;2A.;__base_ctor::(0,238)=#(0,232),(0,2),(0,235),(0,236),(0,2);:_ZN11XScopedLockI6XMutexEC2ERKS1_;2A.;__comp_ctor::(0,238):_ZN11XScopedLockI6XMutexEC1ERKS1_;2A.;XScopedLock::(0,239)=#(0,232),(0,2),(0,235),(0,117),(0,2);:_ZN11XScopedLockI6XMutexEC4ERS0_;2A.;__base_ctor::(0,239):_ZN11XScopedLockI6XMutexEC2ERS0_;2A.;__comp_ctor::(0,239):_ZN11XScopedLockI6XMutexEC1ERS0_;2A.;XScopedLock::(0,240)=#(0,232),(0,2),(0,235),(0,10),(0,2);:_ZN11XScopedLockI6XMutexED4Ev;2A.;__base_dtor::(0,241)=#(0,232),(0,2),(0,235),(0,2);:_ZN11XScopedLockI6XMutexED2Ev;2A.;__comp_dtor::(0,241):_ZN11XScopedLockI6XMutexED1Ev;2A.;;",128,0,19,0
	.stabn	192,0,0,LBB215
	.stabs	"mutex:r(0,117)",64,0,20,0
	.stabn	192,0,0,LBB216
	.stabn	224,0,0,LBE216
	.stabs	"this:r(0,242)=k(0,141)",64,0,217,0
	.stabn	192,0,0,LBB226
	.stabs	"ret:r(0,6)",64,0,161,2
	.stabn	192,0,0,LBB229
	.stabn	224,0,0,LBE229
	.stabn	224,0,0,LBE226
	.stabn	224,0,0,LBE215
	.stabn	224,0,0,LBE176
.section __TEXT,__text,regular,pure_instructions
	.align 2
Lscope25:
	.stabs	"",36,0,0,Lscope25-__ZN13XSignalBuffer23registerTransactionListEP13_XTransaction
.data
.cstring
	.align 2
LC2:
	.ascii "Unhandled Exception Occurs!\0"
.section __TEXT,__text,regular,pure_instructions
	.align 2
	.align 2
	.globl __ZN13XSignalBuffer11synchronizeEv
.section __TEXT,__text,regular,pure_instructions
	.align 2
__ZN13XSignalBuffer11synchronizeEv:
	.stabs	"../../kame/xsignal.cpp",132,0,0,0
	.stabd	68,0,86
LFB1637:
	mflr r0
	stmw r23,-36(r1)
LCFI28:
	stw r0,8(r1)
LCFI29:
	bcl 20,31,"L00000000011$pb"
"L00000000011$pb":
	stwu r1,-208(r1)
LCFI30:
	mflr r31
	mr r29,r3
	.stabd	68,0,88
LBB230:
	addi r3,r1,64
LEHB11:
	bl L__ZN5XTime3nowEv$stub
	.stabd	68,0,87
	li r23,1
	.stabd	68,0,89
	bl L__Z9timeStampv$stub
	.stabd	68,0,91
	li r25,0
	.stabd	68,0,89
	mr r24,r3
	.stabd	68,0,93
L338:
	.stabs	"../../kame/atomic.h",132,0,0,0
	.stabd	68,0,120
LBB231:
LBB232:
LBB233:
LBB234:
LBB235:
LBB236:
LBB237:
	isync
	.stabd	68,0,215
LBE237:
LBE236:
	lwz r0,88(r29)
LBE235:
LBE234:
	cmplw cr7,r0,r25
	bgt+ cr7,L341
	.stabd	68,0,120
LBB238:
LBB239:
LBB240:
LBB241:
	isync
	.stabd	68,0,215
LBE241:
LBE240:
	lwz r0,88(r29)
LBE239:
LBE238:
	addic r2,r0,-1
	subfe r23,r2,r0
	.stabs	"../../kame/xsignal.cpp",132,0,0,0
	.stabd	68,0,96
	b L339
L341:
	.stabs	"../../kame/thread.h",132,0,0,0
	.stabd	68,0,20
LBB242:
LBB243:
	addi r28,r29,40
	.stabd	68,0,21
LBB244:
LBB245:
	mr r3,r28
	.stabd	68,0,20
	stw r28,80(r1)
	.stabd	68,0,21
	bl L__ZN6XMutex4lockEv$stub
LEHE11:
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_deque.h",132,0,0,0
	.stabd	68,0,1118
LBE245:
LBE244:
LBE243:
LBB246:
LBB247:
	lwz r2,16(r29)
	lwz r11,8(r29)
	addi r2,r2,-4
	cmpw cr7,r11,r2
	.stabd	68,0,985
LBE247:
LBE246:
	lwz r0,0(r11)
	stw r0,144(r1)
	.stabd	68,0,1118
LBB248:
LBB249:
	beq- cr7,L350
	.stabd	68,0,1120
	addi r0,r11,4
	stw r0,8(r29)
	b L353
L350:
	.stabd	68,0,1123
	mr r3,r29
LEHB12:
	bl L__ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_pop_front_auxEv$stub
LEHE12:
L353:
	.stabs	"../../kame/atomic.h",132,0,0,0
	.stabd	68,0,222
LBE249:
LBE248:
LBB250:
	addi r27,r29,88
	.stabd	68,0,185
LBB251:
LBB252:
LBB253:
LBB254:
	li r0,-1
	.stabd	68,0,187
LBB255:
	1: 
lwarx r2, 0, r27 
add r2, r2, r0 
stwcx. r2, 0, r27 
bne- 1b
	.stabd	68,0,222
LBE255:
LBE254:
LBE253:
LBE252:
LBE251:
LBE250:
	cmpwi cr7,r2,0
	beq- cr7,L363
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_deque.h",132,0,0,0
	.stabd	68,0,136
LBB256:
LBB257:
LBB258:
LBB259:
LBB260:
	lwz r2,8(r29)
	.stabd	68,0,985
LBE260:
LBE259:
LBE258:
LBE257:
	lwz r9,0(r2)
	lwz r0,4(r9)
	.stabs	"../../kame/atomic.h",132,0,0,0
	.stabd	68,0,216
LBB261:
	stw r0,84(r29)
	.stabd	68,0,123
LBB262:
LBB263:
	sync
	.stabd	68,0,985
LBE263:
LBE262:
LBE261:
LBE256:
	b L363
L456:
L362:
L368:
	mr r30,r3
	.stabs	"../../kame/thread.h",132,0,0,0
	.stabd	68,0,24
LBB264:
LBB265:
LBB266:
	lwz r3,80(r1)
	b L458
L363:
LBE266:
LBE265:
LBE264:
	lwz r3,80(r1)
	.stabs	"../../kame/xsignal.cpp",132,0,0,0
	.stabd	68,0,106
LBE242:
	li r26,0
	.stabs	"../../kame/thread.h",132,0,0,0
	.stabd	68,0,24
LEHB13:
LBB267:
	bl L__ZN6XMutex6unlockEv$stub
LEHE13:
	.stabs	"../../kame/xsignal.cpp",132,0,0,0
	.stabd	68,0,108
LBE267:
	lwz r3,144(r1)
	lwz r2,0(r3)
	lwz r12,8(r2)
	mtctr r12
LEHB14:
	bctrl
LEHE14:
	mr r26,r3
	b L374
	.stabd	68,0,110
L455:
	cmpwi cr7,r4,2
	bne+ cr7,L380
L375:
LBB268:
	bl L___cxa_begin_catch$stub
	.stabd	68,0,111
LEHB15:
	bl L__ZN10XKameError5printEv$stub
LEHE15:
	b L377
L454:
	b L453
L376:
L378:
L377:
LEHB16:
	bl L___cxa_end_catch$stub
LEHE16:
LBE268:
	b L403
	.stabd	68,0,113
L380:
	bl L___cxa_begin_catch$stub
	.stabd	68,0,114
	addi r30,r1,96
	addis r4,r31,ha16(LC2-"L00000000011$pb")
	mr r3,r30
	la r4,lo16(LC2-"L00000000011$pb")(r4)
LEHB17:
	bl L__ZN7QStringC1EPKc$stub
LEHE17:
	addis r4,r31,ha16(LC0-"L00000000011$pb")
	mr r3,r30
	la r4,lo16(LC0-"L00000000011$pb")(r4)
	li r5,114
LEHB18:
	bl L__Z10_gErrPrintRK7QStringPKci$stub
LEHE18:
	b L382
L452:
L381:
	.stabs	"/sw/include/qt/qshared.h",132,0,0,0
	.stabd	68,0,50
	lwz r2,96(r1)
	.stabs	"../../kame/xsignal.cpp",132,0,0,0
	.stabd	68,0,114
	mr r30,r3
	.stabs	"/sw/include/qt/qshared.h",132,0,0,0
	.stabd	68,0,50
	lwz r9,0(r2)
	addi r9,r9,-1
	cmpwi cr7,r9,0
	stw r9,0(r2)
	bne+ cr7,L399
	.stabs	"/sw/include/qt/qstring.h",132,0,0,0
	.stabd	68,0,847
	addis r2,r31,ha16(L__ZN7QString11shared_nullE$non_lazy_ptr-"L00000000011$pb")
	lwz r3,96(r1)
	lwz r2,lo16(L__ZN7QString11shared_nullE$non_lazy_ptr-"L00000000011$pb")(r2)
	lwz r0,0(r2)
	cmpw cr7,r3,r0
	beq+ cr7,L399
	.stabd	68,0,848
L390:
	bl L__ZN11QStringData10deleteSelfEv$stub
	b L399
L382:
	.stabs	"/sw/include/qt/qshared.h",132,0,0,0
	.stabd	68,0,50
	lwz r2,96(r1)
	lwz r9,0(r2)
	addi r9,r9,-1
	cmpwi cr7,r9,0
	stw r9,0(r2)
	bne- cr7,L400
	.stabs	"/sw/include/qt/qstring.h",132,0,0,0
	.stabd	68,0,847
	addis r2,r31,ha16(L__ZN7QString11shared_nullE$non_lazy_ptr-"L00000000011$pb")
	lwz r3,96(r1)
	lwz r2,lo16(L__ZN7QString11shared_nullE$non_lazy_ptr-"L00000000011$pb")(r2)
	lwz r0,0(r2)
	cmpw cr7,r3,r0
	beq- cr7,L400
	.stabd	68,0,848
LEHB19:
	bl L__ZN11QStringData10deleteSelfEv$stub
LEHE19:
	b L400
L453:
	mr r30,r3
L399:
L401:
	bl L___cxa_end_catch$stub
	b L457
L400:
LEHB20:
	bl L___cxa_end_catch$stub
L374:
	.stabs	"../../kame/xsignal.cpp",132,0,0,0
	.stabd	68,0,117
LBB269:
	cmpwi cr7,r26,0
	beq+ cr7,L403
	.stabs	"../../kame/thread.h",132,0,0,0
	.stabd	68,0,21
LBB270:
LBB271:
LBB272:
LBB273:
LBB274:
	mr r3,r28
	.stabd	68,0,20
	stw r28,112(r1)
	.stabd	68,0,21
	bl L__ZN6XMutex4lockEv$stub
LEHE20:
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_deque.h",132,0,0,0
	.stabd	68,0,1074
LBE274:
LBE273:
LBE272:
LBB275:
LBB276:
	lwz r2,32(r29)
	lwz r9,24(r29)
	.stabs	"../../kame/xsignal.cpp",132,0,0,0
	.stabd	68,0,118
LBE276:
LBE275:
LBE271:
	addi r25,r25,1
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_deque.h",132,0,0,0
	.stabd	68,0,1074
LBB277:
LBB278:
LBB279:
	addi r2,r2,-4
	cmpw cr7,r9,r2
	beq- cr7,L406
	.stabs	"/usr/include/gcc/darwin/3.3/c++/new",132,0,0,0
	.stabd	68,0,92
L408:
LBB280:
LBB281:
	cmpwi cr7,r9,0
	beq- cr7,L423
	lwz r0,144(r1)
	stw r0,0(r9)
	lwz r9,24(r29)
L412:
	.stabd	68,0,96
L418:
L421:
L423:
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_deque.h",132,0,0,0
	.stabd	68,0,1076
LBE281:
LBE280:
	addi r0,r9,4
	stw r0,24(r29)
	b L425
L406:
	.stabd	68,0,1079
	addi r4,r1,144
	mr r3,r29
LEHB21:
	bl L__ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_push_back_auxERKS1_$stub
LEHE21:
L425:
	.stabs	"../../kame/atomic.h",132,0,0,0
	.stabd	68,0,162
LBE279:
LBE278:
LBB282:
LBB283:
LBB284:
LBB285:
	1: 
lwarx r9, 0, r27 
addi r9, r9, 1 
stwcx. r9, 0, r27 
bne- 1b
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_deque.h",132,0,0,0
	.stabd	68,0,136
LBE285:
LBE284:
LBE283:
LBE282:
LBB286:
LBB287:
LBB288:
LBB289:
LBB290:
	lwz r2,8(r29)
	.stabd	68,0,985
LBE290:
LBE289:
LBE288:
LBE287:
	lwz r9,0(r2)
	lwz r0,4(r9)
	.stabs	"../../kame/atomic.h",132,0,0,0
	.stabd	68,0,216
LBB291:
	stw r0,84(r29)
	.stabd	68,0,123
LBB292:
LBB293:
	sync
	.stabs	"../../kame/thread.h",132,0,0,0
	.stabd	68,0,24
LBE293:
LBE292:
LBE291:
LBE286:
	lwz r3,112(r1)
LEHB22:
	bl L__ZN6XMutex6unlockEv$stub
LEHE22:
LBE277:
LBE270:
	b L444
L451:
L432:
L438:
	mr r30,r3
LBB294:
LBB295:
LBB296:
LBB297:
LBB298:
	lwz r3,112(r1)
L458:
	bl L__ZN6XMutex6unlockEv$stub
L457:
	mr r3,r30
LEHB23:
	bl L__Unwind_Resume$stub
L403:
	.stabs	"../../kame/xsignal.cpp",132,0,0,0
	.stabd	68,0,126
LBE298:
LBE297:
LBE296:
LBE295:
LBE294:
	lwz r3,144(r1)
	cmpwi cr7,r3,0
	beq+ cr7,L444
	lwz r2,0(r3)
	lwz r12,4(r2)
	mtctr r12
	bctrl
L444:
	.stabd	68,0,128
LBE269:
	bl L__Z9timeStampv$stub
	lis r0,0x1062
	subf r3,r24,r3
	ori r0,r0,19923
	mulhwu r3,r3,r0
	srwi r3,r3,6
	cmplwi cr7,r3,30
	ble+ cr7,L338
L339:
	.stabs	"../../kame/xtime.h",132,0,0,0
	.stabd	68,0,27
LBE233:
LBE232:
LBE231:
LBB299:
	addi r3,r1,128
	bl L__ZN5XTime3nowEv$stub
	lwz r0,68(r1)
	lwz r11,132(r1)
LBB300:
	lis r2,0x1062
	ori r2,r2,19923
	lwz r9,64(r1)
	subf r11,r0,r11
	lwz r0,128(r1)
	mulhw r2,r11,r2
	srawi r11,r11,31
	subf r0,r9,r0
	mulli r0,r0,1000
	srawi r2,r2,6
	subf r2,r11,r2
	add r0,r0,r2
	.stabs	"../../kame/xsignal.cpp",132,0,0,0
	.stabd	68,0,131
LBE300:
LBE299:
	cmplwi cr7,r0,10
	ble+ cr7,L449
	.stabd	68,0,132
	bl L__Z9timeStampv$stub
LEHE23:
L449:
	.stabd	68,0,134
	lwz r0,216(r1)
	addi r1,r1,208
LBE230:
	xori r3,r23,1
	lmw r23,-36(r1)
	mtlr r0
	blr
LFE1637:
.data
.section __DATA,__gcc_except_tab
	.align 2
LLSDA1637:
	.byte	0xff
	.byte	0x0
	.byte	0xb8,0x1
	.byte	0x3
	.byte	0xa9,0x1
	.set L$set$28,LEHB11-LFB1637
	.long L$set$28
	.set L$set$29,LEHE11-LEHB11
	.long L$set$29
	.long	0x0
	.byte	0x0
	.set L$set$30,LEHB12-LFB1637
	.long L$set$30
	.set L$set$31,LEHE12-LEHB12
	.long L$set$31
	.set L$set$32,L456-LFB1637
	.long L$set$32
	.byte	0x0
	.set L$set$33,LEHB13-LFB1637
	.long L$set$33
	.set L$set$34,LEHE13-LEHB13
	.long L$set$34
	.long	0x0
	.byte	0x0
	.set L$set$35,LEHB14-LFB1637
	.long L$set$35
	.set L$set$36,LEHE14-LEHB14
	.long L$set$36
	.set L$set$37,L455-LFB1637
	.long L$set$37
	.byte	0x3
	.set L$set$38,LEHB15-LFB1637
	.long L$set$38
	.set L$set$39,LEHE15-LEHB15
	.long L$set$39
	.set L$set$40,L454-LFB1637
	.long L$set$40
	.byte	0x0
	.set L$set$41,LEHB16-LFB1637
	.long L$set$41
	.set L$set$42,LEHE16-LEHB16
	.long L$set$42
	.long	0x0
	.byte	0x0
	.set L$set$43,LEHB17-LFB1637
	.long L$set$43
	.set L$set$44,LEHE17-LEHB17
	.long L$set$44
	.set L$set$45,L453-LFB1637
	.long L$set$45
	.byte	0x0
	.set L$set$46,LEHB18-LFB1637
	.long L$set$46
	.set L$set$47,LEHE18-LEHB18
	.long L$set$47
	.set L$set$48,L452-LFB1637
	.long L$set$48
	.byte	0x0
	.set L$set$49,LEHB19-LFB1637
	.long L$set$49
	.set L$set$50,LEHE19-LEHB19
	.long L$set$50
	.set L$set$51,L453-LFB1637
	.long L$set$51
	.byte	0x0
	.set L$set$52,LEHB20-LFB1637
	.long L$set$52
	.set L$set$53,LEHE20-LEHB20
	.long L$set$53
	.long	0x0
	.byte	0x0
	.set L$set$54,LEHB21-LFB1637
	.long L$set$54
	.set L$set$55,LEHE21-LEHB21
	.long L$set$55
	.set L$set$56,L451-LFB1637
	.long L$set$56
	.byte	0x0
	.set L$set$57,LEHB22-LFB1637
	.long L$set$57
	.set L$set$58,LEHE22-LEHB22
	.long L$set$58
	.long	0x0
	.byte	0x0
	.set L$set$59,LEHB23-LFB1637
	.long L$set$59
	.set L$set$60,LEHE23-LEHB23
	.long L$set$60
	.long	0x0
	.byte	0x0
	.byte	0x1
	.byte	0x0
	.byte	0x2
	.byte	0x7d
	.align 2
	.long	__ZTI10XKameError
	.long	0
.section __TEXT,__text,regular,pure_instructions
	.align 2
	.stabs	"_ZN13XSignalBuffer11synchronizeEv:F(0,1)",36,0,86,__ZN13XSignalBuffer11synchronizeEv
	.stabs	"this:P(0,48)",64,0,86,29
	.stabs	"dotalk:r(0,1)",64,0,87,23
	.stabs	"time_start:(0,243)=xsXTime:",128,0,88,64
	.stabs	"XTime:Tt(0,243)=s8tv_sec:/0(0,208),0,32;tv_usec:/0(0,208),32,32;XTime::(0,244)=#(0,243),(0,2),(0,245)=*(0,243),(0,2);:_ZN5XTimeC4Ev;2A.;__base_ctor::(0,244):_ZN5XTimeC2Ev;2A.;__comp_ctor::(0,244):_ZN5XTimeC1Ev;2A.;XTime::(0,246)=#(0,243),(0,2),(0,245),(0,208),(0,208),(0,2);:_ZN5XTimeC4Ell;2A.;__base_ctor::(0,246):_ZN5XTimeC2Ell;2A.;__comp_ctor::(0,246):_ZN5XTimeC1Ell;2A.;XTime::(0,247)=#(0,243),(0,2),(0,245),(0,248)=&(0,249)=k(0,243),(0,2);:_ZN5XTimeC4ERKS_;2A.;__base_ctor::(0,247):_ZN5XTimeC2ERKS_;2A.;__comp_ctor::(0,247):_ZN5XTimeC1ERKS_;2A.;operator=::(0,250)=#(0,243),(0,251)=&(0,243),(0,245),(0,248),(0,2);:_ZN5XTimeaSERKS_;2A.;operator-::(0,252)=#(0,243),(0,253)=r(0,10);8;0;,(0,254)=*(0,249),(0,248),(0,2);:_ZNK5XTimemiERKS_;2B.;diff_msec::(0,255)=#(0,243),(0,208),(0,254),(0,248),(0,2);:_ZNK5XTime9diff_msecERKS_;2B.;diff_sec::(0,255):_ZNK5XTime8diff_secERKS_;2B.;operator+=::(0,256)=#(0,243),(0,251),(0,245),(0,253),(0,2);:_ZN5XTimepLEd;2A.;operator-=::(0,256):_ZN5XTimemIEd;2A.;operator==::(0,257)=#(0,243),(0,1),(0,254),(0,248),(0,2);:_ZNK5XTimeeqERKS_;2B.;operator!=::(0,257):_ZNK5XTimeneERKS_;2B.;operator<::(0,257):_ZNK5XTimeltERKS_;2B.;operator<=::(0,257):_ZNK5XTimeleERKS_;2B.;operator>::(0,257):_ZNK5XTimegtERKS_;2B.;operator>=::(0,257):_ZNK5XTimegeERKS_;2B.;operator!::(0,258)=#(0,243),(0,1),(0,254),(0,2);:_ZNK5XTimentEv;2B.;operator b::(0,258):_ZNK5XTimecvbEv;2B.;sec::(0,259)=#(0,243),(0,208),(0,254),(0,2);:_ZNK5XTime3secEv;2B.;usec::(0,259):_ZNK5XTime4usecEv;2B.;now::(0,260)=f(0,243):_ZN5XTime3nowEv;2A?;getTimeStr::(0,261)=#(0,243),(0,262)=xsQString:,(0,254),(0,1),(0,2);:_ZNK5XTime10getTimeStrEb;2B.;getTimeFmtStr::(0,263)=#(0,243),(0,262),(0,254),(0,36),(0,1),(0,2);:_ZNK5XTime13getTimeFmtStrEPKcb;2B.;;",128,0,14,0
	.stabs	"double:t(0,253)",128,0,0,0
	.stabs	"QString:Tt(0,262)=s4null:/2(0,264)=k(0,262):_ZN7QString4nullE;d:/0(0,265)=*(0,266)=xsQStringData:,0,32;shared_null:/0(0,265):_ZN7QString11shared_nullE;QString::(0,267)=#(0,262),(0,2),(0,268)=*(0,262),(0,2);:_ZN7QStringC4Ev;2A.;__base_ctor::(0,267):_ZN7QStringC2Ev;2A.;__comp_ctor::(0,267):_ZN7QStringC1Ev;2A.;__base_ctor::(0,269)=#(0,262),(0,2),(0,268),(0,270)=xsQChar:,(0,2);:_ZN7QStringC2E5QChar;2A.;__comp_ctor::(0,269):_ZN7QStringC1E5QChar;2A.;__base_ctor::(0,271)=#(0,262),(0,2),(0,268),(0,272)=&(0,264),(0,2);:_ZN7QStringC2ERKS_;2A.;__comp_ctor::(0,271):_ZN7QStringC1ERKS_;2A.;__base_ctor::(0,273)=#(0,262),(0,2),(0,268),(0,274)=&(0,275)=k(0,276)=(0,277)=xsQMemArray<char>:,(0,2);:_ZN7QStringC2ERK9QMemArrayIcE;2A.;__comp_ctor::(0,273):_ZN7QStringC1ERK9QMemArrayIcE;2A.;__base_ctor::(0,278)=#(0,262),(0,2),(0,268),(0,279)=*(0,280)=k(0,270),(0,6),(0,2);:_ZN7QStringC2EPK5QCharj;2A.;__comp_ctor::(0,278):_ZN7QStringC1EPK5QCharj;2A.;__base_ctor::(0,281)=#(0,262),(0,2),(0,268),(0,36),(0,2);:_ZN7QStringC2EPKc;2A.;__comp_ctor::(0,281):_ZN7QStringC1EPKc;2A.;__base_ctor::(0,282)=#(0,262),(0,2),(0,268),(0,283)=&(0,284)=k(0,285)=(0,286)=xsbasic_string<char,std::char_traits<char>,std::allocator<char> >:,(0,2);:_ZN7QStringC2ERKSs;2A.;__comp_ctor::(0,282):_ZN7QStringC1ERKSs;2A.;QString::(0,287)=#(0,262),(0,2),(0,268),(0,10),(0,2);:_ZN7QStringD4Ev;2A.;__base_dtor::(0,267):_ZN7QStringD2Ev;2A.;__comp_dtor::(0,267):_ZN7QStringD1Ev;2A.;operator=::(0,288)=#(0,262),(0,289)=&(0,262),(0,268),(0,272),(0,2);:_ZN7QStringaSERKS_;2A.(0,290)=#(0,262),(0,289),(0,268),(0,36),(0,2);:_ZN7QStringaSEPKc;2A.(0,291)=#(0,262),(0,289),(0,268),(0,283),(0,2);:_ZN7QStringaSERKSs;2A.(0,292)=#(0,262),(0,289),(0,268),(0,293)=&(0,294)=k(0,295)=xsQCString:,(0,2);:_ZN7QStringaSERK8QCString;2A.(0,296)=#(0,262),(0,289),(0,268),(0,270),(0,2);:_ZN7QStringaSE5QChar;2A.(0,297)=#(0,262),(0,289),(0,268),(0,38),(0,2);:_ZN7QStringaSEc;2A.;isNull::(0,298)=#(0,262),(0,1),(0,299)=*(0,264),(0,2);:_ZNK7QString6isNullEv;2B.;isEmpty::(0,298):_ZNK7QString7isEmptyEv;2B.;length::(0,300)=#(0,262),(0,301)=(0,6),(0,299),(0,2);:_ZNK7QString6lengthEv;2B.;truncate::(0,302)=#(0,262),(0,2),(0,268),(0,6),(0,2);:_ZN7QString8truncateEj;2A.;fill::(0,303)=#(0,262),(0,289),(0,268),(0,270),(0,10),(0,2);:_ZN7QString4fillE5QChari;2A.;copy::(0,304)=#(0,262),(0,262),(0,299),(0,2);:_ZNK7QString4copyEv;2B.;arg::(0,305)=#(0,262),(0,262),(0,299),(0,208),(0,10),(0,10),(0,2);:_ZNK7QString3argElii;2B.(0,306)=#(0,262),(0,262),(0,299),(0,69),(0,10),(0,10),(0,2);:_ZNK7QString3argEmii;2B.(0,307)=#(0,262),(0,262),(0,299),(0,308)=@s64;r(0,308);01000000000000000000000;0777777777777777777777;,(0,10),(0,10),(0,2);:_ZNK7QString3argExii;2B.(0,309)=#(0,262),(0,262),(0,299),(0,310)=@s64;r(0,310);0000000000000;01777777777777777777777;,(0,10),(0,10),(0,2);:_ZNK7QString3argEyii;2B.(0,311)=#(0,262),(0,262),(0,299),(0,10),(0,10),(0,10),(0,2);:_ZNK7QString3argEiii;2B.(0,312)=#(0,262),(0,262),(0,299),(0,6),(0,10),(0,10),(0,2);:_ZNK7QString3argEjii;2B.(0,313)=#(0,262),(0,262),(0,299),(0,314)=@s16;r(0,314);-32768;32767;,(0,10),(0,10),(0,2);:_ZNK7QString3argEsii;2B.(0,315)=#(0,262),(0,262),(0,299),(0,316)=@s16;r(0,316);0;65535;,(0,10),(0,10),(0,2);:_ZNK7QString3argEtii;2B.(0,317)=#(0,262),(0,262),(0,299),(0,253),(0,10),(0,38),(0,10),(0,2);:_ZNK7QString3argEdici;2B.(0,318)=#(0,262),(0,262),(0,299),(0,38),(0,10),(0,2);:_ZNK7QString3argEci;2B.(0,319)=#(0,262),(0,262),(0,299),(0,270),(0,10),(0,2);:_ZNK7QString3argE5QChari;2B.(0,320)=#(0,262),(0,262),(0,299),(0,272),(0,10),(0,2);:_ZNK7QString3argERKS_i;2B.(0,321)=#(0,262),(0,262),(0,299),(0,272),(0,272),(0,2);:_ZNK7QString3argERKS_S1_;2B.(0,322)=#(0,262),(0,262),(0,299),(0,272),(0,272),(0,272),(0,2);:_ZNK7QString3argERKS_S1_S1_;2B.(0,323)=#(0,262),(0,262),(0,299),(0,272),(0,272),(0,272),(0,272),(0,2);:_ZNK7QString3argERKS_S1_S1_S1_;2B.;sprintf::(0,324)=#(0,262),(0,289),(0,268),(0,36);:_ZN7QString7sprintfEPKcz;2A.;find::(0,325)=#(0,262),(0,10),(0,299),(0,270),(0,10),(0,1),(0,2);:_ZNK7QString4findE5QCharib;2B.(0,326)=#(0,262),(0,10),(0,299),(0,38),(0,10),(0,1),(0,2);:_ZNK7QString4findEcib;2B.(0,327)=#(0,262),(0,10),(0,299),(0,272),(0,10),(0,1),(0,2);:_ZNK7QString4findERKS_ib;2B.(0,328)=#(0,262),(0,10),(0,299),(0,329)=&(0,330)=k(0,331)=xsQRegExp:,(0,10),(0,2);:_ZNK7QString4findERK7QRegExpi;2B.(0,332)=#(0,262),(0,10),(0,299),(0,36),(0,10),(0,2);:_ZNK7QString4findEPKci;2B.;findRev::(0,333)=#(0,262),(0,10),(0,299),(0,270),(0,10),(0,1),(0,2);:_ZNK7QString7findRevE5QCharib;2B.(0,334)=#(0,262),(0,10),(0,299),(0,38),(0,10),(0,1),(0,2);:_ZNK7QString7findRevEcib;2B.(0,335)=#(0,262),(0,10),(0,299),(0,272),(0,10),(0,1),(0,2);:_ZNK7QString7findRevERKS_ib;2B.(0,336)=#(0,262),(0,10),(0,299),(0,329),(0,10),(0,2);:_ZNK7QString7findRevERK7QRegExpi;2B.(0,337)=#(0,262),(0,10),(0,299),(0,36),(0,10),(0,2);:_ZNK7QString7findRevEPKci;2B.;contains::(0,338)=#(0,262),(0,10),(0,299),(0,270),(0,1),(0,2);:_ZNK7QString8containsE5QCharb;2B.(0,339)=#(0,262),(0,10),(0,299),(0,38),(0,1),(0,2);:_ZNK7QString8containsEcb;2B.(0,340)=#(0,262),(0,10),(0,299),(0,36),(0,1),(0,2);:_ZNK7QString8containsEPKcb;2B.(0,341)=#(0,262),(0,10),(0,299),(0,272),(0,1),(0,2);:_ZNK7QString8containsERKS_b;2B.(0,342)=#(0,262),(0,10),(0,299),(0,329),(0,2);:_ZNK7QString8containsERK7QRegExp;2B.;section::(0,343)=#(0,262),(0,262),(0,299),(0,270),(0,10),(0,10),(0,10),(0,2);:_ZNK7QString7sectionE5QChariii;2B.(0,344)=#(0,262),(0,262),(0,299),(0,38),(0,10),(0,10),(0,10),(0,2);:_ZNK7QString7sectionEciii;2B.(0,345)=#(0,262),(0,262),(0,299),(0,36),(0,10),(0,10),(0,10),(0,2);:_ZNK7QString7sectionEPKciii;2B.(0,346)=#(0,262),(0,262),(0,299),(0,272),(0,10),(0,10),(0,10),(0,2);:_ZNK7QString7sectionERKS_iii;2B.(0,347)=#(0,262),(0,262),(0,299),(0,329),(0,10),(0,10),(0,10),(0,2);:_ZNK7QString7sectionERK7QRegExpiii;2B.;left::(0,348)=#(0,262),(0,262),(0,299),(0,6),(0,2);:_ZNK7QString4leftEj;2B.;right::(0,348):_ZNK7QString5rightEj;2B.;mid::(0,349)=#(0,262),(0,262),(0,299),(0,6),(0,6),(0,2);:_ZNK7QString3midEjj;2B.;leftJustify::(0,350)=#(0,262),(0,262),(0,299),(0,6),(0,270),(0,1),(0,2);:_ZNK7QString11leftJustifyEj5QCharb;2B.;rightJustify::(0,351)=#(0,262),(0,262),(0,299),(0,6),(0,270),(0,1),(0,2);:_ZNK7QString12rightJustifyEj5QCharb;2B.;lower::(0,304):_ZNK7QString5lowerEv;2B.;upper::(0,304):_ZNK7QString5upperEv;2B.;stripWhiteSpace::(0,304):_ZNK7QString15stripWhiteSpaceEv;2B.;simplifyWhiteSpace::(0,304):_ZNK7QString18simplifyWhiteSpaceEv;2B.;insert::(0,352)=#(0,262),(0,289),(0,268),(0,6),(0,272),(0,2);:_ZN7QString6insertEjRKS_;2A.(0,353)=#(0,262),(0,289),(0,268),(0,6),(0,274),(0,2);:_ZN7QString6insertEjRK9QMemArrayIcE;2A.(0,354)=#(0,262),(0,289),(0,268),(0,6),(0,36),(0,2);:_ZN7QString6insertEjPKc;2A.(0,355)=#(0,262),(0,289),(0,268),(0,6),(0,279),(0,6),(0,2);:_ZN7QString6insertEjPK5QCharj;2A.(0,356)=#(0,262),(0,289),(0,268),(0,6),(0,270),(0,2);:_ZN7QString6insertEj5QChar;2A.(0,357)=#(0,262),(0,289),(0,268),(0,6),(0,38),(0,2);:_ZN7QString6insertEjc;2A.;append::(0,297):_ZN7QString6appendEc;2A.(0,296):_ZN7QString6appendE5QChar;2A.(0,288):_ZN7QString6appendERKS_;2A.(0,358)=#(0,262),(0,289),(0,268),(0,274),(0,2);:_ZN7QString6appendERK9QMemArrayIcE;2A.(0,290):_ZN7QString6appendEPKc;2A.(0,291):_ZN7QString6appendERKSs;2A.;prepend::(0,297):_ZN7QString7prependEc;2A.(0,296):_ZN7QString7prependE5QChar;2A.(0,288):_ZN7QString7prependERKS_;2A.(0,358):_ZN7QString7prependERK9QMemArrayIcE;2A.(0,290):_ZN7QString7prependEPKc;2A.(0,291):_ZN7QString7prependERKSs;2A.;remove::(0,359)=#(0,262),(0,289),(0,268),(0,6),(0,6),(0,2);:_ZN7QString6removeEjj;2A.(0,288):_ZN7QString6removeERKS_;2A.(0,360)=#(0,262),(0,289),(0,268),(0,272),(0,1),(0,2);:_ZN7QString6removeERKS_b;2A.(0,296):_ZN7QString6removeE5QChar;2A.(0,297):_ZN7QString6removeEc;2A.(0,290):_ZN7QString6removeEPKc;2A.(0,361)=#(0,262),(0,289),(0,268),(0,329),(0,2);:_ZN7QString6removeERK7QRegExp;2A.;replace::(0,362)=#(0,262),(0,289),(0,268),(0,6),(0,6),(0,272),(0,2);:_ZN7QString7replaceEjjRKS_;2A.(0,363)=#(0,262),(0,289),(0,268),(0,6),(0,6),(0,279),(0,6),(0,2);:_ZN7QString7replaceEjjPK5QCharj;2A.(0,364)=#(0,262),(0,289),(0,268),(0,6),(0,6),(0,270),(0,2);:_ZN7QString7replaceEjj5QChar;2A.(0,365)=#(0,262),(0,289),(0,268),(0,6),(0,6),(0,38),(0,2);:_ZN7QString7replaceEjjc;2A.(0,366)=#(0,262),(0,289),(0,268),(0,270),(0,272),(0,2);:_ZN7QString7replaceE5QCharRKS_;2A.(0,367)=#(0,262),(0,289),(0,268),(0,270),(0,272),(0,1),(0,2);:_ZN7QString7replaceE5QCharRKS_b;2A.(0,368)=#(0,262),(0,289),(0,268),(0,38),(0,272),(0,2);:_ZN7QString7replaceEcRKS_;2A.(0,369)=#(0,262),(0,289),(0,268),(0,38),(0,272),(0,1),(0,2);:_ZN7QString7replaceEcRKS_b;2A.(0,370)=#(0,262),(0,289),(0,268),(0,272),(0,272),(0,2);:_ZN7QString7replaceERKS_S1_;2A.(0,371)=#(0,262),(0,289),(0,268),(0,272),(0,272),(0,1),(0,2);:_ZN7QString7replaceERKS_S1_b;2A.(0,372)=#(0,262),(0,289),(0,268),(0,329),(0,272),(0,2);:_ZN7QString7replaceERK7QRegExpRKS_;2A.(0,373)=#(0,262),(0,289),(0,268),(0,270),(0,270),(0,2);:_ZN7QString7replaceE5QCharS0_;2A.;toShort::(0,374)=#(0,262),(0,314),(0,299),(0,375)=*(0,1),(0,10),(0,2);:_ZNK7QString7toShortEPbi;2B.;toUShort::(0,376)=#(0,262),(0,377)=(0,316),(0,299),(0,375),(0,10),(0,2);:_ZNK7QString8toUShortEPbi;2B.;toInt::(0,378)=#(0,262),(0,10),(0,299),(0,375),(0,10),(0,2);:_ZNK7QString5toIntEPbi;2B.;toUInt::(0,379)=#(0,262),(0,301),(0,299),(0,375),(0,10),(0,2);:_ZNK7QString6toUIntEPbi;2B.;toLong::(0,380)=#(0,262),(0,208),(0,299),(0,375),(0,10),(0,2);:_ZNK7QString6toLongEPbi;2B.;toULong::(0,381)=#(0,262),(0,382)=(0,69),(0,299),(0,375),(0,10),(0,2);:_ZNK7QString7toULongEPbi;2B.;toLongLong::(0,383)=#(0,262),(0,384)=(0,385)=(0,308),(0,299),(0,375),(0,10),(0,2);:_ZNK7QString10toLongLongEPbi;2B.;toULongLong::(0,386)=#(0,262),(0,387)=(0,388)=(0,310),(0,299),(0,375),(0,10),(0,2);:_ZNK7QString11toULongLongEPbi;2B.;toFloat::(0,389)=#(0,262),(0,390)=r(0,10);4;0;,(0,299),(0,375),(0,2);:_ZNK7QString7toFloatEPb;2B.;toDouble::(0,391)=#(0,262),(0,253),(0,299),(0,375),(0,2);:_ZNK7QString8toDoubleEPb;2B.;setNum::(0,392)=#(0,262),(0,289),(0,268),(0,314),(0,10),(0,2);:_ZN7QString6setNumEsi;2A.(0,393)=#(0,262),(0,289),(0,268),(0,316),(0,10),(0,2);:_ZN7QString6setNumEti;2A.(0,394)=#(0,262),(0,289),(0,268),(0,10),(0,10),(0,2);:_ZN7QString6setNumEii;2A.(0,395)=#(0,262),(0,289),(0,268),(0,6),(0,10),(0,2);:_ZN7QString6setNumEji;2A.(0,396)=#(0,262),(0,289),(0,268),(0,208),(0,10),(0,2);:_ZN7QString6setNumEli;2A.(0,397)=#(0,262),(0,289),(0,268),(0,69),(0,10),(0,2);:_ZN7QString6setNumEmi;2A.(0,398)=#(0,262),(0,289),(0,268),(0,308),(0,10),(0,2);:_ZN7QString6setNumExi;2A.(0,399)=#(0,262),(0,289),(0,268),(0,310),(0,10),(0,2);:_ZN7QString6setNumEyi;2A.(0,400)=#(0,262),(0,289),(0,268),(0,390),(0,38),(0,10),(0,2);:_ZN7QString6setNumEfci;2A.(0,401)=#(0,262),(0,289),(0,268),(0,253),(0,38),(0,10),(0,2);:_ZN7QString6setNumEdci;2A.;number::(0,402)=f(0,262):_ZN7QString6numberEli;2A?(0,403)=f(0,262):_ZN7QString6numberEmi;2A?(0,404)=f(0,262):_ZN7QString6numberExi;2A?(0,405)=f(0,262):_ZN7QString6numberEyi;2A?(0,406)=f(0,262):_ZN7QString6numberEii;2A?(0,407)=f(0,262):_ZN7QString6numberEji;2A?(0,408)=f(0,262):_ZN7QString6numberEdci;2A?;setExpand::(0,409)=#(0,262),(0,2),(0,268),(0,6),(0,270),(0,2);:_ZN7QString9setExpandEj5QChar;2A.;operator+=::(0,288):_ZN7QStringpLERKS_;2A.(0,358):_ZN7QStringpLERK9QMemArrayIcE;2A.(0,290):_ZN7QStringpLEPKc;2A.(0,291):_ZN7QStringpLERKSs;2A.(0,296):_ZN7QStringpLE5QChar;2A.(0,297):_ZN7QStringpLEc;2A.;at::(0,410)=#(0,262),(0,270),(0,299),(0,6),(0,2);:_ZNK7QString2atEj;2B.;operator[]::(0,411)=#(0,262),(0,270),(0,299),(0,10),(0,2);:_ZNK7QStringixEi;2B.;at::(0,412)=#(0,262),(0,413)=xsQCharRef:,(0,268),(0,6),(0,2);:_ZN7QString2atEj;2A.;operator[]::(0,414)=#(0,262),(0,413),(0,268),(0,10),(0,2);:_ZN7QStringixEi;2A.;constref::(0,410):_ZNK7QString8constrefEj;2B.;ref::(0,415)=#(0,262),(0,416)=&(0,270),(0,268),(0,6),(0,2);:_ZN7QString3refEj;2A.;unicode::(0,417)=#(0,262),(0,279),(0,299),(0,2);:_ZNK7QString7unicodeEv;2B.;ascii::(0,418)=#(0,262),(0,36),(0,299),(0,2);:_ZNK7QString5asciiEv;2B.;fromAscii::(0,419)=f(0,262):_ZN7QString9fromAsciiEPKci;2A?;latin1::(0,418):_ZNK7QString6latin1Ev;2B.;fromLatin1::(0,420)=f(0,262):_ZN7QString10fromLatin1EPKci;2A?;utf8::(0,421)=#(0,262),(0,295),(0,299),(0,2);:_ZNK7QString4utf8Ev;2B.;fromUtf8::(0,422)=f(0,262):_ZN7QString8fromUtf8EPKci;2A?;local8Bit::(0,421):_ZNK7QString9local8BitEv;2B.;fromLocal8Bit::(0,423)=f(0,262):_ZN7QString13fromLocal8BitEPKci;2A?;operator!::(0,298):_ZNK7QStringntEv;2B.;operator PKc::(0,418):_ZNK7QStringcvPKcEv;2B.;operator Ss::(0,424)=#(0,262),(0,285),(0,299),(0,2);:_ZNK7QStringcvSsEv;2B.;fromUcs2::(0,425)=f(0,262):_ZN7QString8fromUcs2EPKt;2A?;ucs2::(0,426)=#(0,262),(0,427)=*(0,428)=k(0,316),(0,299),(0,2);:_ZNK7QString4ucs2Ev;2B.;setUnicode::(0,429)=#(0,262),(0,289),(0,268),(0,279),(0,6),(0,2);:_ZN7QString10setUnicodeEPK5QCharj;2A.;setUnicodeCodes::(0,430)=#(0,262),(0,289),(0,268),(0,431)=*(0,432)=k(0,377),(0,6),(0,2);:_ZN7QString15setUnicodeCodesEPKtj;2A.;setAscii::(0,433)=#(0,262),(0,289),(0,268),(0,36),(0,10),(0,2);:_ZN7QString8setAsciiEPKci;2A.;setLatin1::(0,434)=#(0,262),(0,289),(0,268),(0,36),(0,10),(0,2);:_ZN7QString9setLatin1EPKci;2A.;compare::(0,435)=#(0,262),(0,10),(0,299),(0,272),(0,2);:_ZNK7QString7compareERKS_;2B.(0,436)=f(0,10):_ZN7QString7compareERKS_S1_;2A?;localeAwareCompare::(0,435):_ZNK7QString18localeAwareCompareERKS_;2B.(0,436):_ZN7QString18localeAwareCompareERKS_S1_;2A?;compose::(0,267):_ZN7QString7composeEv;2A.;data::(0,418):_ZNK7QString4dataEv;2B.;startsWith::(0,437)=#(0,262),(0,1),(0,299),(0,272),(0,2);:_ZNK7QString10startsWithERKS_;2B.(0,438)=#(0,262),(0,1),(0,299),(0,272),(0,1),(0,2);:_ZNK7QString10startsWithERKS_b;2B.;endsWith::(0,437):_ZNK7QString8endsWithERKS_;2B.(0,438):_ZNK7QString8endsWithERKS_b;2B.;setLength::(0,302):_ZN7QString9setLengthEj;2A.;capacity::(0,300):_ZNK7QString8capacityEv;2B.;reserve::(0,302):_ZN7QString7reserveEj;2A.;squeeze::(0,267):_ZN7QString7squeezeEv;2A.;simpleText::(0,298):_ZNK7QString10simpleTextEv;2B.;isRightToLeft::(0,298):_ZNK7QString13isRightToLeftEv;2B.;__base_ctor::(0,439)=#(0,262),(0,2),(0,268),(0,10),(0,1),(0,2);:_ZN7QStringC2Eib;0A.;__comp_ctor::(0,439):_ZN7QStringC1Eib;0A.;deref::(0,267):_ZN7QString5derefEv;0A.;real_detach::(0,267):_ZN7QString11real_detachEv;0A.;subat::(0,302):_ZN7QString5subatEj;0A.;multiArg::(0,440)=#(0,262),(0,262),(0,299),(0,10),(0,272),(0,272),(0,272),(0,272),(0,2);:_ZNK7QString8multiArgEiRKS_S1_S1_S1_;0B.;checkSimpleText::(0,441)=#(0,262),(0,2),(0,299),(0,2);:_ZNK7QString15checkSimpleTextEv;0B.;grow::(0,302):_ZN7QString4growEj;0A.;insertHelper::(0,442)=#(0,262),(0,289),(0,268),(0,6),(0,36),(0,6),(0,2);:_ZN7QString12insertHelperEjPKcj;0A.;operatorPlusEqHelper::(0,443)=#(0,262),(0,289),(0,268),(0,36),(0,6),(0,2);:_ZN7QString20operatorPlusEqHelperEPKcj;0A.;latin1ToUnicode::(0,444)=f(0,445)=*(0,270):_ZN7QString15latin1ToUnicodeEPKcPjj;0A?(0,446)=f(0,445):_ZN7QString15latin1ToUnicodeERK9QMemArrayIcEPj;0A?;unicodeToLatin1::(0,447)=f(0,448)=*(0,38):_ZN7QString15unicodeToLatin1EPK5QCharj;0A?;makeSharedNull::(0,449)=f(0,265):_ZN7QString14makeSharedNullEv;0A?;QString::(0,450)=#(0,262),(0,2),(0,268),(0,265),(0,1),(0,2);:_ZN7QStringC4EP11QStringDatab;0A.;__base_ctor::(0,450):_ZN7QStringC2EP11QStringDatab;0A.;__comp_ctor::(0,450):_ZN7QStringC1EP11QStringDatab;0A.;detach::(0,267):_ZN7QString6detachEv;0A.;;",128,0,398,0
	.stabs	"QStringData:Tt(0,266)=s20!1,020,(0,451)=xsQShared:;unicode:(0,445),32,32;ascii:(0,448),64,32;len:(0,301),96,30;issimpletext:(0,301),126,1;maxl:(0,301),128,30;islatin1:(0,301),158,1;QStringData::(0,452)=#(0,266),(0,2),(0,265),(0,2);:_ZN11QStringDataC4Ev;2A.;__base_ctor::(0,452):_ZN11QStringDataC2Ev;2A.;__comp_ctor::(0,452):_ZN11QStringDataC1Ev;2A.;QStringData::(0,453)=#(0,266),(0,2),(0,265),(0,445),(0,6),(0,6),(0,2);:_ZN11QStringDataC4EP5QCharjj;2A.;__base_ctor::(0,453):_ZN11QStringDataC2EP5QCharjj;2A.;__comp_ctor::(0,453):_ZN11QStringDataC1EP5QCharjj;2A.;QStringData::(0,454)=#(0,266),(0,2),(0,265),(0,10),(0,2);:_ZN11QStringDataD4Ev;2A.;__base_dtor::(0,452):_ZN11QStringDataD2Ev;2A.;__comp_dtor::(0,452):_ZN11QStringDataD1Ev;2A.;deleteSelf::(0,452):_ZN11QStringData10deleteSelfEv;2A.;setDirty::(0,452):_ZN11QStringData8setDirtyEv;2A.;__base_ctor::(0,455)=#(0,266),(0,2),(0,265),(0,456)=&(0,457)=k(0,266),(0,2);:_ZN11QStringDataC2ERKS_;0A.;__comp_ctor::(0,455):_ZN11QStringDataC1ERKS_;0A.;operator=::(0,458)=#(0,266),(0,459)=&(0,266),(0,265),(0,456),(0,2);:_ZN11QStringDataaSERKS_;0A.;;",128,0,358,0
	.stabs	"QChar:Tt(0,270)=s2null:/2(0,280):_ZN5QChar4nullE;replacement:/2(0,280):_ZN5QChar11replacementE;byteOrderMark:/2(0,280):_ZN5QChar13byteOrderMarkE;byteOrderSwapped:/2(0,280):_ZN5QChar16byteOrderSwappedE;nbsp:/2(0,280):_ZN5QChar4nbspE;ucs:/0(0,377),0,16;operator=::(0,460)=#(0,270),(0,416),(0,445),(0,461)=&(0,280),(0,2);:_ZN5QCharaSERKS_;2A.;QChar::(0,462)=#(0,270),(0,2),(0,445),(0,2);:_ZN5QCharC4Ev;2A.;__base_ctor::(0,462):_ZN5QCharC2Ev;2A.;__comp_ctor::(0,462):_ZN5QCharC1Ev;2A.;QChar::(0,463)=#(0,270),(0,2),(0,445),(0,38),(0,2);:_ZN5QCharC4Ec;2A.;__base_ctor::(0,463):_ZN5QCharC2Ec;2A.;__comp_ctor::(0,463):_ZN5QCharC1Ec;2A.;QChar::(0,464)=#(0,270),(0,2),(0,445),(0,465)=@s8;r(0,465);0;255;,(0,2);:_ZN5QCharC4Eh;2A.;__base_ctor::(0,464):_ZN5QCharC2Eh;2A.;__comp_ctor::(0,464):_ZN5QCharC1Eh;2A.;QChar::(0,466)=#(0,270),(0,2),(0,445),(0,465),(0,465),(0,2);:_ZN5QCharC4Ehh;2A.;__base_ctor::(0,466):_ZN5QCharC2Ehh;2A.;__comp_ctor::(0,466):_ZN5QCharC1Ehh;2A.;QChar::(0,467)=#(0,270),(0,2),(0,445),(0,461),(0,2);:_ZN5QCharC4ERKS_;2A.;__base_ctor::(0,467):_ZN5QCharC2ERKS_;2A.;__comp_ctor::(0,467):_ZN5QCharC1ERKS_;2A.;QChar::(0,468)=#(0,270),(0,2),(0,445),(0,316),(0,2);:_ZN5QCharC4Et;2A.;__base_ctor::(0,468):_ZN5QCharC2Et;2A.;__comp_ctor::(0,468):_ZN5QCharC1Et;2A.;QChar::(0,469)=#(0,270),(0,2),(0,445),(0,314),(0,2);:_ZN5QCharC4Es;2A.;__base_ctor::(0,469):_ZN5QCharC2Es;2A.;__comp_ctor::(0,469):_ZN5QCharC1Es;2A.;QChar::(0,470)=#(0,270),(0,2),(0,445),(0,6),(0,2);:_ZN5QCharC4Ej;2A.;__base_ctor::(0,470):_ZN5QCharC2Ej;2A.;__comp_ctor::(0,470):_ZN5QCharC1Ej;2A.;QChar::(0,471)=#(0,270),(0,2),(0,445),(0,10),(0,2);:_ZN5QCharC4Ei;2A.;__base_ctor::(0,471):_ZN5QCharC2Ei;2A.;__comp_ctor::(0,471):_ZN5QCharC1Ei;2A.;digitValue::(0,472)=#(0,270),(0,10),(0,279),(0,2);:_ZNK5QChar10digitValueEv;2B.;lower::(0,473)=#(0,270),(0,270),(0,279),(0,2);:_ZNK5QChar5lowerEv;2B.;upper::(0,473):_ZNK5QChar5upperEv;2B.;category::(0,474)=#(0,270),(0,475)=xeCategory:,(0,279),(0,2);:_ZNK5QChar8categoryEv;2B.;direction::(0,476)=#(0,270),(0,477)=xeDirection:,(0,279),(0,2);:_ZNK5QChar9directionEv;2B.;joining::(0,478)=#(0,270),(0,479)=xeJoining:,(0,279),(0,2);:_ZNK5QChar7joiningEv;2B.;mirrored::(0,480)=#(0,270),(0,1),(0,279),(0,2);:_ZNK5QChar8mirroredEv;2B.;mirroredChar::(0,473):_ZNK5QChar12mirroredCharEv;2B.;decomposition::(0,481)=#(0,270),(0,272),(0,279),(0,2);:_ZNK5QChar13decompositionEv;2B.;decompositionTag::(0,482)=#(0,270),(0,483)=xeDecomposition:,(0,279),(0,2);:_ZNK5QChar16decompositionTagEv;2B.;combiningClass::(0,484)=#(0,270),(0,465),(0,279),(0,2);:_ZNK5QChar14combiningClassEv;2B.;latin1::(0,485)=#(0,270),(0,38),(0,279),(0,2);:_ZNK5QChar6latin1Ev;2B.;unicode::(0,486)=#(0,270),(0,377),(0,279),(0,2);:_ZNK5QChar7unicodeEv;2B.(0,487)=#(0,270),(0,488)=&(0,377),(0,445),(0,2);:_ZN5QChar7unicodeEv;2A.;operator c::(0,485):_ZNK5QCharcvcEv;2B.;isNull::(0,480):_ZNK5QChar6isNullEv;2B.;isPrint::(0,480):_ZNK5QChar7isPrintEv;2B.;isPunct::(0,480):_ZNK5QChar7isPunctEv;2B.;isSpace::(0,480):_ZNK5QChar7isSpaceEv;2B.;isMark::(0,480):_ZNK5QChar6isMarkEv;2B.;isLetter::(0,480):_ZNK5QChar8isLetterEv;2B.;isNumber::(0,480):_ZNK5QChar8isNumberEv;2B.;isLetterOrNumber::(0,480):_ZNK5QChar16isLetterOrNumberEv;2B.;isDigit::(0,480):_ZNK5QChar7isDigitEv;2B.;isSymbol::(0,480):_ZNK5QChar8isSymbolEv;2B.;cell::(0,489)=#(0,270),(0,490)=(0,465),(0,279),(0,2);:_ZNK5QChar4cellEv;2B.;row::(0,489):_ZNK5QChar3rowEv;2B.;setCell::(0,464):_ZN5QChar7setCellEh;2A.;setRow::(0,464):_ZN5QChar6setRowEh;2A.;networkOrdered::(0,491)=f(0,1):_ZN5QChar14networkOrderedEv;2A?;;",128,0,73,0
	.stabs	"QByteArray:t(0,276)",128,0,116,0
	.stabs	"QMemArray<char>:Tt(0,277)=s8!1,020,(0,492)=xsQGArray:;__base_ctor::(0,493)=#(0,277),(0,2),(0,494)=*(0,277),(0,10),(0,10),(0,2);:_ZN9QMemArrayIcEC2Eii;1A.;__comp_ctor::(0,493):_ZN9QMemArrayIcEC1Eii;1A.;__base_ctor::(0,495)=#(0,277),(0,2),(0,494),(0,2);:_ZN9QMemArrayIcEC2Ev;2A.;__comp_ctor::(0,495):_ZN9QMemArrayIcEC1Ev;2A.;__base_ctor::(0,496)=#(0,277),(0,2),(0,494),(0,10),(0,2);:_ZN9QMemArrayIcEC2Ei;2A.;__comp_ctor::(0,496):_ZN9QMemArrayIcEC1Ei;2A.;__base_ctor::(0,497)=#(0,277),(0,2),(0,494),(0,498)=&(0,499)=k(0,277),(0,2);:_ZN9QMemArrayIcEC2ERKS0_;2A.;__comp_ctor::(0,497):_ZN9QMemArrayIcEC1ERKS0_;2A.;__base_dtor::(0,495):_ZN9QMemArrayIcED2Ev;2A.;__comp_dtor::(0,495):_ZN9QMemArrayIcED1Ev;2A*0;(0,277);;__deleting_dtor::(0,495):_ZN9QMemArrayIcED0Ev;2A*1;(0,277);;operator=::(0,500)=#(0,277),(0,501)=&(0,277),(0,494),(0,498),(0,2);:_ZN9QMemArrayIcEaSERKS0_;2A.;data::(0,502)=#(0,277),(0,448),(0,503)=*(0,499),(0,2);:_ZNK9QMemArrayIcE4dataEv;2B.;nrefs::(0,504)=#(0,277),(0,301),(0,503),(0,2);:_ZNK9QMemArrayIcE5nrefsEv;2B.;size::(0,504):_ZNK9QMemArrayIcE4sizeEv;2B.;count::(0,504):_ZNK9QMemArrayIcE5countEv;2B.;isEmpty::(0,505)=#(0,277),(0,1),(0,503),(0,2);:_ZNK9QMemArrayIcE7isEmptyEv;2B.;isNull::(0,505):_ZNK9QMemArrayIcE6isNullEv;2B.;resize::(0,506)=#(0,277),(0,1),(0,494),(0,6),(0,2);:_ZN9QMemArrayIcE6resizeEj;2A.(0,507)=#(0,277),(0,1),(0,494),(0,6),(0,508)=xeOptimization:,(0,2);:_ZN9QMemArrayIcE6resizeEjN7QGArray12OptimizationE;2A.;truncate::(0,506):_ZN9QMemArrayIcE8truncateEj;2A.;fill::(0,509)=#(0,277),(0,1),(0,494),(0,510)=&(0,37),(0,10),(0,2);:_ZN9QMemArrayIcE4fillERKci;2A.;detach::(0,495):_ZN9QMemArrayIcE6detachEv;2A*2;(0,277);;copy::(0,511)=#(0,277),(0,277),(0,503),(0,2);:_ZNK9QMemArrayIcE4copyEv;2B.;assign::(0,500):_ZN9QMemArrayIcE6assignERKS0_;2A.(0,512)=#(0,277),(0,501),(0,494),(0,36),(0,6),(0,2);:_ZN9QMemArrayIcE6assignEPKcj;2A.;duplicate::(0,500):_ZN9QMemArrayIcE9duplicateERKS0_;2A.(0,512):_ZN9QMemArrayIcE9duplicateEPKcj;2A.;setRawData::(0,512):_ZN9QMemArrayIcE10setRawDataEPKcj;2A.;resetRawData::(0,513)=#(0,277),(0,2),(0,494),(0,36),(0,6),(0,2);:_ZN9QMemArrayIcE12resetRawDataEPKcj;2A.;find::(0,514)=#(0,277),(0,10),(0,503),(0,510),(0,6),(0,2);:_ZNK9QMemArrayIcE4findERKcj;2B.;contains::(0,515)=#(0,277),(0,10),(0,503),(0,510),(0,2);:_ZNK9QMemArrayIcE8containsERKc;2B.;sort::(0,495):_ZN9QMemArrayIcE4sortEv;2A.;bsearch::(0,515):_ZNK9QMemArrayIcE7bsearchERKc;2B.;operator[]::(0,516)=#(0,277),(0,517)=&(0,38),(0,503),(0,10),(0,2);:_ZNK9QMemArrayIcEixEi;2B.;at::(0,518)=#(0,277),(0,517),(0,503),(0,6),(0,2);:_ZNK9QMemArrayIcE2atEj;2B.;operator PKc::(0,519)=#(0,277),(0,36),(0,503),(0,2);:_ZNK9QMemArrayIcEcvPKcEv;2B.;operator==::(0,520)=#(0,277),(0,1),(0,503),(0,498),(0,2);:_ZNK9QMemArrayIcEeqERKS0_;2B.;operator!=::(0,520):_ZNK9QMemArrayIcEneERKS0_;2B.;begin::(0,521)=#(0,277),(0,448),(0,494),(0,2);:_ZN9QMemArrayIcE5beginEv;2A.;end::(0,521):_ZN9QMemArrayIcE3endEv;2A.;begin::(0,519):_ZNK9QMemArrayIcE5beginEv;2B.;end::(0,519):_ZNK9QMemArrayIcE3endEv;2B.;;~%(0,492);",128,0,48,0
	.stabs	"string:t(0,285)",128,0,60,0
	.stabs	"basic_string<char,std::char_traits<char>,std::allocator<char> >:Tt(0,286)=s4npos:/2(0,522)=k(0,67):_ZNSs4nposE;_M_dataplus:/0(0,523)=xs_Alloc_hider:,0,32;_S_empty_rep_storage:/0(0,524)=ar(0,210);0;3;(0,67):_ZNSs20_S_empty_rep_storageE;_M_data::(0,525)=#(0,286),(0,448),(0,526)=*(0,527)=k(0,286),(0,2);:_ZNKSs7_M_dataEv;0B.(0,528)=#(0,286),(0,448),(0,529)=*(0,286),(0,448),(0,2);:_ZNSs7_M_dataEPc;0A.;_M_rep::(0,530)=#(0,286),(0,531)=*(0,532)=xs_Rep:,(0,526),(0,2);:_ZNKSs6_M_repEv;0B.;_M_ibegin::(0,533)=#(0,286),(0,534)=xs__normal_iterator<char*,std::basic_string<char, std::char_traits<char>, std::allocator<char> > >:,(0,526),(0,2);:_ZNKSs9_M_ibeginEv;0B.;_M_iend::(0,533):_ZNKSs7_M_iendEv;0B.;_M_leak::(0,535)=#(0,286),(0,2),(0,529),(0,2);:_ZNSs7_M_leakEv;0A.;_M_check::(0,536)=#(0,286),(0,534),(0,526),(0,69),(0,2);:_ZNKSs8_M_checkEm;0B.;_M_fold::(0,537)=#(0,286),(0,534),(0,526),(0,69),(0,69),(0,2);:_ZNKSs7_M_foldEmm;0B.;_S_copy_chars::(0,538)=f(0,2):_ZNSs13_S_copy_charsEPcN9__gnu_cxx17__normal_iteratorIS_SsEES2_;0A?(0,539)=f(0,2):_ZNSs13_S_copy_charsEPcN9__gnu_cxx17__normal_iteratorIPKcSsEES4_;0A?(0,540)=f(0,2):_ZNSs13_S_copy_charsEPcS_S_;0A?(0,541)=f(0,2):_ZNSs13_S_copy_charsEPcPKcS1_;0A?;_M_mutate::(0,542)=#(0,286),(0,2),(0,529),(0,69),(0,69),(0,69),(0,2);:_ZNSs9_M_mutateEmmm;0A.;_M_leak_hard::(0,535):_ZNSs12_M_leak_hardEv;0A.;_S_empty_rep::(0,543)=f(0,544)=&(0,532):_ZNSs12_S_empty_repEv;0A?;__base_ctor::(0,535):_ZNSsC2Ev;2A.;__comp_ctor::(0,535):_ZNSsC1Ev;2A.;__base_ctor::(0,545)=#(0,286),(0,2),(0,529),(0,546)=&(0,547)=k(0,548)=xsallocator<char>:,(0,2);:_ZNSsC2ERKSaIcE;2A.;__comp_ctor::(0,545):_ZNSsC1ERKSaIcE;2A.;__base_ctor::(0,549)=#(0,286),(0,2),(0,529),(0,550)=&(0,527),(0,2);:_ZNSsC2ERKSs;2A.;__comp_ctor::(0,549):_ZNSsC1ERKSs;2A.;__base_ctor::(0,551)=#(0,286),(0,2),(0,529),(0,550),(0,69),(0,69),(0,2);:_ZNSsC2ERKSsmm;2A.;__comp_ctor::(0,551):_ZNSsC1ERKSsmm;2A.;__base_ctor::(0,552)=#(0,286),(0,2),(0,529),(0,550),(0,69),(0,69),(0,546),(0,2);:_ZNSsC2ERKSsmmRKSaIcE;2A.;__comp_ctor::(0,552):_ZNSsC1ERKSsmmRKSaIcE;2A.;__base_ctor::(0,553)=#(0,286),(0,2),(0,529),(0,36),(0,69),(0,546),(0,2);:_ZNSsC2EPKcmRKSaIcE;2A.;__comp_ctor::(0,553):_ZNSsC1EPKcmRKSaIcE;2A.;__base_ctor::(0,554)=#(0,286),(0,2),(0,529),(0,36),(0,546),(0,2);:_ZNSsC2EPKcRKSaIcE;2A.;__comp_ctor::(0,554):_ZNSsC1EPKcRKSaIcE;2A.;__base_ctor::(0,555)=#(0,286),(0,2),(0,529),(0,69),(0,38),(0,546),(0,2);:_ZNSsC2EmcRKSaIcE;2A.;__comp_ctor::(0,555):_ZNSsC1EmcRKSaIcE;2A.;__base_dtor::(0,535):_ZNSsD2Ev;2A.;__comp_dtor::(0,535):_ZNSsD1Ev;2A.;operator=::(0,556)=#(0,286),(0,557)=&(0,286),(0,529),(0,550),(0,2);:_ZNSsaSERKSs;2A.(0,558)=#(0,286),(0,557),(0,529),(0,36),(0,2);:_ZNSsaSEPKc;2A.(0,559)=#(0,286),(0,557),(0,529),(0,38),(0,2);:_ZNSsaSEc;2A.;begin::(0,560)=#(0,286),(0,534),(0,529),(0,2);:_ZNSs5beginEv;2A.(0,561)=#(0,286),(0,562)=xs__normal_iterator<const char*,std::basic_string<char, std::char_traits<char>, std::allocator<char> > >:,(0,526),(0,2);:_ZNKSs5beginEv;2B.;end::(0,560):_ZNSs3endEv;2A.(0,561):_ZNKSs3endEv;2B.;rbegin::(0,563)=#(0,286),(0,564)=xsreverse_iterator<__gnu_cxx::__normal_iterator<char*, std::basic_string<char, std::char_traits<char>, std::allocator<char> > > >:,(0,529),(0,2);:_ZNSs6rbeginEv;2A.(0,565)=#(0,286),(0,566)=xsreverse_iterator<__gnu_cxx::__normal_iterator<const char*, std::basic_string<char, std::char_traits<char>, std::allocator<char> > > >:,(0,526),(0,2);:_ZNKSs6rbeginEv;2B.;rend::(0,563):_ZNSs4rendEv;2A.(0,565):_ZNKSs4rendEv;2B.;size::(0,567)=#(0,286),(0,67),(0,526),(0,2);:_ZNKSs4sizeEv;2B.;length::(0,567):_ZNKSs6lengthEv;2B.;max_size::(0,567):_ZNKSs8max_sizeEv;2B.;resize::(0,568)=#(0,286),(0,2),(0,529),(0,69),(0,38),(0,2);:_ZNSs6resizeEmc;2A.(0,569)=#(0,286),(0,2),(0,529),(0,69),(0,2);:_ZNSs6resizeEm;2A.;capacity::(0,567):_ZNKSs8capacityEv;2B.;reserve::(0,570)=#(0,286),(0,2),(0,529),(0,69),(0,2);:_ZNSs7reserveEm;2A.;clear::(0,535):_ZNSs5clearEv;2A.;empty::(0,571)=#(0,286),(0,1),(0,526),(0,2);:_ZNKSs5emptyEv;2B.;operator[]::(0,572)=#(0,286),(0,510),(0,526),(0,69),(0,2);:_ZNKSsixEm;2B.(0,573)=#(0,286),(0,517),(0,529),(0,69),(0,2);:_ZNSsixEm;2A.;at::(0,572):_ZNKSs2atEm;2B.(0,573):_ZNSs2atEm;2A.;operator+=::(0,556):_ZNSspLERKSs;2A.(0,558):_ZNSspLEPKc;2A.(0,559):_ZNSspLEc;2A.;append::(0,556):_ZNSs6appendERKSs;2A.(0,574)=#(0,286),(0,557),(0,529),(0,550),(0,69),(0,69),(0,2);:_ZNSs6appendERKSsmm;2A.(0,575)=#(0,286),(0,557),(0,529),(0,36),(0,69),(0,2);:_ZNSs6appendEPKcm;2A.(0,558):_ZNSs6appendEPKc;2A.(0,576)=#(0,286),(0,557),(0,529),(0,69),(0,38),(0,2);:_ZNSs6appendEmc;2A.;push_back::(0,577)=#(0,286),(0,2),(0,529),(0,38),(0,2);:_ZNSs9push_backEc;2A.;assign::(0,556):_ZNSs6assignERKSs;2A.(0,574):_ZNSs6assignERKSsmm;2A.(0,575):_ZNSs6assignEPKcm;2A.(0,558):_ZNSs6assignEPKc;2A.(0,576):_ZNSs6assignEmc;2A.;insert::(0,578)=#(0,286),(0,2),(0,529),(0,534),(0,69),(0,38),(0,2);:_ZNSs6insertEN9__gnu_cxx17__normal_iteratorIPcSsEEmc;2A.(0,579)=#(0,286),(0,557),(0,529),(0,69),(0,550),(0,2);:_ZNSs6insertEmRKSs;2A.(0,580)=#(0,286),(0,557),(0,529),(0,69),(0,550),(0,69),(0,69),(0,2);:_ZNSs6insertEmRKSsmm;2A.(0,581)=#(0,286),(0,557),(0,529),(0,69),(0,36),(0,69),(0,2);:_ZNSs6insertEmPKcm;2A.(0,582)=#(0,286),(0,557),(0,529),(0,69),(0,36),(0,2);:_ZNSs6insertEmPKc;2A.(0,583)=#(0,286),(0,557),(0,529),(0,69),(0,69),(0,38),(0,2);:_ZNSs6insertEmmc;2A.(0,584)=#(0,286),(0,534),(0,529),(0,534),(0,38),(0,2);:_ZNSs6insertEN9__gnu_cxx17__normal_iteratorIPcSsEEc;2A.;erase::(0,585)=#(0,286),(0,557),(0,529),(0,69),(0,69),(0,2);:_ZNSs5eraseEmm;2A.(0,586)=#(0,286),(0,534),(0,529),(0,534),(0,2);:_ZNSs5eraseEN9__gnu_cxx17__normal_iteratorIPcSsEE;2A.(0,587)=#(0,286),(0,534),(0,529),(0,534),(0,534),(0,2);:_ZNSs5eraseEN9__gnu_cxx17__normal_iteratorIPcSsEES2_;2A.;replace::(0,588)=#(0,286),(0,557),(0,529),(0,69),(0,69),(0,550),(0,2);:_ZNSs7replaceEmmRKSs;2A.(0,589)=#(0,286),(0,557),(0,529),(0,69),(0,69),(0,550),(0,69),(0,69),(0,2);:_ZNSs7replaceEmmRKSsmm;2A.(0,590)=#(0,286),(0,557),(0,529),(0,69),(0,69),(0,36),(0,69),(0,2);:_ZNSs7replaceEmmPKcm;2A.(0,591)=#(0,286),(0,557),(0,529),(0,69),(0,69),(0,36),(0,2);:_ZNSs7replaceEmmPKc;2A.(0,592)=#(0,286),(0,557),(0,529),(0,69),(0,69),(0,69),(0,38),(0,2);:_ZNSs7replaceEmmmc;2A.(0,593)=#(0,286),(0,557),(0,529),(0,534),(0,534),(0,550),(0,2);:_ZNSs7replaceEN9__gnu_cxx17__normal_iteratorIPcSsEES2_RKSs;2A.(0,594)=#(0,286),(0,557),(0,529),(0,534),(0,534),(0,36),(0,69),(0,2);:_ZNSs7replaceEN9__gnu_cxx17__normal_iteratorIPcSsEES2_PKcm;2A.(0,595)=#(0,286),(0,557),(0,529),(0,534),(0,534),(0,36),(0,2);:_ZNSs7replaceEN9__gnu_cxx17__normal_iteratorIPcSsEES2_PKc;2A.(0,596)=#(0,286),(0,557),(0,529),(0,534),(0,534),(0,69),(0,38),(0,2);:_ZNSs7replaceEN9__gnu_cxx17__normal_iteratorIPcSsEES2_mc;2A.(0,597)=#(0,286),(0,557),(0,529),(0,534),(0,534),(0,448),(0,448),(0,2);:_ZNSs7replaceEN9__gnu_cxx17__normal_iteratorIPcSsEES2_S1_S1_;2A.(0,598)=#(0,286),(0,557),(0,529),(0,534),(0,534),(0,36),(0,36),(0,2);:_ZNSs7replaceEN9__gnu_cxx17__normal_iteratorIPcSsEES2_PKcS4_;2A.(0,599)=#(0,286),(0,557),(0,529),(0,534),(0,534),(0,534),(0,534),(0,2);:_ZNSs7replaceEN9__gnu_cxx17__normal_iteratorIPcSsEES2_S2_S2_;2A.(0,600)=#(0,286),(0,557),(0,529),(0,534),(0,534),(0,562),(0,562),(0,2);:_ZNSs7replaceEN9__gnu_cxx17__normal_iteratorIPcSsEES2_NS0_IPKcSsEES5_;2A.;_S_construct::(0,601)=f(0,448):_ZNSs12_S_constructEmcRKSaIcE;0A?;copy::(0,602)=#(0,286),(0,67),(0,526),(0,448),(0,69),(0,69),(0,2);:_ZNKSs4copyEPcmm;2B.;swap::(0,603)=#(0,286),(0,2),(0,529),(0,557),(0,2);:_ZNSs4swapERSs;2A.;c_str::(0,604)=#(0,286),(0,36),(0,526),(0,2);:_ZNKSs5c_strEv;2B.;data::(0,604):_ZNKSs4dataEv;2B.;get_allocator::(0,605)=#(0,286),(0,548),(0,526),(0,2);:_ZNKSs13get_allocatorEv;2B.;find::(0,606)=#(0,286),(0,67),(0,526),(0,36),(0,69),(0,69),(0,2);:_ZNKSs4findEPKcmm;2B.(0,607)=#(0,286),(0,67),(0,526),(0,550),(0,69),(0,2);:_ZNKSs4findERKSsm;2B.(0,608)=#(0,286),(0,67),(0,526),(0,36),(0,69),(0,2);:_ZNKSs4findEPKcm;2B.(0,609)=#(0,286),(0,67),(0,526),(0,38),(0,69),(0,2);:_ZNKSs4findEcm;2B.;rfind::(0,610)=#(0,286),(0,67),(0,526),(0,550),(0,69),(0,2);:_ZNKSs5rfindERKSsm;2B.(0,606):_ZNKSs5rfindEPKcmm;2B.(0,611)=#(0,286),(0,67),(0,526),(0,36),(0,69),(0,2);:_ZNKSs5rfindEPKcm;2B.(0,612)=#(0,286),(0,67),(0,526),(0,38),(0,69),(0,2);:_ZNKSs5rfindEcm;2B.;find_first_of::(0,607):_ZNKSs13find_first_ofERKSsm;2B.(0,606):_ZNKSs13find_first_ofEPKcmm;2B.(0,608):_ZNKSs13find_first_ofEPKcm;2B.(0,609):_ZNKSs13find_first_ofEcm;2B.;find_last_of::(0,610):_ZNKSs12find_last_ofERKSsm;2B.(0,606):_ZNKSs12find_last_ofEPKcmm;2B.(0,611):_ZNKSs12find_last_ofEPKcm;2B.(0,612):_ZNKSs12find_last_ofEcm;2B.;find_first_not_of::(0,607):_ZNKSs17find_first_not_ofERKSsm;2B.(0,606):_ZNKSs17find_first_not_ofEPKcmm;2B.(0,608):_ZNKSs17find_first_not_ofEPKcm;2B.(0,609):_ZNKSs17find_first_not_ofEcm;2B.;find_last_not_of::(0,610):_ZNKSs16find_last_not_ofERKSsm;2B.(0,606):_ZNKSs16find_last_not_ofEPKcmm;2B.(0,611):_ZNKSs16find_last_not_ofEPKcm;2B.(0,612):_ZNKSs16find_last_not_ofEcm;2B.;substr::(0,613)=#(0,286),(0,286),(0,526),(0,69),(0,69),(0,2);:_ZNKSs6substrEmm;2B.;compare::(0,614)=#(0,286),(0,10),(0,526),(0,550),(0,2);:_ZNKSs7compareERKSs;2B.(0,615)=#(0,286),(0,10),(0,526),(0,69),(0,69),(0,550),(0,2);:_ZNKSs7compareEmmRKSs;2B.(0,616)=#(0,286),(0,10),(0,526),(0,69),(0,69),(0,550),(0,69),(0,69),(0,2);:_ZNKSs7compareEmmRKSsmm;2B.(0,617)=#(0,286),(0,10),(0,526),(0,36),(0,2);:_ZNKSs7compareEPKc;2B.(0,618)=#(0,286),(0,10),(0,526),(0,69),(0,69),(0,36),(0,2);:_ZNKSs7compareEmmPKc;2B.(0,619)=#(0,286),(0,10),(0,526),(0,69),(0,69),(0,36),(0,69),(0,2);:_ZNKSs7compareEmmPKcm;2B.;;",128,0,56,0
	.stabs	"QCString:Tt(0,295)=s8!1,020,(0,277);QCString::(0,620)=#(0,295),(0,2),(0,621)=*(0,295),(0,2);:_ZN8QCStringC4Ev;2A.;__base_ctor::(0,620):_ZN8QCStringC2Ev;2A.;__comp_ctor::(0,620):_ZN8QCStringC1Ev;2A.;__base_ctor::(0,622)=#(0,295),(0,2),(0,621),(0,10),(0,2);:_ZN8QCStringC2Ei;2A.;__comp_ctor::(0,622):_ZN8QCStringC1Ei;2A.;QCString::(0,623)=#(0,295),(0,2),(0,621),(0,293),(0,2);:_ZN8QCStringC4ERKS_;2A.;__base_ctor::(0,623):_ZN8QCStringC2ERKS_;2A.;__comp_ctor::(0,623):_ZN8QCStringC1ERKS_;2A.;__base_ctor::(0,624)=#(0,295),(0,2),(0,621),(0,36),(0,2);:_ZN8QCStringC2EPKc;2A.;__comp_ctor::(0,624):_ZN8QCStringC1EPKc;2A.;__base_ctor::(0,625)=#(0,295),(0,2),(0,621),(0,36),(0,6),(0,2);:_ZN8QCStringC2EPKcj;2A.;__comp_ctor::(0,625):_ZN8QCStringC1EPKcj;2A.;__base_dtor::(0,620):_ZN8QCStringD2Ev;2A.;__comp_dtor::(0,620):_ZN8QCStringD1Ev;2A*0;(0,295);;__deleting_dtor::(0,620):_ZN8QCStringD0Ev;2A*1;(0,295);;operator=::(0,626)=#(0,295),(0,627)=&(0,295),(0,621),(0,293),(0,2);:_ZN8QCStringaSERKS_;2A.(0,628)=#(0,295),(0,627),(0,621),(0,36),(0,2);:_ZN8QCStringaSEPKc;2A.;isNull::(0,629)=#(0,295),(0,1),(0,630)=*(0,294),(0,2);:_ZNK8QCString6isNullEv;2B.;isEmpty::(0,629):_ZNK8QCString7isEmptyEv;2B.;length::(0,631)=#(0,295),(0,301),(0,630),(0,2);:_ZNK8QCString6lengthEv;2B.;resize::(0,632)=#(0,295),(0,1),(0,621),(0,6),(0,2);:_ZN8QCString6resizeEj;2A.;truncate::(0,632):_ZN8QCString8truncateEj;2A.;fill::(0,633)=#(0,295),(0,1),(0,621),(0,38),(0,10),(0,2);:_ZN8QCString4fillEci;2A.;copy::(0,634)=#(0,295),(0,295),(0,630),(0,2);:_ZNK8QCString4copyEv;2B.;sprintf::(0,635)=#(0,295),(0,627),(0,621),(0,36);:_ZN8QCString7sprintfEPKcz;2A.;find::(0,636)=#(0,295),(0,10),(0,630),(0,38),(0,10),(0,1),(0,2);:_ZNK8QCString4findEcib;2B.(0,637)=#(0,295),(0,10),(0,630),(0,36),(0,10),(0,1),(0,2);:_ZNK8QCString4findEPKcib;2B.(0,638)=#(0,295),(0,10),(0,630),(0,329),(0,10),(0,2);:_ZNK8QCString4findERK7QRegExpi;2B.;findRev::(0,639)=#(0,295),(0,10),(0,630),(0,38),(0,10),(0,1),(0,2);:_ZNK8QCString7findRevEcib;2B.(0,640)=#(0,295),(0,10),(0,630),(0,36),(0,10),(0,1),(0,2);:_ZNK8QCString7findRevEPKcib;2B.(0,641)=#(0,295),(0,10),(0,630),(0,329),(0,10),(0,2);:_ZNK8QCString7findRevERK7QRegExpi;2B.;contains::(0,642)=#(0,295),(0,10),(0,630),(0,38),(0,1),(0,2);:_ZNK8QCString8containsEcb;2B.(0,643)=#(0,295),(0,10),(0,630),(0,36),(0,1),(0,2);:_ZNK8QCString8containsEPKcb;2B.(0,644)=#(0,295),(0,10),(0,630),(0,329),(0,2);:_ZNK8QCString8containsERK7QRegExp;2B.;left::(0,645)=#(0,295),(0,295),(0,630),(0,6),(0,2);:_ZNK8QCString4leftEj;2B.;right::(0,645):_ZNK8QCString5rightEj;2B.;mid::(0,646)=#(0,295),(0,295),(0,630),(0,6),(0,6),(0,2);:_ZNK8QCString3midEjj;2B.;leftJustify::(0,647)=#(0,295),(0,295),(0,630),(0,6),(0,38),(0,1),(0,2);:_ZNK8QCString11leftJustifyEjcb;2B.;rightJustify::(0,648)=#(0,295),(0,295),(0,630),(0,6),(0,38),(0,1),(0,2);:_ZNK8QCString12rightJustifyEjcb;2B.;lower::(0,634):_ZNK8QCString5lowerEv;2B.;upper::(0,634):_ZNK8QCString5upperEv;2B.;stripWhiteSpace::(0,634):_ZNK8QCString15stripWhiteSpaceEv;2B.;simplifyWhiteSpace::(0,634):_ZNK8QCString18simplifyWhiteSpaceEv;2B.;insert::(0,649)=#(0,295),(0,627),(0,621),(0,6),(0,36),(0,2);:_ZN8QCString6insertEjPKc;2A.(0,650)=#(0,295),(0,627),(0,621),(0,6),(0,38),(0,2);:_ZN8QCString6insertEjc;2A.;append::(0,628):_ZN8QCString6appendEPKc;2A.;prepend::(0,628):_ZN8QCString7prependEPKc;2A.;remove::(0,651)=#(0,295),(0,627),(0,621),(0,6),(0,6),(0,2);:_ZN8QCString6removeEjj;2A.;replace::(0,652)=#(0,295),(0,627),(0,621),(0,6),(0,6),(0,36),(0,2);:_ZN8QCString7replaceEjjPKc;2A.(0,653)=#(0,295),(0,627),(0,621),(0,329),(0,36),(0,2);:_ZN8QCString7replaceERK7QRegExpPKc;2A.(0,654)=#(0,295),(0,627),(0,621),(0,38),(0,36),(0,2);:_ZN8QCString7replaceEcPKc;2A.(0,655)=#(0,295),(0,627),(0,621),(0,36),(0,36),(0,2);:_ZN8QCString7replaceEPKcS1_;2A.(0,656)=#(0,295),(0,627),(0,621),(0,38),(0,38),(0,2);:_ZN8QCString7replaceEcc;2A.;toShort::(0,657)=#(0,295),(0,314),(0,630),(0,375),(0,2);:_ZNK8QCString7toShortEPb;2B.;toUShort::(0,658)=#(0,295),(0,377),(0,630),(0,375),(0,2);:_ZNK8QCString8toUShortEPb;2B.;toInt::(0,659)=#(0,295),(0,10),(0,630),(0,375),(0,2);:_ZNK8QCString5toIntEPb;2B.;toUInt::(0,660)=#(0,295),(0,301),(0,630),(0,375),(0,2);:_ZNK8QCString6toUIntEPb;2B.;toLong::(0,661)=#(0,295),(0,208),(0,630),(0,375),(0,2);:_ZNK8QCString6toLongEPb;2B.;toULong::(0,662)=#(0,295),(0,382),(0,630),(0,375),(0,2);:_ZNK8QCString7toULongEPb;2B.;toFloat::(0,663)=#(0,295),(0,390),(0,630),(0,375),(0,2);:_ZNK8QCString7toFloatEPb;2B.;toDouble::(0,664)=#(0,295),(0,253),(0,630),(0,375),(0,2);:_ZNK8QCString8toDoubleEPb;2B.;setStr::(0,628):_ZN8QCString6setStrEPKc;2A.;setNum::(0,665)=#(0,295),(0,627),(0,621),(0,314),(0,2);:_ZN8QCString6setNumEs;2A.(0,666)=#(0,295),(0,627),(0,621),(0,316),(0,2);:_ZN8QCString6setNumEt;2A.(0,667)=#(0,295),(0,627),(0,621),(0,10),(0,2);:_ZN8QCString6setNumEi;2A.(0,668)=#(0,295),(0,627),(0,621),(0,6),(0,2);:_ZN8QCString6setNumEj;2A.(0,669)=#(0,295),(0,627),(0,621),(0,208),(0,2);:_ZN8QCString6setNumEl;2A.(0,670)=#(0,295),(0,627),(0,621),(0,69),(0,2);:_ZN8QCString6setNumEm;2A.(0,671)=#(0,295),(0,627),(0,621),(0,390),(0,38),(0,10),(0,2);:_ZN8QCString6setNumEfci;2A.(0,672)=#(0,295),(0,627),(0,621),(0,253),(0,38),(0,10),(0,2);:_ZN8QCString6setNumEdci;2A.;setExpand::(0,673)=#(0,295),(0,1),(0,621),(0,6),(0,38),(0,2);:_ZN8QCString9setExpandEjc;2A.;operator PKc::(0,674)=#(0,295),(0,36),(0,630),(0,2);:_ZNK8QCStringcvPKcEv;2B.;operator+=::(0,628):_ZN8QCStringpLEPKc;2A.(0,675)=#(0,295),(0,627),(0,621),(0,38),(0,2);:_ZN8QCStringpLEc;2A.;find::(0,676)=#(0,295),(0,10),(0,630),(0,36),(0,10),(0,1),(0,6),(0,2);:_ZNK8QCString4findEPKcibj;0B.;;~%(0,492);",128,0,143,0
	.stabs	"uint:t(0,301)",128,0,617,0
	.stabs	"long long int:t(0,308)",128,0,0,0
	.stabs	"long long unsigned int:t(0,310)",128,0,0,0
	.stabs	"short int:t(0,314)",128,0,0,0
	.stabs	"short unsigned int:t(0,316)",128,0,0,0
	.stabs	"ushort:t(0,377)",128,0,616,0
	.stabs	"ulong:t(0,382)",128,0,618,0
	.stabs	"Q_LLONG:t(0,384)",128,0,723,0
	.stabs	"Q_INT64:t(0,385)",128,0,720,0
	.stabs	"Q_ULLONG:t(0,387)",128,0,724,0
	.stabs	"Q_UINT64:t(0,388)",128,0,721,0
	.stabs	"float:t(0,390)",128,0,0,0
	.stabs	"QCharRef:Tt(0,413)=s8s:/0(0,289),0,32;p:/0(0,301),32,32;__base_ctor::(0,677)=#(0,413),(0,2),(0,678)=*(0,413),(0,679)=&(0,680)=k(0,413),(0,2);:_ZN8QCharRefC2ERKS_;2A.;__comp_ctor::(0,677):_ZN8QCharRefC1ERKS_;2A.;QCharRef::(0,681)=#(0,413),(0,2),(0,678),(0,268),(0,6),(0,2);:_ZN8QCharRefC4EP7QStringj;0A.;__base_ctor::(0,681):_ZN8QCharRefC2EP7QStringj;0A.;__comp_ctor::(0,681):_ZN8QCharRefC1EP7QStringj;0A.;unicode::(0,682)=#(0,413),(0,377),(0,683)=*(0,680),(0,2);:_ZNK8QCharRef7unicodeEv;2B.;latin1::(0,684)=#(0,413),(0,38),(0,683),(0,2);:_ZNK8QCharRef6latin1Ev;2B.;operator=::(0,685)=#(0,413),(0,413),(0,678),(0,38),(0,2);:_ZN8QCharRefaSEc;2A.(0,686)=#(0,413),(0,413),(0,678),(0,465),(0,2);:_ZN8QCharRefaSEh;2A.(0,687)=#(0,413),(0,413),(0,678),(0,270),(0,2);:_ZN8QCharRefaSE5QChar;2A.(0,688)=#(0,413),(0,413),(0,678),(0,679),(0,2);:_ZN8QCharRefaSERKS_;2A.(0,689)=#(0,413),(0,413),(0,678),(0,316),(0,2);:_ZN8QCharRefaSEt;2A.(0,690)=#(0,413),(0,413),(0,678),(0,314),(0,2);:_ZN8QCharRefaSEs;2A.(0,691)=#(0,413),(0,413),(0,678),(0,6),(0,2);:_ZN8QCharRefaSEj;2A.(0,692)=#(0,413),(0,413),(0,678),(0,10),(0,2);:_ZN8QCharRefaSEi;2A.;operator 5QChar::(0,693)=#(0,413),(0,270),(0,683),(0,2);:_ZNK8QCharRefcv5QCharEv;2B.;isNull::(0,694)=#(0,413),(0,1),(0,683),(0,2);:_ZNK8QCharRef6isNullEv;2B.;isPrint::(0,694):_ZNK8QCharRef7isPrintEv;2B.;isPunct::(0,694):_ZNK8QCharRef7isPunctEv;2B.;isSpace::(0,694):_ZNK8QCharRef7isSpaceEv;2B.;isMark::(0,694):_ZNK8QCharRef6isMarkEv;2B.;isLetter::(0,694):_ZNK8QCharRef8isLetterEv;2B.;isNumber::(0,694):_ZNK8QCharRef8isNumberEv;2B.;isLetterOrNumber::(0,695)=#(0,413),(0,1),(0,678),(0,2);:_ZN8QCharRef16isLetterOrNumberEv;2A.;isDigit::(0,694):_ZNK8QCharRef7isDigitEv;2B.;digitValue::(0,696)=#(0,413),(0,10),(0,683),(0,2);:_ZNK8QCharRef10digitValueEv;2B.;lower::(0,693):_ZNK8QCharRef5lowerEv;2B.;upper::(0,693):_ZNK8QCharRef5upperEv;2B.;category::(0,697)=#(0,413),(0,475),(0,683),(0,2);:_ZNK8QCharRef8categoryEv;2B.;direction::(0,698)=#(0,413),(0,477),(0,683),(0,2);:_ZNK8QCharRef9directionEv;2B.;joining::(0,699)=#(0,413),(0,479),(0,683),(0,2);:_ZNK8QCharRef7joiningEv;2B.;mirrored::(0,694):_ZNK8QCharRef8mirroredEv;2B.;mirroredChar::(0,693):_ZNK8QCharRef12mirroredCharEv;2B.;decomposition::(0,700)=#(0,413),(0,272),(0,683),(0,2);:_ZNK8QCharRef13decompositionEv;2B.;decompositionTag::(0,701)=#(0,413),(0,483),(0,683),(0,2);:_ZNK8QCharRef16decompositionTagEv;2B.;combiningClass::(0,702)=#(0,413),(0,465),(0,683),(0,2);:_ZNK8QCharRef14combiningClassEv;2B.;cell::(0,703)=#(0,413),(0,490),(0,683),(0,2);:_ZNK8QCharRef4cellEv;2B.;row::(0,703):_ZNK8QCharRef3rowEv;2B.;;",128,0,753,0
	.stabs	"QShared:Tt(0,451)=s4count:(0,301),0,32;operator=::(0,704)=#(0,451),(0,705)=&(0,451),(0,706)=*(0,451),(0,707)=&(0,708)=k(0,451),(0,2);:_ZN7QSharedaSERKS_;2A.;__base_ctor::(0,709)=#(0,451),(0,2),(0,706),(0,707),(0,2);:_ZN7QSharedC2ERKS_;2A.;__comp_ctor::(0,709):_ZN7QSharedC1ERKS_;2A.;QShared::(0,710)=#(0,451),(0,2),(0,706),(0,2);:_ZN7QSharedC4Ev;2A.;__base_ctor::(0,710):_ZN7QSharedC2Ev;2A.;__comp_ctor::(0,710):_ZN7QSharedC1Ev;2A.;ref::(0,710):_ZN7QShared3refEv;2A.;deref::(0,711)=#(0,451),(0,1),(0,706),(0,2);:_ZN7QShared5derefEv;2A.;;",128,0,47,0
	.stabs	"unsigned char:t(0,465)",128,0,0,0
	.stabs	"QChar::Category:t(0,475)=eNoCategory:0,Mark_NonSpacing:1,Mark_SpacingCombining:2,Mark_Enclosing:3,Number_DecimalDigit:4,Number_Letter:5,Number_Other:6,Separator_Space:7,Separator_Line:8,Separator_Paragraph:9,Other_Control:10,Other_Format:11,Other_Surrogate:12,Other_PrivateUse:13,Other_NotAssigned:14,Letter_Uppercase:15,Letter_Lowercase:16,Letter_Titlecase:17,Letter_Modifier:18,Letter_Other:19,Punctuation_Connector:20,Punctuation_Dash:21,Punctuation_Dask:21,Punctuation_Open:22,Punctuation_Close:23,Punctuation_InitialQuote:24,Punctuation_FinalQuote:25,Punctuation_Other:26,Symbol_Math:27,Symbol_Currency:28,Symbol_Modifier:29,Symbol_Other:30,;",128,0,94,0
	.stabs	"QChar::Direction:t(0,477)=eDirL:0,DirR:1,DirEN:2,DirES:3,DirET:4,DirAN:5,DirCS:6,DirB:7,DirS:8,DirWS:9,DirON:10,DirLRE:11,DirLRO:12,DirAL:13,DirRLE:14,DirRLO:15,DirPDF:16,DirNSM:17,DirBN:18,;",128,0,137,0
	.stabs	"QChar::Joining:t(0,479)=eOtherJoining:0,Dual:1,Right:2,Center:3,;",128,0,150,0
	.stabs	"QChar::Decomposition:t(0,483)=eSingle:0,Canonical:1,Font:2,NoBreak:3,Initial:4,Medial:5,Final:6,Isolated:7,Circle:8,Super:9,Sub:10,Vertical:11,Wide:12,Narrow:13,Small:14,Square:15,Compat:16,Fraction:17,;",128,0,143,0
	.stabs	"uchar:t(0,490)",128,0,615,0
	.stabs	"QGArray:Tt(0,492)=s8_vptr$QGArray:(0,7),0,32;shd:/0(0,712)=*(0,713)=xsarray_data:,32,32;__base_ctor::(0,714)=#(0,492),(0,2),(0,715)=*(0,492),(0,2);:_ZN7QGArrayC2Ev;2A.;__comp_ctor::(0,714):_ZN7QGArrayC1Ev;2A.;__base_ctor::(0,716)=#(0,492),(0,2),(0,715),(0,10),(0,10),(0,2);:_ZN7QGArrayC2Eii;1A.;__comp_ctor::(0,716):_ZN7QGArrayC1Eii;1A.;__base_ctor::(0,717)=#(0,492),(0,2),(0,715),(0,10),(0,2);:_ZN7QGArrayC2Ei;1A.;__comp_ctor::(0,717):_ZN7QGArrayC1Ei;1A.;__base_ctor::(0,718)=#(0,492),(0,2),(0,715),(0,719)=&(0,720)=k(0,492),(0,2);:_ZN7QGArrayC2ERKS_;1A.;__comp_ctor::(0,718):_ZN7QGArrayC1ERKS_;1A.;__base_dtor::(0,714):_ZN7QGArrayD2Ev;1A.;__comp_dtor::(0,714):_ZN7QGArrayD1Ev;1A*0;(0,492);;__deleting_dtor::(0,714):_ZN7QGArrayD0Ev;1A*1;(0,492);;operator=::(0,721)=#(0,492),(0,722)=&(0,492),(0,715),(0,719),(0,2);:_ZN7QGArrayaSERKS_;1A.;detach::(0,714):_ZN7QGArray6detachEv;1A*2;(0,492);;data::(0,723)=#(0,492),(0,448),(0,724)=*(0,720),(0,2);:_ZNK7QGArray4dataEv;1B.;nrefs::(0,725)=#(0,492),(0,301),(0,724),(0,2);:_ZNK7QGArray5nrefsEv;1B.;size::(0,725):_ZNK7QGArray4sizeEv;1B.;isEqual::(0,726)=#(0,492),(0,1),(0,724),(0,719),(0,2);:_ZNK7QGArray7isEqualERKS_;1B.;resize::(0,727)=#(0,492),(0,1),(0,715),(0,6),(0,508),(0,2);:_ZN7QGArray6resizeEjNS_12OptimizationE;1A.(0,728)=#(0,492),(0,1),(0,715),(0,6),(0,2);:_ZN7QGArray6resizeEj;1A.;fill::(0,729)=#(0,492),(0,1),(0,715),(0,36),(0,10),(0,6),(0,2);:_ZN7QGArray4fillEPKcij;1A.;assign::(0,721):_ZN7QGArray6assignERKS_;1A.(0,730)=#(0,492),(0,722),(0,715),(0,36),(0,6),(0,2);:_ZN7QGArray6assignEPKcj;1A.;duplicate::(0,721):_ZN7QGArray9duplicateERKS_;1A.(0,730):_ZN7QGArray9duplicateEPKcj;1A.;store::(0,731)=#(0,492),(0,2),(0,715),(0,36),(0,6),(0,2);:_ZN7QGArray5storeEPKcj;1A.;sharedBlock::(0,732)=#(0,492),(0,712),(0,724),(0,2);:_ZNK7QGArray11sharedBlockEv;1B.;setSharedBlock::(0,733)=#(0,492),(0,2),(0,715),(0,712),(0,2);:_ZN7QGArray14setSharedBlockEPNS_10array_dataE;1A.;setRawData::(0,730):_ZN7QGArray10setRawDataEPKcj;1A.;resetRawData::(0,731):_ZN7QGArray12resetRawDataEPKcj;1A.;find::(0,734)=#(0,492),(0,10),(0,724),(0,36),(0,6),(0,6),(0,2);:_ZNK7QGArray4findEPKcjj;1B.;contains::(0,735)=#(0,492),(0,10),(0,724),(0,36),(0,6),(0,2);:_ZNK7QGArray8containsEPKcj;1B.;sort::(0,736)=#(0,492),(0,2),(0,715),(0,6),(0,2);:_ZN7QGArray4sortEj;1A.;bsearch::(0,735):_ZNK7QGArray7bsearchEPKcj;1B.;at::(0,737)=#(0,492),(0,448),(0,724),(0,6),(0,2);:_ZNK7QGArray2atEj;1B.;setExpand::(0,738)=#(0,492),(0,1),(0,715),(0,6),(0,36),(0,6),(0,2);:_ZN7QGArray9setExpandEjPKcj;1A.;newData::(0,739)=#(0,492),(0,712),(0,715),(0,2);:_ZN7QGArray7newDataEv;1A*3;(0,492);;deleteData::(0,733):_ZN7QGArray10deleteDataEPNS_10array_dataE;1A*4;(0,492);;msg_index::(0,740)=f(0,2):_ZN7QGArray9msg_indexEj;0A?;;~%(0,492);",128,0,47,0
	.stabs	"QGArray::Optimization:t(0,508)=eMemOptim:0,SpeedOptim:1,;",128,0,65,0
	.stabs	"basic_string<char,std::char_traits<char>,std::allocator<char> >::_Alloc_hider:Tt(0,523)=s4!1,020,(0,548);_M_p:(0,448),0,32;operator=::(0,741)=#(0,523),(0,742)=&(0,523),(0,743)=*(0,523),(0,744)=&(0,745)=k(0,523),(0,2);:_ZNSs12_Alloc_hideraSERKS_;2A.;__base_ctor::(0,746)=#(0,523),(0,2),(0,743),(0,744),(0,2);:_ZNSs12_Alloc_hiderC2ERKS_;2A.;__comp_ctor::(0,746):_ZNSs12_Alloc_hiderC1ERKS_;2A.;__base_dtor::(0,747)=#(0,523),(0,2),(0,743),(0,2);:_ZNSs12_Alloc_hiderD2Ev;2A.;__comp_dtor::(0,747):_ZNSs12_Alloc_hiderD1Ev;2A.;__base_ctor::(0,748)=#(0,523),(0,2),(0,743),(0,448),(0,546),(0,2);:_ZNSs12_Alloc_hiderC2EPcRKSaIcE;2A.;__comp_ctor::(0,748):_ZNSs12_Alloc_hiderC1EPcRKSaIcE;2A.;;",128,0,230,0
	.stabs	"basic_string<char,std::char_traits<char>,std::allocator<char> >::_Rep:Tt(0,532)=s12_S_max_size:/2(0,522):_ZNSs4_Rep11_S_max_sizeE;_S_terminal:/2(0,37):_ZNSs4_Rep11_S_terminalE;_M_length:(0,67),0,32;_M_capacity:(0,67),32,32;_M_references:(0,749)=(0,10),64,32;operator=::(0,750)=#(0,532),(0,544),(0,531),(0,751)=&(0,752)=k(0,532),(0,2);:_ZNSs4_RepaSERKS_;2A.;__base_ctor::(0,753)=#(0,532),(0,2),(0,531),(0,751),(0,2);:_ZNSs4_RepC2ERKS_;2A.;__comp_ctor::(0,753):_ZNSs4_RepC1ERKS_;2A.;__base_ctor::(0,754)=#(0,532),(0,2),(0,531),(0,2);:_ZNSs4_RepC2Ev;2A.;__comp_ctor::(0,754):_ZNSs4_RepC1Ev;2A.;_M_is_leaked::(0,755)=#(0,532),(0,1),(0,756)=*(0,752),(0,2);:_ZNKSs4_Rep12_M_is_leakedEv;2B.;_M_is_shared::(0,755):_ZNKSs4_Rep12_M_is_sharedEv;2B.;_M_set_leaked::(0,757)=#(0,532),(0,2),(0,531),(0,2);:_ZNSs4_Rep13_M_set_leakedEv;2A.;_M_set_sharable::(0,757):_ZNSs4_Rep15_M_set_sharableEv;2A.;_M_refdata::(0,758)=#(0,532),(0,448),(0,531),(0,2);:_ZNSs4_Rep10_M_refdataEv;2A.;operator[]::(0,759)=#(0,532),(0,517),(0,531),(0,69),(0,2);:_ZNSs4_RepixEm;2A.;_M_grab::(0,760)=#(0,532),(0,448),(0,531),(0,546),(0,546),(0,2);:_ZNSs4_Rep7_M_grabERKSaIcES2_;2A.;_S_create::(0,761)=f(0,531):_ZNSs4_Rep9_S_createEmRKSaIcE;2A?;_M_dispose::(0,762)=#(0,532),(0,2),(0,531),(0,546),(0,2);:_ZNSs4_Rep10_M_disposeERKSaIcE;2A.;_M_destroy::(0,763)=#(0,532),(0,2),(0,531),(0,546),(0,2);:_ZNSs4_Rep10_M_destroyERKSaIcE;2A.;_M_refcopy::(0,758):_ZNSs4_Rep10_M_refcopyEv;2A.;_M_clone::(0,764)=#(0,532),(0,448),(0,531),(0,546),(0,69),(0,2);:_ZNSs4_Rep8_M_cloneERKSaIcEm;2A.;;",128,0,148,0
	.stabs	"allocator<char>:Tt(0,548)=s1operator=::(0,765)=#(0,548),(0,766)=&(0,548),(0,767)=*(0,548),(0,546),(0,2);:_ZNSaIcEaSERKS_;2A.;__base_ctor::(0,768)=#(0,548),(0,2),(0,767),(0,2);:_ZNSaIcEC2Ev;2A.;__comp_ctor::(0,768):_ZNSaIcEC1Ev;2A.;__base_ctor::(0,769)=#(0,548),(0,2),(0,767),(0,546),(0,2);:_ZNSaIcEC2ERKS_;2A.;__comp_ctor::(0,769):_ZNSaIcEC1ERKS_;2A.;__base_dtor::(0,768):_ZNSaIcED2Ev;2A.;__comp_dtor::(0,768):_ZNSaIcED1Ev;2A.;address::(0,770)=#(0,548),(0,448),(0,771)=*(0,547),(0,517),(0,2);:_ZNKSaIcE7addressERc;2B.(0,772)=#(0,548),(0,36),(0,771),(0,510),(0,2);:_ZNKSaIcE7addressERKc;2B.;allocate::(0,773)=#(0,548),(0,448),(0,767),(0,69),(0,185),(0,2);:_ZNSaIcE8allocateEmPKv;2A.;deallocate::(0,774)=#(0,548),(0,2),(0,767),(0,448),(0,69),(0,2);:_ZNSaIcE10deallocateEPcm;2A.;max_size::(0,775)=#(0,548),(0,67),(0,771),(0,2);:_ZNKSaIcE8max_sizeEv;2B.;construct::(0,776)=#(0,548),(0,2),(0,767),(0,448),(0,510),(0,2);:_ZNSaIcE9constructEPcRKc;2A.;destroy::(0,777)=#(0,548),(0,2),(0,767),(0,448),(0,2);:_ZNSaIcE7destroyEPc;2A.;;",128,0,49,0
	.stabs	"QGArray::array_data:Tt(0,713)=s12!1,020,(0,451);data:(0,448),32,32;len:(0,301),64,32;operator=::(0,778)=#(0,713),(0,779)=&(0,713),(0,712),(0,780)=&(0,781)=k(0,713),(0,2);:_ZN7QGArray10array_dataaSERKS0_;2A.;__base_ctor::(0,782)=#(0,713),(0,2),(0,712),(0,780),(0,2);:_ZN7QGArray10array_dataC2ERKS0_;2A.;__comp_ctor::(0,782):_ZN7QGArray10array_dataC1ERKS0_;2A.;array_data::(0,783)=#(0,713),(0,2),(0,712),(0,2);:_ZN7QGArray10array_dataC4Ev;2A.;__base_ctor::(0,783):_ZN7QGArray10array_dataC2Ev;2A.;__comp_ctor::(0,783):_ZN7QGArray10array_dataC1Ev;2A.;;",128,0,52,0
	.stabs	"_Atomic_word:t(0,749)",128,0,39,0
	.stabs	"time_stamp_start:r(0,69)",64,0,89,24
	.stabs	"skipped_cnt:r(0,6)",64,0,91,25
	.stabn	192,0,0,LBB230
	.stabs	"transaction:(0,62)",128,0,98,144
	.stabs	"skip:r(0,1)",64,0,106,26
	.stabn	192,0,0,LBB233
	.stabs	"lock:(0,232)",128,0,99,80
	.stabn	192,0,0,LBB242
	.stabs	"mutex:r(0,117)",64,0,20,28
	.stabn	192,0,0,LBB243
	.stabn	224,0,0,LBE243
	.stabs	"this:r(0,242)",64,0,222,27
	.stabn	192,0,0,LBB250
	.stabs	"x:r(0,6)",64,0,185,0
	.stabn	192,0,0,LBB254
	.stabs	"ret:r(0,6)",64,0,186,2
	.stabn	192,0,0,LBB255
	.stabn	224,0,0,LBE255
	.stabn	224,0,0,LBE254
	.stabn	224,0,0,LBE250
	.stabs	"t:r(0,69)",64,0,216,0
	.stabn	192,0,0,LBB256
	.stabn	224,0,0,LBE256
	.stabn	224,0,0,LBE242
	.stabs	"lock:(0,232)",128,0,99,80
	.stabn	192,0,0,LBB267
	.stabn	224,0,0,LBE267
	.stabs	"XKameError:Tt(0,784)=s16m_msg:/0(0,262),0,32;m_file:/0(0,36),32,32;m_line:/0(0,10),64,32;m_errno:/0(0,10),96,32;operator=::(0,785)=#(0,784),(0,786)=&(0,784),(0,787)=*(0,784),(0,788)=&(0,789)=k(0,784),(0,2);:_ZN10XKameErroraSERKS_;2A.;__base_ctor::(0,790)=#(0,784),(0,2),(0,787),(0,788),(0,2);:_ZN10XKameErrorC2ERKS_;2A.;__comp_ctor::(0,790):_ZN10XKameErrorC1ERKS_;2A.;__base_dtor::(0,791)=#(0,784),(0,2),(0,787),(0,2);:_ZN10XKameErrorD2Ev;2A.;__comp_dtor::(0,791):_ZN10XKameErrorD1Ev;2A.;__base_ctor::(0,792)=#(0,784),(0,2),(0,787),(0,272),(0,36),(0,10),(0,2);:_ZN10XKameErrorC2ERK7QStringPKci;2A.;__comp_ctor::(0,792):_ZN10XKameErrorC1ERK7QStringPKci;2A.;print::(0,791):_ZN10XKameError5printEv;2A.(0,793)=#(0,784),(0,2),(0,787),(0,272),(0,2);:_ZN10XKameError5printERK7QString;2A.(0,794)=f(0,2):_ZN10XKameError5printERK7QStringPKcii;2A?;msg::(0,795)=#(0,784),(0,272),(0,796)=*(0,789),(0,2);:_ZNK10XKameError3msgEv;2B.;;",128,0,117,0
	.stabs	"lock:(0,232)",128,0,119,112
	.stabn	192,0,0,LBB271
	.stabn	224,0,0,LBE271
	.stabs	"lock:(0,232)",128,0,119,112
	.stabn	192,0,0,LBB277
	.stabs	"ret:r(0,6)",64,0,161,9
	.stabn	192,0,0,LBB285
	.stabn	224,0,0,LBE285
	.stabs	"t:r(0,69)",64,0,216,0
	.stabn	192,0,0,LBB286
	.stabn	224,0,0,LBE286
	.stabn	224,0,0,LBE277
	.stabs	"lock:(0,232)",128,0,119,112
	.stabn	192,0,0,LBB295
	.stabn	224,0,0,LBE295
	.stabn	224,0,0,LBE233
	.stabn	224,0,0,LBE230
.section __TEXT,__text,regular,pure_instructions
	.align 2
Lscope26:
	.stabs	"",36,0,0,Lscope26-__ZN13XSignalBuffer11synchronizeEv
.section __TEXT,__textcoal_nt,coalesced,no_toc
	.align 2
.weak_definition __ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED2Ev
.private_extern __ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED2Ev
.section __TEXT,__textcoal_nt,coalesced,no_toc
	.align 2
__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED2Ev:
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_deque.h",132,0,0,0
	.stabd	68,0,483
LFB1681:
	li r4,0
	b L__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED4Ev$stub
LFE1681:
	.stabs	"_ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED2Ev:F(0,2)",36,0,483,__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED2Ev
	.stabs	"this:P(0,797)=k(0,164)",64,0,483,3
.section __TEXT,__textcoal_nt,coalesced,no_toc
	.align 2
Lscope27:
	.stabs	"",36,0,0,Lscope27-__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED2Ev
	.align 2
.weak_definition __ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED4Ev
.private_extern __ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED4Ev
.section __TEXT,__textcoal_nt,coalesced,no_toc
	.align 2
__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED4Ev:
	.stabd	68,0,483
LFB1693:
	mflr r0
	stmw r27,-20(r1)
LCFI31:
	stw r0,8(r1)
LCFI32:
	bcl 20,31,"L00000000012$pb"
"L00000000012$pb":
	stwu r1,-112(r1)
LCFI33:
	mflr r31
	.stabd	68,0,484
LBB301:
	lwz r0,0(r3)
	.stabd	68,0,483
	mr r28,r3
	.stabd	68,0,484
	cmpwi cr7,r0,0
	beq+ cr7,L460
	.stabd	68,0,486
	lwz r5,36(r3)
	lwz r4,20(r3)
	addi r5,r5,4
	bl L__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE16_M_destroy_nodesEPPS1_S5_$stub
	.stabd	68,0,432
LBB302:
	lwz r29,4(r28)
	lwz r30,0(r28)
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_alloc.h",132,0,0,0
	.stabd	68,0,242
LBB303:
LBB304:
LBB305:
	cmpwi cr7,r29,0
	beq- cr7,L460
	.stabd	68,0,429
LBB306:
	slwi r29,r29,2
	.stabd	68,0,430
LBB307:
LBB308:
	cmplwi cr7,r29,128
	bgt- cr7,L465
	addis r2,r31,ha16(L__ZNSt24__default_alloc_templateILb1ELi0EE12_S_force_newE$non_lazy_ptr-"L00000000012$pb")
	lwz r2,lo16(L__ZNSt24__default_alloc_templateILb1ELi0EE12_S_force_newE$non_lazy_ptr-"L00000000012$pb")(r2)
	lwz r0,0(r2)
	cmpwi cr7,r0,0
	ble- cr7,L464
L465:
	.stabd	68,0,112
LBB309:
LBB310:
	mr r3,r30
	bl L__ZdlPv$stub
LBE310:
LBE309:
	b L460
L464:
	.stabd	68,0,361
L477:
L489:
	.stabs	"/usr/include/gcc/darwin/3.3/c++/ppc-darwin/bits/gthr-default.h",132,0,0,0
	.stabd	68,0,494
LBB311:
LBB312:
LBB313:
LBB314:
LBB315:
LBB316:
LBB317:
	addis r27,r31,ha16(L__ZNSt24__default_alloc_templateILb1ELi0EE22_S_node_allocator_lockE$non_lazy_ptr-"L00000000012$pb")
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_alloc.h",132,0,0,0
	.stabd	68,0,361
LBE317:
LBE316:
LBE315:
LBE314:
LBE313:
LBE312:
LBB318:
LBB319:
	addi r29,r29,7
	.stabs	"/usr/include/gcc/darwin/3.3/c++/ppc-darwin/bits/gthr-default.h",132,0,0,0
	.stabd	68,0,494
LBE319:
LBE318:
LBB320:
LBB321:
LBB322:
LBB323:
LBB324:
LBB325:
	lwz r27,lo16(L__ZNSt24__default_alloc_templateILb1ELi0EE22_S_node_allocator_lockE$non_lazy_ptr-"L00000000012$pb")(r27)
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_alloc.h",132,0,0,0
	.stabd	68,0,361
LBE325:
LBE324:
LBE323:
LBE322:
LBE321:
LBE320:
	addis r28,r31,ha16(L__ZNSt24__default_alloc_templateILb1ELi0EE12_S_free_listE$non_lazy_ptr-"L00000000012$pb")
	lwz r28,lo16(L__ZNSt24__default_alloc_templateILb1ELi0EE12_S_free_listE$non_lazy_ptr-"L00000000012$pb")(r28)
LBB326:
LBB327:
	rlwinm r29,r29,31,1,29
LBE327:
LBE326:
	addi r29,r29,-4
	.stabs	"/usr/include/gcc/darwin/3.3/c++/ppc-darwin/bits/gthr-default.h",132,0,0,0
	.stabd	68,0,496
LBB328:
LBB329:
LBB330:
LBB331:
LBB332:
LBB333:
LBB334:
	mr r3,r27
	bl L_pthread_mutex_lock$stub
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_alloc.h",132,0,0,0
	.stabd	68,0,442
LBE334:
LBE333:
LBE332:
LBE331:
LBE330:
LBE329:
LBE328:
	lwzx r0,r29,r28
	.stabs	"/usr/include/gcc/darwin/3.3/c++/ppc-darwin/bits/gthr-default.h",132,0,0,0
	.stabd	68,0,514
	mr r3,r27
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_alloc.h",132,0,0,0
	.stabd	68,0,442
	stw r0,0(r30)
	.stabd	68,0,443
	stwx r30,r29,r28
	.stabs	"/usr/include/gcc/darwin/3.3/c++/ppc-darwin/bits/gthr-default.h",132,0,0,0
	.stabd	68,0,514
	bl L_pthread_mutex_unlock$stub
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_deque.h",132,0,0,0
	.stabd	68,0,487
L460:
	lwz r0,120(r1)
	addi r1,r1,112
	lmw r27,-20(r1)
	mtlr r0
	blr
LBE311:
LBE308:
LBE307:
LBE306:
LBE305:
LBE304:
LBE303:
LBE302:
LBE301:
LFE1693:
	.stabs	"_ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED4Ev:F(0,2)",36,0,483,__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED4Ev
	.stabs	"this:P(0,797)",64,0,483,28
	.stabs	"__in_chrg:P(0,10)",64,0,483,4
	.stabs	"__p:r(0,173)",64,0,432,30
	.stabs	"__n:r(0,67)",64,0,432,29
	.stabn	192,0,0,LBB302
	.stabs	"__n:r(0,67)",64,0,429,29
	.stabn	192,0,0,LBB306
	.stabs	"__default_alloc_template<true,0>::_Obj:Tt(0,798)=u4_M_free_list_link:(0,799)=*(0,798),0,32;_M_client_data:(0,800)=ar(0,210);0;0;(0,38),0,8;operator=::(0,801)=#(0,798),(0,802)=&(0,798),(0,799),(0,803)=&(0,804)=k(0,798),(0,2);:_ZNSt24__default_alloc_templateILb1ELi0EE4_ObjaSERKS1_;2A.;__base_ctor::(0,805)=#(0,798),(0,2),(0,799),(0,803),(0,2);:_ZNSt24__default_alloc_templateILb1ELi0EE4_ObjC2ERKS1_;2A.;__comp_ctor::(0,805):_ZNSt24__default_alloc_templateILb1ELi0EE4_ObjC1ERKS1_;2A.;__base_ctor::(0,806)=#(0,798),(0,2),(0,799),(0,2);:_ZNSt24__default_alloc_templateILb1ELi0EE4_ObjC2Ev;2A.;__comp_ctor::(0,806):_ZNSt24__default_alloc_templateILb1ELi0EE4_ObjC1Ev;2A.;;",128,0,341,0
	.stabs	"__lock_instance:(0,807)=xs_Lock:",128,0,441,64
	.stabs	"__default_alloc_template<true,0>::_Lock:Tt(0,807)=s1operator=::(0,808)=#(0,807),(0,809)=&(0,807),(0,810)=*(0,807),(0,811)=&(0,812)=k(0,807),(0,2);:_ZNSt24__default_alloc_templateILb1ELi0EE5_LockaSERKS1_;2A.;__base_ctor::(0,813)=#(0,807),(0,2),(0,810),(0,811),(0,2);:_ZNSt24__default_alloc_templateILb1ELi0EE5_LockC2ERKS1_;2A.;__comp_ctor::(0,813):_ZNSt24__default_alloc_templateILb1ELi0EE5_LockC1ERKS1_;2A.;_Lock::(0,814)=#(0,807),(0,2),(0,810),(0,2);:_ZNSt24__default_alloc_templateILb1ELi0EE5_LockC4Ev;2A.;__base_ctor::(0,814):_ZNSt24__default_alloc_templateILb1ELi0EE5_LockC2Ev;2A.;__comp_ctor::(0,814):_ZNSt24__default_alloc_templateILb1ELi0EE5_LockC1Ev;2A.;_Lock::(0,815)=#(0,807),(0,2),(0,810),(0,10),(0,2);:_ZNSt24__default_alloc_templateILb1ELi0EE5_LockD4Ev;2A.;__base_dtor::(0,814):_ZNSt24__default_alloc_templateILb1ELi0EE5_LockD2Ev;2A.;__comp_dtor::(0,814):_ZNSt24__default_alloc_templateILb1ELi0EE5_LockD1Ev;2A.;;",128,0,376,0
	.stabn	192,0,0,LBB311
	.stabs	"mutex:r(0,816)=*(0,817)=(0,113)",64,0,494,27
	.stabs	"__gthread_mutex_t:t(0,817)",128,0,42,0
	.stabn	192,0,0,LBB317
	.stabn	224,0,0,LBE317
	.stabs	"mutex:r(0,816)",64,0,494,27
	.stabn	192,0,0,LBB325
	.stabn	224,0,0,LBE325
	.stabs	"mutex:r(0,816)",64,0,494,27
	.stabn	192,0,0,LBB333
	.stabn	224,0,0,LBE333
	.stabn	224,0,0,LBE311
	.stabn	224,0,0,LBE306
	.stabn	224,0,0,LBE302
.section __TEXT,__textcoal_nt,coalesced,no_toc
	.align 2
Lscope28:
	.stabs	"",36,0,0,Lscope28-__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED4Ev
.data
.cstring
	.align 2
LC3:
	.ascii "GLIBCPP_FORCE_NEW\0"
.section __TEXT,__textcoal_nt,coalesced,no_toc
	.align 2
.weak_definition __ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE17_M_initialize_mapEm
.private_extern __ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE17_M_initialize_mapEm
.section __TEXT,__textcoal_nt,coalesced,no_toc
	.align 2
__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE17_M_initialize_mapEm:
	.stabd	68,0,504
LFB1702:
	stmw r23,-36(r1)
LCFI34:
	mflr r0
	.stabd	68,0,92
LBB335:
	srwi r27,r4,7
	.stabd	68,0,504
	stw r0,8(r1)
LCFI35:
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_algobase.h",132,0,0,0
	.stabd	68,0,172
LBB336:
	addi r2,r27,3
	li r0,8
	.stabd	68,0,176
LBB337:
	cmplw cr7,r0,r2
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_deque.h",132,0,0,0
	.stabd	68,0,504
	stwu r1,-144(r1)
LCFI36:
	bcl 20,31,"L00000000013$pb"
"L00000000013$pb":
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_algobase.h",132,0,0,0
	.stabd	68,0,172
LBE337:
	stw r2,84(r1)
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_deque.h",132,0,0,0
	.stabd	68,0,504
	mflr r31
	mr r24,r4
	mr r28,r3
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_algobase.h",132,0,0,0
	.stabd	68,0,172
	stw r0,80(r1)
	.stabd	68,0,176
LBB338:
	addi r2,r1,84
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_deque.h",132,0,0,0
	.stabd	68,0,92
LBE338:
LBE336:
	addi r25,r27,1
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_algobase.h",132,0,0,0
	.stabd	68,0,176
LBB339:
LBB340:
	blt- cr7,L510
	addi r2,r1,80
L510:
	.stabd	68,0,172
LBE340:
LBE339:
	lwz r3,0(r2)
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_alloc.h",132,0,0,0
	.stabd	68,0,230
LBB341:
LBB342:
LBB343:
LBB344:
	li r9,0
	.stabd	68,0,231
	cmpwi cr7,r3,0
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_algobase.h",132,0,0,0
	.stabd	68,0,172
LBE344:
LBE343:
LBE342:
LBE341:
	stw r3,4(r28)
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_alloc.h",132,0,0,0
	.stabd	68,0,231
LBB345:
LBB346:
LBB347:
LBB348:
	beq- cr7,L511
	.stabd	68,0,394
LBB349:
LBB350:
	addis r26,r31,ha16(L__ZNSt24__default_alloc_templateILb1ELi0EE12_S_force_newE$non_lazy_ptr-"L00000000013$pb")
	.stabd	68,0,388
LBE350:
	slwi r30,r3,2
	.stabd	68,0,394
LBB351:
	lwz r29,lo16(L__ZNSt24__default_alloc_templateILb1ELi0EE12_S_force_newE$non_lazy_ptr-"L00000000013$pb")(r26)
	lwz r0,0(r29)
	cmpwi cr7,r0,0
	bne+ cr7,L512
	.stabd	68,0,396
	addis r3,r31,ha16(LC3-"L00000000013$pb")
	la r3,lo16(LC3-"L00000000013$pb")(r3)
	bl L_getenv$stub
	cmpwi cr7,r3,0
	beq- cr7,L513
	.stabs	"/usr/include/gcc/darwin/3.3/c++/ppc-darwin/bits/atomicity.h",132,0,0,0
	.stabd	68,0,65
LBB352:
LBB353:
	/* Inline atomic add */
0:	lwarx    r2,0,r29 
	addi   r2,r2,1 
	stwcx.   r2,0,r29 
	bne-     0b 
	/* End atomic add */
LBE353:
LBE352:
	b L512
L513:
LBB354:
LBB355:
	/* Inline atomic add */
0:	lwarx    r2,0,r29 
	addi   r2,r2,-1 
	stwcx.   r2,0,r29 
	bne-     0b 
	/* End atomic add */
	.stabd	68,0,63
L512:
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_alloc.h",132,0,0,0
	.stabd	68,0,402
LBE355:
LBE354:
LBB356:
	cmplwi cr7,r30,128
	bgt- cr7,L518
	lwz r2,lo16(L__ZNSt24__default_alloc_templateILb1ELi0EE12_S_force_newE$non_lazy_ptr-"L00000000013$pb")(r26)
	lwz r0,0(r2)
	cmpwi cr7,r0,0
	ble- cr7,L517
L518:
	.stabd	68,0,108
LBB357:
LBB358:
	mr r3,r30
LEHB24:
	bl L__Znwm$stub
	mr r30,r3
LBE358:
LBE357:
	b L520
L517:
	.stabd	68,0,361
LBB359:
LBB360:
LBB361:
	addi r30,r30,7
	addis r2,r31,ha16(L__ZNSt24__default_alloc_templateILb1ELi0EE12_S_free_listE$non_lazy_ptr-"L00000000013$pb")
	.stabs	"/usr/include/gcc/darwin/3.3/c++/ppc-darwin/bits/gthr-default.h",132,0,0,0
	.stabd	68,0,494
LBE361:
LBE360:
LBB362:
LBB363:
LBB364:
LBB365:
LBB366:
LBB367:
	addis r26,r31,ha16(L__ZNSt24__default_alloc_templateILb1ELi0EE22_S_node_allocator_lockE$non_lazy_ptr-"L00000000013$pb")
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_alloc.h",132,0,0,0
	.stabd	68,0,361
LBE367:
LBE366:
LBE365:
LBE364:
LBE363:
LBE362:
	lwz r23,lo16(L__ZNSt24__default_alloc_templateILb1ELi0EE12_S_free_listE$non_lazy_ptr-"L00000000013$pb")(r2)
LBB368:
LBB369:
	rlwinm r9,r30,31,1,29
	.stabs	"/usr/include/gcc/darwin/3.3/c++/ppc-darwin/bits/gthr-default.h",132,0,0,0
	.stabd	68,0,496
LBE369:
LBE368:
LBB370:
LBB371:
LBB372:
LBB373:
LBB374:
LBB375:
LBB376:
	lwz r3,lo16(L__ZNSt24__default_alloc_templateILb1ELi0EE22_S_node_allocator_lockE$non_lazy_ptr-"L00000000013$pb")(r26)
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_alloc.h",132,0,0,0
	.stabd	68,0,361
LBE376:
LBE375:
LBE374:
LBE373:
LBE372:
LBE371:
LBE370:
	addi r29,r9,-4
	.stabs	"/usr/include/gcc/darwin/3.3/c++/ppc-darwin/bits/gthr-default.h",132,0,0,0
	.stabd	68,0,496
LBB377:
LBB378:
LBB379:
LBB380:
LBB381:
LBB382:
LBB383:
	bl L_pthread_mutex_lock$stub
LEHE24:
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_alloc.h",132,0,0,0
	.stabd	68,0,412
LBE383:
LBE382:
LBE381:
LBE380:
LBE379:
LBE378:
LBE377:
	lwzx r3,r29,r23
	.stabd	68,0,413
	cmpwi cr7,r3,0
	bne+ cr7,L530
	.stabd	68,0,357
	rlwinm r3,r30,0,0,28
LEHB25:
	bl L__ZNSt24__default_alloc_templateILb1ELi0EE9_S_refillEm$stub
	mr r30,r3
	b L532
L530:
	.stabd	68,0,417
	lwz r0,0(r3)
	.stabd	68,0,418
	mr r30,r3
	.stabd	68,0,417
	stwx r0,r29,r23
L532:
	.stabd	68,0,420
	cmpwi cr7,r30,0
	bne+ cr7,L535
	.stabd	68,0,421
	bl L__ZSt17__throw_bad_allocv$stub
LEHE25:
	b L535
L586:
L534:
L546:
	mr r30,r3
	.stabs	"/usr/include/gcc/darwin/3.3/c++/ppc-darwin/bits/gthr-default.h",132,0,0,0
	.stabd	68,0,514
LBB384:
LBB385:
LBB386:
LBB387:
LBB388:
LBB389:
LBB390:
	lwz r3,lo16(L__ZNSt24__default_alloc_templateILb1ELi0EE22_S_node_allocator_lockE$non_lazy_ptr-"L00000000013$pb")(r26)
	bl L_pthread_mutex_unlock$stub
	.stabd	68,0,512
	b L587
L535:
	.stabd	68,0,514
LBE390:
LBE389:
LBE388:
LBE387:
LBE386:
LBE385:
LBE384:
	lwz r3,lo16(L__ZNSt24__default_alloc_templateILb1ELi0EE22_S_node_allocator_lockE$non_lazy_ptr-"L00000000013$pb")(r26)
LEHB26:
	bl L_pthread_mutex_unlock$stub
LEHE26:
L520:
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_alloc.h",132,0,0,0
	.stabd	68,0,388
LBE359:
LBE356:
LBE351:
LBE349:
	mr r9,r30
L511:
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_deque.h",132,0,0,0
	.stabd	68,0,515
LBE348:
LBE347:
LBE346:
LBE345:
	lwz r0,4(r28)
	.stabd	68,0,516
	slwi r2,r25,2
	.stabd	68,0,428
	stw r9,0(r28)
	.stabd	68,0,519
	mr r3,r28
	.stabd	68,0,515
	subf r0,r25,r0
	rlwinm r0,r0,1,0,29
	add r30,r9,r0
	.stabd	68,0,516
	add r29,r30,r2
	.stabd	68,0,519
	mr r4,r30
	mr r5,r29
LEHB27:
	bl L__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE15_M_create_nodesEPPS1_S5_$stub
LEHE27:
	b L561
	.stabd	68,0,520
L585:
L562:
	bl L___cxa_begin_catch$stub
	.stabd	68,0,432
LBB391:
	lwz r4,4(r28)
	lwz r3,0(r28)
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_alloc.h",132,0,0,0
	.stabd	68,0,242
LBB392:
LBB393:
LBB394:
	cmpwi cr7,r4,0
	beq- cr7,L565
	slwi r4,r4,2
LEHB28:
	bl L__ZNSt24__default_alloc_templateILb1ELi0EE10deallocateEPvm$stub
L565:
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_deque.h",132,0,0,0
	.stabd	68,0,523
LBE394:
LBE393:
LBE392:
LBE391:
	li r0,0
	.stabd	68,0,524
	stw r0,4(r28)
	.stabd	68,0,523
	stw r0,0(r28)
	.stabd	68,0,525
	bl L___cxa_rethrow$stub
LEHE28:
L584:
L566:
L568:
	mr r30,r3
	bl L___cxa_end_catch$stub
L587:
	mr r3,r30
LEHB29:
	bl L__Unwind_Resume$stub
LEHE29:
L561:
	.stabd	68,0,206
LBB395:
	addi r2,r28,8
LBE395:
LBB396:
	addi r11,r28,24
	.stabd	68,0,207
LBE396:
LBB397:
LBB398:
	stw r30,12(r2)
	.stabd	68,0,206
LBE398:
LBE397:
LBB399:
	addi r9,r29,-4
	.stabd	68,0,208
LBE399:
LBB400:
LBB401:
	lwz r10,0(r30)
	.stabd	68,0,207
LBE401:
LBE400:
LBB402:
LBB403:
	stw r9,12(r11)
	.stabd	68,0,112
LBE403:
LBE402:
LBB404:
LBB405:
	addi r0,r10,512
	.stabd	68,0,208
	stw r10,4(r2)
	.stabd	68,0,112
	stw r0,8(r2)
	.stabd	68,0,92
LBE405:
LBE404:
	slwi r0,r27,7
	.stabd	68,0,208
LBB406:
LBB407:
	lwz r9,-4(r29)
	.stabd	68,0,92
LBE407:
LBE406:
	subf r0,r0,r24
	slwi r0,r0,2
	.stabd	68,0,208
LBB408:
LBB409:
	stw r9,4(r11)
	.stabd	68,0,112
	addi r9,r9,512
	.stabd	68,0,92
LBE409:
LBE408:
	lwz r2,28(r28)
	.stabd	68,0,530
	lwz r10,12(r28)
	.stabd	68,0,92
	add r2,r2,r0
	.stabd	68,0,112
LBB410:
LBB411:
	stw r9,8(r11)
	.stabd	68,0,92
LBE411:
LBE410:
	stw r2,24(r28)
	.stabd	68,0,530
	stw r10,8(r28)
	lwz r0,152(r1)
	addi r1,r1,144
	lmw r23,-36(r1)
	mtlr r0
	blr
LBE335:
LFE1702:
.data
.section __DATA,__gcc_except_tab
	.align 2
LLSDA1702:
	.byte	0xff
	.byte	0x0
	.byte	0x59
	.byte	0x3
	.byte	0x4e
	.set L$set$61,LEHB24-LFB1702
	.long L$set$61
	.set L$set$62,LEHE24-LEHB24
	.long L$set$62
	.long	0x0
	.byte	0x0
	.set L$set$63,LEHB25-LFB1702
	.long L$set$63
	.set L$set$64,LEHE25-LEHB25
	.long L$set$64
	.set L$set$65,L586-LFB1702
	.long L$set$65
	.byte	0x0
	.set L$set$66,LEHB26-LFB1702
	.long L$set$66
	.set L$set$67,LEHE26-LEHB26
	.long L$set$67
	.long	0x0
	.byte	0x0
	.set L$set$68,LEHB27-LFB1702
	.long L$set$68
	.set L$set$69,LEHE27-LEHB27
	.long L$set$69
	.set L$set$70,L585-LFB1702
	.long L$set$70
	.byte	0x1
	.set L$set$71,LEHB28-LFB1702
	.long L$set$71
	.set L$set$72,LEHE28-LEHB28
	.long L$set$72
	.set L$set$73,L584-LFB1702
	.long L$set$73
	.byte	0x0
	.set L$set$74,LEHB29-LFB1702
	.long L$set$74
	.set L$set$75,LEHE29-LEHB29
	.long L$set$75
	.long	0x0
	.byte	0x0
	.byte	0x1
	.byte	0x0
	.align 2
	.long	0
.section __TEXT,__textcoal_nt,coalesced,no_toc
	.stabs	"_ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE17_M_initialize_mapEm:F(0,2)",36,0,504,__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE17_M_initialize_mapEm
	.stabs	"this:P(0,797)",64,0,504,28
	.stabs	"__num_elements:P(0,67)",64,0,504,24
	.stabs	"__num_nodes:r(0,67)",64,0,505,25
	.stabs	"__nstart:r(0,173)",64,0,515,30
	.stabs	"__nfinish:r(0,173)",64,0,516,29
	.stabn	192,0,0,LBB335
	.stabs	"__ret:r(0,173)",64,0,230,9
	.stabn	192,0,0,LBB344
	.stabn	224,0,0,LBE344
	.stabs	"__ret:r(0,173)",64,0,230,9
	.stabn	192,0,0,LBB348
	.stabs	"__n:r(0,67)",64,0,388,30
	.stabn	192,0,0,LBB349
	.stabs	"__ret:r(0,818)=*(0,2)",64,0,389,30
	.stabn	192,0,0,LBB350
	.stabn	224,0,0,LBE350
	.stabs	"__ret:r(0,818)",64,0,389,30
	.stabn	192,0,0,LBB351
	.stabs	"__tmp:r(0,749)",64,0,64,2
	.stabn	192,0,0,LBB353
	.stabn	224,0,0,LBE353
	.stabs	"__tmp:r(0,749)",64,0,64,2
	.stabn	192,0,0,LBB355
	.stabn	224,0,0,LBE355
	.stabs	"__lock_instance:(0,807)",128,0,411,64
	.stabs	"__result:r(0,819)=*(0,798)",64,0,412,3
	.stabn	192,0,0,LBB359
	.stabn	224,0,0,LBE359
	.stabn	224,0,0,LBE351
	.stabn	224,0,0,LBE349
	.stabn	224,0,0,LBE348
	.stabs	"__p:r(0,173)",64,0,432,3
	.stabs	"__n:r(0,67)",64,0,432,4
	.stabn	192,0,0,LBB391
	.stabn	224,0,0,LBE391
	.stabs	"this:r(0,231)",64,0,206,2
	.stabn	192,0,0,LBB395
	.stabn	224,0,0,LBE395
	.stabs	"this:r(0,231)",64,0,206,11
	.stabs	"__new_node:r(0,173)",64,0,206,9
	.stabn	192,0,0,LBB396
	.stabn	224,0,0,LBE396
	.stabs	"this:r(0,231)",64,0,206,2
	.stabn	192,0,0,LBB397
	.stabn	224,0,0,LBE397
	.stabs	"this:r(0,231)",64,0,206,11
	.stabs	"__new_node:r(0,173)",64,0,206,9
	.stabn	192,0,0,LBB399
	.stabn	224,0,0,LBE399
	.stabs	"this:r(0,231)",64,0,206,2
	.stabn	192,0,0,LBB400
	.stabn	224,0,0,LBE400
	.stabs	"this:r(0,231)",64,0,206,11
	.stabs	"__new_node:r(0,173)",64,0,206,9
	.stabn	192,0,0,LBB402
	.stabn	224,0,0,LBE402
	.stabs	"this:r(0,231)",64,0,206,2
	.stabn	192,0,0,LBB404
	.stabn	224,0,0,LBE404
	.stabs	"this:r(0,231)",64,0,206,11
	.stabs	"__new_node:r(0,173)",64,0,206,9
	.stabn	192,0,0,LBB406
	.stabn	224,0,0,LBE406
	.stabs	"this:r(0,231)",64,0,206,11
	.stabs	"__new_node:r(0,173)",64,0,206,9
	.stabn	192,0,0,LBB408
	.stabn	224,0,0,LBE408
	.stabs	"this:r(0,231)",64,0,206,11
	.stabs	"__new_node:r(0,173)",64,0,206,9
	.stabn	192,0,0,LBB410
	.stabn	224,0,0,LBE410
	.stabn	224,0,0,LBE335
.section __TEXT,__textcoal_nt,coalesced,no_toc
	.align 2
Lscope29:
	.stabs	"",36,0,0,Lscope29-__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE17_M_initialize_mapEm
	.align 2
.weak_definition __ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_push_back_auxERKS1_
.private_extern __ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_push_back_auxERKS1_
.section __TEXT,__textcoal_nt,coalesced,no_toc
	.align 2
__ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_push_back_auxERKS1_:
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/deque.tcc",132,0,0,0
	.stabd	68,0,320
LFB1709:
	mflr r0
	stmw r28,-16(r1)
LCFI37:
	stw r0,8(r1)
LCFI38:
	mr r29,r3
	stwu r1,-80(r1)
LCFI39:
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_deque.h",132,0,0,0
	.stabd	68,0,1523
LBB412:
LBB413:
LBB414:
	lwz r30,36(r3)
	lwz r0,0(r3)
	lwz r2,4(r3)
	subf r0,r0,r30
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/deque.tcc",132,0,0,0
	.stabd	68,0,321
LBE414:
LBE413:
	lwz r28,0(r4)
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_deque.h",132,0,0,0
	.stabd	68,0,1523
LBB415:
LBB416:
	srawi r0,r0,2
	subf r2,r0,r2
	cmplwi cr7,r2,1
	bgt+ cr7,L590
	.stabd	68,0,1524
	li r4,1
	li r5,0
	bl L__ZNSt5dequeIP13_XTransactionSaIS1_EE17_M_reallocate_mapEmb$stub
	lwz r30,36(r29)
L590:
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_alloc.h",132,0,0,0
	.stabd	68,0,232
L598:
LBE416:
LBE415:
LBB417:
LBB418:
LBB419:
LBB420:
	li r3,512
	bl L__ZNSt24__default_alloc_templateILb1ELi0EE8allocateEm$stub
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_deque.h",132,0,0,0
	.stabd	68,0,416
LBE420:
LBE419:
LBE418:
LBE417:
	stw r3,4(r30)
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_construct.h",132,0,0,0
	.stabd	68,0,78
LBB421:
	lwz r2,24(r29)
	.stabs	"/usr/include/gcc/darwin/3.3/c++/new",132,0,0,0
	.stabd	68,0,92
LBB422:
	cmpwi cr7,r2,0
	beq- cr7,L613
	stw r28,0(r2)
L602:
	.stabd	68,0,96
L608:
L611:
L613:
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_deque.h",132,0,0,0
	.stabd	68,0,206
LBE422:
LBE421:
LBB423:
	lwz r11,36(r29)
	addi r9,r29,24
	addi r0,r11,4
	.stabd	68,0,207
LBB424:
	stw r0,12(r9)
	.stabd	68,0,208
	lwz r2,4(r11)
	stw r2,4(r9)
	.stabd	68,0,112
	addi r2,r2,512
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/deque.tcc",132,0,0,0
	.stabd	68,0,328
LBE424:
LBE423:
	lwz r0,28(r29)
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_deque.h",132,0,0,0
	.stabd	68,0,112
LBB425:
LBB426:
	stw r2,8(r9)
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/deque.tcc",132,0,0,0
	.stabd	68,0,328
LBE426:
LBE425:
	stw r0,24(r29)
	.stabd	68,0,333
	lwz r0,88(r1)
	addi r1,r1,80
	lmw r28,-16(r1)
	mtlr r0
	blr
L620:
L627:
L629:
LBE412:
LFE1709:
	.stabs	"_ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_push_back_auxERKS1_:F(0,2)",36,0,320,__ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_push_back_auxERKS1_
	.stabs	"this:P(0,820)=k(0,71)",64,0,320,29
	.stabs	"__t:P(0,76)",64,0,320,4
	.stabs	"__p:r(0,174)",64,0,78,2
	.stabn	192,0,0,LBB421
	.stabn	224,0,0,LBE421
	.stabs	"this:r(0,231)",64,0,206,9
	.stabs	"__new_node:r(0,173)",64,0,206,0
	.stabn	192,0,0,LBB423
	.stabn	224,0,0,LBE423
	.stabs	"this:r(0,231)",64,0,206,9
	.stabs	"__new_node:r(0,173)",64,0,206,0
	.stabn	192,0,0,LBB425
	.stabn	224,0,0,LBE425
.section __TEXT,__textcoal_nt,coalesced,no_toc
	.align 2
Lscope30:
	.stabs	"",36,0,0,Lscope30-__ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_push_back_auxERKS1_
	.align 2
.weak_definition __ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_pop_front_auxEv
.private_extern __ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_pop_front_auxEv
.section __TEXT,__textcoal_nt,coalesced,no_toc
	.align 2
__ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_pop_front_auxEv:
	.stabd	68,0,425
LFB1711:
	mflr r0
	stmw r29,-12(r1)
LCFI40:
	stw r0,8(r1)
LCFI41:
	mr r29,r3
	stwu r1,-80(r1)
LCFI42:
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_alloc.h",132,0,0,0
	.stabd	68,0,242
LBB427:
LBB428:
LBB429:
LBB430:
LBB431:
	li r4,512
	lwz r3,12(r3)
	bl L__ZNSt24__default_alloc_templateILb1ELi0EE10deallocateEPvm$stub
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_deque.h",132,0,0,0
	.stabd	68,0,206
LBE431:
LBE430:
LBE429:
LBE428:
LBB432:
	lwz r11,20(r29)
	addi r9,r29,8
	addi r0,r11,4
	.stabd	68,0,207
LBB433:
	stw r0,12(r9)
	.stabd	68,0,208
	lwz r2,4(r11)
	stw r2,4(r9)
	.stabd	68,0,112
	addi r2,r2,512
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/deque.tcc",132,0,0,0
	.stabd	68,0,429
LBE433:
LBE432:
	lwz r0,12(r29)
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_deque.h",132,0,0,0
	.stabd	68,0,112
LBB434:
LBB435:
	stw r2,8(r9)
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/deque.tcc",132,0,0,0
	.stabd	68,0,429
LBE435:
LBE434:
	stw r0,8(r29)
	lwz r0,88(r1)
	addi r1,r1,80
	lmw r29,-12(r1)
	mtlr r0
	blr
LBE427:
LFE1711:
	.stabs	"_ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_pop_front_auxEv:F(0,2)",36,0,425,__ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_pop_front_auxEv
	.stabs	"this:P(0,820)",64,0,425,29
	.stabs	"this:r(0,231)",64,0,206,9
	.stabs	"__new_node:r(0,173)",64,0,206,0
	.stabn	192,0,0,LBB432
	.stabn	224,0,0,LBE432
	.stabs	"this:r(0,231)",64,0,206,9
	.stabs	"__new_node:r(0,173)",64,0,206,0
	.stabn	192,0,0,LBB434
	.stabn	224,0,0,LBE434
.section __TEXT,__textcoal_nt,coalesced,no_toc
	.align 2
Lscope31:
	.stabs	"",36,0,0,Lscope31-__ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_pop_front_auxEv
	.align 2
.weak_definition __ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE16_M_destroy_nodesEPPS1_S5_
.private_extern __ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE16_M_destroy_nodesEPPS1_S5_
.section __TEXT,__textcoal_nt,coalesced,no_toc
	.align 2
__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE16_M_destroy_nodesEPPS1_S5_:
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_deque.h",132,0,0,0
	.stabd	68,0,554
LFB1718:
	.stabd	68,0,555
LBB436:
LBB437:
	cmplw cr7,r4,r5
	.stabd	68,0,554
	mflr r0
	stmw r29,-12(r1)
LCFI43:
	stw r0,8(r1)
LCFI44:
	mr r29,r5
	stwu r1,-80(r1)
LCFI45:
	.stabd	68,0,555
	mr r30,r4
	bge+ cr7,L657
L655:
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_alloc.h",132,0,0,0
	.stabd	68,0,242
LBB438:
LBB439:
LBB440:
LBB441:
	lwz r3,0(r30)
	li r4,512
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_deque.h",132,0,0,0
	.stabd	68,0,555
LBE441:
LBE440:
LBE439:
LBE438:
	addi r30,r30,4
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_alloc.h",132,0,0,0
	.stabd	68,0,242
LBB442:
LBB443:
LBB444:
LBB445:
	bl L__ZNSt24__default_alloc_templateILb1ELi0EE10deallocateEPvm$stub
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_deque.h",132,0,0,0
	.stabd	68,0,555
LBE445:
LBE444:
LBE443:
LBE442:
	cmplw cr7,r30,r29
	blt+ cr7,L655
L657:
	lwz r0,88(r1)
	addi r1,r1,80
	lmw r29,-12(r1)
	mtlr r0
	blr
LBE437:
LBE436:
LFE1718:
	.stabs	"_ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE16_M_destroy_nodesEPPS1_S5_:F(0,2)",36,0,554,__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE16_M_destroy_nodesEPPS1_S5_
	.stabs	"this:P(0,797)",64,0,554,3
	.stabs	"__nstart:P(0,173)",64,0,554,4
	.stabs	"__nfinish:P(0,173)",64,0,554,29
	.stabs	"__n:r(0,173)",64,0,555,30
	.stabn	192,0,0,LBB437
	.stabn	224,0,0,LBE437
.section __TEXT,__textcoal_nt,coalesced,no_toc
	.align 2
Lscope32:
	.stabs	"",36,0,0,Lscope32-__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE16_M_destroy_nodesEPPS1_S5_
	.align 2
.weak_definition __ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE15_M_create_nodesEPPS1_S5_
.private_extern __ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE15_M_create_nodesEPPS1_S5_
.section __TEXT,__textcoal_nt,coalesced,no_toc
	.align 2
__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE15_M_create_nodesEPPS1_S5_:
	.stabd	68,0,537
LFB1720:
	.stabd	68,0,541
LBB446:
	cmplw cr7,r4,r5
	.stabd	68,0,537
	mflr r0
	stmw r27,-20(r1)
LCFI46:
	stw r0,8(r1)
LCFI47:
	mr r28,r4
	stwu r1,-96(r1)
LCFI48:
	mr r30,r5
	.stabd	68,0,541
	mr r29,r4
	.stabd	68,0,537
	mr r27,r3
	.stabd	68,0,541
	bge+ cr7,L658
L669:
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_alloc.h",132,0,0,0
	.stabd	68,0,232
LBB447:
LBB448:
LBB449:
LBB450:
	li r3,512
LEHB30:
	bl L__ZNSt24__default_alloc_templateILb1ELi0EE8allocateEm$stub
LEHE30:
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_deque.h",132,0,0,0
	.stabd	68,0,416
LBE450:
LBE449:
LBE448:
LBE447:
	stw r3,0(r29)
	.stabd	68,0,541
	addi r29,r29,4
	cmplw cr7,r29,r30
	blt+ cr7,L669
	b L658
	.stabd	68,0,544
L678:
L671:
	bl L___cxa_begin_catch$stub
	.stabd	68,0,546
	mr r3,r27
	mr r4,r28
	mr r5,r29
LEHB31:
	bl L__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE16_M_destroy_nodesEPPS1_S5_$stub
	.stabd	68,0,547
	bl L___cxa_rethrow$stub
LEHE31:
L677:
L672:
L674:
	mr r29,r3
	bl L___cxa_end_catch$stub
	mr r3,r29
LEHB32:
	bl L__Unwind_Resume$stub
LEHE32:
L658:
	lwz r0,104(r1)
	addi r1,r1,96
	lmw r27,-20(r1)
	mtlr r0
	blr
LBE446:
LFE1720:
.data
.section __DATA,__gcc_except_tab
	.align 2
LLSDA1720:
	.byte	0xff
	.byte	0x0
	.byte	0x31
	.byte	0x3
	.byte	0x27
	.set L$set$76,LEHB30-LFB1720
	.long L$set$76
	.set L$set$77,LEHE30-LEHB30
	.long L$set$77
	.set L$set$78,L678-LFB1720
	.long L$set$78
	.byte	0x1
	.set L$set$79,LEHB31-LFB1720
	.long L$set$79
	.set L$set$80,LEHE31-LEHB31
	.long L$set$80
	.set L$set$81,L677-LFB1720
	.long L$set$81
	.byte	0x0
	.set L$set$82,LEHB32-LFB1720
	.long L$set$82
	.set L$set$83,LEHE32-LEHB32
	.long L$set$83
	.long	0x0
	.byte	0x0
	.byte	0x1
	.byte	0x0
	.align 2
	.long	0
.section __TEXT,__textcoal_nt,coalesced,no_toc
	.stabs	"_ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE15_M_create_nodesEPPS1_S5_:F(0,2)",36,0,537,__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE15_M_create_nodesEPPS1_S5_
	.stabs	"this:P(0,797)",64,0,537,27
	.stabs	"__nstart:P(0,173)",64,0,537,28
	.stabs	"__nfinish:P(0,173)",64,0,537,30
	.stabs	"__cur:r(0,173)",64,0,538,29
	.stabn	192,0,0,LBB446
	.stabn	224,0,0,LBE446
.section __TEXT,__textcoal_nt,coalesced,no_toc
	.align 2
Lscope33:
	.stabs	"",36,0,0,Lscope33-__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE15_M_create_nodesEPPS1_S5_
	.align 2
.weak_definition __ZNSt5dequeIP13_XTransactionSaIS1_EE17_M_reallocate_mapEmb
.private_extern __ZNSt5dequeIP13_XTransactionSaIS1_EE17_M_reallocate_mapEmb
.section __TEXT,__textcoal_nt,coalesced,no_toc
	.align 2
__ZNSt5dequeIP13_XTransactionSaIS1_EE17_M_reallocate_mapEmb:
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/deque.tcc",132,0,0,0
	.stabd	68,0,749
LFB1732:
	mflr r0
	stmw r25,-28(r1)
LCFI49:
	stw r0,8(r1)
LCFI50:
	mr r28,r3
	stwu r1,-96(r1)
LCFI51:
	mr r25,r5
	.stabd	68,0,750
LBB451:
	lwz r10,36(r3)
	lwz r11,20(r3)
	subf r2,r11,r10
	srawi r2,r2,2
	addi r27,r2,1
	.stabd	68,0,754
LBB452:
	lwz r2,4(r3)
	.stabd	68,0,751
LBE452:
	add r30,r27,r4
	.stabd	68,0,749
	stw r4,124(r1)
	.stabd	68,0,754
LBB453:
	slwi r0,r30,1
	cmplw cr7,r2,r0
	ble- cr7,L682
	.stabd	68,0,756
	cmpwi cr7,r5,0
	subf r0,r30,r2
	lwz r2,0(r3)
	rlwinm r0,r0,1,0,29
	add r3,r2,r0
	beq- cr7,L683
	slwi r0,r4,2
	add r3,r3,r0
L683:
	.stabd	68,0,758
	cmplw cr7,r3,r11
	.stabd	68,0,756
	mr r30,r3
	.stabd	68,0,758
	bge- cr7,L684
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_algobase.h",132,0,0,0
	.stabd	68,0,345
LBB454:
	addi r5,r10,4
	.stabd	68,0,255
LBB455:
LBB456:
LBB457:
LBB458:
LBB459:
LBB460:
LBB461:
LBB462:
LBB463:
	mr r4,r11
	subf r5,r11,r5
	bl L_memmove$stub
	.stabd	68,0,345
	b L721
L684:
	.stabd	68,0,486
LBE463:
LBE462:
LBE461:
LBE460:
LBE459:
LBE458:
LBE457:
LBE456:
LBE455:
LBE454:
LBB464:
	addi r5,r10,4
	slwi r29,r27,2
	.stabd	68,0,410
LBB465:
LBB466:
LBB467:
LBB468:
LBB469:
LBB470:
LBB471:
LBB472:
LBB473:
	subf r5,r11,r5
	.stabd	68,0,486
LBE473:
LBE472:
LBE471:
LBE470:
LBE469:
LBE468:
LBE467:
LBE466:
LBE465:
	add r3,r3,r29
	.stabd	68,0,411
LBB474:
LBB475:
LBB476:
LBB477:
LBB478:
LBB479:
LBB480:
LBB481:
LBB482:
	rlwinm r5,r5,0,0,29
	mr r4,r11
	subf r3,r5,r3
	bl L_memmove$stub
	.stabd	68,0,486
LBE482:
LBE481:
LBE480:
LBE479:
LBE478:
LBE477:
LBE476:
LBE475:
LBE474:
LBE464:
	b L696
L682:
	.stabd	68,0,176
LBB483:
LBB484:
LBB485:
	cmplw cr7,r2,r4
	addi r9,r1,124
	.stabd	68,0,172
LBE485:
	addi r0,r3,4
	.stabd	68,0,176
LBB486:
	blt- cr7,L698
	mr r9,r0
L698:
	.stabd	68,0,172
LBE486:
LBE484:
	lwz r0,0(r9)
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_alloc.h",132,0,0,0
	.stabd	68,0,230
LBB487:
LBB488:
LBB489:
LBB490:
	li r26,0
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_algobase.h",132,0,0,0
	.stabd	68,0,172
LBE490:
LBE489:
LBE488:
LBE487:
	add r0,r2,r0
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_alloc.h",132,0,0,0
	.stabd	68,0,231
LBB491:
LBB492:
LBB493:
LBB494:
	addic. r29,r0,2
	beq+ cr0,L699
	.stabd	68,0,232
	slwi r3,r29,2
	bl L__ZNSt24__default_alloc_templateILb1ELi0EE8allocateEm$stub
	lwz r10,36(r28)
	lwz r11,20(r28)
	mr r26,r3
L699:
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/deque.tcc",132,0,0,0
	.stabd	68,0,770
LBE494:
LBE493:
LBE492:
LBE491:
	cmpwi cr7,r25,0
	subf r0,r30,r29
	rlwinm r0,r0,1,0,29
	add r3,r26,r0
	beq- cr7,L702
	lwz r0,124(r1)
	slwi r0,r0,2
	add r3,r3,r0
L702:
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_algobase.h",132,0,0,0
	.stabd	68,0,345
LBB495:
	addi r5,r10,4
	.stabd	68,0,255
LBB496:
LBB497:
LBB498:
LBB499:
LBB500:
LBB501:
LBB502:
LBB503:
LBB504:
	mr r4,r11
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/deque.tcc",132,0,0,0
	.stabd	68,0,770
LBE504:
LBE503:
LBE502:
LBE501:
LBE500:
LBE499:
LBE498:
LBE497:
LBE496:
LBE495:
	mr r30,r3
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_algobase.h",132,0,0,0
	.stabd	68,0,255
LBB505:
LBB506:
LBB507:
LBB508:
LBB509:
LBB510:
LBB511:
LBB512:
LBB513:
LBB514:
	subf r5,r11,r5
	bl L_memmove$stub
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_deque.h",132,0,0,0
	.stabd	68,0,432
LBE514:
LBE513:
LBE512:
LBE511:
LBE510:
LBE509:
LBE508:
LBE507:
LBE506:
LBE505:
LBB515:
	lwz r4,4(r28)
	lwz r3,0(r28)
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_alloc.h",132,0,0,0
	.stabd	68,0,242
LBB516:
LBB517:
LBB518:
	cmpwi cr7,r4,0
	beq+ cr7,L710
	slwi r4,r4,2
	bl L__ZNSt24__default_alloc_templateILb1ELi0EE10deallocateEPvm$stub
L710:
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/deque.tcc",132,0,0,0
	.stabd	68,0,775
LBE518:
LBE517:
LBE516:
LBE515:
	stw r26,0(r28)
	.stabd	68,0,776
	stw r29,4(r28)
L721:
	slwi r29,r27,2
L696:
	.stabs	"/usr/include/gcc/darwin/3.3/c++/bits/stl_deque.h",132,0,0,0
	.stabd	68,0,206
LBE483:
LBE453:
LBB519:
	addi r9,r28,8
LBE519:
LBB520:
	add r8,r30,r29
	.stabd	68,0,207
LBE520:
LBB521:
LBB522:
	stw r30,12(r9)
	.stabd	68,0,206
LBE522:
LBE521:
LBB523:
	addi r11,r28,24
	.stabd	68,0,208
LBE523:
LBB524:
LBB525:
	lwz r10,0(r30)
	.stabd	68,0,206
LBE525:
LBE524:
LBB526:
	addi r2,r8,-4
	.stabd	68,0,207
LBB527:
	stw r2,12(r11)
	.stabd	68,0,112
LBE527:
LBE526:
LBB528:
LBB529:
	addi r0,r10,512
	.stabd	68,0,208
	stw r10,4(r9)
	.stabd	68,0,112
	stw r0,8(r9)
	.stabd	68,0,208
LBE529:
LBE528:
LBB530:
LBB531:
	lwz r2,-4(r8)
	.stabd	68,0,112
	addi r0,r2,512
	.stabd	68,0,208
	stw r2,4(r11)
	.stabd	68,0,112
	stw r0,8(r11)
	.stabd	68,0,208
	lwz r0,104(r1)
	addi r1,r1,96
	lmw r25,-28(r1)
	mtlr r0
	blr
LBE531:
LBE530:
LBE451:
LFE1732:
	.stabs	"_ZNSt5dequeIP13_XTransactionSaIS1_EE17_M_reallocate_mapEmb:F(0,2)",36,0,749,__ZNSt5dequeIP13_XTransactionSaIS1_EE17_M_reallocate_mapEmb
	.stabs	"this:P(0,820)",64,0,749,28
	.stabs	"__nodes_to_add:p(0,67)",160,0,749,124
	.stabs	"__add_at_front:P(0,1)",64,0,749,25
	.stabs	"__old_num_nodes:r(0,67)",64,0,750,27
	.stabs	"__new_num_nodes:r(0,67)",64,0,751,30
	.stabs	"__new_nstart:r(0,173)",64,0,753,30
	.stabn	192,0,0,LBB451
	.stabs	"__last:r(0,173)",64,0,345,5
	.stabn	192,0,0,LBB454
	.stabn	224,0,0,LBE454
	.stabs	"__last:r(0,173)",64,0,486,5
	.stabs	"__result:r(0,173)",64,0,486,3
	.stabn	192,0,0,LBB464
	.stabs	"ptrdiff_t:t(0,821)=(0,822)=(0,10)",128,0,74,0
	.stabs	"__darwin_ptrdiff_t:t(0,822)",128,0,78,0
	.stabn	224,0,0,LBE464
	.stabs	"__new_map_size:r(0,67)",64,0,766,29
	.stabn	192,0,0,LBB483
	.stabs	"__a:r(0,823)=&(0,522)",64,0,172,0
	.stabn	192,0,0,LBB484
	.stabn	224,0,0,LBE484
	.stabs	"__ret:r(0,173)",64,0,230,26
	.stabn	192,0,0,LBB490
	.stabn	224,0,0,LBE490
	.stabs	"__ret:r(0,173)",64,0,230,26
	.stabn	192,0,0,LBB494
	.stabn	224,0,0,LBE494
	.stabs	"__last:r(0,173)",64,0,345,5
	.stabn	192,0,0,LBB495
	.stabn	224,0,0,LBE495
	.stabs	"__last:r(0,173)",64,0,345,5
	.stabn	192,0,0,LBB505
	.stabn	224,0,0,LBE505
	.stabs	"__p:r(0,173)",64,0,432,3
	.stabs	"__n:r(0,67)",64,0,432,4
	.stabn	192,0,0,LBB515
	.stabn	224,0,0,LBE515
	.stabn	224,0,0,LBE483
	.stabs	"this:r(0,231)",64,0,206,9
	.stabn	192,0,0,LBB519
	.stabn	224,0,0,LBE519
	.stabs	"this:r(0,231)",64,0,206,11
	.stabs	"__new_node:r(0,173)",64,0,206,2
	.stabn	192,0,0,LBB520
	.stabn	224,0,0,LBE520
	.stabs	"this:r(0,231)",64,0,206,9
	.stabn	192,0,0,LBB521
	.stabn	224,0,0,LBE521
	.stabs	"this:r(0,231)",64,0,206,11
	.stabs	"__new_node:r(0,173)",64,0,206,2
	.stabn	192,0,0,LBB523
	.stabn	224,0,0,LBE523
	.stabs	"this:r(0,231)",64,0,206,9
	.stabn	192,0,0,LBB524
	.stabn	224,0,0,LBE524
	.stabs	"this:r(0,231)",64,0,206,11
	.stabs	"__new_node:r(0,173)",64,0,206,2
	.stabn	192,0,0,LBB526
	.stabn	224,0,0,LBE526
	.stabs	"this:r(0,231)",64,0,206,9
	.stabn	192,0,0,LBB528
	.stabn	224,0,0,LBE528
	.stabs	"this:r(0,231)",64,0,206,11
	.stabs	"__new_node:r(0,173)",64,0,206,2
	.stabn	192,0,0,LBB530
	.stabn	224,0,0,LBE530
	.stabn	224,0,0,LBE451
.section __TEXT,__textcoal_nt,coalesced,no_toc
	.align 2
Lscope34:
	.stabs	"",36,0,0,Lscope34-__ZNSt5dequeIP13_XTransactionSaIS1_EE17_M_reallocate_mapEmb
	.globl __ZTV12_XTalkerBase
.data
.const_data
	.align 3
__ZTV12_XTalkerBase:
	.long	0
	.long	__ZTI12_XTalkerBase
	.long	__ZN12_XTalkerBaseD1Ev
	.long	__ZN12_XTalkerBaseD0Ev
	.globl __ZTV9XListener
	.align 3
__ZTV9XListener:
	.long	0
	.long	__ZTI9XListener
	.long	__ZN9XListenerD1Ev
	.long	__ZN9XListenerD0Ev
.section __TEXT,__StaticInit,regular,pure_instructions
	.align 2
.section __TEXT,__StaticInit,regular,pure_instructions
	.align 2
__Z41__static_initialization_and_destruction_0ii:
	.stabs	"../../kame/xsignal.cpp",132,0,0,0
	.stabd	68,0,135
LFB1740:
	mflr r0
	bcl 20,31,"L00000000014$pb"
"L00000000014$pb":
	stw r0,8(r1)
LCFI52:
	.stabd	68,0,5
	li r0,0
	ori r0,r0,65535
	.stabd	68,0,135
	stmw r29,-12(r1)
LCFI53:
	.stabd	68,0,5
	cmpw cr7,r4,r0
	.stabd	68,0,135
	stwu r1,-176(r1)
LCFI54:
	mflr r31
	mr r29,r4
	mr r30,r3
	.stabd	68,0,5
	bne+ cr7,L723
	cmpwi cr7,r3,1
	bne+ cr7,L723
	.stabs	"/sw/include/boost/shared_ptr.hpp",132,0,0,0
	.stabd	68,0,120
LBB532:
LBB533:
LBB534:
	addis r2,r31,ha16(_g_signalBuffer-"L00000000014$pb")
	li r0,0
	la r2,lo16(_g_signalBuffer-"L00000000014$pb")(r2)
	.stabs	"/sw/include/boost/detail/shared_count.hpp",132,0,0,0
	.stabd	68,0,328
LBB535:
LBB536:
LBB537:
	stw r0,4(r2)
	.stabs	"/sw/include/boost/shared_ptr.hpp",132,0,0,0
	.stabd	68,0,120
LBE537:
LBE536:
LBE535:
	stw r0,0(r2)
	.stabs	"/sw/include/boost/detail/shared_count.hpp",132,0,0,0
	.stabd	68,0,324
L726:
	.stabs	"/sw/include/boost/detail/lwm_pthreads.hpp",132,0,0,0
	.stabd	68,0,77
L734:
L740:
L752:
	.stabd	68,0,77
L758:
L769:
L723:
	.stabs	"../../kame/xsignal.cpp",132,0,0,0
	.stabd	68,0,7
LBE534:
LBE533:
LBE532:
	li r0,0
	ori r0,r0,65535
	cmpw cr7,r29,r0
	bne+ cr7,L773
	cmpwi cr7,r30,1
	bne+ cr7,L773
	bl L_pthread_self$stub
	addis r2,r31,ha16(_g_main_thread_id-"L00000000014$pb")
	stw r3,lo16(_g_main_thread_id-"L00000000014$pb")(r2)
L773:
	.stabd	68,0,5
	li r0,0
	ori r0,r0,65535
	cmpw cr7,r29,r0
	bne+ cr7,L722
	cmpwi cr7,r30,0
	bne- cr7,L722
L776:
	.stabs	"/sw/include/boost/detail/lwm_pthreads.hpp",132,0,0,0
	.stabd	68,0,77
L784:
L790:
L802:
	.stabd	68,0,77
L808:
	.stabs	"/sw/include/boost/detail/shared_count.hpp",132,0,0,0
	.stabd	68,0,172
L819:
	.stabd	68,0,377
LBB538:
LBB539:
LBB540:
	addis r2,r31,ha16(_g_signalBuffer-"L00000000014$pb")
	la r2,lo16(_g_signalBuffer-"L00000000014$pb")(r2)
	.stabd	68,0,378
	lwz r30,4(r2)
	cmpwi cr7,r30,0
	beq+ cr7,L722
	.stabs	"/sw/include/boost/detail/lwm_pthreads.hpp",132,0,0,0
	.stabd	68,0,70
	addi r29,r30,12
	.stabd	68,0,72
	mr r3,r29
	.stabd	68,0,71
	stw r29,128(r1)
	.stabd	68,0,72
	bl L_pthread_mutex_lock$stub
	.stabs	"/sw/include/boost/detail/shared_count.hpp",132,0,0,0
	.stabd	68,0,142
	lwz r2,4(r30)
	addi r2,r2,-1
	.stabd	68,0,144
	cmpwi cr7,r2,0
	.stabd	68,0,142
	stw r2,4(r30)
	.stabd	68,0,144
	beq- cr7,L828
	.stabs	"/sw/include/boost/detail/lwm_pthreads.hpp",132,0,0,0
	.stabd	68,0,77
LBB541:
	lwz r3,128(r1)
	bl L_pthread_mutex_unlock$stub
LBE541:
	b L722
L827:
L833:
L828:
L845:
L851:
	lwz r3,128(r1)
	bl L_pthread_mutex_unlock$stub
	.stabs	"/sw/include/boost/detail/shared_count.hpp",132,0,0,0
	.stabd	68,0,147
	lwz r2,0(r30)
	mr r3,r30
	lwz r12,8(r2)
	mtctr r12
	bctrl
	.stabs	"/sw/include/boost/detail/lwm_pthreads.hpp",132,0,0,0
	.stabd	68,0,72
	mr r3,r29
	.stabd	68,0,71
	stw r29,144(r1)
	.stabd	68,0,72
	bl L_pthread_mutex_lock$stub
	.stabs	"/sw/include/boost/detail/shared_count.hpp",132,0,0,0
	.stabd	68,0,167
	lwz r29,8(r30)
	addi r29,r29,-1
	stw r29,8(r30)
	.stabs	"/sw/include/boost/detail/lwm_pthreads.hpp",132,0,0,0
	.stabd	68,0,77
	lwz r3,144(r1)
	bl L_pthread_mutex_unlock$stub
	.stabs	"/sw/include/boost/detail/shared_count.hpp",132,0,0,0
	.stabd	68,0,170
	cmpwi cr7,r29,0
	bne+ cr7,L722
	.stabd	68,0,172
	lwz r2,0(r30)
	mr r3,r30
	lwz r12,12(r2)
	mtctr r12
	bctrl
	.stabs	"../../kame/xsignal.cpp",132,0,0,0
	.stabd	68,0,5
L722:
	lwz r0,184(r1)
	addi r1,r1,176
	lmw r29,-12(r1)
	mtlr r0
	blr
LBE540:
LBE539:
LBE538:
LFE1740:
	.stabs	"_Z41__static_initialization_and_destruction_0ii:f(0,2)",36,0,135,__Z41__static_initialization_and_destruction_0ii
	.stabs	"__initialize_p:P(0,10)",64,0,135,30
	.stabs	"__priority:P(0,10)",64,0,135,29
	.stabs	"shared_count:Tt(0,824)=s4pi_:/0(0,825)=*(0,826)=xssp_counted_base:,0,32;shared_count::(0,827)=#(0,824),(0,2),(0,828)=*(0,824),(0,2);:_ZN5boost6detail12shared_countC4Ev;2A.;__base_ctor::(0,827):_ZN5boost6detail12shared_countC2Ev;2A.;__comp_ctor::(0,827):_ZN5boost6detail12shared_countC1Ev;2A.;shared_count::(0,829)=#(0,824),(0,2),(0,828),(0,10),(0,2);:_ZN5boost6detail12shared_countD4Ev;2A.;__base_dtor::(0,827):_ZN5boost6detail12shared_countD2Ev;2A.;__comp_dtor::(0,827):_ZN5boost6detail12shared_countD1Ev;2A.;shared_count::(0,830)=#(0,824),(0,2),(0,828),(0,831)=&(0,832)=k(0,824),(0,2);:_ZN5boost6detail12shared_countC4ERKS1_;2A.;__base_ctor::(0,830):_ZN5boost6detail12shared_countC2ERKS1_;2A.;__comp_ctor::(0,830):_ZN5boost6detail12shared_countC1ERKS1_;2A.;shared_count::(0,833)=#(0,824),(0,2),(0,828),(0,834)=&(0,835)=k(0,836)=xsweak_count:,(0,2);:_ZN5boost6detail12shared_countC4ERKNS0_10weak_countE;2A.;__base_ctor::(0,833):_ZN5boost6detail12shared_countC2ERKNS0_10weak_countE;2A.;__comp_ctor::(0,833):_ZN5boost6detail12shared_countC1ERKNS0_10weak_countE;2A.;operator=::(0,837)=#(0,824),(0,838)=&(0,824),(0,828),(0,831),(0,2);:_ZN5boost6detail12shared_countaSERKS1_;2A.;swap::(0,839)=#(0,824),(0,2),(0,828),(0,838),(0,2);:_ZN5boost6detail12shared_count4swapERS1_;2A.;use_count::(0,840)=#(0,824),(0,208),(0,841)=*(0,832),(0,2);:_ZNK5boost6detail12shared_count9use_countEv;2B.;unique::(0,842)=#(0,824),(0,1),(0,841),(0,2);:_ZNK5boost6detail12shared_count6uniqueEv;2B.;get_deleter::(0,843)=#(0,824),(0,818),(0,841),(0,844)=&(0,845)=k(0,846)=xstype_info:,(0,2);:_ZNK5boost6detail12shared_count11get_deleterERKSt9type_info;2B.;;",128,0,311,0
	.stabs	"sp_counted_base:Tt(0,826)=s56_vptr$sp_counted_base:(0,7),0,32;use_count_:/0(0,208),32,32;weak_count_:/0(0,208),64,32;mtx_:/0(0,847)=xslightweight_mutex:,96,352;sp_counted_base::(0,848)=#(0,826),(0,2),(0,825),(0,2);:_ZN5boost6detail15sp_counted_baseC4Ev;2A.;__base_ctor::(0,848):_ZN5boost6detail15sp_counted_baseC2Ev;2A.;__comp_ctor::(0,848):_ZN5boost6detail15sp_counted_baseC1Ev;2A.;__base_dtor::(0,848):_ZN5boost6detail15sp_counted_baseD2Ev;2A.;__comp_dtor::(0,848):_ZN5boost6detail15sp_counted_baseD1Ev;2A*0;(0,826);;__deleting_dtor::(0,848):_ZN5boost6detail15sp_counted_baseD0Ev;2A*1;(0,826);;dispose::(0,848):_ZN5boost6detail15sp_counted_base7disposeEv;2A*2;(0,826);;destruct::(0,848):_ZN5boost6detail15sp_counted_base8destructEv;2A*3;(0,826);;get_deleter::(0,849)=#(0,826),(0,818),(0,825),(0,844),(0,2);:_ZN5boost6detail15sp_counted_base11get_deleterERKSt9type_info;2A*4;(0,826);;add_ref_copy::(0,848):_ZN5boost6detail15sp_counted_base12add_ref_copyEv;2A.;add_ref_lock::(0,848):_ZN5boost6detail15sp_counted_base12add_ref_lockEv;2A.;release::(0,848):_ZN5boost6detail15sp_counted_base7releaseEv;2A.;weak_add_ref::(0,848):_ZN5boost6detail15sp_counted_base12weak_add_refEv;2A.;weak_release::(0,848):_ZN5boost6detail15sp_counted_base12weak_releaseEv;2A.;use_count::(0,850)=#(0,826),(0,208),(0,851)=*(0,852)=k(0,826),(0,2);:_ZNK5boost6detail15sp_counted_base9use_countEv;2B.;__base_ctor::(0,853)=#(0,826),(0,2),(0,825),(0,854)=&(0,852),(0,2);:_ZN5boost6detail15sp_counted_baseC2ERKS1_;0A.;__comp_ctor::(0,853):_ZN5boost6detail15sp_counted_baseC1ERKS1_;0A.;operator=::(0,855)=#(0,826),(0,856)=&(0,826),(0,825),(0,854),(0,2);:_ZN5boost6detail15sp_counted_baseaSERKS1_;0A.;;~%(0,826);",128,0,90,0
	.stabs	"weak_count:Tt(0,836)=s4pi_:/0(0,825),0,32;weak_count::(0,857)=#(0,836),(0,2),(0,858)=*(0,836),(0,2);:_ZN5boost6detail10weak_countC4Ev;2A.;__base_ctor::(0,857):_ZN5boost6detail10weak_countC2Ev;2A.;__comp_ctor::(0,857):_ZN5boost6detail10weak_countC1Ev;2A.;weak_count::(0,859)=#(0,836),(0,2),(0,858),(0,831),(0,2);:_ZN5boost6detail10weak_countC4ERKNS0_12shared_countE;2A.;__base_ctor::(0,859):_ZN5boost6detail10weak_countC2ERKNS0_12shared_countE;2A.;__comp_ctor::(0,859):_ZN5boost6detail10weak_countC1ERKNS0_12shared_countE;2A.;weak_count::(0,860)=#(0,836),(0,2),(0,858),(0,834),(0,2);:_ZN5boost6detail10weak_countC4ERKS1_;2A.;__base_ctor::(0,860):_ZN5boost6detail10weak_countC2ERKS1_;2A.;__comp_ctor::(0,860):_ZN5boost6detail10weak_countC1ERKS1_;2A.;weak_count::(0,861)=#(0,836),(0,2),(0,858),(0,10),(0,2);:_ZN5boost6detail10weak_countD4Ev;2A.;__base_dtor::(0,857):_ZN5boost6detail10weak_countD2Ev;2A.;__comp_dtor::(0,857):_ZN5boost6detail10weak_countD1Ev;2A.;operator=::(0,862)=#(0,836),(0,863)=&(0,836),(0,858),(0,831),(0,2);:_ZN5boost6detail10weak_countaSERKNS0_12shared_countE;2A.(0,864)=#(0,836),(0,863),(0,858),(0,834),(0,2);:_ZN5boost6detail10weak_countaSERKS1_;2A.;swap::(0,865)=#(0,836),(0,2),(0,858),(0,863),(0,2);:_ZN5boost6detail10weak_count4swapERS1_;2A.;use_count::(0,866)=#(0,836),(0,208),(0,867)=*(0,835),(0,2);:_ZNK5boost6detail10weak_count9use_countEv;2B.;;",128,0,447,0
	.stabs	"type_info:Tt(0,846)=s8_vptr$type_info:(0,7),0,32;__name:/1(0,36),32,32;__base_dtor::(0,868)=#(0,846),(0,2),(0,869)=*(0,846),(0,2);:_ZNSt9type_infoD2Ev;2A.;__comp_dtor::(0,868):_ZNSt9type_infoD1Ev;2A*0;(0,846);;__deleting_dtor::(0,868):_ZNSt9type_infoD0Ev;2A*1;(0,846);;operator=::(0,870)=#(0,846),(0,871)=&(0,846),(0,869),(0,844),(0,2);:_ZNSt9type_infoaSERKS_;0A.;__base_ctor::(0,872)=#(0,846),(0,2),(0,869),(0,844),(0,2);:_ZNSt9type_infoC2ERKS_;0A.;__comp_ctor::(0,872):_ZNSt9type_infoC1ERKS_;0A.;type_info::(0,873)=#(0,846),(0,2),(0,869),(0,36),(0,2);:_ZNSt9type_infoC4EPKc;1A.;__base_ctor::(0,873):_ZNSt9type_infoC2EPKc;1A.;__comp_ctor::(0,873):_ZNSt9type_infoC1EPKc;1A.;name::(0,874)=#(0,846),(0,36),(0,875)=*(0,845),(0,2);:_ZNKSt9type_info4nameEv;2B.;before::(0,876)=#(0,846),(0,1),(0,875),(0,844),(0,2);:_ZNKSt9type_info6beforeERKS_;2B.;operator==::(0,876):_ZNKSt9type_infoeqERKS_;2B.;operator!=::(0,876):_ZNKSt9type_infoneERKS_;2B.;__is_pointer_p::(0,877)=#(0,846),(0,1),(0,875),(0,2);:_ZNKSt9type_info14__is_pointer_pEv;2B*2;(0,846);;__is_function_p::(0,877):_ZNKSt9type_info15__is_function_pEv;2B*3;(0,846);;__do_catch::(0,878)=#(0,846),(0,1),(0,875),(0,875),(0,879)=*(0,818),(0,6),(0,2);:_ZNKSt9type_info10__do_catchEPKS_PPvj;2B*4;(0,846);;__do_upcast::(0,880)=#(0,846),(0,1),(0,875),(0,881)=*(0,882)=k(0,883)=xs__class_type_info:,(0,879),(0,2);:_ZNKSt9type_info11__do_upcastEPKN10__cxxabiv117__class_type_infoEPPv;2B*5;(0,846);;;~%(0,846);",128,0,64,0
	.stabs	"lightweight_mutex:Tt(0,847)=s44m_:/0(0,113),0,352;__base_ctor::(0,884)=#(0,847),(0,2),(0,885)=*(0,847),(0,886)=&(0,887)=k(0,847),(0,2);:_ZN5boost6detail17lightweight_mutexC2ERKS1_;0A.;__comp_ctor::(0,884):_ZN5boost6detail17lightweight_mutexC1ERKS1_;0A.;operator=::(0,888)=#(0,847),(0,889)=&(0,847),(0,885),(0,886),(0,2);:_ZN5boost6detail17lightweight_mutexaSERKS1_;0A.;lightweight_mutex::(0,890)=#(0,847),(0,2),(0,885),(0,2);:_ZN5boost6detail17lightweight_mutexC4Ev;2A.;__base_ctor::(0,890):_ZN5boost6detail17lightweight_mutexC2Ev;2A.;__comp_ctor::(0,890):_ZN5boost6detail17lightweight_mutexC1Ev;2A.;lightweight_mutex::(0,891)=#(0,847),(0,2),(0,885),(0,10),(0,2);:_ZN5boost6detail17lightweight_mutexD4Ev;2A.;__base_dtor::(0,890):_ZN5boost6detail17lightweight_mutexD2Ev;2A.;__comp_dtor::(0,890):_ZN5boost6detail17lightweight_mutexD1Ev;2A.;;",128,0,29,0
.section __TEXT,__StaticInit,regular,pure_instructions
	.align 2
Lscope35:
	.stabs	"",36,0,0,Lscope35-__Z41__static_initialization_and_destruction_0ii
.weak_definition __ZTI9XListener
.private_extern __ZTI9XListener
.section __DATA,__datacoal_nt,coalesced,no_toc
	.align 2
__ZTI9XListener:
	.long	__ZTVN10__cxxabiv117__class_type_infoE+8
	.long	__ZTS9XListener
.weak_definition __ZTI12_XTalkerBase
.private_extern __ZTI12_XTalkerBase
	.align 2
__ZTI12_XTalkerBase:
	.long	__ZTVN10__cxxabiv117__class_type_infoE+8
	.long	__ZTS12_XTalkerBase
.weak_definition __ZTI10XKameError
.private_extern __ZTI10XKameError
	.align 2
__ZTI10XKameError:
	.long	__ZTVN10__cxxabiv117__class_type_infoE+8
	.long	__ZTS10XKameError
.weak_definition __ZTS9XListener
.private_extern __ZTS9XListener
	.align 2
__ZTS9XListener:
	.ascii "9XListener\0"
.weak_definition __ZTS12_XTalkerBase
.private_extern __ZTS12_XTalkerBase
	.align 2
__ZTS12_XTalkerBase:
	.ascii "12_XTalkerBase\0"
.weak_definition __ZTS10XKameError
.private_extern __ZTS10XKameError
	.align 2
__ZTS10XKameError:
	.ascii "10XKameError\0"
.section __TEXT,__StaticInit,regular,pure_instructions
	.align 2
.section __TEXT,__StaticInit,regular,pure_instructions
	.align 2
__GLOBAL__I_g_signalBuffer:
	.stabd	68,0,5
LFB1742:
	li r4,0
	li r3,1
	ori r4,r4,65535
	b __Z41__static_initialization_and_destruction_0ii
LFE1742:
	.stabs	"_GLOBAL__I_g_signalBuffer:f(0,2)",36,0,5,__GLOBAL__I_g_signalBuffer
.section __TEXT,__StaticInit,regular,pure_instructions
	.align 2
Lscope36:
	.stabs	"",36,0,0,Lscope36-__GLOBAL__I_g_signalBuffer
.data
.mod_init_func
	.align 2
	.long	__GLOBAL__I_g_signalBuffer
.section __TEXT,__StaticInit,regular,pure_instructions
	.align 2
.section __TEXT,__StaticInit,regular,pure_instructions
	.align 2
__GLOBAL__D_g_signalBuffer:
	.stabd	68,0,5
LFB1744:
	li r4,0
	li r3,0
	ori r4,r4,65535
	b __Z41__static_initialization_and_destruction_0ii
LFE1744:
	.stabs	"_GLOBAL__D_g_signalBuffer:f(0,2)",36,0,5,__GLOBAL__D_g_signalBuffer
.section __TEXT,__StaticInit,regular,pure_instructions
	.align 2
Lscope37:
	.stabs	"",36,0,0,Lscope37-__GLOBAL__D_g_signalBuffer
.data
.mod_term_func
	.align 2
	.long	__GLOBAL__D_g_signalBuffer
	.stabs	"_ZTI9XListener:S(0,892)=k(0,893)=xs__class_type_info_pseudo:",38,0,24,__ZTI9XListener
	.stabs	"__class_type_info_pseudo:t(0,893)=s8:(0,894)=xs__type_info_pseudo:,0,64;;",128,0,0,0
	.stabs	"__type_info_pseudo:t(0,894)=s8:(0,895)=*(0,896)=k(0,897)=(0,897),0,32;:(0,36),32,32;;",128,0,0,0
	.stabs	"_ZTI12_XTalkerBase:S(0,892)",38,0,50,__ZTI12_XTalkerBase
	.stabs	"g_main_thread_id:G(0,898)=(0,899)=(0,900)=*(0,901)=xs_opaque_pthread_t:",32,0,7,0
	.stabs	"_opaque_pthread_t:Tt(0,901)=s604__sig:(0,208),0,32;__cleanup_stack:(0,902)=*(0,903)=xs__darwin_pthread_handler_rec:,32,32;__opaque:(0,904)=ar(0,210);0;595;(0,38),64,4768;operator=::(0,905)=#(0,901),(0,906)=&(0,901),(0,900),(0,907)=&(0,908)=k(0,901),(0,2);:_ZN17_opaque_pthread_taSERKS_;2A.;__base_ctor::(0,909)=#(0,901),(0,2),(0,900),(0,907),(0,2);:_ZN17_opaque_pthread_tC2ERKS_;2A.;__comp_ctor::(0,909):_ZN17_opaque_pthread_tC1ERKS_;2A.;__base_ctor::(0,910)=#(0,901),(0,2),(0,900),(0,2);:_ZN17_opaque_pthread_tC2Ev;2A.;__comp_ctor::(0,910):_ZN17_opaque_pthread_tC1Ev;2A.;;",128,0,74,0
	.stabs	"pthread_t:t(0,898)",128,0,109,0
	.stabs	"__darwin_pthread_t:t(0,899)",128,0,133,0
	.stabs	"__darwin_pthread_handler_rec:Tt(0,903)=s12__routine:(0,911)=*(0,912)=f(0,2),0,32;__arg:(0,818),32,32;__next:(0,902),64,32;operator=::(0,913)=#(0,903),(0,914)=&(0,903),(0,902),(0,915)=&(0,916)=k(0,903),(0,2);:_ZN28__darwin_pthread_handler_recaSERKS_;2A.;__base_ctor::(0,917)=#(0,903),(0,2),(0,902),(0,915),(0,2);:_ZN28__darwin_pthread_handler_recC2ERKS_;2A.;__comp_ctor::(0,917):_ZN28__darwin_pthread_handler_recC1ERKS_;2A.;__base_ctor::(0,918)=#(0,903),(0,2),(0,902),(0,2);:_ZN28__darwin_pthread_handler_recC2Ev;2A.;__comp_ctor::(0,918):_ZN28__darwin_pthread_handler_recC1Ev;2A.;;",128,0,61,0
	.stabs	"_ZTI10XKameError:S(0,892)",38,0,110,__ZTI10XKameError
	.stabs	"_ZTS9XListener:S(0,919)=ar(0,210);0;10;(0,37)",38,0,135,__ZTS9XListener
	.stabs	"_ZTS12_XTalkerBase:S(0,920)=ar(0,210);0;14;(0,37)",38,0,135,__ZTS12_XTalkerBase
	.stabs	"_ZTS10XKameError:S(0,921)=ar(0,210);0;12;(0,37)",38,0,135,__ZTS10XKameError
.globl _ZN9XListenerC2Ebbj.eh
	_ZN9XListenerC2Ebbj.eh = 0
.no_dead_strip _ZN9XListenerC2Ebbj.eh
.globl _ZN9XListenerC1Ebbj.eh
	_ZN9XListenerC1Ebbj.eh = 0
.no_dead_strip _ZN9XListenerC1Ebbj.eh
.globl _ZN9XListenerD2Ev.eh
	_ZN9XListenerD2Ev.eh = 0
.no_dead_strip _ZN9XListenerD2Ev.eh
.globl _ZN9XListenerD1Ev.eh
	_ZN9XListenerD1Ev.eh = 0
.no_dead_strip _ZN9XListenerD1Ev.eh
.globl _ZN9XListenerD0Ev.eh
	_ZN9XListenerD0Ev.eh = 0
.no_dead_strip _ZN9XListenerD0Ev.eh
.globl _ZN9XListenerD4Ev.eh
	_ZN9XListenerD4Ev.eh = 0
.no_dead_strip _ZN9XListenerD4Ev.eh
.globl _ZN12_XTalkerBaseC2Ev.eh
	_ZN12_XTalkerBaseC2Ev.eh = 0
.no_dead_strip _ZN12_XTalkerBaseC2Ev.eh
.globl _ZN12_XTalkerBaseC1Ev.eh
	_ZN12_XTalkerBaseC1Ev.eh = 0
.no_dead_strip _ZN12_XTalkerBaseC1Ev.eh
.globl _ZN12_XTalkerBaseC4Ev.eh
	_ZN12_XTalkerBaseC4Ev.eh = 0
.no_dead_strip _ZN12_XTalkerBaseC4Ev.eh
.globl _ZN12_XTalkerBaseD2Ev.eh
	_ZN12_XTalkerBaseD2Ev.eh = 0
.no_dead_strip _ZN12_XTalkerBaseD2Ev.eh
.globl _ZN12_XTalkerBaseD1Ev.eh
	_ZN12_XTalkerBaseD1Ev.eh = 0
.no_dead_strip _ZN12_XTalkerBaseD1Ev.eh
.globl _ZN12_XTalkerBaseD0Ev.eh
	_ZN12_XTalkerBaseD0Ev.eh = 0
.no_dead_strip _ZN12_XTalkerBaseD0Ev.eh
.globl _ZN12_XTalkerBaseD4Ev.eh
	_ZN12_XTalkerBaseD4Ev.eh = 0
.no_dead_strip _ZN12_XTalkerBaseD4Ev.eh
.globl _ZN13XSignalBufferC2Ev.eh
	_ZN13XSignalBufferC2Ev.eh = 0
.no_dead_strip _ZN13XSignalBufferC2Ev.eh
.globl _ZN13XSignalBufferC1Ev.eh
	_ZN13XSignalBufferC1Ev.eh = 0
.no_dead_strip _ZN13XSignalBufferC1Ev.eh
.globl _ZN13XSignalBufferD2Ev.eh
	_ZN13XSignalBufferD2Ev.eh = 0
.no_dead_strip _ZN13XSignalBufferD2Ev.eh
.globl _ZN13XSignalBufferD1Ev.eh
	_ZN13XSignalBufferD1Ev.eh = 0
.no_dead_strip _ZN13XSignalBufferD1Ev.eh
	_GLOBAL__I_g_signalBuffer.eh = 0
.no_dead_strip _GLOBAL__I_g_signalBuffer.eh
	_GLOBAL__D_g_signalBuffer.eh = 0
.no_dead_strip _GLOBAL__D_g_signalBuffer.eh
.data
.section __TEXT,__eh_frame,coalesced,no_toc+strip_static_syms+live_support
EH_frame1:
	.set L$set$84,LECIE1-LSCIE1
	.long L$set$84
LSCIE1:
	.long	0x0
	.byte	0x1
	.ascii "zPLR\0"
	.byte	0x1
	.byte	0x7c
	.byte	0x41
	.byte	0x7
	.byte	0x90
	.long	L___gxx_personality_v0$non_lazy_ptr-.
	.byte	0x10
	.byte	0x10
	.byte	0xc
	.byte	0x1
	.byte	0x0
	.align 2
LECIE1:
.globl _Z12isMainThreadv.eh
_Z12isMainThreadv.eh:
LSFDE1:
	.set L$set$85,LEFDE1-LASFDE1
	.long L$set$85
LASFDE1:
	.long	LASFDE1-EH_frame1
	.long	LFB1518-.
	.set L$set$86,LFE1518-LFB1518
	.long L$set$86
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$87,LCFI1-LFB1518
	.long L$set$87
	.byte	0x9f
	.byte	0x1
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$88,LCFI2-LCFI1
	.long L$set$88
	.byte	0xe
	.byte	0x50
	.align 2
LEFDE1:
.globl _ZN9XListenerC4Ebbj.eh
_ZN9XListenerC4Ebbj.eh:
LSFDE7:
	.set L$set$89,LEFDE7-LASFDE7
	.long L$set$89
LASFDE7:
	.long	LASFDE7-EH_frame1
	.long	LFB1530-.
	.set L$set$90,LFE1530-LFB1530
	.long L$set$90
	.byte	0x4
	.long	LLSDA1530-.
	.byte	0x4
	.set L$set$91,LCFI3-LFB1530
	.long L$set$91
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x4
	.set L$set$92,LCFI6-LCFI3
	.long L$set$92
	.byte	0xe
	.byte	0x50
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x11
	.byte	0x46
	.byte	0x7f
	.align 2
LEFDE7:
.globl _ZN9XListener4maskEv.eh
_ZN9XListener4maskEv.eh:
LSFDE17:
	.set L$set$93,LEFDE17-LASFDE17
	.long L$set$93
LASFDE17:
	.long	LASFDE17-EH_frame1
	.long	LFB1542-.
	.set L$set$94,LFE1542-LFB1542
	.long L$set$94
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$95,LCFI8-LFB1542
	.long L$set$95
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x4
	.set L$set$96,LCFI9-LCFI8
	.long L$set$96
	.byte	0xe
	.byte	0x50
	.align 2
LEFDE17:
.globl _ZN9XListener6unmaskEv.eh
_ZN9XListener6unmaskEv.eh:
LSFDE19:
	.set L$set$97,LEFDE19-LASFDE19
	.long L$set$97
LASFDE19:
	.long	LASFDE19-EH_frame1
	.long	LFB1544-.
	.set L$set$98,LFE1544-LFB1544
	.long L$set$98
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$99,LCFI11-LFB1544
	.long L$set$99
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x4
	.set L$set$100,LCFI12-LCFI11
	.long L$set$100
	.byte	0xe
	.byte	0x50
	.align 2
LEFDE19:
.globl _ZN12_XTalkerBase4maskEv.eh
_ZN12_XTalkerBase4maskEv.eh:
LSFDE35:
	.set L$set$101,LEFDE35-LASFDE35
	.long L$set$101
LASFDE35:
	.long	LASFDE35-EH_frame1
	.long	LFB1560-.
	.set L$set$102,LFE1560-LFB1560
	.long L$set$102
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$103,LCFI14-LFB1560
	.long L$set$103
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x4
	.set L$set$104,LCFI15-LCFI14
	.long L$set$104
	.byte	0xe
	.byte	0x50
	.align 2
LEFDE35:
.globl _ZN12_XTalkerBase6unmaskEv.eh
_ZN12_XTalkerBase6unmaskEv.eh:
LSFDE37:
	.set L$set$105,LEFDE37-LASFDE37
	.long L$set$105
LASFDE37:
	.long	LASFDE37-EH_frame1
	.long	LFB1562-.
	.set L$set$106,LFE1562-LFB1562
	.long L$set$106
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$107,LCFI17-LFB1562
	.long L$set$107
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x4
	.set L$set$108,LCFI18-LCFI17
	.long L$set$108
	.byte	0xe
	.byte	0x50
	.align 2
LEFDE37:
.globl _ZN13XSignalBufferC4Ev.eh
_ZN13XSignalBufferC4Ev.eh:
LSFDE43:
	.set L$set$109,LEFDE43-LASFDE43
	.long L$set$109
LASFDE43:
	.long	LASFDE43-EH_frame1
	.long	LFB1606-.
	.set L$set$110,LFE1606-LFB1606
	.long L$set$110
	.byte	0x4
	.long	LLSDA1606-.
	.byte	0x4
	.set L$set$111,LCFI20-LFB1606
	.long L$set$111
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x9d
	.byte	0x3
	.byte	0x9c
	.byte	0x4
	.byte	0x9b
	.byte	0x5
	.byte	0x9a
	.byte	0x6
	.byte	0x4
	.set L$set$112,LCFI21-LCFI20
	.long L$set$112
	.byte	0xe
	.byte	0xb0,0x1
	.align 2
LEFDE43:
.globl _ZN13XSignalBufferD4Ev.eh
_ZN13XSignalBufferD4Ev.eh:
LSFDE49:
	.set L$set$113,LEFDE49-LASFDE49
	.long L$set$113
LASFDE49:
	.long	LASFDE49-EH_frame1
	.long	LFB1612-.
	.set L$set$114,LFE1612-LFB1612
	.long L$set$114
	.byte	0x4
	.long	LLSDA1612-.
	.byte	0x4
	.set L$set$115,LCFI23-LFB1612
	.long L$set$115
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x9d
	.byte	0x3
	.byte	0x4
	.set L$set$116,LCFI24-LCFI23
	.long L$set$116
	.byte	0xe
	.byte	0x90,0x1
	.align 2
LEFDE49:
.globl _ZN13XSignalBuffer23registerTransactionListEP13_XTransaction.eh
_ZN13XSignalBuffer23registerTransactionListEP13_XTransaction.eh:
LSFDE51:
	.set L$set$117,LEFDE51-LASFDE51
	.long L$set$117
LASFDE51:
	.long	LASFDE51-EH_frame1
	.long	LFB1628-.
	.set L$set$118,LFE1628-LFB1628
	.long L$set$118
	.byte	0x4
	.long	LLSDA1628-.
	.byte	0x4
	.set L$set$119,LCFI26-LFB1628
	.long L$set$119
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x9d
	.byte	0x3
	.byte	0x9c
	.byte	0x4
	.byte	0x9b
	.byte	0x5
	.byte	0x9a
	.byte	0x6
	.byte	0x99
	.byte	0x7
	.byte	0x98
	.byte	0x8
	.byte	0x97
	.byte	0x9
	.byte	0x96
	.byte	0xa
	.byte	0x95
	.byte	0xb
	.byte	0x4
	.set L$set$120,LCFI27-LCFI26
	.long L$set$120
	.byte	0xe
	.byte	0x90,0x1
	.align 2
LEFDE51:
.globl _ZN13XSignalBuffer11synchronizeEv.eh
_ZN13XSignalBuffer11synchronizeEv.eh:
LSFDE53:
	.set L$set$121,LEFDE53-LASFDE53
	.long L$set$121
LASFDE53:
	.long	LASFDE53-EH_frame1
	.long	LFB1637-.
	.set L$set$122,LFE1637-LFB1637
	.long L$set$122
	.byte	0x4
	.long	LLSDA1637-.
	.byte	0x4
	.set L$set$123,LCFI29-LFB1637
	.long L$set$123
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x9d
	.byte	0x3
	.byte	0x9c
	.byte	0x4
	.byte	0x9b
	.byte	0x5
	.byte	0x9a
	.byte	0x6
	.byte	0x99
	.byte	0x7
	.byte	0x98
	.byte	0x8
	.byte	0x97
	.byte	0x9
	.byte	0x4
	.set L$set$124,LCFI30-LCFI29
	.long L$set$124
	.byte	0xe
	.byte	0xd0,0x1
	.align 2
LEFDE53:
.private_extern _ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED2Ev.eh
.weak_definition _ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED2Ev.eh
_ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED2Ev.eh:
LSFDE55:
	.set L$set$125,LEFDE55-LASFDE55
	.long L$set$125
LASFDE55:
	.long	LASFDE55-EH_frame1
	.long	__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED2Ev-.
	.set L$set$126,LFE1681-LFB1681
	.long L$set$126
	.byte	0x4
	.long	0x0
	.align 2
LEFDE55:
.private_extern _ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED4Ev.eh
.weak_definition _ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED4Ev.eh
_ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED4Ev.eh:
LSFDE57:
	.set L$set$127,LEFDE57-LASFDE57
	.long L$set$127
LASFDE57:
	.long	LASFDE57-EH_frame1
	.long	__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED4Ev-.
	.set L$set$128,LFE1693-LFB1693
	.long L$set$128
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$129,LCFI32-LFB1693
	.long L$set$129
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x9d
	.byte	0x3
	.byte	0x9c
	.byte	0x4
	.byte	0x9b
	.byte	0x5
	.byte	0x4
	.set L$set$130,LCFI33-LCFI32
	.long L$set$130
	.byte	0xe
	.byte	0x70
	.align 2
LEFDE57:
.private_extern _ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE17_M_initialize_mapEm.eh
.weak_definition _ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE17_M_initialize_mapEm.eh
_ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE17_M_initialize_mapEm.eh:
LSFDE59:
	.set L$set$131,LEFDE59-LASFDE59
	.long L$set$131
LASFDE59:
	.long	LASFDE59-EH_frame1
	.long	__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE17_M_initialize_mapEm-.
	.set L$set$132,LFE1702-LFB1702
	.long L$set$132
	.byte	0x4
	.long	LLSDA1702-.
	.byte	0x4
	.set L$set$133,LCFI34-LFB1702
	.long L$set$133
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x9d
	.byte	0x3
	.byte	0x9c
	.byte	0x4
	.byte	0x9b
	.byte	0x5
	.byte	0x9a
	.byte	0x6
	.byte	0x99
	.byte	0x7
	.byte	0x98
	.byte	0x8
	.byte	0x97
	.byte	0x9
	.byte	0x4
	.set L$set$134,LCFI36-LCFI34
	.long L$set$134
	.byte	0xe
	.byte	0x90,0x1
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.align 2
LEFDE59:
.private_extern _ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_push_back_auxERKS1_.eh
.weak_definition _ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_push_back_auxERKS1_.eh
_ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_push_back_auxERKS1_.eh:
LSFDE61:
	.set L$set$135,LEFDE61-LASFDE61
	.long L$set$135
LASFDE61:
	.long	LASFDE61-EH_frame1
	.long	__ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_push_back_auxERKS1_-.
	.set L$set$136,LFE1709-LFB1709
	.long L$set$136
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$137,LCFI38-LFB1709
	.long L$set$137
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x9d
	.byte	0x3
	.byte	0x9c
	.byte	0x4
	.byte	0x4
	.set L$set$138,LCFI39-LCFI38
	.long L$set$138
	.byte	0xe
	.byte	0x50
	.align 2
LEFDE61:
.private_extern _ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_pop_front_auxEv.eh
.weak_definition _ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_pop_front_auxEv.eh
_ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_pop_front_auxEv.eh:
LSFDE63:
	.set L$set$139,LEFDE63-LASFDE63
	.long L$set$139
LASFDE63:
	.long	LASFDE63-EH_frame1
	.long	__ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_pop_front_auxEv-.
	.set L$set$140,LFE1711-LFB1711
	.long L$set$140
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$141,LCFI41-LFB1711
	.long L$set$141
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x9d
	.byte	0x3
	.byte	0x4
	.set L$set$142,LCFI42-LCFI41
	.long L$set$142
	.byte	0xe
	.byte	0x50
	.align 2
LEFDE63:
.private_extern _ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE16_M_destroy_nodesEPPS1_S5_.eh
.weak_definition _ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE16_M_destroy_nodesEPPS1_S5_.eh
_ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE16_M_destroy_nodesEPPS1_S5_.eh:
LSFDE65:
	.set L$set$143,LEFDE65-LASFDE65
	.long L$set$143
LASFDE65:
	.long	LASFDE65-EH_frame1
	.long	__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE16_M_destroy_nodesEPPS1_S5_-.
	.set L$set$144,LFE1718-LFB1718
	.long L$set$144
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$145,LCFI44-LFB1718
	.long L$set$145
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x9d
	.byte	0x3
	.byte	0x4
	.set L$set$146,LCFI45-LCFI44
	.long L$set$146
	.byte	0xe
	.byte	0x50
	.align 2
LEFDE65:
.private_extern _ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE15_M_create_nodesEPPS1_S5_.eh
.weak_definition _ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE15_M_create_nodesEPPS1_S5_.eh
_ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE15_M_create_nodesEPPS1_S5_.eh:
LSFDE67:
	.set L$set$147,LEFDE67-LASFDE67
	.long L$set$147
LASFDE67:
	.long	LASFDE67-EH_frame1
	.long	__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE15_M_create_nodesEPPS1_S5_-.
	.set L$set$148,LFE1720-LFB1720
	.long L$set$148
	.byte	0x4
	.long	LLSDA1720-.
	.byte	0x4
	.set L$set$149,LCFI47-LFB1720
	.long L$set$149
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x9d
	.byte	0x3
	.byte	0x9c
	.byte	0x4
	.byte	0x9b
	.byte	0x5
	.byte	0x4
	.set L$set$150,LCFI48-LCFI47
	.long L$set$150
	.byte	0xe
	.byte	0x60
	.align 2
LEFDE67:
.private_extern _ZNSt5dequeIP13_XTransactionSaIS1_EE17_M_reallocate_mapEmb.eh
.weak_definition _ZNSt5dequeIP13_XTransactionSaIS1_EE17_M_reallocate_mapEmb.eh
_ZNSt5dequeIP13_XTransactionSaIS1_EE17_M_reallocate_mapEmb.eh:
LSFDE69:
	.set L$set$151,LEFDE69-LASFDE69
	.long L$set$151
LASFDE69:
	.long	LASFDE69-EH_frame1
	.long	__ZNSt5dequeIP13_XTransactionSaIS1_EE17_M_reallocate_mapEmb-.
	.set L$set$152,LFE1732-LFB1732
	.long L$set$152
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$153,LCFI50-LFB1732
	.long L$set$153
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x9d
	.byte	0x3
	.byte	0x9c
	.byte	0x4
	.byte	0x9b
	.byte	0x5
	.byte	0x9a
	.byte	0x6
	.byte	0x99
	.byte	0x7
	.byte	0x4
	.set L$set$154,LCFI51-LCFI50
	.long L$set$154
	.byte	0xe
	.byte	0x60
	.align 2
LEFDE69:
_Z41__static_initialization_and_destruction_0ii.eh:
LSFDE71:
	.set L$set$155,LEFDE71-LASFDE71
	.long L$set$155
LASFDE71:
	.long	LASFDE71-EH_frame1
	.long	LFB1740-.
	.set L$set$156,LFE1740-LFB1740
	.long L$set$156
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$157,LCFI54-LFB1740
	.long L$set$157
	.byte	0xe
	.byte	0xb0,0x1
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x9d
	.byte	0x3
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.align 2
LEFDE71:
.section __TEXT,__text,regular,pure_instructions
	.align 2
	.stabs "",100,0,0,Letext
Letext:
.data
.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 2
L_memmove$stub:
	.indirect_symbol _memmove
	mflr r0
	bcl 20,31,L0$_memmove
L0$_memmove:
	mflr r11
	addis r11,r11,ha16(L_memmove$lazy_ptr-L0$_memmove)
	mtlr r0
	lwzu r12,lo16(L_memmove$lazy_ptr-L0$_memmove)(r11)
	mtctr r12
	bctr
.data
.lazy_symbol_pointer
L_memmove$lazy_ptr:
	.indirect_symbol _memmove
	.long dyld_stub_binding_helper
.data
.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 2
L__ZNSt24__default_alloc_templateILb1ELi0EE8allocateEm$stub:
	.indirect_symbol __ZNSt24__default_alloc_templateILb1ELi0EE8allocateEm
	mflr r0
	bcl 20,31,L1$__ZNSt24__default_alloc_templateILb1ELi0EE8allocateEm
L1$__ZNSt24__default_alloc_templateILb1ELi0EE8allocateEm:
	mflr r11
	addis r11,r11,ha16(L__ZNSt24__default_alloc_templateILb1ELi0EE8allocateEm$lazy_ptr-L1$__ZNSt24__default_alloc_templateILb1ELi0EE8allocateEm)
	mtlr r0
	lwzu r12,lo16(L__ZNSt24__default_alloc_templateILb1ELi0EE8allocateEm$lazy_ptr-L1$__ZNSt24__default_alloc_templateILb1ELi0EE8allocateEm)(r11)
	mtctr r12
	bctr
.data
.lazy_symbol_pointer
L__ZNSt24__default_alloc_templateILb1ELi0EE8allocateEm$lazy_ptr:
	.indirect_symbol __ZNSt24__default_alloc_templateILb1ELi0EE8allocateEm
	.long dyld_stub_binding_helper
.data
.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 2
L__ZNSt5dequeIP13_XTransactionSaIS1_EE17_M_reallocate_mapEmb$stub:
	.indirect_symbol __ZNSt5dequeIP13_XTransactionSaIS1_EE17_M_reallocate_mapEmb
	mflr r0
	bcl 20,31,L2$__ZNSt5dequeIP13_XTransactionSaIS1_EE17_M_reallocate_mapEmb
L2$__ZNSt5dequeIP13_XTransactionSaIS1_EE17_M_reallocate_mapEmb:
	mflr r11
	addis r11,r11,ha16(L__ZNSt5dequeIP13_XTransactionSaIS1_EE17_M_reallocate_mapEmb$lazy_ptr-L2$__ZNSt5dequeIP13_XTransactionSaIS1_EE17_M_reallocate_mapEmb)
	mtlr r0
	lwzu r12,lo16(L__ZNSt5dequeIP13_XTransactionSaIS1_EE17_M_reallocate_mapEmb$lazy_ptr-L2$__ZNSt5dequeIP13_XTransactionSaIS1_EE17_M_reallocate_mapEmb)(r11)
	mtctr r12
	bctr
.data
.lazy_symbol_pointer
L__ZNSt5dequeIP13_XTransactionSaIS1_EE17_M_reallocate_mapEmb$lazy_ptr:
	.indirect_symbol __ZNSt5dequeIP13_XTransactionSaIS1_EE17_M_reallocate_mapEmb
	.long dyld_stub_binding_helper
.data
.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 2
L___cxa_rethrow$stub:
	.indirect_symbol ___cxa_rethrow
	mflr r0
	bcl 20,31,L3$___cxa_rethrow
L3$___cxa_rethrow:
	mflr r11
	addis r11,r11,ha16(L___cxa_rethrow$lazy_ptr-L3$___cxa_rethrow)
	mtlr r0
	lwzu r12,lo16(L___cxa_rethrow$lazy_ptr-L3$___cxa_rethrow)(r11)
	mtctr r12
	bctr
.data
.lazy_symbol_pointer
L___cxa_rethrow$lazy_ptr:
	.indirect_symbol ___cxa_rethrow
	.long dyld_stub_binding_helper
.data
.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 2
L__ZNSt24__default_alloc_templateILb1ELi0EE10deallocateEPvm$stub:
	.indirect_symbol __ZNSt24__default_alloc_templateILb1ELi0EE10deallocateEPvm
	mflr r0
	bcl 20,31,L4$__ZNSt24__default_alloc_templateILb1ELi0EE10deallocateEPvm
L4$__ZNSt24__default_alloc_templateILb1ELi0EE10deallocateEPvm:
	mflr r11
	addis r11,r11,ha16(L__ZNSt24__default_alloc_templateILb1ELi0EE10deallocateEPvm$lazy_ptr-L4$__ZNSt24__default_alloc_templateILb1ELi0EE10deallocateEPvm)
	mtlr r0
	lwzu r12,lo16(L__ZNSt24__default_alloc_templateILb1ELi0EE10deallocateEPvm$lazy_ptr-L4$__ZNSt24__default_alloc_templateILb1ELi0EE10deallocateEPvm)(r11)
	mtctr r12
	bctr
.data
.lazy_symbol_pointer
L__ZNSt24__default_alloc_templateILb1ELi0EE10deallocateEPvm$lazy_ptr:
	.indirect_symbol __ZNSt24__default_alloc_templateILb1ELi0EE10deallocateEPvm
	.long dyld_stub_binding_helper
.data
.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 2
L__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE15_M_create_nodesEPPS1_S5_$stub:
	.indirect_symbol __ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE15_M_create_nodesEPPS1_S5_
	mflr r0
	bcl 20,31,L5$__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE15_M_create_nodesEPPS1_S5_
L5$__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE15_M_create_nodesEPPS1_S5_:
	mflr r11
	addis r11,r11,ha16(L__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE15_M_create_nodesEPPS1_S5_$lazy_ptr-L5$__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE15_M_create_nodesEPPS1_S5_)
	mtlr r0
	lwzu r12,lo16(L__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE15_M_create_nodesEPPS1_S5_$lazy_ptr-L5$__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE15_M_create_nodesEPPS1_S5_)(r11)
	mtctr r12
	bctr
.data
.lazy_symbol_pointer
L__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE15_M_create_nodesEPPS1_S5_$lazy_ptr:
	.indirect_symbol __ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE15_M_create_nodesEPPS1_S5_
	.long dyld_stub_binding_helper
.data
.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 2
L__ZSt17__throw_bad_allocv$stub:
	.indirect_symbol __ZSt17__throw_bad_allocv
	mflr r0
	bcl 20,31,L6$__ZSt17__throw_bad_allocv
L6$__ZSt17__throw_bad_allocv:
	mflr r11
	addis r11,r11,ha16(L__ZSt17__throw_bad_allocv$lazy_ptr-L6$__ZSt17__throw_bad_allocv)
	mtlr r0
	lwzu r12,lo16(L__ZSt17__throw_bad_allocv$lazy_ptr-L6$__ZSt17__throw_bad_allocv)(r11)
	mtctr r12
	bctr
.data
.lazy_symbol_pointer
L__ZSt17__throw_bad_allocv$lazy_ptr:
	.indirect_symbol __ZSt17__throw_bad_allocv
	.long dyld_stub_binding_helper
.data
.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 2
L__ZNSt24__default_alloc_templateILb1ELi0EE9_S_refillEm$stub:
	.indirect_symbol __ZNSt24__default_alloc_templateILb1ELi0EE9_S_refillEm
	mflr r0
	bcl 20,31,L7$__ZNSt24__default_alloc_templateILb1ELi0EE9_S_refillEm
L7$__ZNSt24__default_alloc_templateILb1ELi0EE9_S_refillEm:
	mflr r11
	addis r11,r11,ha16(L__ZNSt24__default_alloc_templateILb1ELi0EE9_S_refillEm$lazy_ptr-L7$__ZNSt24__default_alloc_templateILb1ELi0EE9_S_refillEm)
	mtlr r0
	lwzu r12,lo16(L__ZNSt24__default_alloc_templateILb1ELi0EE9_S_refillEm$lazy_ptr-L7$__ZNSt24__default_alloc_templateILb1ELi0EE9_S_refillEm)(r11)
	mtctr r12
	bctr
.data
.lazy_symbol_pointer
L__ZNSt24__default_alloc_templateILb1ELi0EE9_S_refillEm$lazy_ptr:
	.indirect_symbol __ZNSt24__default_alloc_templateILb1ELi0EE9_S_refillEm
	.long dyld_stub_binding_helper
.data
.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 2
L__Znwm$stub:
	.indirect_symbol __Znwm
	mflr r0
	bcl 20,31,L8$__Znwm
L8$__Znwm:
	mflr r11
	addis r11,r11,ha16(L__Znwm$lazy_ptr-L8$__Znwm)
	mtlr r0
	lwzu r12,lo16(L__Znwm$lazy_ptr-L8$__Znwm)(r11)
	mtctr r12
	bctr
.data
.lazy_symbol_pointer
L__Znwm$lazy_ptr:
	.indirect_symbol __Znwm
	.long dyld_stub_binding_helper
.data
.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 2
L_getenv$stub:
	.indirect_symbol _getenv
	mflr r0
	bcl 20,31,L9$_getenv
L9$_getenv:
	mflr r11
	addis r11,r11,ha16(L_getenv$lazy_ptr-L9$_getenv)
	mtlr r0
	lwzu r12,lo16(L_getenv$lazy_ptr-L9$_getenv)(r11)
	mtctr r12
	bctr
.data
.lazy_symbol_pointer
L_getenv$lazy_ptr:
	.indirect_symbol _getenv
	.long dyld_stub_binding_helper
.data
.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 2
L_pthread_mutex_unlock$stub:
	.indirect_symbol _pthread_mutex_unlock
	mflr r0
	bcl 20,31,L10$_pthread_mutex_unlock
L10$_pthread_mutex_unlock:
	mflr r11
	addis r11,r11,ha16(L_pthread_mutex_unlock$lazy_ptr-L10$_pthread_mutex_unlock)
	mtlr r0
	lwzu r12,lo16(L_pthread_mutex_unlock$lazy_ptr-L10$_pthread_mutex_unlock)(r11)
	mtctr r12
	bctr
.data
.lazy_symbol_pointer
L_pthread_mutex_unlock$lazy_ptr:
	.indirect_symbol _pthread_mutex_unlock
	.long dyld_stub_binding_helper
.data
.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 2
L_pthread_mutex_lock$stub:
	.indirect_symbol _pthread_mutex_lock
	mflr r0
	bcl 20,31,L11$_pthread_mutex_lock
L11$_pthread_mutex_lock:
	mflr r11
	addis r11,r11,ha16(L_pthread_mutex_lock$lazy_ptr-L11$_pthread_mutex_lock)
	mtlr r0
	lwzu r12,lo16(L_pthread_mutex_lock$lazy_ptr-L11$_pthread_mutex_lock)(r11)
	mtctr r12
	bctr
.data
.lazy_symbol_pointer
L_pthread_mutex_lock$lazy_ptr:
	.indirect_symbol _pthread_mutex_lock
	.long dyld_stub_binding_helper
.data
.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 2
L__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE16_M_destroy_nodesEPPS1_S5_$stub:
	.indirect_symbol __ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE16_M_destroy_nodesEPPS1_S5_
	mflr r0
	bcl 20,31,L12$__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE16_M_destroy_nodesEPPS1_S5_
L12$__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE16_M_destroy_nodesEPPS1_S5_:
	mflr r11
	addis r11,r11,ha16(L__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE16_M_destroy_nodesEPPS1_S5_$lazy_ptr-L12$__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE16_M_destroy_nodesEPPS1_S5_)
	mtlr r0
	lwzu r12,lo16(L__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE16_M_destroy_nodesEPPS1_S5_$lazy_ptr-L12$__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE16_M_destroy_nodesEPPS1_S5_)(r11)
	mtctr r12
	bctr
.data
.lazy_symbol_pointer
L__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE16_M_destroy_nodesEPPS1_S5_$lazy_ptr:
	.indirect_symbol __ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE16_M_destroy_nodesEPPS1_S5_
	.long dyld_stub_binding_helper
.data
.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 2
L__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED4Ev$stub:
	.indirect_symbol __ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED4Ev
	mflr r0
	bcl 20,31,L13$__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED4Ev
L13$__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED4Ev:
	mflr r11
	addis r11,r11,ha16(L__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED4Ev$lazy_ptr-L13$__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED4Ev)
	mtlr r0
	lwzu r12,lo16(L__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED4Ev$lazy_ptr-L13$__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED4Ev)(r11)
	mtctr r12
	bctr
.data
.lazy_symbol_pointer
L__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED4Ev$lazy_ptr:
	.indirect_symbol __ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED4Ev
	.long dyld_stub_binding_helper
.data
.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 2
L__ZN11QStringData10deleteSelfEv$stub:
	.indirect_symbol __ZN11QStringData10deleteSelfEv
	mflr r0
	bcl 20,31,L14$__ZN11QStringData10deleteSelfEv
L14$__ZN11QStringData10deleteSelfEv:
	mflr r11
	addis r11,r11,ha16(L__ZN11QStringData10deleteSelfEv$lazy_ptr-L14$__ZN11QStringData10deleteSelfEv)
	mtlr r0
	lwzu r12,lo16(L__ZN11QStringData10deleteSelfEv$lazy_ptr-L14$__ZN11QStringData10deleteSelfEv)(r11)
	mtctr r12
	bctr
.data
.lazy_symbol_pointer
L__ZN11QStringData10deleteSelfEv$lazy_ptr:
	.indirect_symbol __ZN11QStringData10deleteSelfEv
	.long dyld_stub_binding_helper
.data
.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 2
L__Z10_gErrPrintRK7QStringPKci$stub:
	.indirect_symbol __Z10_gErrPrintRK7QStringPKci
	mflr r0
	bcl 20,31,L15$__Z10_gErrPrintRK7QStringPKci
L15$__Z10_gErrPrintRK7QStringPKci:
	mflr r11
	addis r11,r11,ha16(L__Z10_gErrPrintRK7QStringPKci$lazy_ptr-L15$__Z10_gErrPrintRK7QStringPKci)
	mtlr r0
	lwzu r12,lo16(L__Z10_gErrPrintRK7QStringPKci$lazy_ptr-L15$__Z10_gErrPrintRK7QStringPKci)(r11)
	mtctr r12
	bctr
.data
.lazy_symbol_pointer
L__Z10_gErrPrintRK7QStringPKci$lazy_ptr:
	.indirect_symbol __Z10_gErrPrintRK7QStringPKci
	.long dyld_stub_binding_helper
.data
.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 2
L__ZN7QStringC1EPKc$stub:
	.indirect_symbol __ZN7QStringC1EPKc
	mflr r0
	bcl 20,31,L16$__ZN7QStringC1EPKc
L16$__ZN7QStringC1EPKc:
	mflr r11
	addis r11,r11,ha16(L__ZN7QStringC1EPKc$lazy_ptr-L16$__ZN7QStringC1EPKc)
	mtlr r0
	lwzu r12,lo16(L__ZN7QStringC1EPKc$lazy_ptr-L16$__ZN7QStringC1EPKc)(r11)
	mtctr r12
	bctr
.data
.lazy_symbol_pointer
L__ZN7QStringC1EPKc$lazy_ptr:
	.indirect_symbol __ZN7QStringC1EPKc
	.long dyld_stub_binding_helper
.data
.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 2
L___cxa_end_catch$stub:
	.indirect_symbol ___cxa_end_catch
	mflr r0
	bcl 20,31,L17$___cxa_end_catch
L17$___cxa_end_catch:
	mflr r11
	addis r11,r11,ha16(L___cxa_end_catch$lazy_ptr-L17$___cxa_end_catch)
	mtlr r0
	lwzu r12,lo16(L___cxa_end_catch$lazy_ptr-L17$___cxa_end_catch)(r11)
	mtctr r12
	bctr
.data
.lazy_symbol_pointer
L___cxa_end_catch$lazy_ptr:
	.indirect_symbol ___cxa_end_catch
	.long dyld_stub_binding_helper
.data
.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 2
L__ZN10XKameError5printEv$stub:
	.indirect_symbol __ZN10XKameError5printEv
	mflr r0
	bcl 20,31,L18$__ZN10XKameError5printEv
L18$__ZN10XKameError5printEv:
	mflr r11
	addis r11,r11,ha16(L__ZN10XKameError5printEv$lazy_ptr-L18$__ZN10XKameError5printEv)
	mtlr r0
	lwzu r12,lo16(L__ZN10XKameError5printEv$lazy_ptr-L18$__ZN10XKameError5printEv)(r11)
	mtctr r12
	bctr
.data
.lazy_symbol_pointer
L__ZN10XKameError5printEv$lazy_ptr:
	.indirect_symbol __ZN10XKameError5printEv
	.long dyld_stub_binding_helper
.data
.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 2
L___cxa_begin_catch$stub:
	.indirect_symbol ___cxa_begin_catch
	mflr r0
	bcl 20,31,L19$___cxa_begin_catch
L19$___cxa_begin_catch:
	mflr r11
	addis r11,r11,ha16(L___cxa_begin_catch$lazy_ptr-L19$___cxa_begin_catch)
	mtlr r0
	lwzu r12,lo16(L___cxa_begin_catch$lazy_ptr-L19$___cxa_begin_catch)(r11)
	mtctr r12
	bctr
.data
.lazy_symbol_pointer
L___cxa_begin_catch$lazy_ptr:
	.indirect_symbol ___cxa_begin_catch
	.long dyld_stub_binding_helper
.data
.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 2
L__ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_pop_front_auxEv$stub:
	.indirect_symbol __ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_pop_front_auxEv
	mflr r0
	bcl 20,31,L20$__ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_pop_front_auxEv
L20$__ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_pop_front_auxEv:
	mflr r11
	addis r11,r11,ha16(L__ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_pop_front_auxEv$lazy_ptr-L20$__ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_pop_front_auxEv)
	mtlr r0
	lwzu r12,lo16(L__ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_pop_front_auxEv$lazy_ptr-L20$__ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_pop_front_auxEv)(r11)
	mtctr r12
	bctr
.data
.lazy_symbol_pointer
L__ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_pop_front_auxEv$lazy_ptr:
	.indirect_symbol __ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_pop_front_auxEv
	.long dyld_stub_binding_helper
.data
.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 2
L__ZN5XTime3nowEv$stub:
	.indirect_symbol __ZN5XTime3nowEv
	mflr r0
	bcl 20,31,L21$__ZN5XTime3nowEv
L21$__ZN5XTime3nowEv:
	mflr r11
	addis r11,r11,ha16(L__ZN5XTime3nowEv$lazy_ptr-L21$__ZN5XTime3nowEv)
	mtlr r0
	lwzu r12,lo16(L__ZN5XTime3nowEv$lazy_ptr-L21$__ZN5XTime3nowEv)(r11)
	mtctr r12
	bctr
.data
.lazy_symbol_pointer
L__ZN5XTime3nowEv$lazy_ptr:
	.indirect_symbol __ZN5XTime3nowEv
	.long dyld_stub_binding_helper
.data
.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 2
L__ZN6XMutex6unlockEv$stub:
	.indirect_symbol __ZN6XMutex6unlockEv
	mflr r0
	bcl 20,31,L22$__ZN6XMutex6unlockEv
L22$__ZN6XMutex6unlockEv:
	mflr r11
	addis r11,r11,ha16(L__ZN6XMutex6unlockEv$lazy_ptr-L22$__ZN6XMutex6unlockEv)
	mtlr r0
	lwzu r12,lo16(L__ZN6XMutex6unlockEv$lazy_ptr-L22$__ZN6XMutex6unlockEv)(r11)
	mtctr r12
	bctr
.data
.lazy_symbol_pointer
L__ZN6XMutex6unlockEv$lazy_ptr:
	.indirect_symbol __ZN6XMutex6unlockEv
	.long dyld_stub_binding_helper
.data
.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 2
L__ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_push_back_auxERKS1_$stub:
	.indirect_symbol __ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_push_back_auxERKS1_
	mflr r0
	bcl 20,31,L23$__ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_push_back_auxERKS1_
L23$__ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_push_back_auxERKS1_:
	mflr r11
	addis r11,r11,ha16(L__ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_push_back_auxERKS1_$lazy_ptr-L23$__ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_push_back_auxERKS1_)
	mtlr r0
	lwzu r12,lo16(L__ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_push_back_auxERKS1_$lazy_ptr-L23$__ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_push_back_auxERKS1_)(r11)
	mtctr r12
	bctr
.data
.lazy_symbol_pointer
L__ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_push_back_auxERKS1_$lazy_ptr:
	.indirect_symbol __ZNSt5dequeIP13_XTransactionSaIS1_EE16_M_push_back_auxERKS1_
	.long dyld_stub_binding_helper
.data
.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 2
L__ZN6XMutex4lockEv$stub:
	.indirect_symbol __ZN6XMutex4lockEv
	mflr r0
	bcl 20,31,L24$__ZN6XMutex4lockEv
L24$__ZN6XMutex4lockEv:
	mflr r11
	addis r11,r11,ha16(L__ZN6XMutex4lockEv$lazy_ptr-L24$__ZN6XMutex4lockEv)
	mtlr r0
	lwzu r12,lo16(L__ZN6XMutex4lockEv$lazy_ptr-L24$__ZN6XMutex4lockEv)(r11)
	mtctr r12
	bctr
.data
.lazy_symbol_pointer
L__ZN6XMutex4lockEv$lazy_ptr:
	.indirect_symbol __ZN6XMutex4lockEv
	.long dyld_stub_binding_helper
.data
.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 2
L__Z9msecsleepj$stub:
	.indirect_symbol __Z9msecsleepj
	mflr r0
	bcl 20,31,L25$__Z9msecsleepj
L25$__Z9msecsleepj:
	mflr r11
	addis r11,r11,ha16(L__Z9msecsleepj$lazy_ptr-L25$__Z9msecsleepj)
	mtlr r0
	lwzu r12,lo16(L__Z9msecsleepj$lazy_ptr-L25$__Z9msecsleepj)(r11)
	mtctr r12
	bctr
.data
.lazy_symbol_pointer
L__Z9msecsleepj$lazy_ptr:
	.indirect_symbol __Z9msecsleepj
	.long dyld_stub_binding_helper
.data
.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 2
L_fprintf$stub:
	.indirect_symbol _fprintf
	mflr r0
	bcl 20,31,L26$_fprintf
L26$_fprintf:
	mflr r11
	addis r11,r11,ha16(L_fprintf$lazy_ptr-L26$_fprintf)
	mtlr r0
	lwzu r12,lo16(L_fprintf$lazy_ptr-L26$_fprintf)(r11)
	mtctr r12
	bctr
.data
.lazy_symbol_pointer
L_fprintf$lazy_ptr:
	.indirect_symbol _fprintf
	.long dyld_stub_binding_helper
.data
.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 2
L__ZN6XMutexD1Ev$stub:
	.indirect_symbol __ZN6XMutexD1Ev
	mflr r0
	bcl 20,31,L27$__ZN6XMutexD1Ev
L27$__ZN6XMutexD1Ev:
	mflr r11
	addis r11,r11,ha16(L__ZN6XMutexD1Ev$lazy_ptr-L27$__ZN6XMutexD1Ev)
	mtlr r0
	lwzu r12,lo16(L__ZN6XMutexD1Ev$lazy_ptr-L27$__ZN6XMutexD1Ev)(r11)
	mtctr r12
	bctr
.data
.lazy_symbol_pointer
L__ZN6XMutexD1Ev$lazy_ptr:
	.indirect_symbol __ZN6XMutexD1Ev
	.long dyld_stub_binding_helper
.data
.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 2
L__Z9timeStampv$stub:
	.indirect_symbol __Z9timeStampv
	mflr r0
	bcl 20,31,L28$__Z9timeStampv
L28$__Z9timeStampv:
	mflr r11
	addis r11,r11,ha16(L__Z9timeStampv$lazy_ptr-L28$__Z9timeStampv)
	mtlr r0
	lwzu r12,lo16(L__Z9timeStampv$lazy_ptr-L28$__Z9timeStampv)(r11)
	mtctr r12
	bctr
.data
.lazy_symbol_pointer
L__Z9timeStampv$lazy_ptr:
	.indirect_symbol __Z9timeStampv
	.long dyld_stub_binding_helper
.data
.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 2
L__ZN6XMutexC1Ev$stub:
	.indirect_symbol __ZN6XMutexC1Ev
	mflr r0
	bcl 20,31,L29$__ZN6XMutexC1Ev
L29$__ZN6XMutexC1Ev:
	mflr r11
	addis r11,r11,ha16(L__ZN6XMutexC1Ev$lazy_ptr-L29$__ZN6XMutexC1Ev)
	mtlr r0
	lwzu r12,lo16(L__ZN6XMutexC1Ev$lazy_ptr-L29$__ZN6XMutexC1Ev)(r11)
	mtctr r12
	bctr
.data
.lazy_symbol_pointer
L__ZN6XMutexC1Ev$lazy_ptr:
	.indirect_symbol __ZN6XMutexC1Ev
	.long dyld_stub_binding_helper
.data
.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 2
L__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED2Ev$stub:
	.indirect_symbol __ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED2Ev
	mflr r0
	bcl 20,31,L30$__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED2Ev
L30$__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED2Ev:
	mflr r11
	addis r11,r11,ha16(L__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED2Ev$lazy_ptr-L30$__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED2Ev)
	mtlr r0
	lwzu r12,lo16(L__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED2Ev$lazy_ptr-L30$__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED2Ev)(r11)
	mtctr r12
	bctr
.data
.lazy_symbol_pointer
L__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED2Ev$lazy_ptr:
	.indirect_symbol __ZNSt11_Deque_baseIP13_XTransactionSaIS1_EED2Ev
	.long dyld_stub_binding_helper
.data
.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 2
L__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE17_M_initialize_mapEm$stub:
	.indirect_symbol __ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE17_M_initialize_mapEm
	mflr r0
	bcl 20,31,L31$__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE17_M_initialize_mapEm
L31$__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE17_M_initialize_mapEm:
	mflr r11
	addis r11,r11,ha16(L__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE17_M_initialize_mapEm$lazy_ptr-L31$__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE17_M_initialize_mapEm)
	mtlr r0
	lwzu r12,lo16(L__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE17_M_initialize_mapEm$lazy_ptr-L31$__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE17_M_initialize_mapEm)(r11)
	mtctr r12
	bctr
.data
.lazy_symbol_pointer
L__ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE17_M_initialize_mapEm$lazy_ptr:
	.indirect_symbol __ZNSt11_Deque_baseIP13_XTransactionSaIS1_EE17_M_initialize_mapEm
	.long dyld_stub_binding_helper
.data
.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 2
L__ZdlPv$stub:
	.indirect_symbol __ZdlPv
	mflr r0
	bcl 20,31,L32$__ZdlPv
L32$__ZdlPv:
	mflr r11
	addis r11,r11,ha16(L__ZdlPv$lazy_ptr-L32$__ZdlPv)
	mtlr r0
	lwzu r12,lo16(L__ZdlPv$lazy_ptr-L32$__ZdlPv)(r11)
	mtctr r12
	bctr
.data
.lazy_symbol_pointer
L__ZdlPv$lazy_ptr:
	.indirect_symbol __ZdlPv
	.long dyld_stub_binding_helper
.data
.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 2
L__Unwind_Resume$stub:
	.indirect_symbol __Unwind_Resume
	mflr r0
	bcl 20,31,L33$__Unwind_Resume
L33$__Unwind_Resume:
	mflr r11
	addis r11,r11,ha16(L__Unwind_Resume$lazy_ptr-L33$__Unwind_Resume)
	mtlr r0
	lwzu r12,lo16(L__Unwind_Resume$lazy_ptr-L33$__Unwind_Resume)(r11)
	mtctr r12
	bctr
.data
.lazy_symbol_pointer
L__Unwind_Resume$lazy_ptr:
	.indirect_symbol __Unwind_Resume
	.long dyld_stub_binding_helper
.data
.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 2
L__Z9my_assertPKci$stub:
	.indirect_symbol __Z9my_assertPKci
	mflr r0
	bcl 20,31,L34$__Z9my_assertPKci
L34$__Z9my_assertPKci:
	mflr r11
	addis r11,r11,ha16(L__Z9my_assertPKci$lazy_ptr-L34$__Z9my_assertPKci)
	mtlr r0
	lwzu r12,lo16(L__Z9my_assertPKci$lazy_ptr-L34$__Z9my_assertPKci)(r11)
	mtctr r12
	bctr
.data
.lazy_symbol_pointer
L__Z9my_assertPKci$lazy_ptr:
	.indirect_symbol __Z9my_assertPKci
	.long dyld_stub_binding_helper
.data
.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 2
L_pthread_equal$stub:
	.indirect_symbol _pthread_equal
	mflr r0
	bcl 20,31,L35$_pthread_equal
L35$_pthread_equal:
	mflr r11
	addis r11,r11,ha16(L_pthread_equal$lazy_ptr-L35$_pthread_equal)
	mtlr r0
	lwzu r12,lo16(L_pthread_equal$lazy_ptr-L35$_pthread_equal)(r11)
	mtctr r12
	bctr
.data
.lazy_symbol_pointer
L_pthread_equal$lazy_ptr:
	.indirect_symbol _pthread_equal
	.long dyld_stub_binding_helper
.data
.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 2
L_pthread_self$stub:
	.indirect_symbol _pthread_self
	mflr r0
	bcl 20,31,L36$_pthread_self
L36$_pthread_self:
	mflr r11
	addis r11,r11,ha16(L_pthread_self$lazy_ptr-L36$_pthread_self)
	mtlr r0
	lwzu r12,lo16(L_pthread_self$lazy_ptr-L36$_pthread_self)(r11)
	mtctr r12
	bctr
.data
.lazy_symbol_pointer
L_pthread_self$lazy_ptr:
	.indirect_symbol _pthread_self
	.long dyld_stub_binding_helper
.data
.non_lazy_symbol_pointer
L___gxx_personality_v0$non_lazy_ptr:
	.indirect_symbol ___gxx_personality_v0
	.long	0
L__ZNSt24__default_alloc_templateILb1ELi0EE22_S_node_allocator_lockE$non_lazy_ptr:
	.indirect_symbol __ZNSt24__default_alloc_templateILb1ELi0EE22_S_node_allocator_lockE
	.long	0
L__ZNSt24__default_alloc_templateILb1ELi0EE12_S_free_listE$non_lazy_ptr:
	.indirect_symbol __ZNSt24__default_alloc_templateILb1ELi0EE12_S_free_listE
	.long	0
L__ZNSt24__default_alloc_templateILb1ELi0EE12_S_force_newE$non_lazy_ptr:
	.indirect_symbol __ZNSt24__default_alloc_templateILb1ELi0EE12_S_force_newE
	.long	0
L__ZN7QString11shared_nullE$non_lazy_ptr:
	.indirect_symbol __ZN7QString11shared_nullE
	.long	0
L___sF$non_lazy_ptr:
	.indirect_symbol ___sF
	.long	0
.data
	.align 2
L__ZTV12_XTalkerBase$non_lazy_ptr:
	.long	__ZTV12_XTalkerBase
	.align 2
L__ZTV9XListener$non_lazy_ptr:
	.long	__ZTV9XListener
.constructor
.data
.destructor
	.align 1
	.subsections_via_symbols
