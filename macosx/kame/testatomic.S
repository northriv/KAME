	.section __TEXT,__text,regular,pure_instructions
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.machine ppc
	.stabs	"/Users/northriv/Documents/Projects/KAME/2.0/macosx/kame/",100,0,4,Ltext0
	.stabs	"../../tools/testatomic.cpp",100,0,4,Ltext0
	.text
Ltext0:
	.stabs	"gcc2_compiled.",60,0,0,0
	.globl __ZTV1B
	.weak_definition __ZTV1B
	.section __DATA,__const_coal,coalesced
	.align 3
__ZTV1B:
	.long	0
	.long	__ZTI1B
	.long	__ZN1BD1Ev
	.long	__ZN1BD0Ev
	.globl __ZTI1B
	.weak_definition __ZTI1B
	.align 2
__ZTI1B:
	.long	__ZTVN10__cxxabiv120__si_class_type_infoE+8
	.long	__ZTS1B
	.long	__ZTI1A
	.globl __ZTI1A
	.weak_definition __ZTI1A
	.align 2
__ZTI1A:
	.long	__ZTVN10__cxxabiv117__class_type_infoE+8
	.long	__ZTS1A
	.globl __ZTS1A
	.weak_definition __ZTS1A
	.section __TEXT,__const_coal,coalesced
	.align 2
__ZTS1A:
	.ascii "1A\0"
	.globl __ZTS1B
	.weak_definition __ZTS1B
	.align 2
__ZTS1B:
	.ascii "1B\0"
	.globl __ZTV1A
	.weak_definition __ZTV1A
	.section __DATA,__const_coal,coalesced
	.align 3
__ZTV1A:
	.long	0
	.long	__ZTI1A
	.long	__ZN1AD1Ev
	.long	__ZN1AD0Ev
	.section __TEXT,__textcoal_nt,coalesced,pure_instructions
	.align 2
	.globl __Z11_my_cassertm
	.weak_definition __Z11_my_cassertm
__Z11_my_cassertm:
	.stabd	46,0,0
	.stabs	"../../kame/support.h",132,0,0,Ltext1
Ltext1:
	.stabd	68,0,87
LFB3:
	stmw r30,-8(r1)
LCFI0:
	stwu r1,-48(r1)
LCFI1:
	mr r30,r1
LCFI2:
	stw r3,72(r30)
	.stabd	68,0,87
	lwz r1,0(r1)
	lmw r30,-8(r1)
	blr
LFE3:
	.stabs	"_Z11_my_cassertm:F(0,1)",36,0,87,__Z11_my_cassertm
	.stabs	":t(0,1)=(0,1)",128,0,0,0
Lscope0:
	.stabs	"",36,0,0,Lscope0-__Z11_my_cassertm
	.stabd	78,0,0
	.align 2
	.globl __Z11readBarrierv
	.weak_definition __Z11readBarrierv
__Z11readBarrierv:
	.stabd	46,0,0
	.stabs	"../../kame/atomic.h",132,0,0,Ltext2
Ltext2:
	.stabd	68,0,119
LFB1383:
	stmw r30,-8(r1)
LCFI3:
	stwu r1,-48(r1)
LCFI4:
	mr r30,r1
LCFI5:
LBB2:
	.stabd	68,0,120
	isync
LBE2:
	.stabd	68,0,121
	lwz r1,0(r1)
	lmw r30,-8(r1)
	blr
LFE1383:
	.stabs	"_Z11readBarrierv:F(0,1)",36,0,119,__Z11readBarrierv
Lscope1:
	.stabs	"",36,0,0,Lscope1-__Z11readBarrierv
	.stabd	78,0,0
	.align 2
	.globl __Z13memoryBarrierv
	.weak_definition __Z13memoryBarrierv
__Z13memoryBarrierv:
	.stabd	46,0,0
	.stabd	68,0,122
LFB1384:
	stmw r30,-8(r1)
LCFI6:
	stwu r1,-48(r1)
LCFI7:
	mr r30,r1
LCFI8:
LBB3:
	.stabd	68,0,123
	sync
LBE3:
	.stabd	68,0,124
	lwz r1,0(r1)
	lmw r30,-8(r1)
	blr
LFE1384:
	.stabs	"_Z13memoryBarrierv:F(0,1)",36,0,122,__Z13memoryBarrierv
Lscope2:
	.stabs	"",36,0,0,Lscope2-__Z13memoryBarrierv
	.stabd	78,0,0
	.text
	.align 2
	.globl __ZNK15XRecursiveMutex23isLockedByCurrentThreadEv
__ZNK15XRecursiveMutex23isLockedByCurrentThreadEv:
	.stabd	46,0,0
	.stabs	"../../kame/thread.cpp",132,0,0,Ltext3
Ltext3:
	.stabd	68,0,100
LFB1472:
	mflr r0
LCFI9:
	stmw r29,-12(r1)
LCFI10:
	stw r0,8(r1)
LCFI11:
	stwu r1,-80(r1)
LCFI12:
	mr r30,r1
LCFI13:
	stw r3,104(r30)
LBB4:
	.stabd	68,0,102
	lwz r2,104(r30)
	lwz r29,44(r2)
	bl L_pthread_self$stub
	mr r0,r3
	mr r3,r29
	mr r4,r0
	bl L_pthread_equal$stub
	mr r0,r3
	srawi r2,r0,31
	xor r0,r2,r0
	subf r0,r2,r0
	neg r0,r0
	srwi r0,r0,31
LBE4:
	.stabd	68,0,103
	mr r3,r0
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r29,-12(r1)
	blr
LFE1472:
	.stabs	"_ZNK15XRecursiveMutex23isLockedByCurrentThreadEv:F(0,2)",36,0,100,__ZNK15XRecursiveMutex23isLockedByCurrentThreadEv
	.stabs	"this:p(0,3)",160,0,100,104
	.stabs	"bool:t(0,2)=@s32;-16;",128,0,0,0
	.stabs	":t(0,3)=k(0,4)",128,0,0,0
	.stabs	":t(0,4)=*(0,5)",128,0,0,0
	.stabs	":t(0,5)=k(0,6)",128,0,0,0
	.stabs	"XRecursiveMutex:t(0,6)=s52m_mutex:/0(0,7)=xsXMutex:,0,352;m_lockingthread:/0(0,8),352,32;m_lockcount:/0(0,10),384,32;__base_ctor ::(0,11)=#(0,6),(0,1),(0,12),(0,13),(0,1);:_ZN15XRecursiveMutexC2ERKS_;2A.;__comp_ctor ::(0,11):_ZN15XRecursiveMutexC1ERKS_;2A.;__base_ctor ::(0,14)=#(0,6),(0,1),(0,12),(0,1);:_ZN15XRecursiveMutexC2Ev;2A.;__comp_ctor ::(0,14):_ZN15XRecursiveMutexC1Ev;2A.;__base_dtor ::(0,14):_ZN15XRecursiveMutexD2Ev;2A.;__comp_dtor ::(0,14):_ZN15XRecursiveMutexD1Ev;2A.;lock::(0,14):_ZN15XRecursiveMutex4lockEv;2A.;unlock::(0,14):_ZN15XRecursiveMutex6unlockEv;2A.;trylock::(0,15)=#(0,6),(0,2),(0,12),(0,1);:_ZN15XRecursiveMutex7trylockEv;2A.;isLockedByCurrentThread::(0,16)=#(0,6),(0,2),(0,4),(0,1);:_ZNK15XRecursiveMutex23isLockedByCurrentThreadEv;2B.;;",128,0,0,0
	.stabs	"XMutex:t(0,7)=s44m_mutex:/1(0,17)=xspthread_mutex_t:,0,352;__base_ctor ::(0,19)=#(0,7),(0,1),(0,20),(0,21),(0,1);:_ZN6XMutexC2ERKS_;2A.;__comp_ctor ::(0,19):_ZN6XMutexC1ERKS_;2A.;__base_ctor ::(0,22)=#(0,7),(0,1),(0,20),(0,1);:_ZN6XMutexC2Ev;2A.;__comp_ctor ::(0,22):_ZN6XMutexC1Ev;2A.;__base_dtor ::(0,22):_ZN6XMutexD2Ev;2A.;__comp_dtor ::(0,22):_ZN6XMutexD1Ev;2A.;lock::(0,22):_ZN6XMutex4lockEv;2A.;unlock::(0,22):_ZN6XMutex6unlockEv;2A.;trylock::(0,23)=#(0,7),(0,2),(0,20),(0,1);:_ZN6XMutex7trylockEv;2A.;;",128,0,0,0
	.stabs	"__darwin_pthread_t:t(0,9)=(0,24)",128,0,0,0
	.stabs	"pthread_t:t(0,8)=(0,9)",128,0,0,0
	.stabs	"int:t(0,10)=r(0,10);-2147483648;2147483647;",128,0,0,0
	.stabs	":t(0,12)=*(0,6)",128,0,0,0
	.stabs	":t(0,13)=&(0,5)",128,0,0,0
	.stabs	"__darwin_pthread_mutex_t:t(0,18)=(0,25)",128,0,0,0
	.stabs	"pthread_mutex_t:t(0,17)=(0,18)",128,0,0,0
	.stabs	":t(0,20)=*(0,7)",128,0,0,0
	.stabs	":t(0,21)=&(0,26)",128,0,0,0
	.stabs	":t(0,24)=*(0,27)",128,0,0,0
	.stabs	"_opaque_pthread_mutex_t:t(0,25)=s44__sig:(0,28),0,32;__opaque:(0,29),32,320;;",128,0,0,0
	.stabs	":t(0,26)=k(0,7)",128,0,0,0
	.stabs	"_opaque_pthread_t:t(0,27)=s604__sig:(0,28),0,32;__cleanup_stack:(0,30),32,32;__opaque:(0,31),64,4768;;",128,0,0,0
	.stabs	"long int:t(0,28)=r(0,28);-2147483648;2147483647;",128,0,0,0
	.stabs	":t(0,29)=ar(0,32);0;39;(0,33)",128,0,0,0
	.stabs	":t(0,30)=*(0,34)",128,0,0,0
	.stabs	":t(0,31)=ar(0,32);0;595;(0,33)",128,0,0,0
	.stabs	"long unsigned int:t(0,32)=r(0,32);0;037777777777;",128,0,0,0
	.stabs	"char:t(0,33)=r(0,33);0;127;",128,0,0,0
	.stabs	"__darwin_pthread_handler_rec:t(0,34)=s12__routine:(0,35),0,32;__arg:(0,36),32,32;__next:(0,30),64,32;;",128,0,0,0
	.stabs	":t(0,35)=*(0,37)",128,0,0,0
	.stabs	":t(0,36)=*(0,1)",128,0,0,0
	.stabs	":t(0,37)=f(0,1)",128,0,0,0
Lscope3:
	.stabs	"",36,0,0,Lscope3-__ZNK15XRecursiveMutex23isLockedByCurrentThreadEv
	.stabd	78,0,0
	.align 2
	.globl __ZN10XCondition4waitEi
__ZN10XCondition4waitEi:
	.stabd	46,0,0
	.stabd	68,0,119
LFB1479:
	mflr r0
LCFI14:
	stmw r30,-8(r1)
LCFI15:
	stw r0,8(r1)
LCFI16:
	stwu r1,-112(r1)
LCFI17:
	mr r30,r1
LCFI18:
	stw r3,136(r30)
	stw r4,140(r30)
LBB5:
LBB6:
	.stabd	68,0,122
	lwz r2,136(r30)
	addi r0,r2,28
	mr r3,r0
	bl L_pthread_mutex_lock$stub
	.stabd	68,0,123
	lwz r0,140(r30)
	cmpwi cr7,r0,0
	ble cr7,L10
LBB7:
	.stabd	68,0,128
	addi r0,r30,72
	mr r3,r0
	li r4,0
	bl L_gettimeofday$stub
	.stabd	68,0,129
	lwz r0,72(r30)
	stw r0,64(r30)
	.stabd	68,0,130
	lwz r2,76(r30)
	lwz r0,140(r30)
	add r0,r2,r0
	mulli r0,r0,1000
	stw r0,56(r30)
	.stabd	68,0,131
	lwz r2,56(r30)
	lis r0,0x3b9a
	ori r0,r0,51711
	cmpw cr7,r2,r0
	ble cr7,L12
	.stabd	68,0,132
	lwz r2,56(r30)
	addis r2,r2,0xc465
	addi r0,r2,13824
	stw r0,56(r30)
	lwz r2,64(r30)
	addi r0,r2,1
	stw r0,64(r30)
L12:
	.stabd	68,0,134
	lwz r0,56(r30)
	stw r0,68(r30)
	.stabd	68,0,135
	lwz r9,136(r30)
	lwz r2,136(r30)
	addi r0,r2,28
	addi r2,r30,64
	mr r3,r9
	mr r4,r0
	mr r5,r2
	bl L_pthread_cond_timedwait$stub
	mr r0,r3
	stw r0,60(r30)
	b L14
L10:
LBE7:
	.stabd	68,0,154
	lwz r9,136(r30)
	lwz r2,136(r30)
	addi r0,r2,28
	mr r3,r9
	mr r4,r0
	bl L_pthread_cond_wait$stub
	mr r0,r3
	stw r0,60(r30)
L14:
	.stabd	68,0,157
	lwz r2,136(r30)
	addi r0,r2,28
	mr r3,r0
	bl L_pthread_mutex_unlock$stub
	.stabd	68,0,158
	lwz r0,60(r30)
LBE6:
LBE5:
	.stabd	68,0,159
	mr r3,r0
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r30,-8(r1)
	blr
LFE1479:
	.stabs	"_ZN10XCondition4waitEi:F(0,10)",36,0,119,__ZN10XCondition4waitEi
	.stabs	"XRecursiveMutex:Tt(0,6)",128,0,71,0
	.stabs	"XMutex:Tt(0,7)",128,0,56,0
	.stabs	"_opaque_pthread_mutex_t:Tt(0,25)",128,0,69,0
	.stabs	"_opaque_pthread_t:Tt(0,27)",128,0,74,0
	.stabs	"__darwin_pthread_handler_rec:Tt(0,34)",128,0,61,0
	.stabs	"this:p(0,38)",160,0,119,136
	.stabs	"usec:p(0,10)",160,0,119,140
	.stabs	"ret:(0,10)",128,0,121,60
	.stabs	"int:t(0,10)",128,0,0,0
	.stabs	":t(0,38)=k(0,39)",128,0,0,0
	.stabs	":t(0,39)=*(0,40)",128,0,0,0
	.stabs	"XCondition:t(0,40)=s72m_cond:/0(0,41)=xspthread_cond_t:,0,224;m_mutex:/0(0,17),224,352;__base_ctor ::(0,43)=#(0,40),(0,1),(0,39),(0,44),(0,1);:_ZN10XConditionC2ERKS_;2A.;__comp_ctor ::(0,43):_ZN10XConditionC1ERKS_;2A.;__base_ctor ::(0,45)=#(0,40),(0,1),(0,39),(0,1);:_ZN10XConditionC2Ev;2A.;__comp_ctor ::(0,45):_ZN10XConditionC1Ev;2A.;__base_dtor ::(0,45):_ZN10XConditionD2Ev;2A.;__comp_dtor ::(0,45):_ZN10XConditionD1Ev;2A.;wait::(0,46)=#(0,40),(0,10),(0,39),(0,10),(0,1);:_ZN10XCondition4waitEi;2A.;signal::(0,45):_ZN10XCondition6signalEv;2A.;broadcast::(0,45):_ZN10XCondition9broadcastEv;2A.;;",128,0,0,0
	.stabs	"__darwin_pthread_cond_t:t(0,42)=(0,47)",128,0,0,0
	.stabs	"pthread_cond_t:t(0,41)=(0,42)",128,0,0,0
	.stabs	":t(0,44)=&(0,48)",128,0,0,0
	.stabs	"_opaque_pthread_cond_t:t(0,47)=s28__sig:(0,28),0,32;__opaque:(0,49),32,192;;",128,0,0,0
	.stabs	":t(0,48)=k(0,40)",128,0,0,0
	.stabs	":t(0,49)=ar(0,32);0;23;(0,33)",128,0,0,0
	.stabn	192,0,0,LBB6
	.stabs	"abstime:(0,50)=xstimespec:",128,0,125,64
	.stabs	"XCondition:Tt(0,40)",128,0,91,0
	.stabs	"_opaque_pthread_cond_t:Tt(0,47)",128,0,67,0
	.stabs	"timespec:Tt(0,50)=s8tv_sec:(0,51),0,32;tv_nsec:(0,28),32,32;;",128,0,89,0
	.stabs	"__darwin_time_t:t(0,52)=(0,28)",128,0,0,0
	.stabs	"time_t:t(0,51)=(0,52)",128,0,0,0
	.stabs	"tv:(0,53)=xstimeval:",128,0,126,72
	.stabs	"timeval:Tt(0,53)=s8tv_sec:(0,52),0,32;tv_usec:(0,54),32,32;;",128,0,74,0
	.stabs	"__int32_t:t(0,55)=(0,10)",128,0,0,0
	.stabs	"__darwin_suseconds_t:t(0,54)=(0,55)",128,0,0,0
	.stabs	"nsec:(0,28)",128,0,127,56
	.stabs	"long int:t(0,28)",128,0,0,0
	.stabn	192,0,0,LBB7
	.stabn	224,0,0,LBE7
	.stabn	224,0,0,LBE6
Lscope4:
	.stabs	"",36,0,0,Lscope4-__ZN10XCondition4waitEi
	.stabd	78,0,0
	.cstring
	.align 2
LC0:
	.ascii "Err:%s:%d\12\0"
	.text
	.align 2
	.globl __Z9my_assertPKci
__Z9my_assertPKci:
	.stabd	46,0,0
	.stabs	"../../tools/testatomic.cpp",132,0,0,Ltext4
Ltext4:
	.stabd	68,0,38
LFB1482:
	mflr r0
LCFI19:
	stmw r30,-8(r1)
LCFI20:
	stw r0,8(r1)
LCFI21:
	stwu r1,-80(r1)
LCFI22:
	mr r30,r1
LCFI23:
	bcl 20,31,"L00000000001$pb"
"L00000000001$pb":
	mflr r31
	stw r3,104(r30)
	stw r4,108(r30)
LBB8:
	.stabd	68,0,39
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000001$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000001$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC0-"L00000000001$pb")
	la r4,lo16(LC0-"L00000000001$pb")(r2)
	lwz r5,104(r30)
	lwz r6,108(r30)
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,40
	bl L_abort$stub
LBE8:
LFE1482:
	.stabs	"_Z9my_assertPKci:F(0,1)",36,0,38,__Z9my_assertPKci
	.stabs	"s:p(0,56)",160,0,38,104
	.stabs	"d:p(0,10)",160,0,38,108
	.stabs	":t(0,56)=*(0,57)",128,0,0,0
	.stabs	":t(0,57)=k(0,33)",128,0,0,0
Lscope5:
	.stabs	"",36,0,0,Lscope5-__Z9my_assertPKci
	.stabd	78,0,0
	.section __TEXT,__textcoal_nt,coalesced,pure_instructions
	.align 2
	.globl __Z10_my_assertbPKci
	.weak_definition __Z10_my_assertbPKci
__Z10_my_assertbPKci:
	.stabd	46,0,0
	.stabs	"../../kame/support.h",132,0,0,Ltext5
Ltext5:
	.stabd	68,0,84
LFB2:
	mflr r0
LCFI24:
	stmw r30,-8(r1)
LCFI25:
	stw r0,8(r1)
LCFI26:
	stwu r1,-80(r1)
LCFI27:
	mr r30,r1
LCFI28:
	stw r3,104(r30)
	stw r4,108(r30)
	stw r5,112(r30)
LBB9:
	.stabd	68,0,85
	lwz r0,104(r30)
	xori r0,r0,1
	cmpwi cr7,r0,0
	beq cr7,L21
	lwz r3,108(r30)
	lwz r4,112(r30)
	bl __Z9my_assertPKci
L21:
LBE9:
	.stabd	68,0,86
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r30,-8(r1)
	blr
LFE2:
	.stabs	"_Z10_my_assertbPKci:F(0,1)",36,0,84,__Z10_my_assertbPKci
	.stabs	"var:p(0,2)",160,0,84,104
	.stabs	"file:p(0,56)",160,0,84,108
	.stabs	"line:p(0,10)",160,0,84,112
Lscope6:
	.stabs	"",36,0,0,Lscope6-__Z10_my_assertbPKci
	.stabd	78,0,0
	.cstring
	.align 2
LC1:
	.ascii "../../tools/testatomic.cpp\0"
	.section __TEXT,__textcoal_nt,coalesced,pure_instructions
	.align 2
	.globl __Z20atomicCompareAndSet2jjjjPj
	.weak_definition __Z20atomicCompareAndSet2jjjjPj
__Z20atomicCompareAndSet2jjjjPj:
	.stabd	46,0,0
	.stabs	"../../tools/testatomic.cpp",132,0,0,Ltext6
Ltext6:
	.stabd	68,0,19
LFB1382:
	mflr r0
LCFI29:
	stmw r30,-8(r1)
LCFI30:
	stw r0,8(r1)
LCFI31:
	stwu r1,-80(r1)
LCFI32:
	mr r30,r1
LCFI33:
	bcl 20,31,"L00000000002$pb"
"L00000000002$pb":
	mflr r31
	stw r3,104(r30)
	stw r4,108(r30)
	stw r5,112(r30)
	stw r6,116(r30)
	stw r7,120(r30)
LBB10:
	.stabd	68,0,20
	lwz r2,120(r30)
	lwz r2,0(r2)
	lwz r0,104(r30)
	cmpw cr7,r2,r0
	mfcr r0
	rlwinm r0,r0,31,1
	mr r3,r0
	addis r2,r31,ha16(LC1-"L00000000002$pb")
	la r4,lo16(LC1-"L00000000002$pb")(r2)
	li r5,20
	bl L__Z10_my_assertbPKci$stub
	.stabd	68,0,21
	lwz r2,120(r30)
	addi r2,r2,4
	lwz r2,0(r2)
	lwz r0,108(r30)
	cmpw cr7,r2,r0
	mfcr r0
	rlwinm r0,r0,31,1
	mr r3,r0
	addis r2,r31,ha16(LC1-"L00000000002$pb")
	la r4,lo16(LC1-"L00000000002$pb")(r2)
	li r5,21
	bl L__Z10_my_assertbPKci$stub
	.stabd	68,0,22
	bl L_rand$stub
	mr r2,r3
	lis r0,0x3fff
	ori r0,r0,65535
	cmpw cr7,r2,r0
	mfcr r0
	rlwinm r0,r0,30,1
	cmpwi cr7,r0,0
	beq cr7,L23
	.stabd	68,0,23
	lwz r2,120(r30)
	lwz r0,112(r30)
	stw r0,0(r2)
	.stabd	68,0,24
	lwz r2,120(r30)
	addi r2,r2,4
	lwz r0,116(r30)
	stw r0,0(r2)
	.stabd	68,0,25
	li r0,1
	stw r0,56(r30)
	b L25
L23:
	.stabd	68,0,27
	li r0,0
	stw r0,56(r30)
L25:
	lwz r0,56(r30)
LBE10:
	.stabd	68,0,28
	mr r3,r0
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r30,-8(r1)
	blr
LFE1382:
	.stabs	"_Z20atomicCompareAndSet2jjjjPj:F(0,2)",36,0,19,__Z20atomicCompareAndSet2jjjjPj
	.stabs	"oldv0:p(0,58)",160,0,19,104
	.stabs	"oldv1:p(0,58)",160,0,19,108
	.stabs	"newv0:p(0,58)",160,0,19,112
	.stabs	"newv1:p(0,58)",160,0,19,116
	.stabs	"target:p(0,60)",160,0,19,120
	.stabs	"unsigned int:t(0,59)=r(0,59);0;037777777777;",128,0,0,0
	.stabs	"uint32_t:t(0,58)=(0,59)",128,0,0,0
	.stabs	":t(0,60)=*(0,58)",128,0,0,0
Lscope7:
	.stabs	"",36,0,0,Lscope7-__Z20atomicCompareAndSet2jjjjPj
	.stabd	78,0,0
	.cstring
	.align 2
LC2:
	.ascii "../../kame/thread.cpp\0"
	.text
	.align 2
	.globl __ZN10XCondition9broadcastEv
__ZN10XCondition9broadcastEv:
	.stabd	46,0,0
	.stabs	"../../kame/thread.cpp",132,0,0,Ltext7
Ltext7:
	.stabd	68,0,169
LFB1481:
	mflr r0
LCFI34:
	stmw r30,-8(r1)
LCFI35:
	stw r0,8(r1)
LCFI36:
	stwu r1,-96(r1)
LCFI37:
	mr r30,r1
LCFI38:
	bcl 20,31,"L00000000003$pb"
"L00000000003$pb":
	mflr r31
	stw r3,120(r30)
LBB11:
LBB12:
	.stabd	68,0,171
	lwz r2,120(r30)
	addi r0,r2,28
	mr r3,r0
	bl L_pthread_mutex_lock$stub
	.stabd	68,0,172
	lwz r0,120(r30)
	mr r3,r0
	bl L_pthread_cond_broadcast$stub
	mr r0,r3
	stw r0,56(r30)
	.stabd	68,0,173
	lwz r2,120(r30)
	addi r0,r2,28
	mr r3,r0
	bl L_pthread_mutex_unlock$stub
	.stabd	68,0,174
	lwz r0,56(r30)
	cmpwi cr7,r0,0
	mfcr r0
	rlwinm r0,r0,31,1
	mr r3,r0
	addis r2,r31,ha16(LC2-"L00000000003$pb")
	la r4,lo16(LC2-"L00000000003$pb")(r2)
	li r5,174
	bl L__Z10_my_assertbPKci$stub
LBE12:
LBE11:
	.stabd	68,0,175
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r30,-8(r1)
	blr
LFE1481:
	.stabs	"_ZN10XCondition9broadcastEv:F(0,1)",36,0,169,__ZN10XCondition9broadcastEv
	.stabs	"this:p(0,38)",160,0,169,120
	.stabs	"ret:(0,10)",128,0,172,56
	.stabn	192,0,0,LBB12
	.stabn	224,0,0,LBE12
Lscope8:
	.stabs	"",36,0,0,Lscope8-__ZN10XCondition9broadcastEv
	.stabd	78,0,0
	.align 2
	.globl __ZN10XCondition6signalEv
__ZN10XCondition6signalEv:
	.stabd	46,0,0
	.stabd	68,0,161
LFB1480:
	mflr r0
LCFI39:
	stmw r30,-8(r1)
LCFI40:
	stw r0,8(r1)
LCFI41:
	stwu r1,-96(r1)
LCFI42:
	mr r30,r1
LCFI43:
	bcl 20,31,"L00000000004$pb"
"L00000000004$pb":
	mflr r31
	stw r3,120(r30)
LBB13:
LBB14:
	.stabd	68,0,163
	lwz r2,120(r30)
	addi r0,r2,28
	mr r3,r0
	bl L_pthread_mutex_lock$stub
	.stabd	68,0,164
	lwz r0,120(r30)
	mr r3,r0
	bl L_pthread_cond_signal$stub
	mr r0,r3
	stw r0,56(r30)
	.stabd	68,0,165
	lwz r2,120(r30)
	addi r0,r2,28
	mr r3,r0
	bl L_pthread_mutex_unlock$stub
	.stabd	68,0,166
	lwz r0,56(r30)
	cmpwi cr7,r0,0
	mfcr r0
	rlwinm r0,r0,31,1
	mr r3,r0
	addis r2,r31,ha16(LC2-"L00000000004$pb")
	la r4,lo16(LC2-"L00000000004$pb")(r2)
	li r5,166
	bl L__Z10_my_assertbPKci$stub
LBE14:
LBE13:
	.stabd	68,0,167
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r30,-8(r1)
	blr
LFE1480:
	.stabs	"_ZN10XCondition6signalEv:F(0,1)",36,0,161,__ZN10XCondition6signalEv
	.stabs	"this:p(0,38)",160,0,161,120
	.stabs	"ret:(0,10)",128,0,164,56
	.stabn	192,0,0,LBB14
	.stabn	224,0,0,LBE14
Lscope9:
	.stabs	"",36,0,0,Lscope9-__ZN10XCondition6signalEv
	.stabd	78,0,0
	.align 2
	.globl __ZN10XConditionD1Ev
__ZN10XConditionD1Ev:
	.stabd	46,0,0
	.stabd	68,0,111
LFB1478:
	mflr r0
LCFI44:
	stmw r30,-8(r1)
LCFI45:
	stw r0,8(r1)
LCFI46:
	stwu r1,-96(r1)
LCFI47:
	mr r30,r1
LCFI48:
	bcl 20,31,"L00000000005$pb"
"L00000000005$pb":
	mflr r31
	stw r3,120(r30)
LBB15:
	.stabd	68,0,113
	lwz r0,120(r30)
	mr r3,r0
	bl L_pthread_cond_destroy$stub
	mr r0,r3
	stw r0,56(r30)
	.stabd	68,0,114
	lwz r0,56(r30)
	cmpwi cr7,r0,0
	mfcr r0
	rlwinm r0,r0,31,1
	mr r3,r0
	addis r2,r31,ha16(LC2-"L00000000005$pb")
	la r4,lo16(LC2-"L00000000005$pb")(r2)
	li r5,114
	bl L__Z10_my_assertbPKci$stub
	.stabd	68,0,115
	lwz r2,120(r30)
	addi r0,r2,28
	mr r3,r0
	bl L_pthread_mutex_destroy$stub
	mr r0,r3
	stw r0,56(r30)
	.stabd	68,0,116
	lwz r0,56(r30)
	cmpwi cr7,r0,0
	mfcr r0
	rlwinm r0,r0,31,1
	mr r3,r0
	addis r2,r31,ha16(LC2-"L00000000005$pb")
	la r4,lo16(LC2-"L00000000005$pb")(r2)
	li r5,116
	bl L__Z10_my_assertbPKci$stub
LBE15:
	.stabd	68,0,117
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r30,-8(r1)
	blr
LFE1478:
	.stabs	"_ZN10XConditionD1Ev:F(0,1)",36,0,111,__ZN10XConditionD1Ev
	.stabs	"this:p(0,38)",160,0,111,120
	.stabs	"ret:(0,10)",128,0,113,56
	.stabn	192,0,0,LBB15
	.stabn	224,0,0,LBE15
Lscope10:
	.stabs	"",36,0,0,Lscope10-__ZN10XConditionD1Ev
	.stabd	78,0,0
	.align 2
	.globl __ZN10XConditionD2Ev
__ZN10XConditionD2Ev:
	.stabd	46,0,0
	.stabd	68,0,111
LFB1477:
	mflr r0
LCFI49:
	stmw r30,-8(r1)
LCFI50:
	stw r0,8(r1)
LCFI51:
	stwu r1,-96(r1)
LCFI52:
	mr r30,r1
LCFI53:
	bcl 20,31,"L00000000006$pb"
"L00000000006$pb":
	mflr r31
	stw r3,120(r30)
LBB16:
	.stabd	68,0,113
	lwz r0,120(r30)
	mr r3,r0
	bl L_pthread_cond_destroy$stub
	mr r0,r3
	stw r0,56(r30)
	.stabd	68,0,114
	lwz r0,56(r30)
	cmpwi cr7,r0,0
	mfcr r0
	rlwinm r0,r0,31,1
	mr r3,r0
	addis r2,r31,ha16(LC2-"L00000000006$pb")
	la r4,lo16(LC2-"L00000000006$pb")(r2)
	li r5,114
	bl L__Z10_my_assertbPKci$stub
	.stabd	68,0,115
	lwz r2,120(r30)
	addi r0,r2,28
	mr r3,r0
	bl L_pthread_mutex_destroy$stub
	mr r0,r3
	stw r0,56(r30)
	.stabd	68,0,116
	lwz r0,56(r30)
	cmpwi cr7,r0,0
	mfcr r0
	rlwinm r0,r0,31,1
	mr r3,r0
	addis r2,r31,ha16(LC2-"L00000000006$pb")
	la r4,lo16(LC2-"L00000000006$pb")(r2)
	li r5,116
	bl L__Z10_my_assertbPKci$stub
LBE16:
	.stabd	68,0,117
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r30,-8(r1)
	blr
LFE1477:
	.stabs	"_ZN10XConditionD2Ev:F(0,1)",36,0,111,__ZN10XConditionD2Ev
	.stabs	"this:p(0,38)",160,0,111,120
	.stabs	"ret:(0,10)",128,0,113,56
	.stabn	192,0,0,LBB16
	.stabn	224,0,0,LBE16
Lscope11:
	.stabs	"",36,0,0,Lscope11-__ZN10XConditionD2Ev
	.stabd	78,0,0
	.align 2
	.globl __ZN10XConditionC1Ev
__ZN10XConditionC1Ev:
	.stabd	46,0,0
	.stabd	68,0,104
LFB1475:
	mflr r0
LCFI54:
	stmw r30,-8(r1)
LCFI55:
	stw r0,8(r1)
LCFI56:
	stwu r1,-96(r1)
LCFI57:
	mr r30,r1
LCFI58:
	bcl 20,31,"L00000000007$pb"
"L00000000007$pb":
	mflr r31
	stw r3,120(r30)
LBB17:
LBB18:
	.stabd	68,0,106
	lwz r2,120(r30)
	addi r0,r2,28
	mr r3,r0
	li r4,0
	bl L_pthread_mutex_init$stub
	mr r0,r3
	stw r0,56(r30)
	.stabd	68,0,107
	lwz r0,56(r30)
	cmpwi cr7,r0,0
	mfcr r0
	rlwinm r0,r0,31,1
	mr r3,r0
	addis r2,r31,ha16(LC2-"L00000000007$pb")
	la r4,lo16(LC2-"L00000000007$pb")(r2)
	li r5,107
	bl L__Z10_my_assertbPKci$stub
	.stabd	68,0,108
	lwz r0,120(r30)
	mr r3,r0
	li r4,0
	bl L_pthread_cond_init$stub
	mr r0,r3
	stw r0,56(r30)
	.stabd	68,0,109
	lwz r0,56(r30)
	cmpwi cr7,r0,0
	mfcr r0
	rlwinm r0,r0,31,1
	mr r3,r0
	addis r2,r31,ha16(LC2-"L00000000007$pb")
	la r4,lo16(LC2-"L00000000007$pb")(r2)
	li r5,109
	bl L__Z10_my_assertbPKci$stub
LBE18:
LBE17:
	.stabd	68,0,110
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r30,-8(r1)
	blr
LFE1475:
	.stabs	"_ZN10XConditionC1Ev:F(0,1)",36,0,104,__ZN10XConditionC1Ev
	.stabs	"this:p(0,38)",160,0,104,120
	.stabs	"ret:(0,10)",128,0,106,56
	.stabn	192,0,0,LBB18
	.stabn	224,0,0,LBE18
Lscope12:
	.stabs	"",36,0,0,Lscope12-__ZN10XConditionC1Ev
	.stabd	78,0,0
	.align 2
	.globl __ZN10XConditionC2Ev
__ZN10XConditionC2Ev:
	.stabd	46,0,0
	.stabd	68,0,104
LFB1474:
	mflr r0
LCFI59:
	stmw r30,-8(r1)
LCFI60:
	stw r0,8(r1)
LCFI61:
	stwu r1,-96(r1)
LCFI62:
	mr r30,r1
LCFI63:
	bcl 20,31,"L00000000008$pb"
"L00000000008$pb":
	mflr r31
	stw r3,120(r30)
LBB19:
LBB20:
	.stabd	68,0,106
	lwz r2,120(r30)
	addi r0,r2,28
	mr r3,r0
	li r4,0
	bl L_pthread_mutex_init$stub
	mr r0,r3
	stw r0,56(r30)
	.stabd	68,0,107
	lwz r0,56(r30)
	cmpwi cr7,r0,0
	mfcr r0
	rlwinm r0,r0,31,1
	mr r3,r0
	addis r2,r31,ha16(LC2-"L00000000008$pb")
	la r4,lo16(LC2-"L00000000008$pb")(r2)
	li r5,107
	bl L__Z10_my_assertbPKci$stub
	.stabd	68,0,108
	lwz r0,120(r30)
	mr r3,r0
	li r4,0
	bl L_pthread_cond_init$stub
	mr r0,r3
	stw r0,56(r30)
	.stabd	68,0,109
	lwz r0,56(r30)
	cmpwi cr7,r0,0
	mfcr r0
	rlwinm r0,r0,31,1
	mr r3,r0
	addis r2,r31,ha16(LC2-"L00000000008$pb")
	la r4,lo16(LC2-"L00000000008$pb")(r2)
	li r5,109
	bl L__Z10_my_assertbPKci$stub
LBE20:
LBE19:
	.stabd	68,0,110
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r30,-8(r1)
	blr
LFE1474:
	.stabs	"_ZN10XConditionC2Ev:F(0,1)",36,0,104,__ZN10XConditionC2Ev
	.stabs	"this:p(0,38)",160,0,104,120
	.stabs	"ret:(0,10)",128,0,106,56
	.stabn	192,0,0,LBB20
	.stabn	224,0,0,LBE20
Lscope13:
	.stabs	"",36,0,0,Lscope13-__ZN10XConditionC2Ev
	.stabd	78,0,0
	.align 2
	.globl __ZN6XMutex6unlockEv
__ZN6XMutex6unlockEv:
	.stabd	46,0,0
	.stabd	68,0,42
LFB1462:
	mflr r0
LCFI64:
	stmw r30,-8(r1)
LCFI65:
	stw r0,8(r1)
LCFI66:
	stwu r1,-96(r1)
LCFI67:
	mr r30,r1
LCFI68:
	bcl 20,31,"L00000000009$pb"
"L00000000009$pb":
	mflr r31
	stw r3,120(r30)
LBB21:
LBB22:
	.stabd	68,0,43
	lwz r0,120(r30)
	mr r3,r0
	bl L_pthread_mutex_unlock$stub
	mr r0,r3
	stw r0,56(r30)
	.stabd	68,0,44
	lwz r0,56(r30)
	cmpwi cr7,r0,0
	mfcr r0
	rlwinm r0,r0,31,1
	mr r3,r0
	addis r2,r31,ha16(LC2-"L00000000009$pb")
	la r4,lo16(LC2-"L00000000009$pb")(r2)
	li r5,44
	bl L__Z10_my_assertbPKci$stub
LBE22:
LBE21:
	.stabd	68,0,45
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r30,-8(r1)
	blr
LFE1462:
	.stabs	"_ZN6XMutex6unlockEv:F(0,1)",36,0,42,__ZN6XMutex6unlockEv
	.stabs	"this:p(0,61)",160,0,42,120
	.stabs	"ret:(0,10)",128,0,43,56
	.stabs	":t(0,61)=k(0,20)",128,0,0,0
	.stabn	192,0,0,LBB22
	.stabn	224,0,0,LBE22
Lscope14:
	.stabs	"",36,0,0,Lscope14-__ZN6XMutex6unlockEv
	.stabd	78,0,0
	.align 2
	.globl __ZN15XRecursiveMutex6unlockEv
__ZN15XRecursiveMutex6unlockEv:
	.stabd	46,0,0
	.stabd	68,0,89
LFB1471:
	mflr r0
LCFI69:
	stmw r29,-12(r1)
LCFI70:
	stw r0,8(r1)
LCFI71:
	stwu r1,-80(r1)
LCFI72:
	mr r30,r1
LCFI73:
	bcl 20,31,"L00000000010$pb"
"L00000000010$pb":
	mflr r31
	stw r3,104(r30)
LBB23:
	.stabd	68,0,91
	lwz r2,104(r30)
	lwz r29,44(r2)
	bl L_pthread_self$stub
	mr r0,r3
	mr r3,r29
	mr r4,r0
	bl L_pthread_equal$stub
	mr r0,r3
	srawi r2,r0,31
	xor r0,r2,r0
	subf r0,r2,r0
	neg r0,r0
	srwi r0,r0,31
	mr r3,r0
	addis r2,r31,ha16(LC2-"L00000000010$pb")
	la r4,lo16(LC2-"L00000000010$pb")(r2)
	li r5,91
	bl L__Z10_my_assertbPKci$stub
	.stabd	68,0,92
	lwz r2,104(r30)
	lwz r2,48(r2)
	addi r0,r2,-1
	lwz r2,104(r30)
	stw r0,48(r2)
	.stabd	68,0,93
	lwz r2,104(r30)
	lwz r0,48(r2)
	cmpwi cr7,r0,0
	bne cr7,L46
	.stabd	68,0,95
	lwz r2,104(r30)
	li r0,-1
	stw r0,44(r2)
	.stabd	68,0,96
	lwz r0,104(r30)
	mr r3,r0
	bl __ZN6XMutex6unlockEv
L46:
LBE23:
	.stabd	68,0,98
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r29,-12(r1)
	blr
LFE1471:
	.stabs	"_ZN15XRecursiveMutex6unlockEv:F(0,1)",36,0,89,__ZN15XRecursiveMutex6unlockEv
	.stabs	"this:p(0,62)",160,0,89,104
	.stabs	":t(0,62)=k(0,12)",128,0,0,0
Lscope15:
	.stabs	"",36,0,0,Lscope15-__ZN15XRecursiveMutex6unlockEv
	.stabd	78,0,0
	.align 2
	.globl __ZN6XMutex7trylockEv
__ZN6XMutex7trylockEv:
	.stabd	46,0,0
	.stabd	68,0,36
LFB1461:
	mflr r0
LCFI74:
	stmw r30,-8(r1)
LCFI75:
	stw r0,8(r1)
LCFI76:
	stwu r1,-96(r1)
LCFI77:
	mr r30,r1
LCFI78:
	bcl 20,31,"L00000000011$pb"
"L00000000011$pb":
	mflr r31
	stw r3,120(r30)
LBB24:
LBB25:
	.stabd	68,0,37
	lwz r0,120(r30)
	mr r3,r0
	bl L_pthread_mutex_trylock$stub
	mr r0,r3
	stw r0,56(r30)
	.stabd	68,0,38
	lwz r0,56(r30)
	xori r0,r0,22
	srawi r2,r0,31
	xor r0,r2,r0
	subf r0,r2,r0
	neg r0,r0
	srwi r0,r0,31
	mr r3,r0
	addis r2,r31,ha16(LC2-"L00000000011$pb")
	la r4,lo16(LC2-"L00000000011$pb")(r2)
	li r5,38
	bl L__Z10_my_assertbPKci$stub
	.stabd	68,0,39
	lwz r0,56(r30)
	cmpwi cr7,r0,0
	mfcr r0
	rlwinm r0,r0,31,1
LBE25:
LBE24:
	.stabd	68,0,40
	mr r3,r0
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r30,-8(r1)
	blr
LFE1461:
	.stabs	"_ZN6XMutex7trylockEv:F(0,2)",36,0,36,__ZN6XMutex7trylockEv
	.stabs	"this:p(0,61)",160,0,36,120
	.stabs	"ret:(0,10)",128,0,37,56
	.stabn	192,0,0,LBB25
	.stabn	224,0,0,LBE25
Lscope16:
	.stabs	"",36,0,0,Lscope16-__ZN6XMutex7trylockEv
	.stabd	78,0,0
	.align 2
	.globl __ZN15XRecursiveMutex7trylockEv
__ZN15XRecursiveMutex7trylockEv:
	.stabd	46,0,0
	.stabd	68,0,70
LFB1470:
	mflr r0
LCFI79:
	stmw r29,-12(r1)
LCFI80:
	stw r0,8(r1)
LCFI81:
	stwu r1,-80(r1)
LCFI82:
	mr r30,r1
LCFI83:
	stw r3,104(r30)
LBB26:
	.stabd	68,0,72
	lwz r2,104(r30)
	lwz r29,44(r2)
	bl L_pthread_self$stub
	mr r0,r3
	mr r3,r29
	mr r4,r0
	bl L_pthread_equal$stub
	mr r0,r3
	cmpwi cr7,r0,0
	mfcr r0
	rlwinm r0,r0,31,1
	cmpwi cr7,r0,0
	beq cr7,L50
	.stabd	68,0,74
	lwz r0,104(r30)
	mr r3,r0
	bl __ZN6XMutex7trylockEv
	mr r0,r3
	cmpwi cr7,r0,0
	beq cr7,L52
	.stabd	68,0,75
	lwz r2,104(r30)
	li r0,1
	stw r0,48(r2)
	.stabd	68,0,76
	bl L_pthread_self$stub
	mr r0,r3
	lwz r2,104(r30)
	stw r0,44(r2)
	b L54
L52:
	.stabd	68,0,79
	li r0,0
	stw r0,56(r30)
	b L55
L50:
	.stabd	68,0,84
	lwz r2,104(r30)
	lwz r2,48(r2)
	addi r0,r2,1
	lwz r2,104(r30)
	stw r0,48(r2)
L54:
	.stabd	68,0,86
	li r0,1
	stw r0,56(r30)
L55:
	lwz r0,56(r30)
LBE26:
	.stabd	68,0,87
	mr r3,r0
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r29,-12(r1)
	blr
LFE1470:
	.stabs	"_ZN15XRecursiveMutex7trylockEv:F(0,2)",36,0,70,__ZN15XRecursiveMutex7trylockEv
	.stabs	"this:p(0,62)",160,0,70,104
Lscope17:
	.stabs	"",36,0,0,Lscope17-__ZN15XRecursiveMutex7trylockEv
	.stabd	78,0,0
	.align 2
	.globl __ZN6XMutex4lockEv
__ZN6XMutex4lockEv:
	.stabd	46,0,0
	.stabd	68,0,31
LFB1460:
	mflr r0
LCFI84:
	stmw r30,-8(r1)
LCFI85:
	stw r0,8(r1)
LCFI86:
	stwu r1,-96(r1)
LCFI87:
	mr r30,r1
LCFI88:
	bcl 20,31,"L00000000012$pb"
"L00000000012$pb":
	mflr r31
	stw r3,120(r30)
LBB27:
LBB28:
	.stabd	68,0,32
	lwz r0,120(r30)
	mr r3,r0
	bl L_pthread_mutex_lock$stub
	mr r0,r3
	stw r0,56(r30)
	.stabd	68,0,33
	lwz r0,56(r30)
	cmpwi cr7,r0,0
	mfcr r0
	rlwinm r0,r0,31,1
	mr r3,r0
	addis r2,r31,ha16(LC2-"L00000000012$pb")
	la r4,lo16(LC2-"L00000000012$pb")(r2)
	li r5,33
	bl L__Z10_my_assertbPKci$stub
LBE28:
LBE27:
	.stabd	68,0,34
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r30,-8(r1)
	blr
LFE1460:
	.stabs	"_ZN6XMutex4lockEv:F(0,1)",36,0,31,__ZN6XMutex4lockEv
	.stabs	"this:p(0,61)",160,0,31,120
	.stabs	"ret:(0,10)",128,0,32,56
	.stabn	192,0,0,LBB28
	.stabn	224,0,0,LBE28
Lscope18:
	.stabs	"",36,0,0,Lscope18-__ZN6XMutex4lockEv
	.stabd	78,0,0
	.align 2
	.globl __ZN15XRecursiveMutex4lockEv
__ZN15XRecursiveMutex4lockEv:
	.stabd	46,0,0
	.stabd	68,0,56
LFB1469:
	mflr r0
LCFI89:
	stmw r29,-12(r1)
LCFI90:
	stw r0,8(r1)
LCFI91:
	stwu r1,-80(r1)
LCFI92:
	mr r30,r1
LCFI93:
	stw r3,104(r30)
LBB29:
	.stabd	68,0,58
	lwz r2,104(r30)
	lwz r29,44(r2)
	bl L_pthread_self$stub
	mr r0,r3
	mr r3,r29
	mr r4,r0
	bl L_pthread_equal$stub
	mr r0,r3
	cmpwi cr7,r0,0
	mfcr r0
	rlwinm r0,r0,31,1
	cmpwi cr7,r0,0
	beq cr7,L60
	.stabd	68,0,60
	lwz r0,104(r30)
	mr r3,r0
	bl __ZN6XMutex4lockEv
	.stabd	68,0,61
	lwz r2,104(r30)
	li r0,1
	stw r0,48(r2)
	.stabd	68,0,62
	bl L_pthread_self$stub
	mr r0,r3
	lwz r2,104(r30)
	stw r0,44(r2)
	b L63
L60:
	.stabd	68,0,66
	lwz r2,104(r30)
	lwz r2,48(r2)
	addi r0,r2,1
	lwz r2,104(r30)
	stw r0,48(r2)
L63:
LBE29:
	.stabd	68,0,68
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r29,-12(r1)
	blr
LFE1469:
	.stabs	"_ZN15XRecursiveMutex4lockEv:F(0,1)",36,0,56,__ZN15XRecursiveMutex4lockEv
	.stabs	"this:p(0,62)",160,0,56,104
Lscope19:
	.stabs	"",36,0,0,Lscope19-__ZN15XRecursiveMutex4lockEv
	.stabd	78,0,0
	.align 2
	.globl __ZN6XMutexD1Ev
__ZN6XMutexD1Ev:
	.stabd	46,0,0
	.stabd	68,0,25
LFB1459:
	mflr r0
LCFI94:
	stmw r30,-8(r1)
LCFI95:
	stw r0,8(r1)
LCFI96:
	stwu r1,-96(r1)
LCFI97:
	mr r30,r1
LCFI98:
	bcl 20,31,"L00000000013$pb"
"L00000000013$pb":
	mflr r31
	stw r3,120(r30)
LBB30:
	.stabd	68,0,27
	lwz r0,120(r30)
	mr r3,r0
	bl L_pthread_mutex_destroy$stub
	mr r0,r3
	stw r0,56(r30)
	.stabd	68,0,28
	lwz r0,56(r30)
	cmpwi cr7,r0,0
	mfcr r0
	rlwinm r0,r0,31,1
	mr r3,r0
	addis r2,r31,ha16(LC2-"L00000000013$pb")
	la r4,lo16(LC2-"L00000000013$pb")(r2)
	li r5,28
	bl L__Z10_my_assertbPKci$stub
LBE30:
	.stabd	68,0,29
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r30,-8(r1)
	blr
LFE1459:
	.stabs	"_ZN6XMutexD1Ev:F(0,1)",36,0,25,__ZN6XMutexD1Ev
	.stabs	"this:p(0,61)",160,0,25,120
	.stabs	"ret:(0,10)",128,0,27,56
	.stabn	192,0,0,LBB30
	.stabn	224,0,0,LBE30
Lscope20:
	.stabs	"",36,0,0,Lscope20-__ZN6XMutexD1Ev
	.stabd	78,0,0
	.align 2
	.globl __ZN15XRecursiveMutexD1Ev
__ZN15XRecursiveMutexD1Ev:
	.stabd	46,0,0
	.stabd	68,0,51
LFB1468:
	mflr r0
LCFI99:
	stmw r29,-12(r1)
LCFI100:
	stw r0,8(r1)
LCFI101:
	stwu r1,-80(r1)
LCFI102:
	mr r30,r1
LCFI103:
	bcl 20,31,"L00000000014$pb"
"L00000000014$pb":
	mflr r31
	stw r3,104(r30)
	.stabd	68,0,53
	lwz r2,104(r30)
	lwz r2,44(r2)
	li r0,-1
	cmpw cr7,r2,r0
	mfcr r0
	rlwinm r0,r0,31,1
	mr r3,r0
	addis r2,r31,ha16(LC2-"L00000000014$pb")
	la r4,lo16(LC2-"L00000000014$pb")(r2)
	li r5,53
LEHB0:
	bl L__Z10_my_assertbPKci$stub
LEHE0:
	b L68
L72:
	stw r3,60(r30)
L69:
	lwz r29,60(r30)
	.stabd	68,0,54
	lwz r0,104(r30)
	stw r0,56(r30)
	lwz r3,56(r30)
	bl __ZN6XMutexD1Ev
	stw r29,60(r30)
	lwz r3,60(r30)
LEHB1:
	bl L__Unwind_Resume$stub
L68:
	lwz r0,104(r30)
	stw r0,56(r30)
	lwz r3,56(r30)
	bl __ZN6XMutexD1Ev
LEHE1:
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r29,-12(r1)
	blr
LFE1468:
	.section __DATA,__gcc_except_tab
LLSDA1468:
	.byte	0xff
	.byte	0xff
	.byte	0x3
	.byte	0x1a
	.set L$set$0,LEHB0-LFB1468
	.long L$set$0
	.set L$set$1,LEHE0-LEHB0
	.long L$set$1
	.set L$set$2,L72-LFB1468
	.long L$set$2
	.byte	0x0
	.set L$set$3,LEHB1-LFB1468
	.long L$set$3
	.set L$set$4,LEHE1-LEHB1
	.long L$set$4
	.long	0x0
	.byte	0x0
	.text
	.stabs	"_ZN15XRecursiveMutexD1Ev:F(0,1)",36,0,51,__ZN15XRecursiveMutexD1Ev
	.stabs	"this:p(0,62)",160,0,51,104
Lscope21:
	.stabs	"",36,0,0,Lscope21-__ZN15XRecursiveMutexD1Ev
	.stabd	78,0,0
	.align 2
	.globl __ZN15XRecursiveMutexD2Ev
__ZN15XRecursiveMutexD2Ev:
	.stabd	46,0,0
	.stabd	68,0,51
LFB1467:
	mflr r0
LCFI104:
	stmw r29,-12(r1)
LCFI105:
	stw r0,8(r1)
LCFI106:
	stwu r1,-80(r1)
LCFI107:
	mr r30,r1
LCFI108:
	bcl 20,31,"L00000000015$pb"
"L00000000015$pb":
	mflr r31
	stw r3,104(r30)
	.stabd	68,0,53
	lwz r2,104(r30)
	lwz r2,44(r2)
	li r0,-1
	cmpw cr7,r2,r0
	mfcr r0
	rlwinm r0,r0,31,1
	mr r3,r0
	addis r2,r31,ha16(LC2-"L00000000015$pb")
	la r4,lo16(LC2-"L00000000015$pb")(r2)
	li r5,53
LEHB2:
	bl L__Z10_my_assertbPKci$stub
LEHE2:
	b L74
L78:
	stw r3,60(r30)
L75:
	lwz r29,60(r30)
	.stabd	68,0,54
	lwz r0,104(r30)
	stw r0,56(r30)
	lwz r3,56(r30)
	bl __ZN6XMutexD1Ev
	stw r29,60(r30)
	lwz r3,60(r30)
LEHB3:
	bl L__Unwind_Resume$stub
L74:
	lwz r0,104(r30)
	stw r0,56(r30)
	lwz r3,56(r30)
	bl __ZN6XMutexD1Ev
LEHE3:
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r29,-12(r1)
	blr
LFE1467:
	.section __DATA,__gcc_except_tab
LLSDA1467:
	.byte	0xff
	.byte	0xff
	.byte	0x3
	.byte	0x1a
	.set L$set$5,LEHB2-LFB1467
	.long L$set$5
	.set L$set$6,LEHE2-LEHB2
	.long L$set$6
	.set L$set$7,L78-LFB1467
	.long L$set$7
	.byte	0x0
	.set L$set$8,LEHB3-LFB1467
	.long L$set$8
	.set L$set$9,LEHE3-LEHB3
	.long L$set$9
	.long	0x0
	.byte	0x0
	.text
	.stabs	"_ZN15XRecursiveMutexD2Ev:F(0,1)",36,0,51,__ZN15XRecursiveMutexD2Ev
	.stabs	"this:p(0,62)",160,0,51,104
Lscope22:
	.stabs	"",36,0,0,Lscope22-__ZN15XRecursiveMutexD2Ev
	.stabd	78,0,0
	.align 2
	.globl __ZN6XMutexD2Ev
__ZN6XMutexD2Ev:
	.stabd	46,0,0
	.stabd	68,0,25
LFB1458:
	mflr r0
LCFI109:
	stmw r30,-8(r1)
LCFI110:
	stw r0,8(r1)
LCFI111:
	stwu r1,-96(r1)
LCFI112:
	mr r30,r1
LCFI113:
	bcl 20,31,"L00000000016$pb"
"L00000000016$pb":
	mflr r31
	stw r3,120(r30)
LBB31:
	.stabd	68,0,27
	lwz r0,120(r30)
	mr r3,r0
	bl L_pthread_mutex_destroy$stub
	mr r0,r3
	stw r0,56(r30)
	.stabd	68,0,28
	lwz r0,56(r30)
	cmpwi cr7,r0,0
	mfcr r0
	rlwinm r0,r0,31,1
	mr r3,r0
	addis r2,r31,ha16(LC2-"L00000000016$pb")
	la r4,lo16(LC2-"L00000000016$pb")(r2)
	li r5,28
	bl L__Z10_my_assertbPKci$stub
LBE31:
	.stabd	68,0,29
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r30,-8(r1)
	blr
LFE1458:
	.stabs	"_ZN6XMutexD2Ev:F(0,1)",36,0,25,__ZN6XMutexD2Ev
	.stabs	"this:p(0,61)",160,0,25,120
	.stabs	"ret:(0,10)",128,0,27,56
	.stabn	192,0,0,LBB31
	.stabn	224,0,0,LBE31
Lscope23:
	.stabs	"",36,0,0,Lscope23-__ZN6XMutexD2Ev
	.stabd	78,0,0
	.align 2
	.globl __ZN6XMutexC1Ev
__ZN6XMutexC1Ev:
	.stabd	46,0,0
	.stabd	68,0,11
LFB1456:
	mflr r0
LCFI114:
	stmw r30,-8(r1)
LCFI115:
	stw r0,8(r1)
LCFI116:
	stwu r1,-96(r1)
LCFI117:
	mr r30,r1
LCFI118:
	bcl 20,31,"L00000000017$pb"
"L00000000017$pb":
	mflr r31
	stw r3,120(r30)
LBB32:
LBB33:
	.stabd	68,0,15
	addi r0,r30,60
	mr r3,r0
	bl L_pthread_mutexattr_init$stub
	mr r0,r3
	stw r0,56(r30)
	.stabd	68,0,16
	lwz r0,56(r30)
	cmpwi cr7,r0,0
	mfcr r0
	rlwinm r0,r0,31,1
	mr r3,r0
	addis r2,r31,ha16(LC2-"L00000000017$pb")
	la r4,lo16(LC2-"L00000000017$pb")(r2)
	li r5,16
	bl L__Z10_my_assertbPKci$stub
	.stabd	68,0,18
	lwz r0,120(r30)
	addi r2,r30,60
	mr r3,r0
	mr r4,r2
	bl L_pthread_mutex_init$stub
	mr r0,r3
	stw r0,56(r30)
	.stabd	68,0,19
	lwz r0,56(r30)
	cmpwi cr7,r0,0
	mfcr r0
	rlwinm r0,r0,31,1
	mr r3,r0
	addis r2,r31,ha16(LC2-"L00000000017$pb")
	la r4,lo16(LC2-"L00000000017$pb")(r2)
	li r5,19
	bl L__Z10_my_assertbPKci$stub
	.stabd	68,0,21
	addi r0,r30,60
	mr r3,r0
	bl L_pthread_mutexattr_destroy$stub
	mr r0,r3
	stw r0,56(r30)
	.stabd	68,0,22
	lwz r0,56(r30)
	cmpwi cr7,r0,0
	mfcr r0
	rlwinm r0,r0,31,1
	mr r3,r0
	addis r2,r31,ha16(LC2-"L00000000017$pb")
	la r4,lo16(LC2-"L00000000017$pb")(r2)
	li r5,22
	bl L__Z10_my_assertbPKci$stub
LBE33:
LBE32:
	.stabd	68,0,23
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r30,-8(r1)
	blr
LFE1456:
	.stabs	"_ZN6XMutexC1Ev:F(0,1)",36,0,11,__ZN6XMutexC1Ev
	.stabs	"this:p(0,61)",160,0,11,120
	.stabs	"attr:(0,63)=xspthread_mutexattr_t:",128,0,13,60
	.stabs	"_opaque_pthread_mutexattr_t:Tt(0,65)=s12__sig:(0,28),0,32;__opaque:(0,66),32,64;;",128,0,70,0
	.stabs	"pthread_mutexattr_t:t(0,63)=(0,64)",128,0,89,0
	.stabs	"__darwin_pthread_mutexattr_t:t(0,64)=(0,65)",128,0,0,0
	.stabs	":t(0,66)=ar(0,32);0;7;(0,33)",128,0,0,0
	.stabs	"ret:(0,10)",128,0,14,56
	.stabn	192,0,0,LBB33
	.stabn	224,0,0,LBE33
Lscope24:
	.stabs	"",36,0,0,Lscope24-__ZN6XMutexC1Ev
	.stabd	78,0,0
	.align 2
	.globl __ZN15XRecursiveMutexC1Ev
__ZN15XRecursiveMutexC1Ev:
	.stabd	46,0,0
	.stabd	68,0,47
LFB1465:
	mflr r0
LCFI119:
	stmw r30,-8(r1)
LCFI120:
	stw r0,8(r1)
LCFI121:
	stwu r1,-80(r1)
LCFI122:
	mr r30,r1
LCFI123:
	stw r3,104(r30)
LBB34:
	.stabd	68,0,47
	lwz r0,104(r30)
	mr r3,r0
	bl __ZN6XMutexC1Ev
	.stabd	68,0,49
	lwz r2,104(r30)
	li r0,-1
	stw r0,44(r2)
LBE34:
	.stabd	68,0,50
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r30,-8(r1)
	blr
LFE1465:
	.stabs	"_ZN15XRecursiveMutexC1Ev:F(0,1)",36,0,47,__ZN15XRecursiveMutexC1Ev
	.stabs	"this:p(0,62)",160,0,47,104
Lscope25:
	.stabs	"",36,0,0,Lscope25-__ZN15XRecursiveMutexC1Ev
	.stabd	78,0,0
	.align 2
	.globl __ZN15XRecursiveMutexC2Ev
__ZN15XRecursiveMutexC2Ev:
	.stabd	46,0,0
	.stabd	68,0,47
LFB1464:
	mflr r0
LCFI124:
	stmw r30,-8(r1)
LCFI125:
	stw r0,8(r1)
LCFI126:
	stwu r1,-80(r1)
LCFI127:
	mr r30,r1
LCFI128:
	stw r3,104(r30)
LBB35:
	.stabd	68,0,47
	lwz r0,104(r30)
	mr r3,r0
	bl __ZN6XMutexC1Ev
	.stabd	68,0,49
	lwz r2,104(r30)
	li r0,-1
	stw r0,44(r2)
LBE35:
	.stabd	68,0,50
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r30,-8(r1)
	blr
LFE1464:
	.stabs	"_ZN15XRecursiveMutexC2Ev:F(0,1)",36,0,47,__ZN15XRecursiveMutexC2Ev
	.stabs	"this:p(0,62)",160,0,47,104
Lscope26:
	.stabs	"",36,0,0,Lscope26-__ZN15XRecursiveMutexC2Ev
	.stabd	78,0,0
	.align 2
	.globl __ZN6XMutexC2Ev
__ZN6XMutexC2Ev:
	.stabd	46,0,0
	.stabd	68,0,11
LFB1455:
	mflr r0
LCFI129:
	stmw r30,-8(r1)
LCFI130:
	stw r0,8(r1)
LCFI131:
	stwu r1,-96(r1)
LCFI132:
	mr r30,r1
LCFI133:
	bcl 20,31,"L00000000018$pb"
"L00000000018$pb":
	mflr r31
	stw r3,120(r30)
LBB36:
LBB37:
	.stabd	68,0,15
	addi r0,r30,60
	mr r3,r0
	bl L_pthread_mutexattr_init$stub
	mr r0,r3
	stw r0,56(r30)
	.stabd	68,0,16
	lwz r0,56(r30)
	cmpwi cr7,r0,0
	mfcr r0
	rlwinm r0,r0,31,1
	mr r3,r0
	addis r2,r31,ha16(LC2-"L00000000018$pb")
	la r4,lo16(LC2-"L00000000018$pb")(r2)
	li r5,16
	bl L__Z10_my_assertbPKci$stub
	.stabd	68,0,18
	lwz r0,120(r30)
	addi r2,r30,60
	mr r3,r0
	mr r4,r2
	bl L_pthread_mutex_init$stub
	mr r0,r3
	stw r0,56(r30)
	.stabd	68,0,19
	lwz r0,56(r30)
	cmpwi cr7,r0,0
	mfcr r0
	rlwinm r0,r0,31,1
	mr r3,r0
	addis r2,r31,ha16(LC2-"L00000000018$pb")
	la r4,lo16(LC2-"L00000000018$pb")(r2)
	li r5,19
	bl L__Z10_my_assertbPKci$stub
	.stabd	68,0,21
	addi r0,r30,60
	mr r3,r0
	bl L_pthread_mutexattr_destroy$stub
	mr r0,r3
	stw r0,56(r30)
	.stabd	68,0,22
	lwz r0,56(r30)
	cmpwi cr7,r0,0
	mfcr r0
	rlwinm r0,r0,31,1
	mr r3,r0
	addis r2,r31,ha16(LC2-"L00000000018$pb")
	la r4,lo16(LC2-"L00000000018$pb")(r2)
	li r5,22
	bl L__Z10_my_assertbPKci$stub
LBE37:
LBE36:
	.stabd	68,0,23
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r30,-8(r1)
	blr
LFE1455:
	.stabs	"_ZN6XMutexC2Ev:F(0,1)",36,0,11,__ZN6XMutexC2Ev
	.stabs	"this:p(0,61)",160,0,11,120
	.stabs	"attr:(0,63)",128,0,13,60
	.stabs	"ret:(0,10)",128,0,14,56
	.stabn	192,0,0,LBB37
	.stabn	224,0,0,LBE37
Lscope27:
	.stabs	"",36,0,0,Lscope27-__ZN6XMutexC2Ev
	.stabd	78,0,0
	.cstring
	.align 2
LC3:
	.ascii "Created %d\12\0"
	.section __TEXT,__textcoal_nt,coalesced,pure_instructions
	.align 2
	.globl __ZN1AC2Ei
	.weak_definition __ZN1AC2Ei
__ZN1AC2Ei:
	.stabd	46,0,0
	.stabs	"../../tools/testatomic.cpp",132,0,0,Ltext8
Ltext8:
	.stabd	68,0,45
LFB1484:
	mflr r0
LCFI134:
	stmw r30,-8(r1)
LCFI135:
	stw r0,8(r1)
LCFI136:
	stwu r1,-80(r1)
LCFI137:
	mr r30,r1
LCFI138:
	bcl 20,31,"L00000000019$pb"
"L00000000019$pb":
	mflr r31
	stw r3,104(r30)
	stw r4,108(r30)
LBB38:
	.stabd	68,0,45
	addis r2,r31,ha16(L__ZTV1A$non_lazy_ptr-"L00000000019$pb")
	lwz r2,lo16(L__ZTV1A$non_lazy_ptr-"L00000000019$pb")(r2)
	addi r0,r2,8
	lwz r2,104(r30)
	stw r0,0(r2)
	lwz r2,104(r30)
	lwz r0,108(r30)
	stw r0,4(r2)
	.stabd	68,0,46
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000019$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000019$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC3-"L00000000019$pb")
	la r4,lo16(LC3-"L00000000019$pb")(r2)
	lwz r5,108(r30)
	bl L_fprintf$LDBLStub$stub
LBE38:
	.stabd	68,0,47
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r30,-8(r1)
	blr
LFE1484:
	.stabs	"_ZN1AC2Ei:F(0,1)",36,0,45,__ZN1AC2Ei
	.stabs	"this:p(0,67)",160,0,45,104
	.stabs	"x:p(0,10)",160,0,45,108
	.stabs	":t(0,67)=k(0,68)",128,0,0,0
	.stabs	":t(0,68)=*(0,69)",128,0,0,0
	.stabs	"A:t(0,69)=s8_vptr$A:(0,70),0,32;m_x:(0,10),32,32;__base_ctor ::(0,71)=#(0,69),(0,1),(0,68),(0,72),(0,1);:_ZN1AC2ERKS_;2A.;__comp_ctor ::(0,71):_ZN1AC1ERKS_;2A.;__base_ctor ::(0,73)=#(0,69),(0,1),(0,68),(0,10),(0,1);:_ZN1AC2Ei;2A.;__comp_ctor ::(0,73):_ZN1AC1Ei;2A.;__base_dtor ::(0,74)=#(0,69),(0,1),(0,68),(0,1);:_ZN1AD2Ev;2A.;__comp_dtor ::(0,74):_ZN1AD1Ev;2A*0;(0,69);;__deleting_dtor ::(0,74):_ZN1AD0Ev;2A*1;(0,69);;;~%(0,69);",128,0,0,0
	.stabs	":t(0,70)=*(0,75)",128,0,0,0
	.stabs	":t(0,72)=&(0,76)",128,0,0,0
	.stabs	"__vtbl_ptr_type:t(0,75)=*(0,77)",128,0,0,0
	.stabs	":t(0,76)=k(0,69)",128,0,0,0
	.stabs	":t(0,77)=f(0,10)",128,0,0,0
Lscope28:
	.stabs	"",36,0,0,Lscope28-__ZN1AC2Ei
	.stabd	78,0,0
	.align 2
	.globl __ZN1AC1Ei
	.weak_definition __ZN1AC1Ei
__ZN1AC1Ei:
	.stabd	46,0,0
	.stabd	68,0,45
LFB1485:
	mflr r0
LCFI139:
	stmw r30,-8(r1)
LCFI140:
	stw r0,8(r1)
LCFI141:
	stwu r1,-80(r1)
LCFI142:
	mr r30,r1
LCFI143:
	bcl 20,31,"L00000000020$pb"
"L00000000020$pb":
	mflr r31
	stw r3,104(r30)
	stw r4,108(r30)
LBB39:
	.stabd	68,0,45
	addis r2,r31,ha16(L__ZTV1A$non_lazy_ptr-"L00000000020$pb")
	lwz r2,lo16(L__ZTV1A$non_lazy_ptr-"L00000000020$pb")(r2)
	addi r0,r2,8
	lwz r2,104(r30)
	stw r0,0(r2)
	lwz r2,104(r30)
	lwz r0,108(r30)
	stw r0,4(r2)
	.stabd	68,0,46
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000020$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000020$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC3-"L00000000020$pb")
	la r4,lo16(LC3-"L00000000020$pb")(r2)
	lwz r5,108(r30)
	bl L_fprintf$LDBLStub$stub
LBE39:
	.stabd	68,0,47
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r30,-8(r1)
	blr
LFE1485:
	.stabs	"_ZN1AC1Ei:F(0,1)",36,0,45,__ZN1AC1Ei
	.stabs	"A:Tt(0,69)",128,0,43,0
	.stabs	"this:p(0,67)",160,0,45,104
	.stabs	"x:p(0,10)",160,0,45,108
Lscope29:
	.stabs	"",36,0,0,Lscope29-__ZN1AC1Ei
	.stabd	78,0,0
	.cstring
	.align 2
LC4:
	.ascii "Destroyed %d\12\0"
	.section __TEXT,__textcoal_nt,coalesced,pure_instructions
	.align 2
	.globl __ZN1AD2Ev
	.weak_definition __ZN1AD2Ev
__ZN1AD2Ev:
	.stabd	46,0,0
	.stabd	68,0,48
LFB1487:
	mflr r0
LCFI144:
	stmw r30,-8(r1)
LCFI145:
	stw r0,8(r1)
LCFI146:
	stwu r1,-80(r1)
LCFI147:
	mr r30,r1
LCFI148:
	bcl 20,31,"L00000000021$pb"
"L00000000021$pb":
	mflr r31
	stw r3,104(r30)
	.stabd	68,0,48
	addis r2,r31,ha16(L__ZTV1A$non_lazy_ptr-"L00000000021$pb")
	lwz r2,lo16(L__ZTV1A$non_lazy_ptr-"L00000000021$pb")(r2)
	addi r0,r2,8
	lwz r2,104(r30)
	stw r0,0(r2)
	.stabd	68,0,49
	lwz r2,104(r30)
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000021$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000021$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC4-"L00000000021$pb")
	la r4,lo16(LC4-"L00000000021$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,50
	li r0,0
	cmpwi cr7,r0,0
	beq cr7,L98
	lwz r3,104(r30)
	bl L__ZdlPv$stub
L98:
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r30,-8(r1)
	blr
LFE1487:
	.stabs	"_ZN1AD2Ev:F(0,1)",36,0,48,__ZN1AD2Ev
	.stabs	"this:p(0,67)",160,0,48,104
Lscope30:
	.stabs	"",36,0,0,Lscope30-__ZN1AD2Ev
	.stabd	78,0,0
	.align 2
	.globl __ZN1AD1Ev
	.weak_definition __ZN1AD1Ev
__ZN1AD1Ev:
	.stabd	46,0,0
	.stabd	68,0,48
LFB1488:
	mflr r0
LCFI149:
	stmw r30,-8(r1)
LCFI150:
	stw r0,8(r1)
LCFI151:
	stwu r1,-80(r1)
LCFI152:
	mr r30,r1
LCFI153:
	bcl 20,31,"L00000000022$pb"
"L00000000022$pb":
	mflr r31
	stw r3,104(r30)
	.stabd	68,0,48
	addis r2,r31,ha16(L__ZTV1A$non_lazy_ptr-"L00000000022$pb")
	lwz r2,lo16(L__ZTV1A$non_lazy_ptr-"L00000000022$pb")(r2)
	addi r0,r2,8
	lwz r2,104(r30)
	stw r0,0(r2)
	.stabd	68,0,49
	lwz r2,104(r30)
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000022$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000022$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC4-"L00000000022$pb")
	la r4,lo16(LC4-"L00000000022$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,50
	li r0,0
	cmpwi cr7,r0,0
	beq cr7,L103
	lwz r3,104(r30)
	bl L__ZdlPv$stub
L103:
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r30,-8(r1)
	blr
LFE1488:
	.stabs	"_ZN1AD1Ev:F(0,1)",36,0,48,__ZN1AD1Ev
	.stabs	"this:p(0,67)",160,0,48,104
Lscope31:
	.stabs	"",36,0,0,Lscope31-__ZN1AD1Ev
	.stabd	78,0,0
	.align 2
	.globl __ZN1AD0Ev
	.weak_definition __ZN1AD0Ev
__ZN1AD0Ev:
	.stabd	46,0,0
	.stabd	68,0,48
LFB1489:
	mflr r0
LCFI154:
	stmw r30,-8(r1)
LCFI155:
	stw r0,8(r1)
LCFI156:
	stwu r1,-80(r1)
LCFI157:
	mr r30,r1
LCFI158:
	bcl 20,31,"L00000000023$pb"
"L00000000023$pb":
	mflr r31
	stw r3,104(r30)
	.stabd	68,0,48
	addis r2,r31,ha16(L__ZTV1A$non_lazy_ptr-"L00000000023$pb")
	lwz r2,lo16(L__ZTV1A$non_lazy_ptr-"L00000000023$pb")(r2)
	addi r0,r2,8
	lwz r2,104(r30)
	stw r0,0(r2)
	.stabd	68,0,49
	lwz r2,104(r30)
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000023$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000023$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC4-"L00000000023$pb")
	la r4,lo16(LC4-"L00000000023$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,50
	li r0,1
	cmpwi cr7,r0,0
	beq cr7,L108
	lwz r3,104(r30)
	bl L__ZdlPv$stub
L108:
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r30,-8(r1)
	blr
LFE1489:
	.stabs	"_ZN1AD0Ev:F(0,1)",36,0,48,__ZN1AD0Ev
	.stabs	"this:p(0,67)",160,0,48,104
Lscope32:
	.stabs	"",36,0,0,Lscope32-__ZN1AD0Ev
	.stabd	78,0,0
	.align 2
	.globl __ZN1BC1Ei
	.weak_definition __ZN1BC1Ei
__ZN1BC1Ei:
	.stabd	46,0,0
	.stabd	68,0,56
LFB1492:
	mflr r0
LCFI159:
	stmw r29,-12(r1)
LCFI160:
	stw r0,8(r1)
LCFI161:
	stwu r1,-80(r1)
LCFI162:
	mr r30,r1
LCFI163:
	bcl 20,31,"L00000000024$pb"
"L00000000024$pb":
	mflr r31
	stw r3,104(r30)
	stw r4,108(r30)
LBB40:
	.stabd	68,0,56
	lwz r0,104(r30)
	mr r3,r0
	lwz r4,108(r30)
LEHB4:
	bl L__ZN1AC2Ei$stub
LEHE4:
	addis r2,r31,ha16(L__ZTV1B$non_lazy_ptr-"L00000000024$pb")
	lwz r2,lo16(L__ZTV1B$non_lazy_ptr-"L00000000024$pb")(r2)
	addi r0,r2,8
	lwz r2,104(r30)
	stw r0,0(r2)
	.stabd	68,0,57
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000024$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000024$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC3-"L00000000024$pb")
	la r4,lo16(LC3-"L00000000024$pb")(r2)
	lwz r5,108(r30)
LEHB5:
	bl L_fprintf$LDBLStub$stub
LEHE5:
	b L112
L113:
	stw r3,56(r30)
L111:
	lwz r29,56(r30)
	.stabd	68,0,58
	lwz r0,104(r30)
	mr r3,r0
	bl L__ZN1AD2Ev$stub
	stw r29,56(r30)
	lwz r3,56(r30)
LEHB6:
	bl L__Unwind_Resume$stub
LEHE6:
L112:
LBE40:
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r29,-12(r1)
	blr
LFE1492:
	.section __DATA,__gcc_except_tab
LLSDA1492:
	.byte	0xff
	.byte	0xff
	.byte	0x3
	.byte	0x27
	.set L$set$10,LEHB4-LFB1492
	.long L$set$10
	.set L$set$11,LEHE4-LEHB4
	.long L$set$11
	.long	0x0
	.byte	0x0
	.set L$set$12,LEHB5-LFB1492
	.long L$set$12
	.set L$set$13,LEHE5-LEHB5
	.long L$set$13
	.set L$set$14,L113-LFB1492
	.long L$set$14
	.byte	0x0
	.set L$set$15,LEHB6-LFB1492
	.long L$set$15
	.set L$set$16,LEHE6-LEHB6
	.long L$set$16
	.long	0x0
	.byte	0x0
	.section __TEXT,__textcoal_nt,coalesced,pure_instructions
	.stabs	"_ZN1BC1Ei:F(0,1)",36,0,56,__ZN1BC1Ei
	.stabs	"this:p(0,78)",160,0,56,104
	.stabs	"x:p(0,10)",160,0,56,108
	.stabs	":t(0,78)=k(0,79)",128,0,0,0
	.stabs	":t(0,79)=*(0,80)",128,0,0,0
	.stabs	"B:t(0,80)=s8!1,020,(0,69)=xsA:;__base_ctor ::(0,81)=#(0,80),(0,1),(0,79),(0,82),(0,1);:_ZN1BC2ERKS_;2A.;__comp_ctor ::(0,81):_ZN1BC1ERKS_;2A.;__base_ctor ::(0,83)=#(0,80),(0,1),(0,79),(0,10),(0,1);:_ZN1BC2Ei;2A.;__comp_ctor ::(0,83):_ZN1BC1Ei;2A.;__base_dtor ::(0,84)=#(0,80),(0,1),(0,79),(0,1);:_ZN1BD2Ev;2A.;__comp_dtor ::(0,84):_ZN1BD1Ev;2A*0;(0,80);;__deleting_dtor ::(0,84):_ZN1BD0Ev;2A*1;(0,80);;;~%(0,69);",128,0,0,0
	.stabs	":t(0,82)=&(0,85)",128,0,0,0
	.stabs	":t(0,85)=k(0,80)",128,0,0,0
Lscope33:
	.stabs	"",36,0,0,Lscope33-__ZN1BC1Ei
	.stabd	78,0,0
	.align 2
	.globl __ZN1BD1Ev
	.weak_definition __ZN1BD1Ev
__ZN1BD1Ev:
	.stabd	46,0,0
	.stabd	68,0,59
LFB1495:
	mflr r0
LCFI164:
	stmw r29,-12(r1)
LCFI165:
	stw r0,8(r1)
LCFI166:
	stwu r1,-80(r1)
LCFI167:
	mr r30,r1
LCFI168:
	bcl 20,31,"L00000000025$pb"
"L00000000025$pb":
	mflr r31
	stw r3,104(r30)
	.stabd	68,0,59
	addis r2,r31,ha16(L__ZTV1B$non_lazy_ptr-"L00000000025$pb")
	lwz r2,lo16(L__ZTV1B$non_lazy_ptr-"L00000000025$pb")(r2)
	addi r0,r2,8
	lwz r2,104(r30)
	stw r0,0(r2)
	.stabd	68,0,60
	lwz r2,104(r30)
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000025$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000025$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC4-"L00000000025$pb")
	la r4,lo16(LC4-"L00000000025$pb")(r2)
	mr r5,r0
LEHB7:
	bl L_fprintf$LDBLStub$stub
LEHE7:
	b L115
L121:
	stw r3,60(r30)
L116:
	lwz r29,60(r30)
	.stabd	68,0,61
	lwz r0,104(r30)
	stw r0,56(r30)
	lwz r3,56(r30)
	bl L__ZN1AD2Ev$stub
	stw r29,60(r30)
	lwz r3,60(r30)
LEHB8:
	bl L__Unwind_Resume$stub
L115:
	lwz r0,104(r30)
	stw r0,56(r30)
	lwz r3,56(r30)
	bl L__ZN1AD2Ev$stub
LEHE8:
	li r0,0
	cmpwi cr7,r0,0
	beq cr7,L120
	lwz r3,104(r30)
	bl L__ZdlPv$stub
L120:
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r29,-12(r1)
	blr
LFE1495:
	.section __DATA,__gcc_except_tab
LLSDA1495:
	.byte	0xff
	.byte	0xff
	.byte	0x3
	.byte	0x1a
	.set L$set$17,LEHB7-LFB1495
	.long L$set$17
	.set L$set$18,LEHE7-LEHB7
	.long L$set$18
	.set L$set$19,L121-LFB1495
	.long L$set$19
	.byte	0x0
	.set L$set$20,LEHB8-LFB1495
	.long L$set$20
	.set L$set$21,LEHE8-LEHB8
	.long L$set$21
	.long	0x0
	.byte	0x0
	.section __TEXT,__textcoal_nt,coalesced,pure_instructions
	.stabs	"_ZN1BD1Ev:F(0,1)",36,0,59,__ZN1BD1Ev
	.stabs	"B:Tt(0,80)",128,0,54,0
	.stabs	"this:p(0,78)",160,0,59,104
Lscope34:
	.stabs	"",36,0,0,Lscope34-__ZN1BD1Ev
	.stabd	78,0,0
	.align 2
	.globl __ZN1BD0Ev
	.weak_definition __ZN1BD0Ev
__ZN1BD0Ev:
	.stabd	46,0,0
	.stabd	68,0,59
LFB1496:
	mflr r0
LCFI169:
	stmw r29,-12(r1)
LCFI170:
	stw r0,8(r1)
LCFI171:
	stwu r1,-80(r1)
LCFI172:
	mr r30,r1
LCFI173:
	bcl 20,31,"L00000000026$pb"
"L00000000026$pb":
	mflr r31
	stw r3,104(r30)
	.stabd	68,0,59
	addis r2,r31,ha16(L__ZTV1B$non_lazy_ptr-"L00000000026$pb")
	lwz r2,lo16(L__ZTV1B$non_lazy_ptr-"L00000000026$pb")(r2)
	addi r0,r2,8
	lwz r2,104(r30)
	stw r0,0(r2)
	.stabd	68,0,60
	lwz r2,104(r30)
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000026$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000026$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC4-"L00000000026$pb")
	la r4,lo16(LC4-"L00000000026$pb")(r2)
	mr r5,r0
LEHB9:
	bl L_fprintf$LDBLStub$stub
LEHE9:
	b L123
L129:
	stw r3,60(r30)
L124:
	lwz r29,60(r30)
	.stabd	68,0,61
	lwz r0,104(r30)
	stw r0,56(r30)
	lwz r3,56(r30)
	bl L__ZN1AD2Ev$stub
	stw r29,60(r30)
	lwz r3,60(r30)
LEHB10:
	bl L__Unwind_Resume$stub
L123:
	lwz r0,104(r30)
	stw r0,56(r30)
	lwz r3,56(r30)
	bl L__ZN1AD2Ev$stub
LEHE10:
	li r0,1
	cmpwi cr7,r0,0
	beq cr7,L128
	lwz r3,104(r30)
	bl L__ZdlPv$stub
L128:
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r29,-12(r1)
	blr
LFE1496:
	.section __DATA,__gcc_except_tab
LLSDA1496:
	.byte	0xff
	.byte	0xff
	.byte	0x3
	.byte	0x1a
	.set L$set$22,LEHB9-LFB1496
	.long L$set$22
	.set L$set$23,LEHE9-LEHB9
	.long L$set$23
	.set L$set$24,L129-LFB1496
	.long L$set$24
	.byte	0x0
	.set L$set$25,LEHB10-LFB1496
	.long L$set$25
	.set L$set$26,LEHE10-LEHB10
	.long L$set$26
	.long	0x0
	.byte	0x0
	.section __TEXT,__textcoal_nt,coalesced,pure_instructions
	.stabs	"_ZN1BD0Ev:F(0,1)",36,0,59,__ZN1BD0Ev
	.stabs	"this:p(0,78)",160,0,59,104
Lscope35:
	.stabs	"",36,0,0,Lscope35-__ZN1BD0Ev
	.stabd	78,0,0
	.align 2
	.globl __ZN17atomic_scoped_ptrI1AEC1EPS0_
	.weak_definition __ZN17atomic_scoped_ptrI1AEC1EPS0_
__ZN17atomic_scoped_ptrI1AEC1EPS0_:
	.stabd	46,0,0
	.stabs	"../../kame/atomic_smart_ptr.h",132,0,0,Ltext9
Ltext9:
	.stabd	68,0,22
LFB1551:
	mflr r0
LCFI174:
	stmw r30,-8(r1)
LCFI175:
	stw r0,8(r1)
LCFI176:
	stwu r1,-80(r1)
LCFI177:
	mr r30,r1
LCFI178:
	stw r3,104(r30)
	stw r4,108(r30)
LBB41:
	.stabd	68,0,22
	lwz r2,104(r30)
	lwz r0,108(r30)
	stw r0,0(r2)
	.stabd	68,0,23
	bl L__Z13memoryBarrierv$stub
LBE41:
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r30,-8(r1)
	blr
LFE1551:
	.stabs	"_ZN17atomic_scoped_ptrI1AEC1EPS0_:F(0,1)",36,0,22,__ZN17atomic_scoped_ptrI1AEC1EPS0_
	.stabs	"this:p(0,86)",160,0,22,104
	.stabs	"t:p(0,68)",160,0,22,108
	.stabs	":t(0,86)=k(0,87)",128,0,0,0
	.stabs	":t(0,87)=*(0,88)",128,0,0,0
	.stabs	"atomic_scoped_ptr<A>:t(0,88)=s4m_ptr:/0(0,68),0,32;__base_ctor ::(0,89)=#(0,88),(0,1),(0,87),(0,90),(0,1);:_ZN17atomic_scoped_ptrI1AEC2ERKS1_;2A.;__comp_ctor ::(0,89):_ZN17atomic_scoped_ptrI1AEC1ERKS1_;2A.;__base_ctor ::(0,91)=#(0,88),(0,1),(0,87),(0,1);:_ZN17atomic_scoped_ptrI1AEC2Ev;2A.;__comp_ctor ::(0,91):_ZN17atomic_scoped_ptrI1AEC1Ev;2A.;__base_ctor ::(0,92)=#(0,88),(0,1),(0,87),(0,68),(0,1);:_ZN17atomic_scoped_ptrI1AEC2EPS0_;2A.;__comp_ctor ::(0,92):_ZN17atomic_scoped_ptrI1AEC1EPS0_;2A.;__base_dtor ::(0,91):_ZN17atomic_scoped_ptrI1AED2Ev;2A.;__comp_dtor ::(0,91):_ZN17atomic_scoped_ptrI1AED1Ev;2A.;reset::(0,93)=#(0,88),(0,1),(0,87),(0,68),(0,1);:_ZN17atomic_scoped_ptrI1AE5resetEPS0_;2A.;swap::(0,94)=#(0,88),(0,1),(0,87),(0,95),(0,1);:_ZN17atomic_scoped_ptrI1AE4swapERS1_;2A.;operator*::(0,96)=#(0,88),(0,97),(0,98),(0,1);:_ZNK17atomic_scoped_ptrI1AEdeEv;2B.;operator->::(0,99)=#(0,88),(0,68),(0,98),(0,1);:_ZNK17atomic_scoped_ptrI1AEptEv;2B.;get::(0,99):_ZNK17atomic_scoped_ptrI1AE3getEv;2B.;;",128,0,0,0
	.stabs	":t(0,90)=&(0,100)",128,0,0,0
	.stabs	":t(0,95)=&(0,88)",128,0,0,0
	.stabs	":t(0,97)=&(0,69)",128,0,0,0
	.stabs	":t(0,98)=*(0,100)",128,0,0,0
	.stabs	":t(0,100)=k(0,88)",128,0,0,0
Lscope36:
	.stabs	"",36,0,0,Lscope36-__ZN17atomic_scoped_ptrI1AEC1EPS0_
	.stabd	78,0,0
	.align 2
	.globl __ZN17atomic_scoped_ptrI1AED1Ev
	.weak_definition __ZN17atomic_scoped_ptrI1AED1Ev
__ZN17atomic_scoped_ptrI1AED1Ev:
	.stabd	46,0,0
	.stabd	68,0,26
LFB1554:
	mflr r0
LCFI179:
	stmw r30,-8(r1)
LCFI180:
	stw r0,8(r1)
LCFI181:
	stwu r1,-80(r1)
LCFI182:
	mr r30,r1
LCFI183:
	stw r3,104(r30)
	.stabd	68,0,26
	lwz r2,104(r30)
	lwz r0,0(r2)
	cmpwi cr7,r0,0
	beq cr7,L135
	lwz r2,104(r30)
	lwz r2,0(r2)
	lwz r2,0(r2)
	addi r2,r2,4
	lwz r9,0(r2)
	lwz r2,104(r30)
	lwz r0,0(r2)
	mr r3,r0
	mr r12,r9
	mtctr r12
	bctrl
L135:
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r30,-8(r1)
	blr
LFE1554:
	.stabs	"_ZN17atomic_scoped_ptrI1AED1Ev:F(0,1)",36,0,26,__ZN17atomic_scoped_ptrI1AED1Ev
	.stabs	"atomic_scoped_ptr<A>:Tt(0,88)",128,0,16,0
	.stabs	"this:p(0,86)",160,0,26,104
Lscope37:
	.stabs	"",36,0,0,Lscope37-__ZN17atomic_scoped_ptrI1AED1Ev
	.stabd	78,0,0
	.align 2
	.globl __ZN6atomicIiEC1Ei
	.weak_definition __ZN6atomicIiEC1Ei
__ZN6atomicIiEC1Ei:
	.stabd	46,0,0
	.stabs	"../../kame/atomic.h",132,0,0,Ltext10
Ltext10:
	.stabd	68,0,207
LFB1559:
	mflr r0
LCFI184:
	stmw r30,-8(r1)
LCFI185:
	stw r0,8(r1)
LCFI186:
	stwu r1,-80(r1)
LCFI187:
	mr r30,r1
LCFI188:
	stw r3,104(r30)
	stw r4,108(r30)
LBB42:
	.stabd	68,0,207
	lwz r2,104(r30)
	lwz r0,108(r30)
	stw r0,0(r2)
	bl L__Z13memoryBarrierv$stub
LBE42:
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r30,-8(r1)
	blr
LFE1559:
	.stabs	"_ZN6atomicIiEC1Ei:F(0,1)",36,0,207,__ZN6atomicIiEC1Ei
	.stabs	"this:p(0,101)",160,0,207,104
	.stabs	"t:p(0,10)",160,0,207,108
	.stabs	":t(0,101)=k(0,102)",128,0,0,0
	.stabs	":t(0,102)=*(0,103)",128,0,0,0
	.stabs	"atomic<int>:t(0,103)=s4m_var:/0(0,10),0,32;__base_ctor ::(0,104)=#(0,103),(0,1),(0,102),(0,1);:_ZN6atomicIiEC2Ev;2A.;__comp_ctor ::(0,104):_ZN6atomicIiEC1Ev;2A.;__base_ctor ::(0,105)=#(0,103),(0,1),(0,102),(0,10),(0,1);:_ZN6atomicIiEC2Ei;2A.;__comp_ctor ::(0,105):_ZN6atomicIiEC1Ei;2A.;__base_ctor ::(0,106)=#(0,103),(0,1),(0,102),(0,107),(0,1);:_ZN6atomicIiEC2ERKS0_;2A.;__comp_ctor ::(0,106):_ZN6atomicIiEC1ERKS0_;2A.;__base_dtor ::(0,104):_ZN6atomicIiED2Ev;2A.;__comp_dtor ::(0,104):_ZN6atomicIiED1Ev;2A.;operator 13::(0,108)=#(0,103),(0,10),(0,109),(0,1);:_ZNK6atomicIiEcviEv;2B.;operator=::(0,110)=#(0,103),(0,111),(0,102),(0,10),(0,1);:_ZN6atomicIiEaSEi;2A.;operator++::(0,112)=#(0,103),(0,111),(0,102),(0,1);:_ZN6atomicIiEppEv;2A.;operator--::(0,112):_ZN6atomicIiEmmEv;2A.;operator+=::(0,110):_ZN6atomicIiEpLEi;2A.;operator-=::(0,110):_ZN6atomicIiEmIEi;2A.;swap::(0,113):_ZN6atomicIiE4swapEiRS0_;2A?;decAndTest::(0,114)=#(0,103),(0,2),(0,102),(0,1);:_ZN6atomicIiE10decAndTestEv;2A.;addAndTest::(0,115)=#(0,103),(0,2),(0,102),(0,10),(0,1);:_ZN6atomicIiE10addAndTestEi;2A.;;",128,0,0,0
	.stabs	":t(0,107)=&(0,116)",128,0,0,0
	.stabs	":t(0,109)=*(0,116)",128,0,0,0
	.stabs	":t(0,111)=&(0,103)",128,0,0,0
	.stabs	":t(0,113)=f(0,10)",128,0,0,0
	.stabs	":t(0,116)=k(0,103)",128,0,0,0
Lscope38:
	.stabs	"",36,0,0,Lscope38-__ZN6atomicIiEC1Ei
	.stabd	78,0,0
	.align 2
	.globl __ZN6atomicIiED1Ev
	.weak_definition __ZN6atomicIiED1Ev
__ZN6atomicIiED1Ev:
	.stabd	46,0,0
	.stabd	68,0,209
LFB1562:
	stmw r30,-8(r1)
LCFI189:
	stwu r1,-48(r1)
LCFI190:
	mr r30,r1
LCFI191:
	stw r3,72(r30)
	.stabd	68,0,209
	lwz r1,0(r1)
	lmw r30,-8(r1)
	blr
LFE1562:
	.stabs	"_ZN6atomicIiED1Ev:F(0,1)",36,0,209,__ZN6atomicIiED1Ev
	.stabs	"atomic<int>:Tt(0,103)",128,0,204,0
	.stabs	"this:p(0,101)",160,0,209,72
Lscope39:
	.stabs	"",36,0,0,Lscope39-__ZN6atomicIiED1Ev
	.stabd	78,0,0
	.align 2
	.globl __ZNK6atomicIiEcviEv
	.weak_definition __ZNK6atomicIiEcviEv
__ZNK6atomicIiEcviEv:
	.stabd	46,0,0
	.stabd	68,0,210
LFB1566:
	mflr r0
LCFI192:
	stmw r30,-8(r1)
LCFI193:
	stw r0,8(r1)
LCFI194:
	stwu r1,-80(r1)
LCFI195:
	mr r30,r1
LCFI196:
	stw r3,104(r30)
LBB43:
	.stabd	68,0,210
	bl L__Z11readBarrierv$stub
	lwz r2,104(r30)
	lwz r0,0(r2)
LBE43:
	mr r3,r0
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r30,-8(r1)
	blr
LFE1566:
	.stabs	"_ZNK6atomicIiEcviEv:F(0,10)",36,0,210,__ZNK6atomicIiEcviEv
	.stabs	"this:p(0,117)",160,0,210,104
	.stabs	":t(0,117)=k(0,109)",128,0,0,0
Lscope40:
	.stabs	"",36,0,0,Lscope40-__ZNK6atomicIiEcviEv
	.stabd	78,0,0
	.cstring
	.align 2
LC5:
	.ascii "../../kame/atomic_smart_ptr.h\0"
	.section __TEXT,__textcoal_nt,coalesced,pure_instructions
	.align 2
	.globl __ZNK17atomic_shared_ptrI1AEdeEv
	.weak_definition __ZNK17atomic_shared_ptrI1AEdeEv
__ZNK17atomic_shared_ptrI1AEdeEv:
	.stabd	46,0,0
	.stabs	"../../kame/atomic_smart_ptr.h",132,0,0,Ltext11
Ltext11:
	.stabd	68,0,93
LFB1568:
	mflr r0
LCFI197:
	stmw r30,-8(r1)
LCFI198:
	stw r0,8(r1)
LCFI199:
	stwu r1,-80(r1)
LCFI200:
	mr r30,r1
LCFI201:
	bcl 20,31,"L00000000027$pb"
"L00000000027$pb":
	mflr r31
	stw r3,104(r30)
LBB44:
	.stabd	68,0,93
	lwz r2,104(r30)
	lwz r0,0(r2)
	srawi r2,r0,31
	xor r0,r2,r0
	subf r0,r2,r0
	neg r0,r0
	srwi r0,r0,31
	mr r3,r0
	addis r2,r31,ha16(LC5-"L00000000027$pb")
	la r4,lo16(LC5-"L00000000027$pb")(r2)
	li r5,93
	bl L__Z10_my_assertbPKci$stub
	lwz r2,104(r30)
	lwz r0,0(r2)
LBE44:
	mr r3,r0
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r30,-8(r1)
	blr
LFE1568:
	.stabs	"_ZNK17atomic_shared_ptrI1AEdeEv:F(0,97)",36,0,93,__ZNK17atomic_shared_ptrI1AEdeEv
	.stabs	"this:p(0,118)",160,0,93,104
	.stabs	":t(0,118)=k(0,119)",128,0,0,0
	.stabs	":t(0,119)=*(0,120)",128,0,0,0
	.stabs	":t(0,120)=k(0,121)",128,0,0,0
	.stabs	"atomic_shared_ptr<A>:t(0,121)=s12m_ptr_instant:/0(0,68),0,32;m_ref:/0(0,122)=xs_RefLocal:,32,64;__base_ctor ::(0,123)=#(0,121),(0,1),(0,124),(0,1);:_ZN17atomic_shared_ptrI1AEC2Ev;2A.;__comp_ctor ::(0,123):_ZN17atomic_shared_ptrI1AEC1Ev;2A.;__base_ctor ::(0,125)=#(0,121),(0,1),(0,124),(0,126),(0,1);:_ZN17atomic_shared_ptrI1AEC2ERKS1_;2A.;__comp_ctor ::(0,125):_ZN17atomic_shared_ptrI1AEC1ERKS1_;2A.;__base_dtor ::(0,123):_ZN17atomic_shared_ptrI1AED2Ev;2A.;__comp_dtor ::(0,123):_ZN17atomic_shared_ptrI1AED1Ev;2A.;operator=::(0,127)=#(0,121),(0,128),(0,124),(0,126),(0,1);:_ZN17atomic_shared_ptrI1AEaSERKS1_;2A.;reset::(0,123):_ZN17atomic_shared_ptrI1AE5resetEv;2A.;swap::(0,129)=#(0,121),(0,1),(0,124),(0,128),(0,1);:_ZN17atomic_shared_ptrI1AE4swapERS1_;2A.;operator*::(0,130)=#(0,121),(0,97),(0,119),(0,1);:_ZNK17atomic_shared_ptrI1AEdeEv;2B.;operator->::(0,131)=#(0,121),(0,68),(0,119),(0,1);:_ZNK17atomic_shared_ptrI1AEptEv;2B.;get::(0,131):_ZNK17atomic_shared_ptrI1AE3getEv;2B.;operator!::(0,132)=#(0,121),(0,2),(0,119),(0,1);:_ZNK17atomic_shared_ptrI1AEntEv;2B.;operator 12::(0,132):_ZNK17atomic_shared_ptrI1AEcvbEv;2B.;_scan_::(0,133)=#(0,121),(0,134),(0,119),(0,1);:_ZNK17atomic_shared_ptrI1AE6_scan_Ev;2B.;;",128,0,0,0
	.stabs	"_RefLocal:t(0,122)=s8pref:(0,134),0,32;refcnt_n_serial:(0,59),32,32;__base_ctor ::(0,135)=#(0,122),(0,1),(0,136),(0,137),(0,1);:_ZN17atomic_shared_ptrI1AE9_RefLocalC2ERKS2_;2A.;__comp_ctor ::(0,135):_ZN17atomic_shared_ptrI1AE9_RefLocalC1ERKS2_;2A.;__base_ctor ::(0,138)=#(0,122),(0,1),(0,136),(0,1);:_ZN17atomic_shared_ptrI1AE9_RefLocalC2Ev;2A.;__comp_ctor ::(0,138):_ZN17atomic_shared_ptrI1AE9_RefLocalC1Ev;2A.;refcnt_cached::(0,139)=#(0,122),(0,59),(0,140),(0,1);:_ZNK17atomic_shared_ptrI1AE9_RefLocal13refcnt_cachedEv;2B.;serial::(0,139):_ZNK17atomic_shared_ptrI1AE9_RefLocal6serialEv;2B.;_refcnt_n_serial::(0,141)=#(0,122),(0,59),(0,140),(0,59),(0,59),(0,1);:_ZNK17atomic_shared_ptrI1AE9_RefLocal16_refcnt_n_serialEjj;2B.;;",128,0,0,0
	.stabs	":t(0,124)=*(0,121)",128,0,0,0
	.stabs	":t(0,126)=&(0,120)",128,0,0,0
	.stabs	":t(0,128)=&(0,121)",128,0,0,0
	.stabs	":t(0,134)=*(0,142)",128,0,0,0
	.stabs	":t(0,136)=*(0,122)",128,0,0,0
	.stabs	":t(0,137)=&(0,143)",128,0,0,0
	.stabs	":t(0,140)=*(0,143)",128,0,0,0
	.stabs	"Ref:t(0,142)=s8ptr:(0,68),0,32;refcnt:(0,59),32,32;__base_ctor ::(0,144)=#(0,142),(0,1),(0,134),(0,145),(0,1);:_ZN17atomic_shared_ptrI1AE3RefC2ERKS2_;2A.;__comp_ctor ::(0,144):_ZN17atomic_shared_ptrI1AE3RefC1ERKS2_;2A.;__base_dtor ::(0,146)=#(0,142),(0,1),(0,134),(0,1);:_ZN17atomic_shared_ptrI1AE3RefD2Ev;2A.;__comp_dtor ::(0,146):_ZN17atomic_shared_ptrI1AE3RefD1Ev;2A.;;",128,0,0,0
	.stabs	":t(0,143)=k(0,122)",128,0,0,0
	.stabs	":t(0,145)=&(0,147)",128,0,0,0
	.stabs	":t(0,147)=k(0,142)",128,0,0,0
Lscope41:
	.stabs	"",36,0,0,Lscope41-__ZNK17atomic_shared_ptrI1AEdeEv
	.stabd	78,0,0
	.align 2
	.globl __ZN17atomic_shared_ptrI1AE3RefC1IS0_EEPT_
	.weak_definition __ZN17atomic_shared_ptrI1AE3RefC1IS0_EEPT_
__ZN17atomic_shared_ptrI1AE3RefC1IS0_EEPT_:
	.stabd	46,0,0
	.stabd	68,0,111
LFB1588:
	stmw r30,-8(r1)
LCFI202:
	stwu r1,-48(r1)
LCFI203:
	mr r30,r1
LCFI204:
	stw r3,72(r30)
	stw r4,76(r30)
LBB45:
	.stabd	68,0,111
	lwz r2,72(r30)
	lwz r0,76(r30)
	stw r0,0(r2)
	lwz r2,72(r30)
	li r0,1
	stw r0,4(r2)
LBE45:
	lwz r1,0(r1)
	lmw r30,-8(r1)
	blr
LFE1588:
	.stabs	"_ZN17atomic_shared_ptrI1AE3RefC1IS0_EEPT_:F(0,1)",36,0,111,__ZN17atomic_shared_ptrI1AE3RefC1IS0_EEPT_
	.stabs	"atomic_shared_ptr<A>:Tt(0,121)",128,0,52,0
	.stabs	"atomic_shared_ptr<A>::_RefLocal:Tt(0,122)",128,0,117,0
	.stabs	"atomic_shared_ptr<A>::Ref:Tt(0,142)",128,0,109,0
	.stabs	"this:p(0,148)",160,0,111,72
	.stabs	"p:p(0,68)",160,0,111,76
	.stabs	":t(0,148)=k(0,134)",128,0,0,0
Lscope42:
	.stabs	"",36,0,0,Lscope42-__ZN17atomic_shared_ptrI1AE3RefC1IS0_EEPT_
	.stabd	78,0,0
	.align 2
	.globl __ZN17atomic_shared_ptrI1AE9_RefLocalC1INS1_3RefEEEPT_
	.weak_definition __ZN17atomic_shared_ptrI1AE9_RefLocalC1INS1_3RefEEEPT_
__ZN17atomic_shared_ptrI1AE9_RefLocalC1INS1_3RefEEEPT_:
	.stabd	46,0,0
	.stabd	68,0,119
LFB1591:
	stmw r30,-8(r1)
LCFI205:
	stwu r1,-48(r1)
LCFI206:
	mr r30,r1
LCFI207:
	stw r3,72(r30)
	stw r4,76(r30)
LBB46:
	.stabd	68,0,121
	lwz r2,72(r30)
	lwz r0,76(r30)
	stw r0,0(r2)
	lwz r2,72(r30)
	li r0,0
	stw r0,4(r2)
LBE46:
	.stabd	68,0,123
	lwz r1,0(r1)
	lmw r30,-8(r1)
	blr
LFE1591:
	.stabs	"_ZN17atomic_shared_ptrI1AE9_RefLocalC1INS1_3RefEEEPT_:F(0,1)",36,0,119,__ZN17atomic_shared_ptrI1AE9_RefLocalC1INS1_3RefEEEPT_
	.stabs	"this:p(0,149)",160,0,119,72
	.stabs	"r:p(0,134)",160,0,119,76
	.stabs	":t(0,149)=k(0,136)",128,0,0,0
Lscope43:
	.stabs	"",36,0,0,Lscope43-__ZN17atomic_shared_ptrI1AE9_RefLocalC1INS1_3RefEEEPT_
	.stabd	78,0,0
	.align 2
	.globl __ZN17atomic_shared_ptrI1AEC1IS0_EEPT_
	.weak_definition __ZN17atomic_shared_ptrI1AEC1IS0_EEPT_
__ZN17atomic_shared_ptrI1AEC1IS0_EEPT_:
	.stabd	46,0,0
	.stabd	68,0,57
LFB1536:
	mflr r0
LCFI208:
	stmw r28,-16(r1)
LCFI209:
	stw r0,8(r1)
LCFI210:
	stwu r1,-80(r1)
LCFI211:
	mr r30,r1
LCFI212:
	stw r3,104(r30)
	stw r4,108(r30)
LBB47:
	.stabd	68,0,57
	lwz r2,104(r30)
	addi r28,r2,4
	li r3,8
	bl L__Znwm$stub
	mr r0,r3
	mr r29,r0
	mr r3,r29
	lwz r4,108(r30)
	bl L__ZN17atomic_shared_ptrI1AE3RefC1IS0_EEPT_$stub
	mr r3,r28
	mr r4,r29
	bl L__ZN17atomic_shared_ptrI1AE9_RefLocalC1INS1_3RefEEEPT_$stub
	.stabd	68,0,58
	lwz r2,104(r30)
	lwz r0,108(r30)
	stw r0,0(r2)
	.stabd	68,0,59
	bl L__Z13memoryBarrierv$stub
LBE47:
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r28,-16(r1)
	blr
LFE1536:
	.stabs	"_ZN17atomic_shared_ptrI1AEC1IS0_EEPT_:F(0,1)",36,0,57,__ZN17atomic_shared_ptrI1AEC1IS0_EEPT_
	.stabs	"this:p(0,150)",160,0,57,104
	.stabs	"y:p(0,68)",160,0,57,108
	.stabs	":t(0,150)=k(0,124)",128,0,0,0
Lscope44:
	.stabs	"",36,0,0,Lscope44-__ZN17atomic_shared_ptrI1AEC1IS0_EEPT_
	.stabd	78,0,0
	.align 2
	.globl __ZNK17atomic_shared_ptrI1AE9_RefLocal13refcnt_cachedEv
	.weak_definition __ZNK17atomic_shared_ptrI1AE9_RefLocal13refcnt_cachedEv
__ZNK17atomic_shared_ptrI1AE9_RefLocal13refcnt_cachedEv:
	.stabd	46,0,0
	.stabd	68,0,132
LFB1592:
	stmw r30,-8(r1)
LCFI213:
	stwu r1,-48(r1)
LCFI214:
	mr r30,r1
LCFI215:
	stw r3,72(r30)
LBB48:
	.stabd	68,0,133
	lwz r2,72(r30)
	lwz r0,4(r2)
	srwi r0,r0,16
LBE48:
	mr r3,r0
	lwz r1,0(r1)
	lmw r30,-8(r1)
	blr
LFE1592:
	.stabs	"_ZNK17atomic_shared_ptrI1AE9_RefLocal13refcnt_cachedEv:F(0,59)",36,0,132,__ZNK17atomic_shared_ptrI1AE9_RefLocal13refcnt_cachedEv
	.stabs	"this:p(0,151)",160,0,132,72
	.stabs	":t(0,151)=k(0,140)",128,0,0,0
Lscope45:
	.stabs	"",36,0,0,Lscope45-__ZNK17atomic_shared_ptrI1AE9_RefLocal13refcnt_cachedEv
	.stabd	78,0,0
	.align 2
	.globl __ZN17atomic_shared_ptrI1AE3RefD1Ev
	.weak_definition __ZN17atomic_shared_ptrI1AE3RefD1Ev
__ZN17atomic_shared_ptrI1AE3RefD1Ev:
	.stabd	46,0,0
	.stabd	68,0,112
LFB1596:
	mflr r0
LCFI216:
	stmw r30,-8(r1)
LCFI217:
	stw r0,8(r1)
LCFI218:
	stwu r1,-80(r1)
LCFI219:
	mr r30,r1
LCFI220:
	bcl 20,31,"L00000000028$pb"
"L00000000028$pb":
	mflr r31
	stw r3,104(r30)
	.stabd	68,0,112
	lwz r2,104(r30)
	lwz r0,4(r2)
	cmpwi cr7,r0,0
	mfcr r0
	rlwinm r0,r0,31,1
	mr r3,r0
	addis r2,r31,ha16(LC5-"L00000000028$pb")
	la r4,lo16(LC5-"L00000000028$pb")(r2)
	li r5,112
	bl L__Z10_my_assertbPKci$stub
	lwz r2,104(r30)
	lwz r0,0(r2)
	cmpwi cr7,r0,0
	beq cr7,L156
	lwz r2,104(r30)
	lwz r2,0(r2)
	lwz r2,0(r2)
	addi r2,r2,4
	lwz r9,0(r2)
	lwz r2,104(r30)
	lwz r0,0(r2)
	mr r3,r0
	mr r12,r9
	mtctr r12
	bctrl
L156:
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r30,-8(r1)
	blr
LFE1596:
	.stabs	"_ZN17atomic_shared_ptrI1AE3RefD1Ev:F(0,1)",36,0,112,__ZN17atomic_shared_ptrI1AE3RefD1Ev
	.stabs	"this:p(0,148)",160,0,112,104
Lscope46:
	.stabs	"",36,0,0,Lscope46-__ZN17atomic_shared_ptrI1AE3RefD1Ev
	.stabd	78,0,0
	.align 2
	.globl __ZN17atomic_shared_ptrI1AE3RefC1I1BEEPT_
	.weak_definition __ZN17atomic_shared_ptrI1AE3RefC1I1BEEPT_
__ZN17atomic_shared_ptrI1AE3RefC1I1BEEPT_:
	.stabd	46,0,0
	.stabd	68,0,111
LFB1599:
	stmw r30,-8(r1)
LCFI221:
	stwu r1,-48(r1)
LCFI222:
	mr r30,r1
LCFI223:
	stw r3,72(r30)
	stw r4,76(r30)
LBB49:
	.stabd	68,0,111
	lwz r0,76(r30)
	lwz r2,72(r30)
	stw r0,0(r2)
	lwz r2,72(r30)
	li r0,1
	stw r0,4(r2)
LBE49:
	lwz r1,0(r1)
	lmw r30,-8(r1)
	blr
LFE1599:
	.stabs	"_ZN17atomic_shared_ptrI1AE3RefC1I1BEEPT_:F(0,1)",36,0,111,__ZN17atomic_shared_ptrI1AE3RefC1I1BEEPT_
	.stabs	"this:p(0,148)",160,0,111,72
	.stabs	"p:p(0,79)",160,0,111,76
Lscope47:
	.stabs	"",36,0,0,Lscope47-__ZN17atomic_shared_ptrI1AE3RefC1I1BEEPT_
	.stabd	78,0,0
	.align 2
	.globl __ZN17atomic_shared_ptrI1AEC1I1BEEPT_
	.weak_definition __ZN17atomic_shared_ptrI1AEC1I1BEEPT_
__ZN17atomic_shared_ptrI1AEC1I1BEEPT_:
	.stabd	46,0,0
	.stabd	68,0,57
LFB1542:
	mflr r0
LCFI224:
	stmw r28,-16(r1)
LCFI225:
	stw r0,8(r1)
LCFI226:
	stwu r1,-80(r1)
LCFI227:
	mr r30,r1
LCFI228:
	stw r3,104(r30)
	stw r4,108(r30)
LBB50:
	.stabd	68,0,57
	lwz r2,104(r30)
	addi r28,r2,4
	li r3,8
	bl L__Znwm$stub
	mr r0,r3
	mr r29,r0
	mr r3,r29
	lwz r4,108(r30)
	bl L__ZN17atomic_shared_ptrI1AE3RefC1I1BEEPT_$stub
	mr r3,r28
	mr r4,r29
	bl L__ZN17atomic_shared_ptrI1AE9_RefLocalC1INS1_3RefEEEPT_$stub
	.stabd	68,0,58
	lwz r0,108(r30)
	lwz r2,104(r30)
	stw r0,0(r2)
	.stabd	68,0,59
	bl L__Z13memoryBarrierv$stub
LBE50:
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r28,-16(r1)
	blr
LFE1542:
	.stabs	"_ZN17atomic_shared_ptrI1AEC1I1BEEPT_:F(0,1)",36,0,57,__ZN17atomic_shared_ptrI1AEC1I1BEEPT_
	.stabs	"this:p(0,150)",160,0,57,104
	.stabs	"y:p(0,79)",160,0,57,108
Lscope48:
	.stabs	"",36,0,0,Lscope48-__ZN17atomic_shared_ptrI1AEC1I1BEEPT_
	.stabd	78,0,0
	.align 2
	.globl __ZN17atomic_shared_ptrI1AE9_RefLocalC1Ev
	.weak_definition __ZN17atomic_shared_ptrI1AE9_RefLocalC1Ev
__ZN17atomic_shared_ptrI1AE9_RefLocalC1Ev:
	.stabd	46,0,0
	.stabd	68,0,124
LFB1602:
	stmw r30,-8(r1)
LCFI229:
	stwu r1,-48(r1)
LCFI230:
	mr r30,r1
LCFI231:
	stw r3,72(r30)
LBB51:
	.stabd	68,0,126
	lwz r2,72(r30)
	li r0,0
	stw r0,0(r2)
	lwz r2,72(r30)
	li r0,0
	stw r0,4(r2)
LBE51:
	.stabd	68,0,128
	lwz r1,0(r1)
	lmw r30,-8(r1)
	blr
LFE1602:
	.stabs	"_ZN17atomic_shared_ptrI1AE9_RefLocalC1Ev:F(0,1)",36,0,124,__ZN17atomic_shared_ptrI1AE9_RefLocalC1Ev
	.stabs	"this:p(0,149)",160,0,124,72
Lscope49:
	.stabs	"",36,0,0,Lscope49-__ZN17atomic_shared_ptrI1AE9_RefLocalC1Ev
	.stabd	78,0,0
	.align 2
	.globl __ZN17atomic_shared_ptrI1AEC1Ev
	.weak_definition __ZN17atomic_shared_ptrI1AEC1Ev
__ZN17atomic_shared_ptrI1AEC1Ev:
	.stabd	46,0,0
	.stabd	68,0,55
LFB1545:
	mflr r0
LCFI232:
	stmw r30,-8(r1)
LCFI233:
	stw r0,8(r1)
LCFI234:
	stwu r1,-80(r1)
LCFI235:
	mr r30,r1
LCFI236:
	stw r3,104(r30)
LBB52:
	.stabd	68,0,55
	lwz r2,104(r30)
	li r0,0
	stw r0,0(r2)
	lwz r2,104(r30)
	addi r0,r2,4
	mr r3,r0
	bl L__ZN17atomic_shared_ptrI1AE9_RefLocalC1Ev$stub
LBE52:
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r30,-8(r1)
	blr
LFE1545:
	.stabs	"_ZN17atomic_shared_ptrI1AEC1Ev:F(0,1)",36,0,55,__ZN17atomic_shared_ptrI1AEC1Ev
	.stabs	"this:p(0,150)",160,0,55,104
Lscope50:
	.stabs	"",36,0,0,Lscope50-__ZN17atomic_shared_ptrI1AEC1Ev
	.stabd	78,0,0
	.align 2
	.globl __Z10atomicSwapIP1AET_S2_PS2_
	.weak_definition __Z10atomicSwapIP1AET_S2_PS2_
__Z10atomicSwapIP1AET_S2_PS2_:
	.stabd	46,0,0
	.stabs	"../../kame/atomic.h",132,0,0,Ltext12
Ltext12:
	.stabd	68,0,147
LFB1604:
	stmw r30,-8(r1)
LCFI237:
	stwu r1,-64(r1)
LCFI238:
	mr r30,r1
LCFI239:
	stw r3,88(r30)
	stw r4,92(r30)
LBB53:
LBB54:
	.stabd	68,0,154
	lwz r2,88(r30)
	lwz r0,92(r30)
	1: lwarx r9, 0, r0;stwcx. r2, 0, r0;bne- 1b
	mr r0,r9
	stw r0,24(r30)
	.stabd	68,0,155
	lwz r0,24(r30)
LBE54:
LBE53:
	mr r3,r0
	lwz r1,0(r1)
	lmw r30,-8(r1)
	blr
LFE1604:
	.stabs	"_Z10atomicSwapIP1AET_S2_PS2_:F(0,68)",36,0,147,__Z10atomicSwapIP1AET_S2_PS2_
	.stabs	"newv:p(0,68)",160,0,147,88
	.stabs	"target:p(0,152)",160,0,147,92
	.stabs	"ret:(0,68)",128,0,148,24
	.stabs	":t(0,152)=*(0,68)",128,0,0,0
	.stabn	192,0,0,LBB54
	.stabn	224,0,0,LBE54
Lscope51:
	.stabs	"",36,0,0,Lscope51-__Z10atomicSwapIP1AET_S2_PS2_
	.stabd	78,0,0
	.align 2
	.globl __ZN17atomic_scoped_ptrI1AE4swapERS1_
	.weak_definition __ZN17atomic_scoped_ptrI1AE4swapERS1_
__ZN17atomic_scoped_ptrI1AE4swapERS1_:
	.stabd	46,0,0
	.stabs	"../../kame/atomic_smart_ptr.h",132,0,0,Ltext13
Ltext13:
	.stabd	68,0,34
LFB1556:
	mflr r0
LCFI240:
	stmw r30,-8(r1)
LCFI241:
	stw r0,8(r1)
LCFI242:
	stwu r1,-80(r1)
LCFI243:
	mr r30,r1
LCFI244:
	stw r3,104(r30)
	stw r4,108(r30)
LBB55:
	.stabd	68,0,35
	lwz r2,104(r30)
	lwz r0,0(r2)
	lwz r2,108(r30)
	mr r3,r0
	mr r4,r2
	bl L__Z10atomicSwapIP1AET_S2_PS2_$stub
	mr r0,r3
	lwz r2,104(r30)
	stw r0,0(r2)
LBE55:
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r30,-8(r1)
	blr
LFE1556:
	.stabs	"_ZN17atomic_scoped_ptrI1AE4swapERS1_:F(0,1)",36,0,34,__ZN17atomic_scoped_ptrI1AE4swapERS1_
	.stabs	"this:p(0,86)",160,0,34,104
	.stabs	"x:p(0,95)",160,0,34,108
Lscope52:
	.stabs	"",36,0,0,Lscope52-__ZN17atomic_scoped_ptrI1AE4swapERS1_
	.stabd	78,0,0
	.align 2
	.globl __ZN17atomic_scoped_ptrI1AE5resetEPS0_
	.weak_definition __ZN17atomic_scoped_ptrI1AE5resetEPS0_
__ZN17atomic_scoped_ptrI1AE5resetEPS0_:
	.stabd	46,0,0
	.stabd	68,0,28
LFB1555:
	mflr r0
LCFI245:
	stmw r30,-8(r1)
LCFI246:
	stw r0,8(r1)
LCFI247:
	stwu r1,-96(r1)
LCFI248:
	mr r30,r1
LCFI249:
	stw r3,120(r30)
	stw r4,124(r30)
LBB56:
LBB57:
	.stabd	68,0,29
	lwz r0,120(r30)
	lwz r3,124(r30)
	mr r4,r0
	bl L__Z10atomicSwapIP1AET_S2_PS2_$stub
	mr r0,r3
	stw r0,56(r30)
	.stabd	68,0,30
	lwz r0,56(r30)
	cmpwi cr7,r0,0
	beq cr7,L172
	lwz r2,56(r30)
	lwz r2,0(r2)
	addi r2,r2,4
	lwz r0,0(r2)
	lwz r3,56(r30)
	mr r12,r0
	mtctr r12
	bctrl
L172:
LBE57:
LBE56:
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r30,-8(r1)
	blr
LFE1555:
	.stabs	"_ZN17atomic_scoped_ptrI1AE5resetEPS0_:F(0,1)",36,0,28,__ZN17atomic_scoped_ptrI1AE5resetEPS0_
	.stabs	"this:p(0,86)",160,0,28,120
	.stabs	"t:p(0,68)",160,0,28,124
	.stabs	"old:(0,68)",128,0,29,56
	.stabn	192,0,0,LBB57
	.stabn	224,0,0,LBE57
Lscope53:
	.stabs	"",36,0,0,Lscope53-__ZN17atomic_scoped_ptrI1AE5resetEPS0_
	.stabd	78,0,0
	.align 2
	.globl __Z9atomicAddIiEvPT_S0_
	.weak_definition __Z9atomicAddIiEvPT_S0_
__Z9atomicAddIiEvPT_S0_:
	.stabd	46,0,0
	.stabs	"../../kame/atomic.h",132,0,0,Ltext14
Ltext14:
	.stabd	68,0,169
LFB1605:
	stmw r30,-8(r1)
LCFI250:
	stwu r1,-64(r1)
LCFI251:
	mr r30,r1
LCFI252:
	stw r3,88(r30)
	stw r4,92(r30)
LBB58:
LBB59:
	.stabd	68,0,177
	lwz r9,88(r30)
	lwz r2,92(r30)
	lwz r0,24(r30)
	1: lwarx r0, 0, r9;add r0, r0, r2;stwcx. r0, 0, r9;bne- 1b
	stw r0,24(r30)
LBE59:
LBE58:
	lwz r1,0(r1)
	lmw r30,-8(r1)
	blr
LFE1605:
	.stabs	"_Z9atomicAddIiEvPT_S0_:F(0,1)",36,0,169,__Z9atomicAddIiEvPT_S0_
	.stabs	"target:p(0,153)",160,0,169,88
	.stabs	"x:p(0,10)",160,0,169,92
	.stabs	"ret:(0,10)",128,0,170,24
	.stabs	":t(0,153)=*(0,10)",128,0,0,0
	.stabn	192,0,0,LBB59
	.stabn	224,0,0,LBE59
Lscope54:
	.stabs	"",36,0,0,Lscope54-__Z9atomicAddIiEvPT_S0_
	.stabd	78,0,0
	.align 2
	.globl __ZN6atomicIiEpLEi
	.weak_definition __ZN6atomicIiEpLEi
__ZN6atomicIiEpLEi:
	.stabd	46,0,0
	.stabd	68,0,214
LFB1567:
	mflr r0
LCFI253:
	stmw r30,-8(r1)
LCFI254:
	stw r0,8(r1)
LCFI255:
	stwu r1,-80(r1)
LCFI256:
	mr r30,r1
LCFI257:
	stw r3,104(r30)
	stw r4,108(r30)
LBB60:
	.stabd	68,0,214
	lwz r0,104(r30)
	mr r3,r0
	lwz r4,108(r30)
	bl L__Z9atomicAddIiEvPT_S0_$stub
	lwz r0,104(r30)
LBE60:
	mr r3,r0
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r30,-8(r1)
	blr
LFE1567:
	.stabs	"_ZN6atomicIiEpLEi:F(0,111)",36,0,214,__ZN6atomicIiEpLEi
	.stabs	"this:p(0,101)",160,0,214,104
	.stabs	"t:p(0,10)",160,0,214,108
Lscope55:
	.stabs	"",36,0,0,Lscope55-__ZN6atomicIiEpLEi
	.stabd	78,0,0
	.align 2
	.globl __Z9atomicAddIjEvPT_S0_
	.weak_definition __Z9atomicAddIjEvPT_S0_
__Z9atomicAddIjEvPT_S0_:
	.stabd	46,0,0
	.stabd	68,0,169
LFB1606:
	stmw r30,-8(r1)
LCFI258:
	stwu r1,-64(r1)
LCFI259:
	mr r30,r1
LCFI260:
	stw r3,88(r30)
	stw r4,92(r30)
LBB61:
LBB62:
	.stabd	68,0,177
	lwz r9,88(r30)
	lwz r2,92(r30)
	lwz r0,24(r30)
	1: lwarx r0, 0, r9;add r0, r0, r2;stwcx. r0, 0, r9;bne- 1b
	stw r0,24(r30)
LBE62:
LBE61:
	lwz r1,0(r1)
	lmw r30,-8(r1)
	blr
LFE1606:
	.stabs	"_Z9atomicAddIjEvPT_S0_:F(0,1)",36,0,169,__Z9atomicAddIjEvPT_S0_
	.stabs	"target:p(0,154)",160,0,169,88
	.stabs	"x:p(0,59)",160,0,169,92
	.stabs	"ret:(0,59)",128,0,170,24
	.stabs	"unsigned int:t(0,59)",128,0,0,0
	.stabs	":t(0,154)=*(0,59)",128,0,0,0
	.stabn	192,0,0,LBB62
	.stabn	224,0,0,LBE62
Lscope56:
	.stabs	"",36,0,0,Lscope56-__Z9atomicAddIjEvPT_S0_
	.stabd	78,0,0
	.align 2
	.globl __ZNK17atomic_shared_ptrI1AE9_RefLocal6serialEv
	.weak_definition __ZNK17atomic_shared_ptrI1AE9_RefLocal6serialEv
__ZNK17atomic_shared_ptrI1AE9_RefLocal6serialEv:
	.stabd	46,0,0
	.stabs	"../../kame/atomic_smart_ptr.h",132,0,0,Ltext15
Ltext15:
	.stabd	68,0,136
LFB1607:
	stmw r30,-8(r1)
LCFI261:
	stwu r1,-48(r1)
LCFI262:
	mr r30,r1
LCFI263:
	stw r3,72(r30)
LBB63:
	.stabd	68,0,137
	lwz r2,72(r30)
	lwz r0,4(r2)
	rlwinm r0,r0,0,16,31
LBE63:
	mr r3,r0
	lwz r1,0(r1)
	lmw r30,-8(r1)
	blr
LFE1607:
	.stabs	"_ZNK17atomic_shared_ptrI1AE9_RefLocal6serialEv:F(0,59)",36,0,136,__ZNK17atomic_shared_ptrI1AE9_RefLocal6serialEv
	.stabs	"this:p(0,151)",160,0,136,72
Lscope57:
	.stabs	"",36,0,0,Lscope57-__ZNK17atomic_shared_ptrI1AE9_RefLocal6serialEv
	.stabd	78,0,0
	.align 2
	.globl __ZNK17atomic_shared_ptrI1AE9_RefLocal16_refcnt_n_serialEjj
	.weak_definition __ZNK17atomic_shared_ptrI1AE9_RefLocal16_refcnt_n_serialEjj
__ZNK17atomic_shared_ptrI1AE9_RefLocal16_refcnt_n_serialEjj:
	.stabd	46,0,0
	.stabd	68,0,140
LFB1608:
	stmw r30,-8(r1)
LCFI264:
	stwu r1,-48(r1)
LCFI265:
	mr r30,r1
LCFI266:
	stw r3,72(r30)
	stw r4,76(r30)
	stw r5,80(r30)
LBB64:
	.stabd	68,0,142
	lwz r0,80(r30)
	rlwinm r2,r0,0,16,31
	lwz r0,76(r30)
	slwi r0,r0,16
	or r0,r2,r0
LBE64:
	mr r3,r0
	lwz r1,0(r1)
	lmw r30,-8(r1)
	blr
LFE1608:
	.stabs	"_ZNK17atomic_shared_ptrI1AE9_RefLocal16_refcnt_n_serialEjj:F(0,59)",36,0,140,__ZNK17atomic_shared_ptrI1AE9_RefLocal16_refcnt_n_serialEjj
	.stabs	"this:p(0,151)",160,0,140,72
	.stabs	"refcnt:p(0,59)",160,0,140,76
	.stabs	"serial:p(0,59)",160,0,140,80
Lscope58:
	.stabs	"",36,0,0,Lscope58-__ZNK17atomic_shared_ptrI1AE9_RefLocal16_refcnt_n_serialEjj
	.stabd	78,0,0
	.align 2
	.globl __Z16atomicAddAndTestIiEbPT_S0_
	.weak_definition __Z16atomicAddAndTestIiEbPT_S0_
__Z16atomicAddAndTestIiEbPT_S0_:
	.stabd	46,0,0
	.stabs	"../../kame/atomic.h",132,0,0,Ltext16
Ltext16:
	.stabd	68,0,181
LFB1609:
	stmw r30,-8(r1)
LCFI267:
	stwu r1,-64(r1)
LCFI268:
	mr r30,r1
LCFI269:
	stw r3,88(r30)
	stw r4,92(r30)
LBB65:
LBB66:
	.stabd	68,0,189
	lwz r2,88(r30)
	lwz r0,92(r30)
	1: lwarx r9, 0, r2;add r9, r9, r0;stwcx. r9, 0, r2;bne- 1b
	mr r0,r9
	stw r0,24(r30)
	.stabd	68,0,190
	lwz r0,24(r30)
	cmpwi cr7,r0,0
	mfcr r0
	rlwinm r0,r0,31,1
LBE66:
LBE65:
	mr r3,r0
	lwz r1,0(r1)
	lmw r30,-8(r1)
	blr
LFE1609:
	.stabs	"_Z16atomicAddAndTestIiEbPT_S0_:F(0,2)",36,0,181,__Z16atomicAddAndTestIiEbPT_S0_
	.stabs	"target:p(0,153)",160,0,181,88
	.stabs	"x:p(0,10)",160,0,181,92
	.stabs	"ret:(0,10)",128,0,182,24
	.stabn	192,0,0,LBB66
	.stabn	224,0,0,LBE66
Lscope59:
	.stabs	"",36,0,0,Lscope59-__Z16atomicAddAndTestIiEbPT_S0_
	.stabd	78,0,0
	.align 2
	.globl __Z16atomicDecAndTestIiEbPT_
	.weak_definition __Z16atomicDecAndTestIiEbPT_
__Z16atomicDecAndTestIiEbPT_:
	.stabd	46,0,0
	.stabd	68,0,194
LFB1610:
	mflr r0
LCFI270:
	stmw r30,-8(r1)
LCFI271:
	stw r0,8(r1)
LCFI272:
	stwu r1,-80(r1)
LCFI273:
	mr r30,r1
LCFI274:
	stw r3,104(r30)
LBB67:
	.stabd	68,0,195
	lwz r3,104(r30)
	li r4,-1
	bl L__Z16atomicAddAndTestIiEbPT_S0_$stub
	mr r0,r3
LBE67:
	mr r3,r0
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r30,-8(r1)
	blr
LFE1610:
	.stabs	"_Z16atomicDecAndTestIiEbPT_:F(0,2)",36,0,194,__Z16atomicDecAndTestIiEbPT_
	.stabs	"target:p(0,153)",160,0,194,104
Lscope60:
	.stabs	"",36,0,0,Lscope60-__Z16atomicDecAndTestIiEbPT_
	.stabd	78,0,0
	.align 2
	.globl __ZN17atomic_shared_ptrI1AE4swapERS1_
	.weak_definition __ZN17atomic_shared_ptrI1AE4swapERS1_
__ZN17atomic_shared_ptrI1AE4swapERS1_:
	.stabd	46,0,0
	.stabs	"../../kame/atomic_smart_ptr.h",132,0,0,Ltext17
Ltext17:
	.stabd	68,0,209
LFB1572:
	mflr r0
LCFI275:
	stmw r27,-20(r1)
LCFI276:
	stw r0,8(r1)
LCFI277:
	stwu r1,-144(r1)
LCFI278:
	mr r30,r1
LCFI279:
	bcl 20,31,"L00000000029$pb"
"L00000000029$pb":
	mflr r31
	stw r3,168(r30)
	stw r4,172(r30)
LBB68:
LBB69:
	.stabd	68,0,212
	lwz r2,168(r30)
	lwz r0,4(r2)
	cmpwi cr7,r0,0
	beq cr7,L219
	.stabd	68,0,213
	lwz r2,168(r30)
	lwz r2,4(r2)
	addi r29,r2,4
	lwz r2,168(r30)
	addi r0,r2,4
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AE9_RefLocal13refcnt_cachedEv$stub
	mr r0,r3
	mr r3,r29
	mr r4,r0
	bl L__Z9atomicAddIjEvPT_S0_$stub
	.stabd	68,0,214
	lwz r2,168(r30)
	addi r29,r2,4
	lwz r2,168(r30)
	addi r0,r2,4
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AE9_RefLocal6serialEv$stub
	mr r2,r3
	addi r0,r2,1
	mr r3,r29
	li r4,0
	mr r5,r0
	bl L__ZNK17atomic_shared_ptrI1AE9_RefLocal16_refcnt_n_serialEjj$stub
	mr r0,r3
	lwz r2,168(r30)
	stw r0,8(r2)
	b L219
L188:
L219:
LBB70:
	.stabd	68,0,217
	li r0,0
	stw r0,60(r30)
L190:
	.stabd	68,0,220
	lwz r2,172(r30)
	lwz r0,4(r2)
	stw r0,64(r30)
	.stabd	68,0,221
	lwz r2,172(r30)
	addi r0,r2,4
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AE9_RefLocal6serialEv$stub
	mr r0,r3
	stw r0,56(r30)
	.stabd	68,0,222
	lwz r0,64(r30)
	cmpwi cr7,r0,0
	beq cr7,L191
	.stabd	68,0,223
	lwz r2,172(r30)
	addi r0,r2,4
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AE9_RefLocal13refcnt_cachedEv$stub
	mr r0,r3
	stw r0,60(r30)
	.stabd	68,0,224
	li r3,1
	bl L__Z11_my_cassertm$stub
	.stabd	68,0,225
	lwz r0,64(r30)
	mr r29,r0
	lwz r2,172(r30)
	addi r0,r2,4
	mr r3,r0
	lwz r4,60(r30)
	lwz r5,56(r30)
	bl L__ZNK17atomic_shared_ptrI1AE9_RefLocal16_refcnt_n_serialEjj$stub
	mr r28,r3
	lwz r0,64(r30)
	mr r27,r0
	lwz r2,172(r30)
	addi r9,r2,4
	lwz r2,60(r30)
	addi r0,r2,1
	mr r3,r9
	mr r4,r0
	lwz r5,56(r30)
	bl L__ZNK17atomic_shared_ptrI1AE9_RefLocal16_refcnt_n_serialEjj$stub
	mr r9,r3
	lwz r2,172(r30)
	addi r0,r2,4
	mr r3,r29
	mr r4,r28
	mr r5,r27
	mr r6,r9
	mr r7,r0
	bl L__Z20atomicCompareAndSet2jjjjPj$stub
	mr r0,r3
	cmpwi cr7,r0,0
	beq cr7,L190
L191:
	.stabd	68,0,231
	lwz r0,64(r30)
	cmpwi cr7,r0,0
	beq cr7,L193
	.stabd	68,0,232
	lwz r2,64(r30)
	addi r0,r2,4
	mr r3,r0
	lwz r4,60(r30)
	bl L__Z9atomicAddIjEvPT_S0_$stub
L193:
	.stabd	68,0,234
	lwz r0,64(r30)
	stw r0,100(r30)
	lwz r2,172(r30)
	addi r2,r2,4
	stw r2,96(r30)
	lwz r0,64(r30)
	cmpwi cr7,r0,0
	beq cr7,L195
	lwz r2,60(r30)
	addi r2,r2,1
	stw r2,92(r30)
	b L197
L195:
	li r0,0
	stw r0,92(r30)
L197:
	lwz r3,96(r30)
	lwz r4,92(r30)
	lwz r5,56(r30)
	bl L__ZNK17atomic_shared_ptrI1AE9_RefLocal16_refcnt_n_serialEjj$stub
	mr r29,r3
	lwz r2,168(r30)
	lwz r0,4(r2)
	mr r28,r0
	lwz r2,172(r30)
	addi r0,r2,4
	lwz r2,56(r30)
	addi r2,r2,1
	mr r3,r0
	li r4,0
	mr r5,r2
	bl L__ZNK17atomic_shared_ptrI1AE9_RefLocal16_refcnt_n_serialEjj$stub
	mr r9,r3
	lwz r2,172(r30)
	addi r0,r2,4
	lwz r3,100(r30)
	mr r4,r29
	mr r5,r28
	mr r6,r9
	mr r7,r0
	bl L__Z20atomicCompareAndSet2jjjjPj$stub
	mr r0,r3
	cmpwi cr7,r0,0
	bne cr7,L198
	.stabd	68,0,239
	lwz r0,64(r30)
	cmpwi cr7,r0,0
	beq cr7,L188
	.stabd	68,0,240
	lwz r2,64(r30)
	addi r0,r2,4
	mr r2,r0
	lwz r0,60(r30)
	neg r0,r0
	mr r3,r2
	mr r4,r0
	bl L__Z16atomicAddAndTestIiEbPT_S0_$stub
	mr r0,r3
	cmpwi cr7,r0,0
	beq cr7,L201
	.stabd	68,0,241
	lwz r0,64(r30)
	stw r0,88(r30)
	lwz r2,88(r30)
	cmpwi cr7,r2,0
	beq cr7,L201
	lwz r3,88(r30)
	bl L__ZN17atomic_shared_ptrI1AE3RefD1Ev$stub
	lwz r3,88(r30)
	bl L__ZdlPv$stub
L201:
	.stabd	68,0,243
	lwz r2,172(r30)
	addi r0,r2,4
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AE9_RefLocal13refcnt_cachedEv$stub
	mr r0,r3
	stw r0,60(r30)
	.stabd	68,0,244
	lwz r0,64(r30)
	mr r29,r0
	lwz r2,172(r30)
	addi r0,r2,4
	mr r3,r0
	lwz r4,60(r30)
	lwz r5,56(r30)
	bl L__ZNK17atomic_shared_ptrI1AE9_RefLocal16_refcnt_n_serialEjj$stub
	mr r28,r3
	lwz r0,64(r30)
	mr r27,r0
	lwz r2,172(r30)
	addi r9,r2,4
	lwz r2,60(r30)
	addi r0,r2,-1
	mr r3,r9
	mr r4,r0
	lwz r5,56(r30)
	bl L__ZNK17atomic_shared_ptrI1AE9_RefLocal16_refcnt_n_serialEjj$stub
	mr r9,r3
	lwz r2,172(r30)
	addi r0,r2,4
	mr r3,r29
	mr r4,r28
	mr r5,r27
	mr r6,r9
	mr r7,r0
	bl L__Z20atomicCompareAndSet2jjjjPj$stub
	mr r0,r3
	cmpwi cr7,r0,0
	bne cr7,L188
	.stabd	68,0,249
	lwz r2,172(r30)
	lwz r2,4(r2)
	lwz r0,64(r30)
	cmpw cr7,r2,r0
	bne cr7,L205
	lwz r2,172(r30)
	addi r0,r2,4
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AE9_RefLocal6serialEv$stub
	mr r2,r3
	lwz r0,56(r30)
	cmpw cr7,r2,r0
	beq cr7,L207
L205:
	li r0,1
	stw r0,84(r30)
	b L208
L207:
	li r2,0
	stw r2,84(r30)
L208:
	lwz r0,84(r30)
	cmpwi cr7,r0,0
	beq cr7,L201
	.stabd	68,0,250
	lwz r2,172(r30)
	addi r0,r2,4
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AE9_RefLocal6serialEv$stub
	mr r2,r3
	lwz r0,56(r30)
	xor r0,r2,r0
	srawi r2,r0,31
	xor r0,r2,r0
	subf r0,r2,r0
	neg r0,r0
	srwi r0,r0,31
	mr r3,r0
	addis r2,r31,ha16(LC5-"L00000000029$pb")
	la r4,lo16(LC5-"L00000000029$pb")(r2)
	li r5,250
	bl L__Z10_my_assertbPKci$stub
	.stabd	68,0,251
	lwz r2,64(r30)
	addi r0,r2,4
	mr r3,r0
	bl L__Z16atomicDecAndTestIiEbPT_$stub
	mr r0,r3
	cmpwi cr7,r0,0
	beq cr7,L188
	.stabd	68,0,252
	lwz r0,64(r30)
	stw r0,80(r30)
	lwz r2,80(r30)
	cmpwi cr7,r2,0
	beq cr7,L188
	lwz r3,80(r30)
	bl L__ZN17atomic_shared_ptrI1AE3RefD1Ev$stub
	lwz r3,80(r30)
	bl L__ZdlPv$stub
	b L188
L198:
LBE70:
	.stabd	68,0,264
	lwz r2,168(r30)
	lwz r0,64(r30)
	stw r0,4(r2)
	.stabd	68,0,265
	lwz r2,168(r30)
	lwz r0,4(r2)
	cmpwi cr7,r0,0
	beq cr7,L212
	lwz r2,168(r30)
	lwz r2,4(r2)
	lwz r2,0(r2)
	stw r2,76(r30)
	b L214
L212:
	li r0,0
	stw r0,76(r30)
L214:
	lwz r2,168(r30)
	lwz r0,76(r30)
	stw r0,0(r2)
	.stabd	68,0,266
	lwz r2,172(r30)
	lwz r0,4(r2)
	cmpwi cr7,r0,0
	beq cr7,L215
	lwz r2,172(r30)
	lwz r2,4(r2)
	lwz r2,0(r2)
	stw r2,72(r30)
	b L217
L215:
	li r2,0
	stw r2,72(r30)
L217:
	lwz r2,172(r30)
	lwz r0,72(r30)
	stw r0,0(r2)
	.stabd	68,0,267
	bl L__Z13memoryBarrierv$stub
LBE69:
LBE68:
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r27,-20(r1)
	blr
LFE1572:
	.stabs	"_ZN17atomic_shared_ptrI1AE4swapERS1_:F(0,1)",36,0,209,__ZN17atomic_shared_ptrI1AE4swapERS1_
	.stabs	"this:p(0,150)",160,0,209,168
	.stabs	"r:p(0,128)",160,0,209,172
	.stabs	"pref:(0,134)",128,0,210,64
	.stabn	192,0,0,LBB69
	.stabs	"refcnt_cached:(0,59)",128,0,217,60
	.stabs	"serial:(0,59)",128,0,218,56
	.stabn	192,0,0,LBB70
	.stabn	224,0,0,LBE70
	.stabn	224,0,0,LBE69
Lscope61:
	.stabs	"",36,0,0,Lscope61-__ZN17atomic_shared_ptrI1AE4swapERS1_
	.stabd	78,0,0
	.align 2
	.globl __Z16atomicAddAndTestIjEbPT_S0_
	.weak_definition __Z16atomicAddAndTestIjEbPT_S0_
__Z16atomicAddAndTestIjEbPT_S0_:
	.stabd	46,0,0
	.stabs	"../../kame/atomic.h",132,0,0,Ltext18
Ltext18:
	.stabd	68,0,181
LFB1612:
	stmw r30,-8(r1)
LCFI280:
	stwu r1,-64(r1)
LCFI281:
	mr r30,r1
LCFI282:
	stw r3,88(r30)
	stw r4,92(r30)
LBB71:
LBB72:
	.stabd	68,0,189
	lwz r2,88(r30)
	lwz r0,92(r30)
	1: lwarx r9, 0, r2;add r9, r9, r0;stwcx. r9, 0, r2;bne- 1b
	mr r0,r9
	stw r0,24(r30)
	.stabd	68,0,190
	lwz r0,24(r30)
	cmpwi cr7,r0,0
	mfcr r0
	rlwinm r0,r0,31,1
LBE72:
LBE71:
	mr r3,r0
	lwz r1,0(r1)
	lmw r30,-8(r1)
	blr
LFE1612:
	.stabs	"_Z16atomicAddAndTestIjEbPT_S0_:F(0,2)",36,0,181,__Z16atomicAddAndTestIjEbPT_S0_
	.stabs	"target:p(0,154)",160,0,181,88
	.stabs	"x:p(0,59)",160,0,181,92
	.stabs	"ret:(0,59)",128,0,182,24
	.stabn	192,0,0,LBB72
	.stabn	224,0,0,LBE72
Lscope62:
	.stabs	"",36,0,0,Lscope62-__Z16atomicAddAndTestIjEbPT_S0_
	.stabd	78,0,0
	.align 2
	.globl __Z16atomicDecAndTestIjEbPT_
	.weak_definition __Z16atomicDecAndTestIjEbPT_
__Z16atomicDecAndTestIjEbPT_:
	.stabd	46,0,0
	.stabd	68,0,194
LFB1593:
	mflr r0
LCFI283:
	stmw r30,-8(r1)
LCFI284:
	stw r0,8(r1)
LCFI285:
	stwu r1,-80(r1)
LCFI286:
	mr r30,r1
LCFI287:
	stw r3,104(r30)
LBB73:
	.stabd	68,0,195
	lwz r3,104(r30)
	li r4,-1
	bl L__Z16atomicAddAndTestIjEbPT_S0_$stub
	mr r0,r3
LBE73:
	mr r3,r0
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r30,-8(r1)
	blr
LFE1593:
	.stabs	"_Z16atomicDecAndTestIjEbPT_:F(0,2)",36,0,194,__Z16atomicDecAndTestIjEbPT_
	.stabs	"target:p(0,154)",160,0,194,104
Lscope63:
	.stabs	"",36,0,0,Lscope63-__Z16atomicDecAndTestIjEbPT_
	.stabd	78,0,0
	.align 2
	.globl __ZN17atomic_shared_ptrI1AED1Ev
	.weak_definition __ZN17atomic_shared_ptrI1AED1Ev
__ZN17atomic_shared_ptrI1AED1Ev:
	.stabd	46,0,0
	.stabs	"../../kame/atomic_smart_ptr.h",132,0,0,Ltext19
Ltext19:
	.stabd	68,0,154
LFB1539:
	mflr r0
LCFI288:
	stmw r30,-8(r1)
LCFI289:
	stw r0,8(r1)
LCFI290:
	stwu r1,-96(r1)
LCFI291:
	mr r30,r1
LCFI292:
	bcl 20,31,"L00000000030$pb"
"L00000000030$pb":
	mflr r31
	stw r3,120(r30)
LBB74:
	.stabd	68,0,155
	lwz r2,120(r30)
	lwz r0,4(r2)
	stw r0,56(r30)
	.stabd	68,0,156
	lwz r0,56(r30)
	cmpwi cr7,r0,0
	beq cr7,L229
	.stabd	68,0,158
	lwz r2,120(r30)
	addi r0,r2,4
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AE9_RefLocal13refcnt_cachedEv$stub
	mr r0,r3
	cmpwi cr7,r0,0
	mfcr r0
	rlwinm r0,r0,31,1
	mr r3,r0
	addis r2,r31,ha16(LC5-"L00000000030$pb")
	la r4,lo16(LC5-"L00000000030$pb")(r2)
	li r5,158
	bl L__Z10_my_assertbPKci$stub
	.stabd	68,0,160
	lwz r2,56(r30)
	addi r0,r2,4
	mr r3,r0
	bl L__Z16atomicDecAndTestIjEbPT_$stub
	mr r0,r3
	cmpwi cr7,r0,0
	beq cr7,L229
	.stabd	68,0,161
	lwz r0,56(r30)
	stw r0,72(r30)
	lwz r0,72(r30)
	cmpwi cr7,r0,0
	beq cr7,L229
	lwz r3,72(r30)
	bl L__ZN17atomic_shared_ptrI1AE3RefD1Ev$stub
	lwz r3,72(r30)
	bl L__ZdlPv$stub
L229:
LBE74:
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r30,-8(r1)
	blr
LFE1539:
	.stabs	"_ZN17atomic_shared_ptrI1AED1Ev:F(0,1)",36,0,154,__ZN17atomic_shared_ptrI1AED1Ev
	.stabs	"this:p(0,150)",160,0,154,120
	.stabs	"pref:(0,134)",128,0,155,56
	.stabn	192,0,0,LBB74
	.stabn	224,0,0,LBE74
Lscope64:
	.stabs	"",36,0,0,Lscope64-__ZN17atomic_shared_ptrI1AED1Ev
	.stabd	78,0,0
	.align 2
	.globl __ZN17atomic_shared_ptrI1AE5resetIS0_EEvPT_
	.weak_definition __ZN17atomic_shared_ptrI1AE5resetIS0_EEvPT_
__ZN17atomic_shared_ptrI1AE5resetIS0_EEvPT_:
	.stabd	46,0,0
	.stabd	68,0,84
LFB1569:
	mflr r0
LCFI293:
	stmw r29,-12(r1)
LCFI294:
	stw r0,8(r1)
LCFI295:
	stwu r1,-96(r1)
LCFI296:
	mr r30,r1
LCFI297:
	stw r3,120(r30)
	stw r4,124(r30)
LBB75:
	.stabd	68,0,85
	addi r3,r30,56
	lwz r4,124(r30)
LEHB11:
	bl L__ZN17atomic_shared_ptrI1AEC1IS0_EEPT_$stub
LEHE11:
	addi r3,r30,56
	lwz r4,120(r30)
LEHB12:
	bl L__ZN17atomic_shared_ptrI1AE4swapERS1_$stub
LEHE12:
	b L231
L234:
	stw r3,72(r30)
L232:
	lwz r29,72(r30)
	addi r3,r30,56
	bl L__ZN17atomic_shared_ptrI1AED1Ev$stub
	stw r29,72(r30)
	lwz r3,72(r30)
LEHB13:
	bl L__Unwind_Resume$stub
L231:
	addi r3,r30,56
	bl L__ZN17atomic_shared_ptrI1AED1Ev$stub
LEHE13:
LBE75:
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r29,-12(r1)
	blr
LFE1569:
	.section __DATA,__gcc_except_tab
LLSDA1569:
	.byte	0xff
	.byte	0xff
	.byte	0x3
	.byte	0x27
	.set L$set$27,LEHB11-LFB1569
	.long L$set$27
	.set L$set$28,LEHE11-LEHB11
	.long L$set$28
	.long	0x0
	.byte	0x0
	.set L$set$29,LEHB12-LFB1569
	.long L$set$29
	.set L$set$30,LEHE12-LEHB12
	.long L$set$30
	.set L$set$31,L234-LFB1569
	.long L$set$31
	.byte	0x0
	.set L$set$32,LEHB13-LFB1569
	.long L$set$32
	.set L$set$33,LEHE13-LEHB13
	.long L$set$33
	.long	0x0
	.byte	0x0
	.section __TEXT,__textcoal_nt,coalesced,pure_instructions
	.stabs	"_ZN17atomic_shared_ptrI1AE5resetIS0_EEvPT_:F(0,1)",36,0,84,__ZN17atomic_shared_ptrI1AE5resetIS0_EEvPT_
	.stabs	"this:p(0,150)",160,0,84,120
	.stabs	"y:p(0,68)",160,0,84,124
Lscope65:
	.stabs	"",36,0,0,Lscope65-__ZN17atomic_shared_ptrI1AE5resetIS0_EEvPT_
	.stabd	78,0,0
	.align 2
	.globl __ZN17atomic_shared_ptrI1AE5resetEv
	.weak_definition __ZN17atomic_shared_ptrI1AE5resetEv
__ZN17atomic_shared_ptrI1AE5resetEv:
	.stabd	46,0,0
	.stabd	68,0,81
LFB1571:
	mflr r0
LCFI298:
	stmw r29,-12(r1)
LCFI299:
	stw r0,8(r1)
LCFI300:
	stwu r1,-96(r1)
LCFI301:
	mr r30,r1
LCFI302:
	stw r3,120(r30)
LBB76:
	.stabd	68,0,82
	addi r3,r30,56
	bl L__ZN17atomic_shared_ptrI1AEC1Ev$stub
	addi r3,r30,56
	lwz r4,120(r30)
LEHB14:
	bl L__ZN17atomic_shared_ptrI1AE4swapERS1_$stub
LEHE14:
	b L236
L239:
	stw r3,72(r30)
L237:
	lwz r29,72(r30)
	addi r3,r30,56
	bl L__ZN17atomic_shared_ptrI1AED1Ev$stub
	stw r29,72(r30)
	lwz r3,72(r30)
LEHB15:
	bl L__Unwind_Resume$stub
L236:
	addi r3,r30,56
	bl L__ZN17atomic_shared_ptrI1AED1Ev$stub
LEHE15:
LBE76:
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r29,-12(r1)
	blr
LFE1571:
	.section __DATA,__gcc_except_tab
LLSDA1571:
	.byte	0xff
	.byte	0xff
	.byte	0x3
	.byte	0x1a
	.set L$set$34,LEHB14-LFB1571
	.long L$set$34
	.set L$set$35,LEHE14-LEHB14
	.long L$set$35
	.set L$set$36,L239-LFB1571
	.long L$set$36
	.byte	0x0
	.set L$set$37,LEHB15-LFB1571
	.long L$set$37
	.set L$set$38,LEHE15-LEHB15
	.long L$set$38
	.long	0x0
	.byte	0x0
	.section __TEXT,__textcoal_nt,coalesced,pure_instructions
	.stabs	"_ZN17atomic_shared_ptrI1AE5resetEv:F(0,1)",36,0,81,__ZN17atomic_shared_ptrI1AE5resetEv
	.stabs	"this:p(0,150)",160,0,81,120
Lscope66:
	.stabs	"",36,0,0,Lscope66-__ZN17atomic_shared_ptrI1AE5resetEv
	.stabd	78,0,0
	.align 2
	.globl __Z9atomicIncIjEvPT_
	.weak_definition __Z9atomicIncIjEvPT_
__Z9atomicIncIjEvPT_:
	.stabd	46,0,0
	.stabs	"../../kame/atomic.h",132,0,0,Ltext20
Ltext20:
	.stabd	68,0,158
LFB1613:
	stmw r30,-8(r1)
LCFI303:
	stwu r1,-64(r1)
LCFI304:
	mr r30,r1
LCFI305:
	stw r3,88(r30)
LBB77:
LBB78:
	.stabd	68,0,166
	lwz r0,88(r30)
	lwz r2,24(r30)
	mr r9,r2
	1: lwarx r9, 0, r0;addi r9, r9, 1;stwcx. r9, 0, r0;bne- 1b
	mr r0,r9
	stw r0,24(r30)
LBE78:
LBE77:
	lwz r1,0(r1)
	lmw r30,-8(r1)
	blr
LFE1613:
	.stabs	"_Z9atomicIncIjEvPT_:F(0,1)",36,0,158,__Z9atomicIncIjEvPT_
	.stabs	"target:p(0,154)",160,0,158,88
	.stabs	"ret:(0,59)",128,0,159,24
	.stabn	192,0,0,LBB78
	.stabn	224,0,0,LBE78
Lscope67:
	.stabs	"",36,0,0,Lscope67-__Z9atomicIncIjEvPT_
	.stabd	78,0,0
	.align 2
	.globl __ZNK17atomic_shared_ptrI1AE6_scan_Ev
	.weak_definition __ZNK17atomic_shared_ptrI1AE6_scan_Ev
__ZNK17atomic_shared_ptrI1AE6_scan_Ev:
	.stabd	46,0,0
	.stabs	"../../kame/atomic_smart_ptr.h",132,0,0,Ltext21
Ltext21:
	.stabd	68,0,166
LFB1603:
	mflr r0
LCFI306:
	stmw r27,-20(r1)
LCFI307:
	stw r0,8(r1)
LCFI308:
	stwu r1,-112(r1)
LCFI309:
	mr r30,r1
LCFI310:
	bcl 20,31,"L00000000031$pb"
"L00000000031$pb":
	mflr r31
	stw r3,136(r30)
L243:
LBB79:
LBB80:
LBB81:
	.stabd	68,0,171
	lwz r2,136(r30)
	lwz r0,4(r2)
	stw r0,68(r30)
	.stabd	68,0,172
	lwz r0,68(r30)
	cmpwi cr7,r0,0
	bne cr7,L244
	li r0,0
	stw r0,76(r30)
	b L246
L244:
	.stabd	68,0,174
	lwz r2,136(r30)
	addi r0,r2,4
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AE9_RefLocal13refcnt_cachedEv$stub
	mr r0,r3
	stw r0,60(r30)
	.stabd	68,0,175
	lwz r2,136(r30)
	addi r0,r2,4
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AE9_RefLocal6serialEv$stub
	mr r0,r3
	stw r0,64(r30)
	.stabd	68,0,176
	li r3,1
	bl L__Z11_my_cassertm$stub
	.stabd	68,0,177
	lwz r0,68(r30)
	mr r29,r0
	lwz r2,136(r30)
	addi r0,r2,4
	mr r3,r0
	lwz r4,60(r30)
	lwz r5,64(r30)
	bl L__ZNK17atomic_shared_ptrI1AE9_RefLocal16_refcnt_n_serialEjj$stub
	mr r28,r3
	lwz r0,68(r30)
	mr r27,r0
	lwz r2,136(r30)
	addi r9,r2,4
	lwz r2,60(r30)
	addi r0,r2,1
	mr r3,r9
	mr r4,r0
	lwz r5,64(r30)
	bl L__ZNK17atomic_shared_ptrI1AE9_RefLocal16_refcnt_n_serialEjj$stub
	mr r9,r3
	lwz r2,136(r30)
	addi r0,r2,4
	mr r3,r29
	mr r4,r28
	mr r5,r27
	mr r6,r9
	mr r7,r0
	bl L__Z20atomicCompareAndSet2jjjjPj$stub
	mr r0,r3
	cmpwi cr7,r0,0
	beq cr7,L243
LBE81:
	.stabd	68,0,183
	lwz r2,68(r30)
	addi r0,r2,4
	mr r3,r0
	bl L__Z9atomicIncIjEvPT_$stub
L248:
LBB82:
	.stabd	68,0,186
	lwz r2,136(r30)
	addi r0,r2,4
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AE9_RefLocal13refcnt_cachedEv$stub
	mr r0,r3
	stw r0,56(r30)
	.stabd	68,0,187
	lwz r0,68(r30)
	mr r29,r0
	lwz r2,136(r30)
	addi r0,r2,4
	mr r3,r0
	lwz r4,56(r30)
	lwz r5,64(r30)
	bl L__ZNK17atomic_shared_ptrI1AE9_RefLocal16_refcnt_n_serialEjj$stub
	mr r28,r3
	lwz r0,68(r30)
	mr r27,r0
	lwz r2,136(r30)
	addi r9,r2,4
	lwz r2,56(r30)
	addi r0,r2,-1
	mr r3,r9
	mr r4,r0
	lwz r5,64(r30)
	bl L__ZNK17atomic_shared_ptrI1AE9_RefLocal16_refcnt_n_serialEjj$stub
	mr r9,r3
	lwz r2,136(r30)
	addi r0,r2,4
	mr r3,r29
	mr r4,r28
	mr r5,r27
	mr r6,r9
	mr r7,r0
	bl L__Z20atomicCompareAndSet2jjjjPj$stub
	mr r0,r3
	cmpwi cr7,r0,0
	bne cr7,L249
	.stabd	68,0,192
	lwz r2,136(r30)
	lwz r2,4(r2)
	lwz r0,68(r30)
	cmpw cr7,r2,r0
	bne cr7,L251
	lwz r2,136(r30)
	addi r0,r2,4
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AE9_RefLocal6serialEv$stub
	mr r2,r3
	lwz r0,64(r30)
	cmpw cr7,r2,r0
	beq cr7,L253
L251:
	li r0,1
	stw r0,72(r30)
	b L254
L253:
	li r2,0
	stw r2,72(r30)
L254:
	lwz r0,72(r30)
	cmpwi cr7,r0,0
	beq cr7,L248
	.stabd	68,0,193
	lwz r2,136(r30)
	addi r0,r2,4
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AE9_RefLocal6serialEv$stub
	mr r2,r3
	lwz r0,64(r30)
	xor r0,r2,r0
	srawi r2,r0,31
	xor r0,r2,r0
	subf r0,r2,r0
	neg r0,r0
	srwi r0,r0,31
	mr r3,r0
	addis r2,r31,ha16(LC5-"L00000000031$pb")
	la r4,lo16(LC5-"L00000000031$pb")(r2)
	li r5,193
	bl L__Z10_my_assertbPKci$stub
	.stabd	68,0,194
	lwz r2,68(r30)
	addi r0,r2,4
	mr r3,r0
	bl L__Z16atomicDecAndTestIiEbPT_$stub
	mr r0,r3
	cmpwi cr7,r0,0
	beq cr7,L249
	.stabd	68,0,195
	li r3,1
	addis r2,r31,ha16(LC5-"L00000000031$pb")
	la r4,lo16(LC5-"L00000000031$pb")(r2)
	li r5,195
	bl L__Z10_my_assertbPKci$stub
L249:
LBE82:
	.stabd	68,0,204
	lwz r0,68(r30)
	stw r0,76(r30)
L246:
	lwz r0,76(r30)
LBE80:
LBE79:
	mr r3,r0
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r27,-20(r1)
	blr
LFE1603:
	.stabs	"_ZNK17atomic_shared_ptrI1AE6_scan_Ev:F(0,134)",36,0,166,__ZNK17atomic_shared_ptrI1AE6_scan_Ev
	.stabs	"this:p(0,118)",160,0,166,136
	.stabs	"pref:(0,134)",128,0,167,68
	.stabs	"serial:(0,59)",128,0,169,64
	.stabn	192,0,0,LBB80
	.stabs	"refcnt_cached:(0,59)",128,0,173,60
	.stabn	192,0,0,LBB81
	.stabn	224,0,0,LBE81
	.stabs	"refcnt_cached:(0,59)",128,0,185,56
	.stabn	192,0,0,LBB82
	.stabn	224,0,0,LBE82
	.stabn	224,0,0,LBE80
Lscope68:
	.stabs	"",36,0,0,Lscope68-__ZNK17atomic_shared_ptrI1AE6_scan_Ev
	.stabd	78,0,0
	.align 2
	.globl __ZN17atomic_shared_ptrI1AEC1ERKS1_
	.weak_definition __ZN17atomic_shared_ptrI1AEC1ERKS1_
__ZN17atomic_shared_ptrI1AEC1ERKS1_:
	.stabd	46,0,0
	.stabd	68,0,62
LFB1548:
	mflr r0
LCFI311:
	stmw r29,-12(r1)
LCFI312:
	stw r0,8(r1)
LCFI313:
	stwu r1,-80(r1)
LCFI314:
	mr r30,r1
LCFI315:
	stw r3,104(r30)
	stw r4,108(r30)
LBB83:
	.stabd	68,0,62
	lwz r2,104(r30)
	addi r29,r2,4
	lwz r3,108(r30)
	bl L__ZNK17atomic_shared_ptrI1AE6_scan_Ev$stub
	mr r0,r3
	mr r3,r29
	mr r4,r0
	bl L__ZN17atomic_shared_ptrI1AE9_RefLocalC1INS1_3RefEEEPT_$stub
	.stabd	68,0,63
	lwz r2,104(r30)
	lwz r0,4(r2)
	cmpwi cr7,r0,0
	beq cr7,L259
	lwz r2,104(r30)
	lwz r2,4(r2)
	lwz r2,0(r2)
	stw r2,56(r30)
	b L261
L259:
	li r0,0
	stw r0,56(r30)
L261:
	lwz r2,104(r30)
	lwz r0,56(r30)
	stw r0,0(r2)
	.stabd	68,0,64
	bl L__Z13memoryBarrierv$stub
LBE83:
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r29,-12(r1)
	blr
LFE1548:
	.stabs	"_ZN17atomic_shared_ptrI1AEC1ERKS1_:F(0,1)",36,0,62,__ZN17atomic_shared_ptrI1AEC1ERKS1_
	.stabs	"this:p(0,150)",160,0,62,104
	.stabs	"t:p(0,126)",160,0,62,108
Lscope69:
	.stabs	"",36,0,0,Lscope69-__ZN17atomic_shared_ptrI1AEC1ERKS1_
	.stabd	78,0,0
	.align 2
	.globl __ZN17atomic_shared_ptrI1AEaSERKS1_
	.weak_definition __ZN17atomic_shared_ptrI1AEaSERKS1_
__ZN17atomic_shared_ptrI1AEaSERKS1_:
	.stabd	46,0,0
	.stabd	68,0,73
LFB1570:
	mflr r0
LCFI316:
	stmw r29,-12(r1)
LCFI317:
	stw r0,8(r1)
LCFI318:
	stwu r1,-96(r1)
LCFI319:
	mr r30,r1
LCFI320:
	stw r3,120(r30)
	stw r4,124(r30)
LBB84:
	.stabd	68,0,74
	addi r3,r30,56
	lwz r4,124(r30)
LEHB16:
	bl L__ZN17atomic_shared_ptrI1AEC1ERKS1_$stub
LEHE16:
	addi r3,r30,56
	lwz r4,120(r30)
LEHB17:
	bl L__ZN17atomic_shared_ptrI1AE4swapERS1_$stub
LEHE17:
	b L264
L267:
	stw r3,72(r30)
L265:
	lwz r29,72(r30)
	addi r3,r30,56
	bl L__ZN17atomic_shared_ptrI1AED1Ev$stub
	stw r29,72(r30)
	lwz r3,72(r30)
LEHB18:
	bl L__Unwind_Resume$stub
L264:
	addi r3,r30,56
	bl L__ZN17atomic_shared_ptrI1AED1Ev$stub
LEHE18:
	.stabd	68,0,75
	lwz r0,120(r30)
LBE84:
	mr r3,r0
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r29,-12(r1)
	blr
LFE1570:
	.section __DATA,__gcc_except_tab
LLSDA1570:
	.byte	0xff
	.byte	0xff
	.byte	0x3
	.byte	0x27
	.set L$set$39,LEHB16-LFB1570
	.long L$set$39
	.set L$set$40,LEHE16-LEHB16
	.long L$set$40
	.long	0x0
	.byte	0x0
	.set L$set$41,LEHB17-LFB1570
	.long L$set$41
	.set L$set$42,LEHE17-LEHB17
	.long L$set$42
	.set L$set$43,L267-LFB1570
	.long L$set$43
	.byte	0x0
	.set L$set$44,LEHB18-LFB1570
	.long L$set$44
	.set L$set$45,LEHE18-LEHB18
	.long L$set$45
	.long	0x0
	.byte	0x0
	.section __TEXT,__textcoal_nt,coalesced,pure_instructions
	.stabs	"_ZN17atomic_shared_ptrI1AEaSERKS1_:F(0,128)",36,0,73,__ZN17atomic_shared_ptrI1AEaSERKS1_
	.stabs	"this:p(0,150)",160,0,73,120
	.stabs	"t:p(0,126)",160,0,73,124
Lscope70:
	.stabs	"",36,0,0,Lscope70-__ZN17atomic_shared_ptrI1AEaSERKS1_
	.stabd	78,0,0
	.cstring
	.align 2
LC6:
	.ascii "p1 %d\12\0"
	.align 2
LC7:
	.ascii "p2 %d\12\0"
	.align 2
LC8:
	.ascii "p4 %d\12\0"
	.align 2
LC9:
	.ascii "p6 %d\12\0"
	.align 2
LC10:
	.ascii "p3 %d\12\0"
	.align 2
LC11:
	.ascii "p5 %d\12\0"
	.text
	.align 2
	.globl _main
_main:
	.stabd	46,0,0
	.stabs	"../../tools/testatomic.cpp",132,0,0,Ltext22
Ltext22:
	.stabd	68,0,65
LFB1497:
	mflr r0
LCFI321:
	stmw r29,-12(r1)
LCFI322:
	stw r0,8(r1)
LCFI323:
	stwu r1,-224(r1)
LCFI324:
	mr r30,r1
LCFI325:
	bcl 20,31,"L00000000032$pb"
"L00000000032$pb":
	mflr r31
	stw r3,248(r30)
	stw r4,252(r30)
LBB85:
	.stabd	68,0,67
	lwz r2,252(r30)
	addi r2,r2,4
	lwz r0,0(r2)
	mr r3,r0
	bl L_atoi$stub
	mr r0,r3
	mr r3,r0
	bl L_srand$stub
	.stabd	68,0,68
	li r3,8
LEHB19:
	bl L__Znwm$stub
LEHE19:
	mr r0,r3
	stw r0,192(r30)
	lwz r3,192(r30)
	li r4,10
LEHB20:
	bl L__ZN1AC1Ei$stub
LEHE20:
	b L269
L327:
	stw r3,196(r30)
L270:
	lwz r29,196(r30)
	lwz r3,192(r30)
	bl L__ZdlPv$stub
	stw r29,196(r30)
	lwz r3,196(r30)
LEHB21:
	bl L__Unwind_Resume$stub
L269:
	addi r3,r30,56
	lwz r4,192(r30)
	bl L__ZN17atomic_shared_ptrI1AEC1IS0_EEPT_$stub
LEHE21:
	.stabd	68,0,69
	li r3,8
LEHB22:
	bl L__Znwm$stub
LEHE22:
	mr r0,r3
	stw r0,188(r30)
	lwz r3,188(r30)
	li r4,11
LEHB23:
	bl L__ZN1BC1Ei$stub
LEHE23:
	b L271
L325:
	stw r3,196(r30)
L272:
	lwz r29,196(r30)
	lwz r3,188(r30)
	bl L__ZdlPv$stub
	stw r29,196(r30)
	b L306
L271:
	addi r0,r30,68
	mr r3,r0
	lwz r4,188(r30)
LEHB24:
	bl L__ZN17atomic_shared_ptrI1AEC1I1BEEPT_$stub
LEHE24:
	.stabd	68,0,70
	addi r0,r30,80
	mr r3,r0
	bl L__ZN17atomic_shared_ptrI1AEC1Ev$stub
	.stabd	68,0,71
	addi r0,r30,92
	mr r3,r0
	addi r4,r30,56
LEHB25:
	bl L__ZN17atomic_shared_ptrI1AEC1ERKS1_$stub
LEHE25:
	.stabd	68,0,72
	addi r0,r30,104
	addi r2,r30,80
	mr r3,r0
	mr r4,r2
LEHB26:
	bl L__ZN17atomic_shared_ptrI1AEC1ERKS1_$stub
LEHE26:
	.stabd	68,0,73
	li r3,8
LEHB27:
	bl L__Znwm$stub
LEHE27:
	mr r0,r3
	stw r0,184(r30)
	lwz r3,184(r30)
	li r4,15
LEHB28:
	bl L__ZN1AC1Ei$stub
LEHE28:
	b L273
L320:
	stw r3,196(r30)
L274:
	lwz r29,196(r30)
	lwz r3,184(r30)
	bl L__ZdlPv$stub
	stw r29,196(r30)
	b L298
L273:
	addi r0,r30,116
	mr r3,r0
	lwz r4,184(r30)
LEHB29:
	bl L__ZN17atomic_shared_ptrI1AEC1IS0_EEPT_$stub
LEHE29:
	.stabd	68,0,74
	li r3,8
LEHB30:
	bl L__Znwm$stub
LEHE30:
	mr r0,r3
	stw r0,180(r30)
	lwz r3,180(r30)
	li r4,20
LEHB31:
	bl L__ZN1AC1Ei$stub
LEHE31:
	b L275
L318:
	stw r3,196(r30)
L276:
	lwz r29,196(r30)
	lwz r3,180(r30)
	bl L__ZdlPv$stub
	stw r29,196(r30)
	b L296
L275:
	addi r0,r30,128
	mr r3,r0
	lwz r4,180(r30)
	bl L__ZN17atomic_scoped_ptrI1AEC1EPS0_$stub
	.stabd	68,0,75
	li r3,8
LEHB32:
	bl L__Znwm$stub
LEHE32:
	mr r0,r3
	stw r0,176(r30)
	lwz r3,176(r30)
	li r4,21
LEHB33:
	bl L__ZN1AC1Ei$stub
LEHE33:
	b L277
L316:
	stw r3,196(r30)
L278:
	lwz r29,196(r30)
	lwz r3,176(r30)
	bl L__ZdlPv$stub
	stw r29,196(r30)
	b L294
L277:
	addi r0,r30,132
	mr r3,r0
	lwz r4,176(r30)
	bl L__ZN17atomic_scoped_ptrI1AEC1EPS0_$stub
	.stabd	68,0,76
	li r3,8
LEHB34:
	bl L__Znwm$stub
LEHE34:
	mr r0,r3
	stw r0,172(r30)
	lwz r3,172(r30)
	li r4,22
LEHB35:
	bl L__ZN1AC1Ei$stub
LEHE35:
	b L279
L314:
	stw r3,196(r30)
L280:
	lwz r29,196(r30)
	lwz r3,172(r30)
	bl L__ZdlPv$stub
	stw r29,196(r30)
	b L292
L279:
	addi r0,r30,136
	mr r3,r0
	lwz r4,172(r30)
	bl L__ZN17atomic_scoped_ptrI1AEC1EPS0_$stub
	.stabd	68,0,77
	addi r0,r30,136
	mr r3,r0
	li r4,0
LEHB36:
	bl L__ZN17atomic_scoped_ptrI1AE5resetEPS0_$stub
	.stabd	68,0,78
	addi r0,r30,128
	addi r2,r30,132
	mr r3,r0
	mr r4,r2
	bl L__ZN17atomic_scoped_ptrI1AE4swapERS1_$stub
	.stabd	68,0,79
	li r3,8
	bl L__Znwm$stub
LEHE36:
	mr r0,r3
	stw r0,168(r30)
	lwz r3,168(r30)
	li r4,23
LEHB37:
	bl L__ZN1AC1Ei$stub
LEHE37:
	b L281
L312:
	stw r3,196(r30)
L282:
	lwz r29,196(r30)
	lwz r3,168(r30)
	bl L__ZdlPv$stub
	stw r29,196(r30)
	b L291
L281:
	addi r0,r30,136
	mr r3,r0
	lwz r4,168(r30)
LEHB38:
	bl L__ZN17atomic_scoped_ptrI1AE5resetEPS0_$stub
	.stabd	68,0,80
	li r3,8
	bl L__Znwm$stub
LEHE38:
	mr r0,r3
	stw r0,164(r30)
	lwz r3,164(r30)
	li r4,24
LEHB39:
	bl L__ZN1AC1Ei$stub
LEHE39:
	b L283
L311:
	stw r3,196(r30)
L284:
	lwz r29,196(r30)
	lwz r3,164(r30)
	bl L__ZdlPv$stub
	stw r29,196(r30)
	b L291
L283:
	addi r0,r30,136
	mr r3,r0
	lwz r4,164(r30)
LEHB40:
	bl L__ZN17atomic_scoped_ptrI1AE5resetEPS0_$stub
LEHE40:
	.stabd	68,0,82
	addi r0,r30,140
	mr r3,r0
	li r4,3
	bl L__ZN6atomicIiEC1Ei$stub
	.stabd	68,0,83
	addi r0,r30,140
	mr r3,r0
	bl L__ZNK6atomicIiEcviEv$stub
	mr r0,r3
	cmpwi cr7,r0,3
	mfcr r0
	rlwinm r0,r0,31,1
	mr r3,r0
	addis r2,r31,ha16(LC1-"L00000000032$pb")
	la r4,lo16(LC1-"L00000000032$pb")(r2)
	li r5,83
LEHB41:
	bl L__Z10_my_assertbPKci$stub
	.stabd	68,0,84
	addi r0,r30,140
	mr r3,r0
	li r4,1
	bl L__ZN6atomicIiEpLEi$stub
	.stabd	68,0,85
	addi r0,r30,140
	mr r3,r0
	bl L__ZNK6atomicIiEcviEv$stub
	mr r0,r3
	cmpwi cr7,r0,4
	mfcr r0
	rlwinm r0,r0,31,1
	mr r3,r0
	addis r2,r31,ha16(LC1-"L00000000032$pb")
	la r4,lo16(LC1-"L00000000032$pb")(r2)
	li r5,85
	bl L__Z10_my_assertbPKci$stub
	.stabd	68,0,86
	addi r3,r30,56
	bl L__ZNK17atomic_shared_ptrI1AEdeEv$stub
	mr r2,r3
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000032$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000032$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC6-"L00000000032$pb")
	la r4,lo16(LC6-"L00000000032$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,87
	addi r0,r30,68
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AEdeEv$stub
	mr r2,r3
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000032$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000032$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC7-"L00000000032$pb")
	la r4,lo16(LC7-"L00000000032$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,89
	addi r0,r30,92
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AEdeEv$stub
	mr r2,r3
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000032$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000032$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC8-"L00000000032$pb")
	la r4,lo16(LC8-"L00000000032$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,91
	addi r0,r30,116
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AEdeEv$stub
	mr r2,r3
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000032$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000032$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC9-"L00000000032$pb")
	la r4,lo16(LC9-"L00000000032$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,92
	li r3,8
	bl L__Znwm$stub
LEHE41:
	mr r0,r3
	stw r0,160(r30)
	lwz r3,160(r30)
	li r4,111
LEHB42:
	bl L__ZN1AC1Ei$stub
LEHE42:
	b L285
L309:
	stw r3,196(r30)
L286:
	lwz r29,196(r30)
	lwz r3,160(r30)
	bl L__ZdlPv$stub
	stw r29,196(r30)
	b L290
L285:
	addi r0,r30,68
	mr r3,r0
	lwz r4,160(r30)
LEHB43:
	bl L__ZN17atomic_shared_ptrI1AE5resetIS0_EEvPT_$stub
	.stabd	68,0,93
	addi r3,r30,56
	bl L__ZNK17atomic_shared_ptrI1AEdeEv$stub
	mr r2,r3
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000032$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000032$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC6-"L00000000032$pb")
	la r4,lo16(LC6-"L00000000032$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,94
	addi r0,r30,68
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AEdeEv$stub
	mr r2,r3
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000032$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000032$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC7-"L00000000032$pb")
	la r4,lo16(LC7-"L00000000032$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,96
	addi r0,r30,92
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AEdeEv$stub
	mr r2,r3
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000032$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000032$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC8-"L00000000032$pb")
	la r4,lo16(LC8-"L00000000032$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,98
	addi r0,r30,116
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AEdeEv$stub
	mr r2,r3
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000032$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000032$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC9-"L00000000032$pb")
	la r4,lo16(LC9-"L00000000032$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,99
	addi r0,r30,80
	addi r2,r30,68
	mr r3,r0
	mr r4,r2
	bl L__ZN17atomic_shared_ptrI1AEaSERKS1_$stub
	.stabd	68,0,100
	addi r3,r30,56
	bl L__ZNK17atomic_shared_ptrI1AEdeEv$stub
	mr r2,r3
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000032$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000032$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC6-"L00000000032$pb")
	la r4,lo16(LC6-"L00000000032$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,101
	addi r0,r30,68
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AEdeEv$stub
	mr r2,r3
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000032$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000032$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC7-"L00000000032$pb")
	la r4,lo16(LC7-"L00000000032$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,102
	addi r0,r30,80
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AEdeEv$stub
	mr r2,r3
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000032$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000032$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC10-"L00000000032$pb")
	la r4,lo16(LC10-"L00000000032$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,103
	addi r0,r30,92
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AEdeEv$stub
	mr r2,r3
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000032$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000032$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC8-"L00000000032$pb")
	la r4,lo16(LC8-"L00000000032$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,105
	addi r0,r30,116
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AEdeEv$stub
	mr r2,r3
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000032$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000032$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC9-"L00000000032$pb")
	la r4,lo16(LC9-"L00000000032$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,106
	addi r0,r30,92
	addi r2,r30,80
	mr r3,r0
	mr r4,r2
	bl L__ZN17atomic_shared_ptrI1AEaSERKS1_$stub
	.stabd	68,0,107
	addi r3,r30,56
	bl L__ZNK17atomic_shared_ptrI1AEdeEv$stub
	mr r2,r3
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000032$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000032$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC6-"L00000000032$pb")
	la r4,lo16(LC6-"L00000000032$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,108
	addi r0,r30,68
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AEdeEv$stub
	mr r2,r3
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000032$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000032$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC7-"L00000000032$pb")
	la r4,lo16(LC7-"L00000000032$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,109
	addi r0,r30,80
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AEdeEv$stub
	mr r2,r3
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000032$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000032$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC10-"L00000000032$pb")
	la r4,lo16(LC10-"L00000000032$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,110
	addi r0,r30,92
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AEdeEv$stub
	mr r2,r3
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000032$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000032$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC8-"L00000000032$pb")
	la r4,lo16(LC8-"L00000000032$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,112
	addi r0,r30,116
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AEdeEv$stub
	mr r2,r3
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000032$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000032$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC9-"L00000000032$pb")
	la r4,lo16(LC9-"L00000000032$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,113
	addi r0,r30,104
	mr r3,r0
	bl L__ZN17atomic_shared_ptrI1AE5resetEv$stub
	.stabd	68,0,114
	addi r3,r30,56
	bl L__ZNK17atomic_shared_ptrI1AEdeEv$stub
	mr r2,r3
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000032$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000032$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC6-"L00000000032$pb")
	la r4,lo16(LC6-"L00000000032$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,115
	addi r0,r30,68
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AEdeEv$stub
	mr r2,r3
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000032$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000032$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC7-"L00000000032$pb")
	la r4,lo16(LC7-"L00000000032$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,116
	addi r0,r30,80
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AEdeEv$stub
	mr r2,r3
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000032$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000032$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC10-"L00000000032$pb")
	la r4,lo16(LC10-"L00000000032$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,117
	addi r0,r30,92
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AEdeEv$stub
	mr r2,r3
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000032$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000032$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC8-"L00000000032$pb")
	la r4,lo16(LC8-"L00000000032$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,119
	addi r0,r30,116
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AEdeEv$stub
	mr r2,r3
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000032$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000032$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC9-"L00000000032$pb")
	la r4,lo16(LC9-"L00000000032$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,120
	addi r0,r30,116
	addi r2,r30,80
	mr r3,r0
	mr r4,r2
	bl L__ZN17atomic_shared_ptrI1AE4swapERS1_$stub
	.stabd	68,0,121
	addi r3,r30,56
	bl L__ZNK17atomic_shared_ptrI1AEdeEv$stub
	mr r2,r3
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000032$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000032$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC6-"L00000000032$pb")
	la r4,lo16(LC6-"L00000000032$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,122
	addi r0,r30,68
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AEdeEv$stub
	mr r2,r3
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000032$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000032$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC7-"L00000000032$pb")
	la r4,lo16(LC7-"L00000000032$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,123
	addi r0,r30,80
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AEdeEv$stub
	mr r2,r3
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000032$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000032$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC10-"L00000000032$pb")
	la r4,lo16(LC10-"L00000000032$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,124
	addi r0,r30,92
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AEdeEv$stub
	mr r2,r3
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000032$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000032$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC8-"L00000000032$pb")
	la r4,lo16(LC8-"L00000000032$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,126
	addi r0,r30,116
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AEdeEv$stub
	mr r2,r3
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000032$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000032$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC9-"L00000000032$pb")
	la r4,lo16(LC9-"L00000000032$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,127
	addi r0,r30,92
	addi r2,r30,104
	mr r3,r0
	mr r4,r2
	bl L__ZN17atomic_shared_ptrI1AE4swapERS1_$stub
	.stabd	68,0,128
	addi r3,r30,56
	bl L__ZNK17atomic_shared_ptrI1AEdeEv$stub
	mr r2,r3
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000032$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000032$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC6-"L00000000032$pb")
	la r4,lo16(LC6-"L00000000032$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,129
	addi r0,r30,68
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AEdeEv$stub
	mr r2,r3
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000032$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000032$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC7-"L00000000032$pb")
	la r4,lo16(LC7-"L00000000032$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,130
	addi r0,r30,80
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AEdeEv$stub
	mr r2,r3
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000032$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000032$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC10-"L00000000032$pb")
	la r4,lo16(LC10-"L00000000032$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,132
	addi r0,r30,104
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AEdeEv$stub
	mr r2,r3
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000032$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000032$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC11-"L00000000032$pb")
	la r4,lo16(LC11-"L00000000032$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,133
	addi r0,r30,116
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AEdeEv$stub
	mr r2,r3
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000032$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000032$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC9-"L00000000032$pb")
	la r4,lo16(LC9-"L00000000032$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,134
	addi r0,r30,92
	addi r2,r30,68
	mr r3,r0
	mr r4,r2
	bl L__ZN17atomic_shared_ptrI1AE4swapERS1_$stub
	.stabd	68,0,135
	addi r3,r30,56
	bl L__ZNK17atomic_shared_ptrI1AEdeEv$stub
	mr r2,r3
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000032$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000032$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC6-"L00000000032$pb")
	la r4,lo16(LC6-"L00000000032$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,137
	addi r0,r30,80
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AEdeEv$stub
	mr r2,r3
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000032$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000032$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC10-"L00000000032$pb")
	la r4,lo16(LC10-"L00000000032$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,138
	addi r0,r30,92
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AEdeEv$stub
	mr r2,r3
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000032$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000032$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC8-"L00000000032$pb")
	la r4,lo16(LC8-"L00000000032$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,139
	addi r0,r30,104
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AEdeEv$stub
	mr r2,r3
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000032$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000032$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC11-"L00000000032$pb")
	la r4,lo16(LC11-"L00000000032$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,140
	addi r0,r30,116
	mr r3,r0
	bl L__ZNK17atomic_shared_ptrI1AEdeEv$stub
	mr r2,r3
	lwz r0,4(r2)
	addis r2,r31,ha16(L___sF$non_lazy_ptr-"L00000000032$pb")
	lwz r2,lo16(L___sF$non_lazy_ptr-"L00000000032$pb")(r2)
	addi r3,r2,176
	addis r2,r31,ha16(LC9-"L00000000032$pb")
	la r4,lo16(LC9-"L00000000032$pb")(r2)
	mr r5,r0
	bl L_fprintf$LDBLStub$stub
	.stabd	68,0,141
	li r3,8
	bl L__Znwm$stub
LEHE43:
	mr r0,r3
	stw r0,156(r30)
	lwz r3,156(r30)
	li r4,99
LEHB44:
	bl L__ZN1AC1Ei$stub
LEHE44:
	b L287
L308:
	stw r3,196(r30)
L288:
	lwz r29,196(r30)
	lwz r3,156(r30)
	bl L__ZdlPv$stub
	stw r29,196(r30)
	b L290
L287:
	.stabd	68,0,142
	li r0,0
	stw r0,152(r30)
	addi r0,r30,140
	mr r3,r0
	bl L__ZN6atomicIiED1Ev$stub
	addi r0,r30,136
	mr r3,r0
LEHB45:
	bl L__ZN17atomic_scoped_ptrI1AED1Ev$stub
LEHE45:
	b L289
L310:
	stw r3,196(r30)
L290:
	lwz r29,196(r30)
	addi r0,r30,140
	mr r3,r0
	bl L__ZN6atomicIiED1Ev$stub
	stw r29,196(r30)
	b L291
L313:
	stw r3,196(r30)
L291:
	lwz r29,196(r30)
	addi r0,r30,136
	mr r3,r0
	bl L__ZN17atomic_scoped_ptrI1AED1Ev$stub
	stw r29,196(r30)
	b L292
L315:
	stw r3,196(r30)
L292:
	lwz r29,196(r30)
	addi r0,r30,132
	mr r3,r0
	bl L__ZN17atomic_scoped_ptrI1AED1Ev$stub
	stw r29,196(r30)
	b L294
L289:
	addi r0,r30,132
	mr r3,r0
LEHB46:
	bl L__ZN17atomic_scoped_ptrI1AED1Ev$stub
LEHE46:
	b L293
L317:
	stw r3,196(r30)
L294:
	lwz r29,196(r30)
	addi r0,r30,128
	mr r3,r0
	bl L__ZN17atomic_scoped_ptrI1AED1Ev$stub
	stw r29,196(r30)
	b L296
L293:
	addi r0,r30,128
	mr r3,r0
LEHB47:
	bl L__ZN17atomic_scoped_ptrI1AED1Ev$stub
LEHE47:
	b L295
L319:
	stw r3,196(r30)
L296:
	lwz r29,196(r30)
	addi r0,r30,116
	mr r3,r0
	bl L__ZN17atomic_shared_ptrI1AED1Ev$stub
	stw r29,196(r30)
	b L298
L295:
	addi r0,r30,116
	mr r3,r0
LEHB48:
	bl L__ZN17atomic_shared_ptrI1AED1Ev$stub
LEHE48:
	b L297
L321:
	stw r3,196(r30)
L298:
	lwz r29,196(r30)
	addi r0,r30,104
	mr r3,r0
	bl L__ZN17atomic_shared_ptrI1AED1Ev$stub
	stw r29,196(r30)
	b L300
L297:
	addi r0,r30,104
	mr r3,r0
LEHB49:
	bl L__ZN17atomic_shared_ptrI1AED1Ev$stub
LEHE49:
	b L299
L322:
	stw r3,196(r30)
L300:
	lwz r29,196(r30)
	addi r0,r30,92
	mr r3,r0
	bl L__ZN17atomic_shared_ptrI1AED1Ev$stub
	stw r29,196(r30)
	b L302
L299:
	addi r0,r30,92
	mr r3,r0
LEHB50:
	bl L__ZN17atomic_shared_ptrI1AED1Ev$stub
LEHE50:
	b L301
L323:
	stw r3,196(r30)
L302:
	lwz r29,196(r30)
	addi r0,r30,80
	mr r3,r0
	bl L__ZN17atomic_shared_ptrI1AED1Ev$stub
	stw r29,196(r30)
	b L304
L301:
	addi r0,r30,80
	mr r3,r0
LEHB51:
	bl L__ZN17atomic_shared_ptrI1AED1Ev$stub
LEHE51:
	b L303
L324:
	stw r3,196(r30)
L304:
	lwz r29,196(r30)
	addi r0,r30,68
	mr r3,r0
	bl L__ZN17atomic_shared_ptrI1AED1Ev$stub
	stw r29,196(r30)
	b L306
L303:
	addi r0,r30,68
	mr r3,r0
LEHB52:
	bl L__ZN17atomic_shared_ptrI1AED1Ev$stub
LEHE52:
	b L305
L326:
	stw r3,196(r30)
L306:
	lwz r29,196(r30)
	addi r3,r30,56
	bl L__ZN17atomic_shared_ptrI1AED1Ev$stub
	stw r29,196(r30)
	lwz r3,196(r30)
LEHB53:
	bl L__Unwind_Resume$stub
L305:
	addi r3,r30,56
	bl L__ZN17atomic_shared_ptrI1AED1Ev$stub
LEHE53:
	lwz r0,152(r30)
LBE85:
	.stabd	68,0,143
	mr r3,r0
	lwz r1,0(r1)
	lwz r0,8(r1)
	mtlr r0
	lmw r29,-12(r1)
	blr
LFE1497:
	.section __DATA,__gcc_except_tab
LLSDA1497:
	.byte	0xff
	.byte	0xff
	.byte	0x3
	.byte	0xc7,0x3
	.set L$set$46,LEHB19-LFB1497
	.long L$set$46
	.set L$set$47,LEHE19-LEHB19
	.long L$set$47
	.long	0x0
	.byte	0x0
	.set L$set$48,LEHB20-LFB1497
	.long L$set$48
	.set L$set$49,LEHE20-LEHB20
	.long L$set$49
	.set L$set$50,L327-LFB1497
	.long L$set$50
	.byte	0x0
	.set L$set$51,LEHB21-LFB1497
	.long L$set$51
	.set L$set$52,LEHE21-LEHB21
	.long L$set$52
	.long	0x0
	.byte	0x0
	.set L$set$53,LEHB22-LFB1497
	.long L$set$53
	.set L$set$54,LEHE22-LEHB22
	.long L$set$54
	.set L$set$55,L326-LFB1497
	.long L$set$55
	.byte	0x0
	.set L$set$56,LEHB23-LFB1497
	.long L$set$56
	.set L$set$57,LEHE23-LEHB23
	.long L$set$57
	.set L$set$58,L325-LFB1497
	.long L$set$58
	.byte	0x0
	.set L$set$59,LEHB24-LFB1497
	.long L$set$59
	.set L$set$60,LEHE24-LEHB24
	.long L$set$60
	.set L$set$61,L326-LFB1497
	.long L$set$61
	.byte	0x0
	.set L$set$62,LEHB25-LFB1497
	.long L$set$62
	.set L$set$63,LEHE25-LEHB25
	.long L$set$63
	.set L$set$64,L323-LFB1497
	.long L$set$64
	.byte	0x0
	.set L$set$65,LEHB26-LFB1497
	.long L$set$65
	.set L$set$66,LEHE26-LEHB26
	.long L$set$66
	.set L$set$67,L322-LFB1497
	.long L$set$67
	.byte	0x0
	.set L$set$68,LEHB27-LFB1497
	.long L$set$68
	.set L$set$69,LEHE27-LEHB27
	.long L$set$69
	.set L$set$70,L321-LFB1497
	.long L$set$70
	.byte	0x0
	.set L$set$71,LEHB28-LFB1497
	.long L$set$71
	.set L$set$72,LEHE28-LEHB28
	.long L$set$72
	.set L$set$73,L320-LFB1497
	.long L$set$73
	.byte	0x0
	.set L$set$74,LEHB29-LFB1497
	.long L$set$74
	.set L$set$75,LEHE29-LEHB29
	.long L$set$75
	.set L$set$76,L321-LFB1497
	.long L$set$76
	.byte	0x0
	.set L$set$77,LEHB30-LFB1497
	.long L$set$77
	.set L$set$78,LEHE30-LEHB30
	.long L$set$78
	.set L$set$79,L319-LFB1497
	.long L$set$79
	.byte	0x0
	.set L$set$80,LEHB31-LFB1497
	.long L$set$80
	.set L$set$81,LEHE31-LEHB31
	.long L$set$81
	.set L$set$82,L318-LFB1497
	.long L$set$82
	.byte	0x0
	.set L$set$83,LEHB32-LFB1497
	.long L$set$83
	.set L$set$84,LEHE32-LEHB32
	.long L$set$84
	.set L$set$85,L317-LFB1497
	.long L$set$85
	.byte	0x0
	.set L$set$86,LEHB33-LFB1497
	.long L$set$86
	.set L$set$87,LEHE33-LEHB33
	.long L$set$87
	.set L$set$88,L316-LFB1497
	.long L$set$88
	.byte	0x0
	.set L$set$89,LEHB34-LFB1497
	.long L$set$89
	.set L$set$90,LEHE34-LEHB34
	.long L$set$90
	.set L$set$91,L315-LFB1497
	.long L$set$91
	.byte	0x0
	.set L$set$92,LEHB35-LFB1497
	.long L$set$92
	.set L$set$93,LEHE35-LEHB35
	.long L$set$93
	.set L$set$94,L314-LFB1497
	.long L$set$94
	.byte	0x0
	.set L$set$95,LEHB36-LFB1497
	.long L$set$95
	.set L$set$96,LEHE36-LEHB36
	.long L$set$96
	.set L$set$97,L313-LFB1497
	.long L$set$97
	.byte	0x0
	.set L$set$98,LEHB37-LFB1497
	.long L$set$98
	.set L$set$99,LEHE37-LEHB37
	.long L$set$99
	.set L$set$100,L312-LFB1497
	.long L$set$100
	.byte	0x0
	.set L$set$101,LEHB38-LFB1497
	.long L$set$101
	.set L$set$102,LEHE38-LEHB38
	.long L$set$102
	.set L$set$103,L313-LFB1497
	.long L$set$103
	.byte	0x0
	.set L$set$104,LEHB39-LFB1497
	.long L$set$104
	.set L$set$105,LEHE39-LEHB39
	.long L$set$105
	.set L$set$106,L311-LFB1497
	.long L$set$106
	.byte	0x0
	.set L$set$107,LEHB40-LFB1497
	.long L$set$107
	.set L$set$108,LEHE40-LEHB40
	.long L$set$108
	.set L$set$109,L313-LFB1497
	.long L$set$109
	.byte	0x0
	.set L$set$110,LEHB41-LFB1497
	.long L$set$110
	.set L$set$111,LEHE41-LEHB41
	.long L$set$111
	.set L$set$112,L310-LFB1497
	.long L$set$112
	.byte	0x0
	.set L$set$113,LEHB42-LFB1497
	.long L$set$113
	.set L$set$114,LEHE42-LEHB42
	.long L$set$114
	.set L$set$115,L309-LFB1497
	.long L$set$115
	.byte	0x0
	.set L$set$116,LEHB43-LFB1497
	.long L$set$116
	.set L$set$117,LEHE43-LEHB43
	.long L$set$117
	.set L$set$118,L310-LFB1497
	.long L$set$118
	.byte	0x0
	.set L$set$119,LEHB44-LFB1497
	.long L$set$119
	.set L$set$120,LEHE44-LEHB44
	.long L$set$120
	.set L$set$121,L308-LFB1497
	.long L$set$121
	.byte	0x0
	.set L$set$122,LEHB45-LFB1497
	.long L$set$122
	.set L$set$123,LEHE45-LEHB45
	.long L$set$123
	.set L$set$124,L315-LFB1497
	.long L$set$124
	.byte	0x0
	.set L$set$125,LEHB46-LFB1497
	.long L$set$125
	.set L$set$126,LEHE46-LEHB46
	.long L$set$126
	.set L$set$127,L317-LFB1497
	.long L$set$127
	.byte	0x0
	.set L$set$128,LEHB47-LFB1497
	.long L$set$128
	.set L$set$129,LEHE47-LEHB47
	.long L$set$129
	.set L$set$130,L319-LFB1497
	.long L$set$130
	.byte	0x0
	.set L$set$131,LEHB48-LFB1497
	.long L$set$131
	.set L$set$132,LEHE48-LEHB48
	.long L$set$132
	.set L$set$133,L321-LFB1497
	.long L$set$133
	.byte	0x0
	.set L$set$134,LEHB49-LFB1497
	.long L$set$134
	.set L$set$135,LEHE49-LEHB49
	.long L$set$135
	.set L$set$136,L322-LFB1497
	.long L$set$136
	.byte	0x0
	.set L$set$137,LEHB50-LFB1497
	.long L$set$137
	.set L$set$138,LEHE50-LEHB50
	.long L$set$138
	.set L$set$139,L323-LFB1497
	.long L$set$139
	.byte	0x0
	.set L$set$140,LEHB51-LFB1497
	.long L$set$140
	.set L$set$141,LEHE51-LEHB51
	.long L$set$141
	.set L$set$142,L324-LFB1497
	.long L$set$142
	.byte	0x0
	.set L$set$143,LEHB52-LFB1497
	.long L$set$143
	.set L$set$144,LEHE52-LEHB52
	.long L$set$144
	.set L$set$145,L326-LFB1497
	.long L$set$145
	.byte	0x0
	.set L$set$146,LEHB53-LFB1497
	.long L$set$146
	.set L$set$147,LEHE53-LEHB53
	.long L$set$147
	.long	0x0
	.byte	0x0
	.text
	.stabs	"main:F(0,10)",36,0,65,_main
	.stabs	"argc:p(0,10)",160,0,65,248
	.stabs	"argv:p(0,155)",160,0,65,252
	.stabs	"p1:(0,121)",128,0,68,56
	.stabs	":t(0,155)=*(0,156)",128,0,0,0
	.stabs	":t(0,156)=*(0,33)",128,0,0,0
	.stabs	"p2:(0,121)",128,0,69,68
	.stabs	"p3:(0,121)",128,0,70,80
	.stabs	"p4:(0,121)",128,0,71,92
	.stabs	"p5:(0,121)",128,0,72,104
	.stabs	"p6:(0,121)",128,0,73,116
	.stabs	"p10:(0,88)",128,0,74,128
	.stabs	"p11:(0,88)",128,0,75,132
	.stabs	"p12:(0,88)",128,0,76,136
	.stabs	"a:(0,103)",128,0,82,140
	.stabn	192,0,0,LBB85
	.stabn	224,0,0,LBE85
Lscope71:
	.stabs	"",36,0,0,Lscope71-_main
	.stabd	78,0,0
	.section __TEXT,__eh_frame,coalesced,no_toc+strip_static_syms+live_support
EH_frame1:
	.set L$set$148,LECIE1-LSCIE1
	.long L$set$148
LSCIE1:
	.long	0x0
	.byte	0x1
	.ascii "zPLR\0"
	.byte	0x1
	.byte	0x7c
	.byte	0x41
	.byte	0x7
	.byte	0x9b
	.long	L___gxx_personality_v0$non_lazy_ptr-.
	.byte	0x10
	.byte	0x10
	.byte	0xc
	.byte	0x1
	.byte	0x0
	.align 2
LECIE1:
	.globl __Z11_my_cassertm.eh
	.weak_definition __Z11_my_cassertm.eh
__Z11_my_cassertm.eh:
LSFDE1:
	.set L$set$149,LEFDE1-LASFDE1
	.long L$set$149
LASFDE1:
	.long	LASFDE1-EH_frame1
	.long	LFB3-.
	.set L$set$150,LFE3-LFB3
	.long L$set$150
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$151,LCFI1-LFB3
	.long L$set$151
	.byte	0xe
	.byte	0x30
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x4
	.set L$set$152,LCFI2-LCFI1
	.long L$set$152
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE1:
	.globl __Z11readBarrierv.eh
	.weak_definition __Z11readBarrierv.eh
__Z11readBarrierv.eh:
LSFDE3:
	.set L$set$153,LEFDE3-LASFDE3
	.long L$set$153
LASFDE3:
	.long	LASFDE3-EH_frame1
	.long	LFB1383-.
	.set L$set$154,LFE1383-LFB1383
	.long L$set$154
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$155,LCFI4-LFB1383
	.long L$set$155
	.byte	0xe
	.byte	0x30
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x4
	.set L$set$156,LCFI5-LCFI4
	.long L$set$156
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE3:
	.globl __Z13memoryBarrierv.eh
	.weak_definition __Z13memoryBarrierv.eh
__Z13memoryBarrierv.eh:
LSFDE5:
	.set L$set$157,LEFDE5-LASFDE5
	.long L$set$157
LASFDE5:
	.long	LASFDE5-EH_frame1
	.long	LFB1384-.
	.set L$set$158,LFE1384-LFB1384
	.long L$set$158
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$159,LCFI7-LFB1384
	.long L$set$159
	.byte	0xe
	.byte	0x30
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x4
	.set L$set$160,LCFI8-LCFI7
	.long L$set$160
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE5:
	.globl __ZNK15XRecursiveMutex23isLockedByCurrentThreadEv.eh
__ZNK15XRecursiveMutex23isLockedByCurrentThreadEv.eh:
LSFDE7:
	.set L$set$161,LEFDE7-LASFDE7
	.long L$set$161
LASFDE7:
	.long	LASFDE7-EH_frame1
	.long	LFB1472-.
	.set L$set$162,LFE1472-LFB1472
	.long L$set$162
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$163,LCFI12-LFB1472
	.long L$set$163
	.byte	0xe
	.byte	0x50
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x9d
	.byte	0x3
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$164,LCFI13-LCFI12
	.long L$set$164
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE7:
	.globl __ZN10XCondition4waitEi.eh
__ZN10XCondition4waitEi.eh:
LSFDE9:
	.set L$set$165,LEFDE9-LASFDE9
	.long L$set$165
LASFDE9:
	.long	LASFDE9-EH_frame1
	.long	LFB1479-.
	.set L$set$166,LFE1479-LFB1479
	.long L$set$166
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$167,LCFI17-LFB1479
	.long L$set$167
	.byte	0xe
	.byte	0x70
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$168,LCFI18-LCFI17
	.long L$set$168
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE9:
	.globl __Z9my_assertPKci.eh
__Z9my_assertPKci.eh:
LSFDE11:
	.set L$set$169,LEFDE11-LASFDE11
	.long L$set$169
LASFDE11:
	.long	LASFDE11-EH_frame1
	.long	LFB1482-.
	.set L$set$170,LFE1482-LFB1482
	.long L$set$170
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$171,LCFI22-LFB1482
	.long L$set$171
	.byte	0xe
	.byte	0x50
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$172,LCFI23-LCFI22
	.long L$set$172
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE11:
	.globl __Z10_my_assertbPKci.eh
	.weak_definition __Z10_my_assertbPKci.eh
__Z10_my_assertbPKci.eh:
LSFDE13:
	.set L$set$173,LEFDE13-LASFDE13
	.long L$set$173
LASFDE13:
	.long	LASFDE13-EH_frame1
	.long	LFB2-.
	.set L$set$174,LFE2-LFB2
	.long L$set$174
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$175,LCFI27-LFB2
	.long L$set$175
	.byte	0xe
	.byte	0x50
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$176,LCFI28-LCFI27
	.long L$set$176
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE13:
	.globl __Z20atomicCompareAndSet2jjjjPj.eh
	.weak_definition __Z20atomicCompareAndSet2jjjjPj.eh
__Z20atomicCompareAndSet2jjjjPj.eh:
LSFDE15:
	.set L$set$177,LEFDE15-LASFDE15
	.long L$set$177
LASFDE15:
	.long	LASFDE15-EH_frame1
	.long	LFB1382-.
	.set L$set$178,LFE1382-LFB1382
	.long L$set$178
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$179,LCFI32-LFB1382
	.long L$set$179
	.byte	0xe
	.byte	0x50
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$180,LCFI33-LCFI32
	.long L$set$180
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE15:
	.globl __ZN10XCondition9broadcastEv.eh
__ZN10XCondition9broadcastEv.eh:
LSFDE17:
	.set L$set$181,LEFDE17-LASFDE17
	.long L$set$181
LASFDE17:
	.long	LASFDE17-EH_frame1
	.long	LFB1481-.
	.set L$set$182,LFE1481-LFB1481
	.long L$set$182
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$183,LCFI37-LFB1481
	.long L$set$183
	.byte	0xe
	.byte	0x60
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$184,LCFI38-LCFI37
	.long L$set$184
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE17:
	.globl __ZN10XCondition6signalEv.eh
__ZN10XCondition6signalEv.eh:
LSFDE19:
	.set L$set$185,LEFDE19-LASFDE19
	.long L$set$185
LASFDE19:
	.long	LASFDE19-EH_frame1
	.long	LFB1480-.
	.set L$set$186,LFE1480-LFB1480
	.long L$set$186
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$187,LCFI42-LFB1480
	.long L$set$187
	.byte	0xe
	.byte	0x60
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$188,LCFI43-LCFI42
	.long L$set$188
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE19:
	.globl __ZN10XConditionD1Ev.eh
__ZN10XConditionD1Ev.eh:
LSFDE21:
	.set L$set$189,LEFDE21-LASFDE21
	.long L$set$189
LASFDE21:
	.long	LASFDE21-EH_frame1
	.long	LFB1478-.
	.set L$set$190,LFE1478-LFB1478
	.long L$set$190
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$191,LCFI47-LFB1478
	.long L$set$191
	.byte	0xe
	.byte	0x60
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$192,LCFI48-LCFI47
	.long L$set$192
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE21:
	.globl __ZN10XConditionD2Ev.eh
__ZN10XConditionD2Ev.eh:
LSFDE23:
	.set L$set$193,LEFDE23-LASFDE23
	.long L$set$193
LASFDE23:
	.long	LASFDE23-EH_frame1
	.long	LFB1477-.
	.set L$set$194,LFE1477-LFB1477
	.long L$set$194
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$195,LCFI52-LFB1477
	.long L$set$195
	.byte	0xe
	.byte	0x60
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$196,LCFI53-LCFI52
	.long L$set$196
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE23:
	.globl __ZN10XConditionC1Ev.eh
__ZN10XConditionC1Ev.eh:
LSFDE25:
	.set L$set$197,LEFDE25-LASFDE25
	.long L$set$197
LASFDE25:
	.long	LASFDE25-EH_frame1
	.long	LFB1475-.
	.set L$set$198,LFE1475-LFB1475
	.long L$set$198
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$199,LCFI57-LFB1475
	.long L$set$199
	.byte	0xe
	.byte	0x60
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$200,LCFI58-LCFI57
	.long L$set$200
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE25:
	.globl __ZN10XConditionC2Ev.eh
__ZN10XConditionC2Ev.eh:
LSFDE27:
	.set L$set$201,LEFDE27-LASFDE27
	.long L$set$201
LASFDE27:
	.long	LASFDE27-EH_frame1
	.long	LFB1474-.
	.set L$set$202,LFE1474-LFB1474
	.long L$set$202
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$203,LCFI62-LFB1474
	.long L$set$203
	.byte	0xe
	.byte	0x60
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$204,LCFI63-LCFI62
	.long L$set$204
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE27:
	.globl __ZN6XMutex6unlockEv.eh
__ZN6XMutex6unlockEv.eh:
LSFDE29:
	.set L$set$205,LEFDE29-LASFDE29
	.long L$set$205
LASFDE29:
	.long	LASFDE29-EH_frame1
	.long	LFB1462-.
	.set L$set$206,LFE1462-LFB1462
	.long L$set$206
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$207,LCFI67-LFB1462
	.long L$set$207
	.byte	0xe
	.byte	0x60
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$208,LCFI68-LCFI67
	.long L$set$208
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE29:
	.globl __ZN15XRecursiveMutex6unlockEv.eh
__ZN15XRecursiveMutex6unlockEv.eh:
LSFDE31:
	.set L$set$209,LEFDE31-LASFDE31
	.long L$set$209
LASFDE31:
	.long	LASFDE31-EH_frame1
	.long	LFB1471-.
	.set L$set$210,LFE1471-LFB1471
	.long L$set$210
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$211,LCFI72-LFB1471
	.long L$set$211
	.byte	0xe
	.byte	0x50
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x9d
	.byte	0x3
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$212,LCFI73-LCFI72
	.long L$set$212
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE31:
	.globl __ZN6XMutex7trylockEv.eh
__ZN6XMutex7trylockEv.eh:
LSFDE33:
	.set L$set$213,LEFDE33-LASFDE33
	.long L$set$213
LASFDE33:
	.long	LASFDE33-EH_frame1
	.long	LFB1461-.
	.set L$set$214,LFE1461-LFB1461
	.long L$set$214
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$215,LCFI77-LFB1461
	.long L$set$215
	.byte	0xe
	.byte	0x60
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$216,LCFI78-LCFI77
	.long L$set$216
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE33:
	.globl __ZN15XRecursiveMutex7trylockEv.eh
__ZN15XRecursiveMutex7trylockEv.eh:
LSFDE35:
	.set L$set$217,LEFDE35-LASFDE35
	.long L$set$217
LASFDE35:
	.long	LASFDE35-EH_frame1
	.long	LFB1470-.
	.set L$set$218,LFE1470-LFB1470
	.long L$set$218
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$219,LCFI82-LFB1470
	.long L$set$219
	.byte	0xe
	.byte	0x50
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x9d
	.byte	0x3
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$220,LCFI83-LCFI82
	.long L$set$220
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE35:
	.globl __ZN6XMutex4lockEv.eh
__ZN6XMutex4lockEv.eh:
LSFDE37:
	.set L$set$221,LEFDE37-LASFDE37
	.long L$set$221
LASFDE37:
	.long	LASFDE37-EH_frame1
	.long	LFB1460-.
	.set L$set$222,LFE1460-LFB1460
	.long L$set$222
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$223,LCFI87-LFB1460
	.long L$set$223
	.byte	0xe
	.byte	0x60
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$224,LCFI88-LCFI87
	.long L$set$224
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE37:
	.globl __ZN15XRecursiveMutex4lockEv.eh
__ZN15XRecursiveMutex4lockEv.eh:
LSFDE39:
	.set L$set$225,LEFDE39-LASFDE39
	.long L$set$225
LASFDE39:
	.long	LASFDE39-EH_frame1
	.long	LFB1469-.
	.set L$set$226,LFE1469-LFB1469
	.long L$set$226
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$227,LCFI92-LFB1469
	.long L$set$227
	.byte	0xe
	.byte	0x50
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x9d
	.byte	0x3
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$228,LCFI93-LCFI92
	.long L$set$228
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE39:
	.globl __ZN6XMutexD1Ev.eh
__ZN6XMutexD1Ev.eh:
LSFDE41:
	.set L$set$229,LEFDE41-LASFDE41
	.long L$set$229
LASFDE41:
	.long	LASFDE41-EH_frame1
	.long	LFB1459-.
	.set L$set$230,LFE1459-LFB1459
	.long L$set$230
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$231,LCFI97-LFB1459
	.long L$set$231
	.byte	0xe
	.byte	0x60
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$232,LCFI98-LCFI97
	.long L$set$232
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE41:
	.globl __ZN15XRecursiveMutexD1Ev.eh
__ZN15XRecursiveMutexD1Ev.eh:
LSFDE43:
	.set L$set$233,LEFDE43-LASFDE43
	.long L$set$233
LASFDE43:
	.long	LASFDE43-EH_frame1
	.long	LFB1468-.
	.set L$set$234,LFE1468-LFB1468
	.long L$set$234
	.byte	0x4
	.long	LLSDA1468-.
	.byte	0x4
	.set L$set$235,LCFI102-LFB1468
	.long L$set$235
	.byte	0xe
	.byte	0x50
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x9d
	.byte	0x3
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$236,LCFI103-LCFI102
	.long L$set$236
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE43:
	.globl __ZN15XRecursiveMutexD2Ev.eh
__ZN15XRecursiveMutexD2Ev.eh:
LSFDE45:
	.set L$set$237,LEFDE45-LASFDE45
	.long L$set$237
LASFDE45:
	.long	LASFDE45-EH_frame1
	.long	LFB1467-.
	.set L$set$238,LFE1467-LFB1467
	.long L$set$238
	.byte	0x4
	.long	LLSDA1467-.
	.byte	0x4
	.set L$set$239,LCFI107-LFB1467
	.long L$set$239
	.byte	0xe
	.byte	0x50
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x9d
	.byte	0x3
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$240,LCFI108-LCFI107
	.long L$set$240
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE45:
	.globl __ZN6XMutexD2Ev.eh
__ZN6XMutexD2Ev.eh:
LSFDE47:
	.set L$set$241,LEFDE47-LASFDE47
	.long L$set$241
LASFDE47:
	.long	LASFDE47-EH_frame1
	.long	LFB1458-.
	.set L$set$242,LFE1458-LFB1458
	.long L$set$242
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$243,LCFI112-LFB1458
	.long L$set$243
	.byte	0xe
	.byte	0x60
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$244,LCFI113-LCFI112
	.long L$set$244
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE47:
	.globl __ZN6XMutexC1Ev.eh
__ZN6XMutexC1Ev.eh:
LSFDE49:
	.set L$set$245,LEFDE49-LASFDE49
	.long L$set$245
LASFDE49:
	.long	LASFDE49-EH_frame1
	.long	LFB1456-.
	.set L$set$246,LFE1456-LFB1456
	.long L$set$246
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$247,LCFI117-LFB1456
	.long L$set$247
	.byte	0xe
	.byte	0x60
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$248,LCFI118-LCFI117
	.long L$set$248
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE49:
	.globl __ZN15XRecursiveMutexC1Ev.eh
__ZN15XRecursiveMutexC1Ev.eh:
LSFDE51:
	.set L$set$249,LEFDE51-LASFDE51
	.long L$set$249
LASFDE51:
	.long	LASFDE51-EH_frame1
	.long	LFB1465-.
	.set L$set$250,LFE1465-LFB1465
	.long L$set$250
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$251,LCFI122-LFB1465
	.long L$set$251
	.byte	0xe
	.byte	0x50
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$252,LCFI123-LCFI122
	.long L$set$252
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE51:
	.globl __ZN15XRecursiveMutexC2Ev.eh
__ZN15XRecursiveMutexC2Ev.eh:
LSFDE53:
	.set L$set$253,LEFDE53-LASFDE53
	.long L$set$253
LASFDE53:
	.long	LASFDE53-EH_frame1
	.long	LFB1464-.
	.set L$set$254,LFE1464-LFB1464
	.long L$set$254
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$255,LCFI127-LFB1464
	.long L$set$255
	.byte	0xe
	.byte	0x50
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$256,LCFI128-LCFI127
	.long L$set$256
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE53:
	.globl __ZN6XMutexC2Ev.eh
__ZN6XMutexC2Ev.eh:
LSFDE55:
	.set L$set$257,LEFDE55-LASFDE55
	.long L$set$257
LASFDE55:
	.long	LASFDE55-EH_frame1
	.long	LFB1455-.
	.set L$set$258,LFE1455-LFB1455
	.long L$set$258
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$259,LCFI132-LFB1455
	.long L$set$259
	.byte	0xe
	.byte	0x60
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$260,LCFI133-LCFI132
	.long L$set$260
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE55:
	.globl __ZN1AC2Ei.eh
	.weak_definition __ZN1AC2Ei.eh
__ZN1AC2Ei.eh:
LSFDE57:
	.set L$set$261,LEFDE57-LASFDE57
	.long L$set$261
LASFDE57:
	.long	LASFDE57-EH_frame1
	.long	LFB1484-.
	.set L$set$262,LFE1484-LFB1484
	.long L$set$262
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$263,LCFI137-LFB1484
	.long L$set$263
	.byte	0xe
	.byte	0x50
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$264,LCFI138-LCFI137
	.long L$set$264
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE57:
	.globl __ZN1AC1Ei.eh
	.weak_definition __ZN1AC1Ei.eh
__ZN1AC1Ei.eh:
LSFDE59:
	.set L$set$265,LEFDE59-LASFDE59
	.long L$set$265
LASFDE59:
	.long	LASFDE59-EH_frame1
	.long	LFB1485-.
	.set L$set$266,LFE1485-LFB1485
	.long L$set$266
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$267,LCFI142-LFB1485
	.long L$set$267
	.byte	0xe
	.byte	0x50
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$268,LCFI143-LCFI142
	.long L$set$268
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE59:
	.globl __ZN1AD2Ev.eh
	.weak_definition __ZN1AD2Ev.eh
__ZN1AD2Ev.eh:
LSFDE61:
	.set L$set$269,LEFDE61-LASFDE61
	.long L$set$269
LASFDE61:
	.long	LASFDE61-EH_frame1
	.long	LFB1487-.
	.set L$set$270,LFE1487-LFB1487
	.long L$set$270
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$271,LCFI147-LFB1487
	.long L$set$271
	.byte	0xe
	.byte	0x50
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$272,LCFI148-LCFI147
	.long L$set$272
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE61:
	.globl __ZN1AD1Ev.eh
	.weak_definition __ZN1AD1Ev.eh
__ZN1AD1Ev.eh:
LSFDE63:
	.set L$set$273,LEFDE63-LASFDE63
	.long L$set$273
LASFDE63:
	.long	LASFDE63-EH_frame1
	.long	LFB1488-.
	.set L$set$274,LFE1488-LFB1488
	.long L$set$274
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$275,LCFI152-LFB1488
	.long L$set$275
	.byte	0xe
	.byte	0x50
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$276,LCFI153-LCFI152
	.long L$set$276
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE63:
	.globl __ZN1AD0Ev.eh
	.weak_definition __ZN1AD0Ev.eh
__ZN1AD0Ev.eh:
LSFDE65:
	.set L$set$277,LEFDE65-LASFDE65
	.long L$set$277
LASFDE65:
	.long	LASFDE65-EH_frame1
	.long	LFB1489-.
	.set L$set$278,LFE1489-LFB1489
	.long L$set$278
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$279,LCFI157-LFB1489
	.long L$set$279
	.byte	0xe
	.byte	0x50
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$280,LCFI158-LCFI157
	.long L$set$280
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE65:
	.globl __ZN1BC1Ei.eh
	.weak_definition __ZN1BC1Ei.eh
__ZN1BC1Ei.eh:
LSFDE67:
	.set L$set$281,LEFDE67-LASFDE67
	.long L$set$281
LASFDE67:
	.long	LASFDE67-EH_frame1
	.long	LFB1492-.
	.set L$set$282,LFE1492-LFB1492
	.long L$set$282
	.byte	0x4
	.long	LLSDA1492-.
	.byte	0x4
	.set L$set$283,LCFI162-LFB1492
	.long L$set$283
	.byte	0xe
	.byte	0x50
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x9d
	.byte	0x3
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$284,LCFI163-LCFI162
	.long L$set$284
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE67:
	.globl __ZN1BD1Ev.eh
	.weak_definition __ZN1BD1Ev.eh
__ZN1BD1Ev.eh:
LSFDE69:
	.set L$set$285,LEFDE69-LASFDE69
	.long L$set$285
LASFDE69:
	.long	LASFDE69-EH_frame1
	.long	LFB1495-.
	.set L$set$286,LFE1495-LFB1495
	.long L$set$286
	.byte	0x4
	.long	LLSDA1495-.
	.byte	0x4
	.set L$set$287,LCFI167-LFB1495
	.long L$set$287
	.byte	0xe
	.byte	0x50
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x9d
	.byte	0x3
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$288,LCFI168-LCFI167
	.long L$set$288
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE69:
	.globl __ZN1BD0Ev.eh
	.weak_definition __ZN1BD0Ev.eh
__ZN1BD0Ev.eh:
LSFDE71:
	.set L$set$289,LEFDE71-LASFDE71
	.long L$set$289
LASFDE71:
	.long	LASFDE71-EH_frame1
	.long	LFB1496-.
	.set L$set$290,LFE1496-LFB1496
	.long L$set$290
	.byte	0x4
	.long	LLSDA1496-.
	.byte	0x4
	.set L$set$291,LCFI172-LFB1496
	.long L$set$291
	.byte	0xe
	.byte	0x50
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x9d
	.byte	0x3
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$292,LCFI173-LCFI172
	.long L$set$292
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE71:
	.globl __ZN17atomic_scoped_ptrI1AEC1EPS0_.eh
	.weak_definition __ZN17atomic_scoped_ptrI1AEC1EPS0_.eh
__ZN17atomic_scoped_ptrI1AEC1EPS0_.eh:
LSFDE73:
	.set L$set$293,LEFDE73-LASFDE73
	.long L$set$293
LASFDE73:
	.long	LASFDE73-EH_frame1
	.long	LFB1551-.
	.set L$set$294,LFE1551-LFB1551
	.long L$set$294
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$295,LCFI177-LFB1551
	.long L$set$295
	.byte	0xe
	.byte	0x50
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$296,LCFI178-LCFI177
	.long L$set$296
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE73:
	.globl __ZN17atomic_scoped_ptrI1AED1Ev.eh
	.weak_definition __ZN17atomic_scoped_ptrI1AED1Ev.eh
__ZN17atomic_scoped_ptrI1AED1Ev.eh:
LSFDE75:
	.set L$set$297,LEFDE75-LASFDE75
	.long L$set$297
LASFDE75:
	.long	LASFDE75-EH_frame1
	.long	LFB1554-.
	.set L$set$298,LFE1554-LFB1554
	.long L$set$298
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$299,LCFI182-LFB1554
	.long L$set$299
	.byte	0xe
	.byte	0x50
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$300,LCFI183-LCFI182
	.long L$set$300
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE75:
	.globl __ZN6atomicIiEC1Ei.eh
	.weak_definition __ZN6atomicIiEC1Ei.eh
__ZN6atomicIiEC1Ei.eh:
LSFDE77:
	.set L$set$301,LEFDE77-LASFDE77
	.long L$set$301
LASFDE77:
	.long	LASFDE77-EH_frame1
	.long	LFB1559-.
	.set L$set$302,LFE1559-LFB1559
	.long L$set$302
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$303,LCFI187-LFB1559
	.long L$set$303
	.byte	0xe
	.byte	0x50
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$304,LCFI188-LCFI187
	.long L$set$304
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE77:
	.globl __ZN6atomicIiED1Ev.eh
	.weak_definition __ZN6atomicIiED1Ev.eh
__ZN6atomicIiED1Ev.eh:
LSFDE79:
	.set L$set$305,LEFDE79-LASFDE79
	.long L$set$305
LASFDE79:
	.long	LASFDE79-EH_frame1
	.long	LFB1562-.
	.set L$set$306,LFE1562-LFB1562
	.long L$set$306
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$307,LCFI190-LFB1562
	.long L$set$307
	.byte	0xe
	.byte	0x30
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x4
	.set L$set$308,LCFI191-LCFI190
	.long L$set$308
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE79:
	.globl __ZNK6atomicIiEcviEv.eh
	.weak_definition __ZNK6atomicIiEcviEv.eh
__ZNK6atomicIiEcviEv.eh:
LSFDE81:
	.set L$set$309,LEFDE81-LASFDE81
	.long L$set$309
LASFDE81:
	.long	LASFDE81-EH_frame1
	.long	LFB1566-.
	.set L$set$310,LFE1566-LFB1566
	.long L$set$310
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$311,LCFI195-LFB1566
	.long L$set$311
	.byte	0xe
	.byte	0x50
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$312,LCFI196-LCFI195
	.long L$set$312
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE81:
	.globl __ZNK17atomic_shared_ptrI1AEdeEv.eh
	.weak_definition __ZNK17atomic_shared_ptrI1AEdeEv.eh
__ZNK17atomic_shared_ptrI1AEdeEv.eh:
LSFDE83:
	.set L$set$313,LEFDE83-LASFDE83
	.long L$set$313
LASFDE83:
	.long	LASFDE83-EH_frame1
	.long	LFB1568-.
	.set L$set$314,LFE1568-LFB1568
	.long L$set$314
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$315,LCFI200-LFB1568
	.long L$set$315
	.byte	0xe
	.byte	0x50
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$316,LCFI201-LCFI200
	.long L$set$316
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE83:
	.globl __ZN17atomic_shared_ptrI1AE3RefC1IS0_EEPT_.eh
	.weak_definition __ZN17atomic_shared_ptrI1AE3RefC1IS0_EEPT_.eh
__ZN17atomic_shared_ptrI1AE3RefC1IS0_EEPT_.eh:
LSFDE85:
	.set L$set$317,LEFDE85-LASFDE85
	.long L$set$317
LASFDE85:
	.long	LASFDE85-EH_frame1
	.long	LFB1588-.
	.set L$set$318,LFE1588-LFB1588
	.long L$set$318
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$319,LCFI203-LFB1588
	.long L$set$319
	.byte	0xe
	.byte	0x30
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x4
	.set L$set$320,LCFI204-LCFI203
	.long L$set$320
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE85:
	.globl __ZN17atomic_shared_ptrI1AE9_RefLocalC1INS1_3RefEEEPT_.eh
	.weak_definition __ZN17atomic_shared_ptrI1AE9_RefLocalC1INS1_3RefEEEPT_.eh
__ZN17atomic_shared_ptrI1AE9_RefLocalC1INS1_3RefEEEPT_.eh:
LSFDE87:
	.set L$set$321,LEFDE87-LASFDE87
	.long L$set$321
LASFDE87:
	.long	LASFDE87-EH_frame1
	.long	LFB1591-.
	.set L$set$322,LFE1591-LFB1591
	.long L$set$322
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$323,LCFI206-LFB1591
	.long L$set$323
	.byte	0xe
	.byte	0x30
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x4
	.set L$set$324,LCFI207-LCFI206
	.long L$set$324
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE87:
	.globl __ZN17atomic_shared_ptrI1AEC1IS0_EEPT_.eh
	.weak_definition __ZN17atomic_shared_ptrI1AEC1IS0_EEPT_.eh
__ZN17atomic_shared_ptrI1AEC1IS0_EEPT_.eh:
LSFDE89:
	.set L$set$325,LEFDE89-LASFDE89
	.long L$set$325
LASFDE89:
	.long	LASFDE89-EH_frame1
	.long	LFB1536-.
	.set L$set$326,LFE1536-LFB1536
	.long L$set$326
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$327,LCFI211-LFB1536
	.long L$set$327
	.byte	0xe
	.byte	0x50
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x9d
	.byte	0x3
	.byte	0x9c
	.byte	0x4
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$328,LCFI212-LCFI211
	.long L$set$328
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE89:
	.globl __ZNK17atomic_shared_ptrI1AE9_RefLocal13refcnt_cachedEv.eh
	.weak_definition __ZNK17atomic_shared_ptrI1AE9_RefLocal13refcnt_cachedEv.eh
__ZNK17atomic_shared_ptrI1AE9_RefLocal13refcnt_cachedEv.eh:
LSFDE91:
	.set L$set$329,LEFDE91-LASFDE91
	.long L$set$329
LASFDE91:
	.long	LASFDE91-EH_frame1
	.long	LFB1592-.
	.set L$set$330,LFE1592-LFB1592
	.long L$set$330
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$331,LCFI214-LFB1592
	.long L$set$331
	.byte	0xe
	.byte	0x30
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x4
	.set L$set$332,LCFI215-LCFI214
	.long L$set$332
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE91:
	.globl __ZN17atomic_shared_ptrI1AE3RefD1Ev.eh
	.weak_definition __ZN17atomic_shared_ptrI1AE3RefD1Ev.eh
__ZN17atomic_shared_ptrI1AE3RefD1Ev.eh:
LSFDE93:
	.set L$set$333,LEFDE93-LASFDE93
	.long L$set$333
LASFDE93:
	.long	LASFDE93-EH_frame1
	.long	LFB1596-.
	.set L$set$334,LFE1596-LFB1596
	.long L$set$334
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$335,LCFI219-LFB1596
	.long L$set$335
	.byte	0xe
	.byte	0x50
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$336,LCFI220-LCFI219
	.long L$set$336
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE93:
	.globl __ZN17atomic_shared_ptrI1AE3RefC1I1BEEPT_.eh
	.weak_definition __ZN17atomic_shared_ptrI1AE3RefC1I1BEEPT_.eh
__ZN17atomic_shared_ptrI1AE3RefC1I1BEEPT_.eh:
LSFDE95:
	.set L$set$337,LEFDE95-LASFDE95
	.long L$set$337
LASFDE95:
	.long	LASFDE95-EH_frame1
	.long	LFB1599-.
	.set L$set$338,LFE1599-LFB1599
	.long L$set$338
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$339,LCFI222-LFB1599
	.long L$set$339
	.byte	0xe
	.byte	0x30
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x4
	.set L$set$340,LCFI223-LCFI222
	.long L$set$340
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE95:
	.globl __ZN17atomic_shared_ptrI1AEC1I1BEEPT_.eh
	.weak_definition __ZN17atomic_shared_ptrI1AEC1I1BEEPT_.eh
__ZN17atomic_shared_ptrI1AEC1I1BEEPT_.eh:
LSFDE97:
	.set L$set$341,LEFDE97-LASFDE97
	.long L$set$341
LASFDE97:
	.long	LASFDE97-EH_frame1
	.long	LFB1542-.
	.set L$set$342,LFE1542-LFB1542
	.long L$set$342
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$343,LCFI227-LFB1542
	.long L$set$343
	.byte	0xe
	.byte	0x50
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x9d
	.byte	0x3
	.byte	0x9c
	.byte	0x4
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$344,LCFI228-LCFI227
	.long L$set$344
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE97:
	.globl __ZN17atomic_shared_ptrI1AE9_RefLocalC1Ev.eh
	.weak_definition __ZN17atomic_shared_ptrI1AE9_RefLocalC1Ev.eh
__ZN17atomic_shared_ptrI1AE9_RefLocalC1Ev.eh:
LSFDE99:
	.set L$set$345,LEFDE99-LASFDE99
	.long L$set$345
LASFDE99:
	.long	LASFDE99-EH_frame1
	.long	LFB1602-.
	.set L$set$346,LFE1602-LFB1602
	.long L$set$346
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$347,LCFI230-LFB1602
	.long L$set$347
	.byte	0xe
	.byte	0x30
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x4
	.set L$set$348,LCFI231-LCFI230
	.long L$set$348
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE99:
	.globl __ZN17atomic_shared_ptrI1AEC1Ev.eh
	.weak_definition __ZN17atomic_shared_ptrI1AEC1Ev.eh
__ZN17atomic_shared_ptrI1AEC1Ev.eh:
LSFDE101:
	.set L$set$349,LEFDE101-LASFDE101
	.long L$set$349
LASFDE101:
	.long	LASFDE101-EH_frame1
	.long	LFB1545-.
	.set L$set$350,LFE1545-LFB1545
	.long L$set$350
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$351,LCFI235-LFB1545
	.long L$set$351
	.byte	0xe
	.byte	0x50
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$352,LCFI236-LCFI235
	.long L$set$352
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE101:
	.globl __Z10atomicSwapIP1AET_S2_PS2_.eh
	.weak_definition __Z10atomicSwapIP1AET_S2_PS2_.eh
__Z10atomicSwapIP1AET_S2_PS2_.eh:
LSFDE103:
	.set L$set$353,LEFDE103-LASFDE103
	.long L$set$353
LASFDE103:
	.long	LASFDE103-EH_frame1
	.long	LFB1604-.
	.set L$set$354,LFE1604-LFB1604
	.long L$set$354
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$355,LCFI238-LFB1604
	.long L$set$355
	.byte	0xe
	.byte	0x40
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x4
	.set L$set$356,LCFI239-LCFI238
	.long L$set$356
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE103:
	.globl __ZN17atomic_scoped_ptrI1AE4swapERS1_.eh
	.weak_definition __ZN17atomic_scoped_ptrI1AE4swapERS1_.eh
__ZN17atomic_scoped_ptrI1AE4swapERS1_.eh:
LSFDE105:
	.set L$set$357,LEFDE105-LASFDE105
	.long L$set$357
LASFDE105:
	.long	LASFDE105-EH_frame1
	.long	LFB1556-.
	.set L$set$358,LFE1556-LFB1556
	.long L$set$358
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$359,LCFI243-LFB1556
	.long L$set$359
	.byte	0xe
	.byte	0x50
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$360,LCFI244-LCFI243
	.long L$set$360
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE105:
	.globl __ZN17atomic_scoped_ptrI1AE5resetEPS0_.eh
	.weak_definition __ZN17atomic_scoped_ptrI1AE5resetEPS0_.eh
__ZN17atomic_scoped_ptrI1AE5resetEPS0_.eh:
LSFDE107:
	.set L$set$361,LEFDE107-LASFDE107
	.long L$set$361
LASFDE107:
	.long	LASFDE107-EH_frame1
	.long	LFB1555-.
	.set L$set$362,LFE1555-LFB1555
	.long L$set$362
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$363,LCFI248-LFB1555
	.long L$set$363
	.byte	0xe
	.byte	0x60
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$364,LCFI249-LCFI248
	.long L$set$364
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE107:
	.globl __Z9atomicAddIiEvPT_S0_.eh
	.weak_definition __Z9atomicAddIiEvPT_S0_.eh
__Z9atomicAddIiEvPT_S0_.eh:
LSFDE109:
	.set L$set$365,LEFDE109-LASFDE109
	.long L$set$365
LASFDE109:
	.long	LASFDE109-EH_frame1
	.long	LFB1605-.
	.set L$set$366,LFE1605-LFB1605
	.long L$set$366
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$367,LCFI251-LFB1605
	.long L$set$367
	.byte	0xe
	.byte	0x40
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x4
	.set L$set$368,LCFI252-LCFI251
	.long L$set$368
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE109:
	.globl __ZN6atomicIiEpLEi.eh
	.weak_definition __ZN6atomicIiEpLEi.eh
__ZN6atomicIiEpLEi.eh:
LSFDE111:
	.set L$set$369,LEFDE111-LASFDE111
	.long L$set$369
LASFDE111:
	.long	LASFDE111-EH_frame1
	.long	LFB1567-.
	.set L$set$370,LFE1567-LFB1567
	.long L$set$370
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$371,LCFI256-LFB1567
	.long L$set$371
	.byte	0xe
	.byte	0x50
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$372,LCFI257-LCFI256
	.long L$set$372
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE111:
	.globl __Z9atomicAddIjEvPT_S0_.eh
	.weak_definition __Z9atomicAddIjEvPT_S0_.eh
__Z9atomicAddIjEvPT_S0_.eh:
LSFDE113:
	.set L$set$373,LEFDE113-LASFDE113
	.long L$set$373
LASFDE113:
	.long	LASFDE113-EH_frame1
	.long	LFB1606-.
	.set L$set$374,LFE1606-LFB1606
	.long L$set$374
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$375,LCFI259-LFB1606
	.long L$set$375
	.byte	0xe
	.byte	0x40
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x4
	.set L$set$376,LCFI260-LCFI259
	.long L$set$376
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE113:
	.globl __ZNK17atomic_shared_ptrI1AE9_RefLocal6serialEv.eh
	.weak_definition __ZNK17atomic_shared_ptrI1AE9_RefLocal6serialEv.eh
__ZNK17atomic_shared_ptrI1AE9_RefLocal6serialEv.eh:
LSFDE115:
	.set L$set$377,LEFDE115-LASFDE115
	.long L$set$377
LASFDE115:
	.long	LASFDE115-EH_frame1
	.long	LFB1607-.
	.set L$set$378,LFE1607-LFB1607
	.long L$set$378
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$379,LCFI262-LFB1607
	.long L$set$379
	.byte	0xe
	.byte	0x30
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x4
	.set L$set$380,LCFI263-LCFI262
	.long L$set$380
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE115:
	.globl __ZNK17atomic_shared_ptrI1AE9_RefLocal16_refcnt_n_serialEjj.eh
	.weak_definition __ZNK17atomic_shared_ptrI1AE9_RefLocal16_refcnt_n_serialEjj.eh
__ZNK17atomic_shared_ptrI1AE9_RefLocal16_refcnt_n_serialEjj.eh:
LSFDE117:
	.set L$set$381,LEFDE117-LASFDE117
	.long L$set$381
LASFDE117:
	.long	LASFDE117-EH_frame1
	.long	LFB1608-.
	.set L$set$382,LFE1608-LFB1608
	.long L$set$382
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$383,LCFI265-LFB1608
	.long L$set$383
	.byte	0xe
	.byte	0x30
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x4
	.set L$set$384,LCFI266-LCFI265
	.long L$set$384
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE117:
	.globl __Z16atomicAddAndTestIiEbPT_S0_.eh
	.weak_definition __Z16atomicAddAndTestIiEbPT_S0_.eh
__Z16atomicAddAndTestIiEbPT_S0_.eh:
LSFDE119:
	.set L$set$385,LEFDE119-LASFDE119
	.long L$set$385
LASFDE119:
	.long	LASFDE119-EH_frame1
	.long	LFB1609-.
	.set L$set$386,LFE1609-LFB1609
	.long L$set$386
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$387,LCFI268-LFB1609
	.long L$set$387
	.byte	0xe
	.byte	0x40
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x4
	.set L$set$388,LCFI269-LCFI268
	.long L$set$388
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE119:
	.globl __Z16atomicDecAndTestIiEbPT_.eh
	.weak_definition __Z16atomicDecAndTestIiEbPT_.eh
__Z16atomicDecAndTestIiEbPT_.eh:
LSFDE121:
	.set L$set$389,LEFDE121-LASFDE121
	.long L$set$389
LASFDE121:
	.long	LASFDE121-EH_frame1
	.long	LFB1610-.
	.set L$set$390,LFE1610-LFB1610
	.long L$set$390
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$391,LCFI273-LFB1610
	.long L$set$391
	.byte	0xe
	.byte	0x50
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$392,LCFI274-LCFI273
	.long L$set$392
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE121:
	.globl __ZN17atomic_shared_ptrI1AE4swapERS1_.eh
	.weak_definition __ZN17atomic_shared_ptrI1AE4swapERS1_.eh
__ZN17atomic_shared_ptrI1AE4swapERS1_.eh:
LSFDE123:
	.set L$set$393,LEFDE123-LASFDE123
	.long L$set$393
LASFDE123:
	.long	LASFDE123-EH_frame1
	.long	LFB1572-.
	.set L$set$394,LFE1572-LFB1572
	.long L$set$394
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$395,LCFI278-LFB1572
	.long L$set$395
	.byte	0xe
	.byte	0x90,0x1
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
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$396,LCFI279-LCFI278
	.long L$set$396
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE123:
	.globl __Z16atomicAddAndTestIjEbPT_S0_.eh
	.weak_definition __Z16atomicAddAndTestIjEbPT_S0_.eh
__Z16atomicAddAndTestIjEbPT_S0_.eh:
LSFDE125:
	.set L$set$397,LEFDE125-LASFDE125
	.long L$set$397
LASFDE125:
	.long	LASFDE125-EH_frame1
	.long	LFB1612-.
	.set L$set$398,LFE1612-LFB1612
	.long L$set$398
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$399,LCFI281-LFB1612
	.long L$set$399
	.byte	0xe
	.byte	0x40
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x4
	.set L$set$400,LCFI282-LCFI281
	.long L$set$400
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE125:
	.globl __Z16atomicDecAndTestIjEbPT_.eh
	.weak_definition __Z16atomicDecAndTestIjEbPT_.eh
__Z16atomicDecAndTestIjEbPT_.eh:
LSFDE127:
	.set L$set$401,LEFDE127-LASFDE127
	.long L$set$401
LASFDE127:
	.long	LASFDE127-EH_frame1
	.long	LFB1593-.
	.set L$set$402,LFE1593-LFB1593
	.long L$set$402
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$403,LCFI286-LFB1593
	.long L$set$403
	.byte	0xe
	.byte	0x50
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$404,LCFI287-LCFI286
	.long L$set$404
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE127:
	.globl __ZN17atomic_shared_ptrI1AED1Ev.eh
	.weak_definition __ZN17atomic_shared_ptrI1AED1Ev.eh
__ZN17atomic_shared_ptrI1AED1Ev.eh:
LSFDE129:
	.set L$set$405,LEFDE129-LASFDE129
	.long L$set$405
LASFDE129:
	.long	LASFDE129-EH_frame1
	.long	LFB1539-.
	.set L$set$406,LFE1539-LFB1539
	.long L$set$406
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$407,LCFI291-LFB1539
	.long L$set$407
	.byte	0xe
	.byte	0x60
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$408,LCFI292-LCFI291
	.long L$set$408
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE129:
	.globl __ZN17atomic_shared_ptrI1AE5resetIS0_EEvPT_.eh
	.weak_definition __ZN17atomic_shared_ptrI1AE5resetIS0_EEvPT_.eh
__ZN17atomic_shared_ptrI1AE5resetIS0_EEvPT_.eh:
LSFDE131:
	.set L$set$409,LEFDE131-LASFDE131
	.long L$set$409
LASFDE131:
	.long	LASFDE131-EH_frame1
	.long	LFB1569-.
	.set L$set$410,LFE1569-LFB1569
	.long L$set$410
	.byte	0x4
	.long	LLSDA1569-.
	.byte	0x4
	.set L$set$411,LCFI296-LFB1569
	.long L$set$411
	.byte	0xe
	.byte	0x60
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x9d
	.byte	0x3
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$412,LCFI297-LCFI296
	.long L$set$412
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE131:
	.globl __ZN17atomic_shared_ptrI1AE5resetEv.eh
	.weak_definition __ZN17atomic_shared_ptrI1AE5resetEv.eh
__ZN17atomic_shared_ptrI1AE5resetEv.eh:
LSFDE133:
	.set L$set$413,LEFDE133-LASFDE133
	.long L$set$413
LASFDE133:
	.long	LASFDE133-EH_frame1
	.long	LFB1571-.
	.set L$set$414,LFE1571-LFB1571
	.long L$set$414
	.byte	0x4
	.long	LLSDA1571-.
	.byte	0x4
	.set L$set$415,LCFI301-LFB1571
	.long L$set$415
	.byte	0xe
	.byte	0x60
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x9d
	.byte	0x3
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$416,LCFI302-LCFI301
	.long L$set$416
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE133:
	.globl __Z9atomicIncIjEvPT_.eh
	.weak_definition __Z9atomicIncIjEvPT_.eh
__Z9atomicIncIjEvPT_.eh:
LSFDE135:
	.set L$set$417,LEFDE135-LASFDE135
	.long L$set$417
LASFDE135:
	.long	LASFDE135-EH_frame1
	.long	LFB1613-.
	.set L$set$418,LFE1613-LFB1613
	.long L$set$418
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$419,LCFI304-LFB1613
	.long L$set$419
	.byte	0xe
	.byte	0x40
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x4
	.set L$set$420,LCFI305-LCFI304
	.long L$set$420
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE135:
	.globl __ZNK17atomic_shared_ptrI1AE6_scan_Ev.eh
	.weak_definition __ZNK17atomic_shared_ptrI1AE6_scan_Ev.eh
__ZNK17atomic_shared_ptrI1AE6_scan_Ev.eh:
LSFDE137:
	.set L$set$421,LEFDE137-LASFDE137
	.long L$set$421
LASFDE137:
	.long	LASFDE137-EH_frame1
	.long	LFB1603-.
	.set L$set$422,LFE1603-LFB1603
	.long L$set$422
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$423,LCFI309-LFB1603
	.long L$set$423
	.byte	0xe
	.byte	0x70
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
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$424,LCFI310-LCFI309
	.long L$set$424
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE137:
	.globl __ZN17atomic_shared_ptrI1AEC1ERKS1_.eh
	.weak_definition __ZN17atomic_shared_ptrI1AEC1ERKS1_.eh
__ZN17atomic_shared_ptrI1AEC1ERKS1_.eh:
LSFDE139:
	.set L$set$425,LEFDE139-LASFDE139
	.long L$set$425
LASFDE139:
	.long	LASFDE139-EH_frame1
	.long	LFB1548-.
	.set L$set$426,LFE1548-LFB1548
	.long L$set$426
	.byte	0x4
	.long	0x0
	.byte	0x4
	.set L$set$427,LCFI314-LFB1548
	.long L$set$427
	.byte	0xe
	.byte	0x50
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x9d
	.byte	0x3
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$428,LCFI315-LCFI314
	.long L$set$428
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE139:
	.globl __ZN17atomic_shared_ptrI1AEaSERKS1_.eh
	.weak_definition __ZN17atomic_shared_ptrI1AEaSERKS1_.eh
__ZN17atomic_shared_ptrI1AEaSERKS1_.eh:
LSFDE141:
	.set L$set$429,LEFDE141-LASFDE141
	.long L$set$429
LASFDE141:
	.long	LASFDE141-EH_frame1
	.long	LFB1570-.
	.set L$set$430,LFE1570-LFB1570
	.long L$set$430
	.byte	0x4
	.long	LLSDA1570-.
	.byte	0x4
	.set L$set$431,LCFI319-LFB1570
	.long L$set$431
	.byte	0xe
	.byte	0x60
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x9d
	.byte	0x3
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$432,LCFI320-LCFI319
	.long L$set$432
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE141:
	.globl _main.eh
_main.eh:
LSFDE143:
	.set L$set$433,LEFDE143-LASFDE143
	.long L$set$433
LASFDE143:
	.long	LASFDE143-EH_frame1
	.long	LFB1497-.
	.set L$set$434,LFE1497-LFB1497
	.long L$set$434
	.byte	0x4
	.long	LLSDA1497-.
	.byte	0x4
	.set L$set$435,LCFI324-LFB1497
	.long L$set$435
	.byte	0xe
	.byte	0xe0,0x1
	.byte	0x9f
	.byte	0x1
	.byte	0x9e
	.byte	0x2
	.byte	0x9d
	.byte	0x3
	.byte	0x11
	.byte	0x41
	.byte	0x7e
	.byte	0x4
	.set L$set$436,LCFI325-LCFI324
	.long L$set$436
	.byte	0xd
	.byte	0x1e
	.align 2
LEFDE143:
	.text
	.stabs	"",100,0,0,Letext0
Letext0:
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L_pthread_cond_timedwait$stub:
	.indirect_symbol _pthread_cond_timedwait
	mflr r0
	bcl 20,31,"L00000000001$spb"
"L00000000001$spb":
	mflr r11
	addis r11,r11,ha16(L_pthread_cond_timedwait$lazy_ptr-"L00000000001$spb")
	mtlr r0
	lwzu r12,lo16(L_pthread_cond_timedwait$lazy_ptr-"L00000000001$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L_pthread_cond_timedwait$lazy_ptr:
	.indirect_symbol _pthread_cond_timedwait
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__Z16atomicAddAndTestIjEbPT_S0_$stub:
	.indirect_symbol __Z16atomicAddAndTestIjEbPT_S0_
	mflr r0
	bcl 20,31,"L00000000002$spb"
"L00000000002$spb":
	mflr r11
	addis r11,r11,ha16(L__Z16atomicAddAndTestIjEbPT_S0_$lazy_ptr-"L00000000002$spb")
	mtlr r0
	lwzu r12,lo16(L__Z16atomicAddAndTestIjEbPT_S0_$lazy_ptr-"L00000000002$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__Z16atomicAddAndTestIjEbPT_S0_$lazy_ptr:
	.indirect_symbol __Z16atomicAddAndTestIjEbPT_S0_
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__Z11_my_cassertm$stub:
	.indirect_symbol __Z11_my_cassertm
	mflr r0
	bcl 20,31,"L00000000003$spb"
"L00000000003$spb":
	mflr r11
	addis r11,r11,ha16(L__Z11_my_cassertm$lazy_ptr-"L00000000003$spb")
	mtlr r0
	lwzu r12,lo16(L__Z11_my_cassertm$lazy_ptr-"L00000000003$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__Z11_my_cassertm$lazy_ptr:
	.indirect_symbol __Z11_my_cassertm
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__ZN17atomic_scoped_ptrI1AEC1EPS0_$stub:
	.indirect_symbol __ZN17atomic_scoped_ptrI1AEC1EPS0_
	mflr r0
	bcl 20,31,"L00000000004$spb"
"L00000000004$spb":
	mflr r11
	addis r11,r11,ha16(L__ZN17atomic_scoped_ptrI1AEC1EPS0_$lazy_ptr-"L00000000004$spb")
	mtlr r0
	lwzu r12,lo16(L__ZN17atomic_scoped_ptrI1AEC1EPS0_$lazy_ptr-"L00000000004$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__ZN17atomic_scoped_ptrI1AEC1EPS0_$lazy_ptr:
	.indirect_symbol __ZN17atomic_scoped_ptrI1AEC1EPS0_
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__Z9atomicIncIjEvPT_$stub:
	.indirect_symbol __Z9atomicIncIjEvPT_
	mflr r0
	bcl 20,31,"L00000000005$spb"
"L00000000005$spb":
	mflr r11
	addis r11,r11,ha16(L__Z9atomicIncIjEvPT_$lazy_ptr-"L00000000005$spb")
	mtlr r0
	lwzu r12,lo16(L__Z9atomicIncIjEvPT_$lazy_ptr-"L00000000005$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__Z9atomicIncIjEvPT_$lazy_ptr:
	.indirect_symbol __Z9atomicIncIjEvPT_
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L_pthread_cond_signal$stub:
	.indirect_symbol _pthread_cond_signal
	mflr r0
	bcl 20,31,"L00000000006$spb"
"L00000000006$spb":
	mflr r11
	addis r11,r11,ha16(L_pthread_cond_signal$lazy_ptr-"L00000000006$spb")
	mtlr r0
	lwzu r12,lo16(L_pthread_cond_signal$lazy_ptr-"L00000000006$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L_pthread_cond_signal$lazy_ptr:
	.indirect_symbol _pthread_cond_signal
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__ZN6atomicIiEC1Ei$stub:
	.indirect_symbol __ZN6atomicIiEC1Ei
	mflr r0
	bcl 20,31,"L00000000007$spb"
"L00000000007$spb":
	mflr r11
	addis r11,r11,ha16(L__ZN6atomicIiEC1Ei$lazy_ptr-"L00000000007$spb")
	mtlr r0
	lwzu r12,lo16(L__ZN6atomicIiEC1Ei$lazy_ptr-"L00000000007$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__ZN6atomicIiEC1Ei$lazy_ptr:
	.indirect_symbol __ZN6atomicIiEC1Ei
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__ZN1AD2Ev$stub:
	.indirect_symbol __ZN1AD2Ev
	mflr r0
	bcl 20,31,"L00000000008$spb"
"L00000000008$spb":
	mflr r11
	addis r11,r11,ha16(L__ZN1AD2Ev$lazy_ptr-"L00000000008$spb")
	mtlr r0
	lwzu r12,lo16(L__ZN1AD2Ev$lazy_ptr-"L00000000008$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__ZN1AD2Ev$lazy_ptr:
	.indirect_symbol __ZN1AD2Ev
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__Z13memoryBarrierv$stub:
	.indirect_symbol __Z13memoryBarrierv
	mflr r0
	bcl 20,31,"L00000000009$spb"
"L00000000009$spb":
	mflr r11
	addis r11,r11,ha16(L__Z13memoryBarrierv$lazy_ptr-"L00000000009$spb")
	mtlr r0
	lwzu r12,lo16(L__Z13memoryBarrierv$lazy_ptr-"L00000000009$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__Z13memoryBarrierv$lazy_ptr:
	.indirect_symbol __Z13memoryBarrierv
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__Z11readBarrierv$stub:
	.indirect_symbol __Z11readBarrierv
	mflr r0
	bcl 20,31,"L00000000010$spb"
"L00000000010$spb":
	mflr r11
	addis r11,r11,ha16(L__Z11readBarrierv$lazy_ptr-"L00000000010$spb")
	mtlr r0
	lwzu r12,lo16(L__Z11readBarrierv$lazy_ptr-"L00000000010$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__Z11readBarrierv$lazy_ptr:
	.indirect_symbol __Z11readBarrierv
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__ZNK17atomic_shared_ptrI1AE9_RefLocal6serialEv$stub:
	.indirect_symbol __ZNK17atomic_shared_ptrI1AE9_RefLocal6serialEv
	mflr r0
	bcl 20,31,"L00000000011$spb"
"L00000000011$spb":
	mflr r11
	addis r11,r11,ha16(L__ZNK17atomic_shared_ptrI1AE9_RefLocal6serialEv$lazy_ptr-"L00000000011$spb")
	mtlr r0
	lwzu r12,lo16(L__ZNK17atomic_shared_ptrI1AE9_RefLocal6serialEv$lazy_ptr-"L00000000011$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__ZNK17atomic_shared_ptrI1AE9_RefLocal6serialEv$lazy_ptr:
	.indirect_symbol __ZNK17atomic_shared_ptrI1AE9_RefLocal6serialEv
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__ZNK17atomic_shared_ptrI1AEdeEv$stub:
	.indirect_symbol __ZNK17atomic_shared_ptrI1AEdeEv
	mflr r0
	bcl 20,31,"L00000000012$spb"
"L00000000012$spb":
	mflr r11
	addis r11,r11,ha16(L__ZNK17atomic_shared_ptrI1AEdeEv$lazy_ptr-"L00000000012$spb")
	mtlr r0
	lwzu r12,lo16(L__ZNK17atomic_shared_ptrI1AEdeEv$lazy_ptr-"L00000000012$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__ZNK17atomic_shared_ptrI1AEdeEv$lazy_ptr:
	.indirect_symbol __ZNK17atomic_shared_ptrI1AEdeEv
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__Unwind_Resume$stub:
	.indirect_symbol __Unwind_Resume
	mflr r0
	bcl 20,31,"L00000000013$spb"
"L00000000013$spb":
	mflr r11
	addis r11,r11,ha16(L__Unwind_Resume$lazy_ptr-"L00000000013$spb")
	mtlr r0
	lwzu r12,lo16(L__Unwind_Resume$lazy_ptr-"L00000000013$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__Unwind_Resume$lazy_ptr:
	.indirect_symbol __Unwind_Resume
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L_pthread_cond_wait$stub:
	.indirect_symbol _pthread_cond_wait
	mflr r0
	bcl 20,31,"L00000000014$spb"
"L00000000014$spb":
	mflr r11
	addis r11,r11,ha16(L_pthread_cond_wait$lazy_ptr-"L00000000014$spb")
	mtlr r0
	lwzu r12,lo16(L_pthread_cond_wait$lazy_ptr-"L00000000014$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L_pthread_cond_wait$lazy_ptr:
	.indirect_symbol _pthread_cond_wait
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L_pthread_mutex_trylock$stub:
	.indirect_symbol _pthread_mutex_trylock
	mflr r0
	bcl 20,31,"L00000000015$spb"
"L00000000015$spb":
	mflr r11
	addis r11,r11,ha16(L_pthread_mutex_trylock$lazy_ptr-"L00000000015$spb")
	mtlr r0
	lwzu r12,lo16(L_pthread_mutex_trylock$lazy_ptr-"L00000000015$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L_pthread_mutex_trylock$lazy_ptr:
	.indirect_symbol _pthread_mutex_trylock
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__Z9atomicAddIjEvPT_S0_$stub:
	.indirect_symbol __Z9atomicAddIjEvPT_S0_
	mflr r0
	bcl 20,31,"L00000000016$spb"
"L00000000016$spb":
	mflr r11
	addis r11,r11,ha16(L__Z9atomicAddIjEvPT_S0_$lazy_ptr-"L00000000016$spb")
	mtlr r0
	lwzu r12,lo16(L__Z9atomicAddIjEvPT_S0_$lazy_ptr-"L00000000016$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__Z9atomicAddIjEvPT_S0_$lazy_ptr:
	.indirect_symbol __Z9atomicAddIjEvPT_S0_
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L_pthread_mutex_lock$stub:
	.indirect_symbol _pthread_mutex_lock
	mflr r0
	bcl 20,31,"L00000000017$spb"
"L00000000017$spb":
	mflr r11
	addis r11,r11,ha16(L_pthread_mutex_lock$lazy_ptr-"L00000000017$spb")
	mtlr r0
	lwzu r12,lo16(L_pthread_mutex_lock$lazy_ptr-"L00000000017$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L_pthread_mutex_lock$lazy_ptr:
	.indirect_symbol _pthread_mutex_lock
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__ZNK17atomic_shared_ptrI1AE9_RefLocal13refcnt_cachedEv$stub:
	.indirect_symbol __ZNK17atomic_shared_ptrI1AE9_RefLocal13refcnt_cachedEv
	mflr r0
	bcl 20,31,"L00000000018$spb"
"L00000000018$spb":
	mflr r11
	addis r11,r11,ha16(L__ZNK17atomic_shared_ptrI1AE9_RefLocal13refcnt_cachedEv$lazy_ptr-"L00000000018$spb")
	mtlr r0
	lwzu r12,lo16(L__ZNK17atomic_shared_ptrI1AE9_RefLocal13refcnt_cachedEv$lazy_ptr-"L00000000018$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__ZNK17atomic_shared_ptrI1AE9_RefLocal13refcnt_cachedEv$lazy_ptr:
	.indirect_symbol __ZNK17atomic_shared_ptrI1AE9_RefLocal13refcnt_cachedEv
	.long	dyld_stub_binding_helper
	.non_lazy_symbol_pointer
L__ZTV1B$non_lazy_ptr:
	.indirect_symbol __ZTV1B
	.long	0
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__Z16atomicDecAndTestIjEbPT_$stub:
	.indirect_symbol __Z16atomicDecAndTestIjEbPT_
	mflr r0
	bcl 20,31,"L00000000019$spb"
"L00000000019$spb":
	mflr r11
	addis r11,r11,ha16(L__Z16atomicDecAndTestIjEbPT_$lazy_ptr-"L00000000019$spb")
	mtlr r0
	lwzu r12,lo16(L__Z16atomicDecAndTestIjEbPT_$lazy_ptr-"L00000000019$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__Z16atomicDecAndTestIjEbPT_$lazy_ptr:
	.indirect_symbol __Z16atomicDecAndTestIjEbPT_
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__ZN17atomic_scoped_ptrI1AE4swapERS1_$stub:
	.indirect_symbol __ZN17atomic_scoped_ptrI1AE4swapERS1_
	mflr r0
	bcl 20,31,"L00000000020$spb"
"L00000000020$spb":
	mflr r11
	addis r11,r11,ha16(L__ZN17atomic_scoped_ptrI1AE4swapERS1_$lazy_ptr-"L00000000020$spb")
	mtlr r0
	lwzu r12,lo16(L__ZN17atomic_scoped_ptrI1AE4swapERS1_$lazy_ptr-"L00000000020$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__ZN17atomic_scoped_ptrI1AE4swapERS1_$lazy_ptr:
	.indirect_symbol __ZN17atomic_scoped_ptrI1AE4swapERS1_
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__ZN17atomic_shared_ptrI1AEaSERKS1_$stub:
	.indirect_symbol __ZN17atomic_shared_ptrI1AEaSERKS1_
	mflr r0
	bcl 20,31,"L00000000021$spb"
"L00000000021$spb":
	mflr r11
	addis r11,r11,ha16(L__ZN17atomic_shared_ptrI1AEaSERKS1_$lazy_ptr-"L00000000021$spb")
	mtlr r0
	lwzu r12,lo16(L__ZN17atomic_shared_ptrI1AEaSERKS1_$lazy_ptr-"L00000000021$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__ZN17atomic_shared_ptrI1AEaSERKS1_$lazy_ptr:
	.indirect_symbol __ZN17atomic_shared_ptrI1AEaSERKS1_
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__ZNK6atomicIiEcviEv$stub:
	.indirect_symbol __ZNK6atomicIiEcviEv
	mflr r0
	bcl 20,31,"L00000000022$spb"
"L00000000022$spb":
	mflr r11
	addis r11,r11,ha16(L__ZNK6atomicIiEcviEv$lazy_ptr-"L00000000022$spb")
	mtlr r0
	lwzu r12,lo16(L__ZNK6atomicIiEcviEv$lazy_ptr-"L00000000022$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__ZNK6atomicIiEcviEv$lazy_ptr:
	.indirect_symbol __ZNK6atomicIiEcviEv
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__ZN17atomic_shared_ptrI1AE9_RefLocalC1INS1_3RefEEEPT_$stub:
	.indirect_symbol __ZN17atomic_shared_ptrI1AE9_RefLocalC1INS1_3RefEEEPT_
	mflr r0
	bcl 20,31,"L00000000023$spb"
"L00000000023$spb":
	mflr r11
	addis r11,r11,ha16(L__ZN17atomic_shared_ptrI1AE9_RefLocalC1INS1_3RefEEEPT_$lazy_ptr-"L00000000023$spb")
	mtlr r0
	lwzu r12,lo16(L__ZN17atomic_shared_ptrI1AE9_RefLocalC1INS1_3RefEEEPT_$lazy_ptr-"L00000000023$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__ZN17atomic_shared_ptrI1AE9_RefLocalC1INS1_3RefEEEPT_$lazy_ptr:
	.indirect_symbol __ZN17atomic_shared_ptrI1AE9_RefLocalC1INS1_3RefEEEPT_
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__ZNK17atomic_shared_ptrI1AE9_RefLocal16_refcnt_n_serialEjj$stub:
	.indirect_symbol __ZNK17atomic_shared_ptrI1AE9_RefLocal16_refcnt_n_serialEjj
	mflr r0
	bcl 20,31,"L00000000024$spb"
"L00000000024$spb":
	mflr r11
	addis r11,r11,ha16(L__ZNK17atomic_shared_ptrI1AE9_RefLocal16_refcnt_n_serialEjj$lazy_ptr-"L00000000024$spb")
	mtlr r0
	lwzu r12,lo16(L__ZNK17atomic_shared_ptrI1AE9_RefLocal16_refcnt_n_serialEjj$lazy_ptr-"L00000000024$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__ZNK17atomic_shared_ptrI1AE9_RefLocal16_refcnt_n_serialEjj$lazy_ptr:
	.indirect_symbol __ZNK17atomic_shared_ptrI1AE9_RefLocal16_refcnt_n_serialEjj
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__ZN17atomic_shared_ptrI1AEC1I1BEEPT_$stub:
	.indirect_symbol __ZN17atomic_shared_ptrI1AEC1I1BEEPT_
	mflr r0
	bcl 20,31,"L00000000025$spb"
"L00000000025$spb":
	mflr r11
	addis r11,r11,ha16(L__ZN17atomic_shared_ptrI1AEC1I1BEEPT_$lazy_ptr-"L00000000025$spb")
	mtlr r0
	lwzu r12,lo16(L__ZN17atomic_shared_ptrI1AEC1I1BEEPT_$lazy_ptr-"L00000000025$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__ZN17atomic_shared_ptrI1AEC1I1BEEPT_$lazy_ptr:
	.indirect_symbol __ZN17atomic_shared_ptrI1AEC1I1BEEPT_
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L_fprintf$LDBLStub$stub:
	.indirect_symbol _fprintf$LDBLStub
	mflr r0
	bcl 20,31,"L00000000026$spb"
"L00000000026$spb":
	mflr r11
	addis r11,r11,ha16(L_fprintf$LDBLStub$lazy_ptr-"L00000000026$spb")
	mtlr r0
	lwzu r12,lo16(L_fprintf$LDBLStub$lazy_ptr-"L00000000026$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L_fprintf$LDBLStub$lazy_ptr:
	.indirect_symbol _fprintf$LDBLStub
	.long	dyld_stub_binding_helper
	.non_lazy_symbol_pointer
L___gxx_personality_v0$non_lazy_ptr:
	.indirect_symbol ___gxx_personality_v0
	.long	0
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__ZN17atomic_shared_ptrI1AEC1Ev$stub:
	.indirect_symbol __ZN17atomic_shared_ptrI1AEC1Ev
	mflr r0
	bcl 20,31,"L00000000027$spb"
"L00000000027$spb":
	mflr r11
	addis r11,r11,ha16(L__ZN17atomic_shared_ptrI1AEC1Ev$lazy_ptr-"L00000000027$spb")
	mtlr r0
	lwzu r12,lo16(L__ZN17atomic_shared_ptrI1AEC1Ev$lazy_ptr-"L00000000027$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__ZN17atomic_shared_ptrI1AEC1Ev$lazy_ptr:
	.indirect_symbol __ZN17atomic_shared_ptrI1AEC1Ev
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__ZN17atomic_shared_ptrI1AE4swapERS1_$stub:
	.indirect_symbol __ZN17atomic_shared_ptrI1AE4swapERS1_
	mflr r0
	bcl 20,31,"L00000000028$spb"
"L00000000028$spb":
	mflr r11
	addis r11,r11,ha16(L__ZN17atomic_shared_ptrI1AE4swapERS1_$lazy_ptr-"L00000000028$spb")
	mtlr r0
	lwzu r12,lo16(L__ZN17atomic_shared_ptrI1AE4swapERS1_$lazy_ptr-"L00000000028$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__ZN17atomic_shared_ptrI1AE4swapERS1_$lazy_ptr:
	.indirect_symbol __ZN17atomic_shared_ptrI1AE4swapERS1_
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__ZN17atomic_scoped_ptrI1AE5resetEPS0_$stub:
	.indirect_symbol __ZN17atomic_scoped_ptrI1AE5resetEPS0_
	mflr r0
	bcl 20,31,"L00000000029$spb"
"L00000000029$spb":
	mflr r11
	addis r11,r11,ha16(L__ZN17atomic_scoped_ptrI1AE5resetEPS0_$lazy_ptr-"L00000000029$spb")
	mtlr r0
	lwzu r12,lo16(L__ZN17atomic_scoped_ptrI1AE5resetEPS0_$lazy_ptr-"L00000000029$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__ZN17atomic_scoped_ptrI1AE5resetEPS0_$lazy_ptr:
	.indirect_symbol __ZN17atomic_scoped_ptrI1AE5resetEPS0_
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__ZN17atomic_shared_ptrI1AE3RefD1Ev$stub:
	.indirect_symbol __ZN17atomic_shared_ptrI1AE3RefD1Ev
	mflr r0
	bcl 20,31,"L00000000030$spb"
"L00000000030$spb":
	mflr r11
	addis r11,r11,ha16(L__ZN17atomic_shared_ptrI1AE3RefD1Ev$lazy_ptr-"L00000000030$spb")
	mtlr r0
	lwzu r12,lo16(L__ZN17atomic_shared_ptrI1AE3RefD1Ev$lazy_ptr-"L00000000030$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__ZN17atomic_shared_ptrI1AE3RefD1Ev$lazy_ptr:
	.indirect_symbol __ZN17atomic_shared_ptrI1AE3RefD1Ev
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__ZN17atomic_shared_ptrI1AEC1ERKS1_$stub:
	.indirect_symbol __ZN17atomic_shared_ptrI1AEC1ERKS1_
	mflr r0
	bcl 20,31,"L00000000031$spb"
"L00000000031$spb":
	mflr r11
	addis r11,r11,ha16(L__ZN17atomic_shared_ptrI1AEC1ERKS1_$lazy_ptr-"L00000000031$spb")
	mtlr r0
	lwzu r12,lo16(L__ZN17atomic_shared_ptrI1AEC1ERKS1_$lazy_ptr-"L00000000031$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__ZN17atomic_shared_ptrI1AEC1ERKS1_$lazy_ptr:
	.indirect_symbol __ZN17atomic_shared_ptrI1AEC1ERKS1_
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__Z16atomicAddAndTestIiEbPT_S0_$stub:
	.indirect_symbol __Z16atomicAddAndTestIiEbPT_S0_
	mflr r0
	bcl 20,31,"L00000000032$spb"
"L00000000032$spb":
	mflr r11
	addis r11,r11,ha16(L__Z16atomicAddAndTestIiEbPT_S0_$lazy_ptr-"L00000000032$spb")
	mtlr r0
	lwzu r12,lo16(L__Z16atomicAddAndTestIiEbPT_S0_$lazy_ptr-"L00000000032$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__Z16atomicAddAndTestIiEbPT_S0_$lazy_ptr:
	.indirect_symbol __Z16atomicAddAndTestIiEbPT_S0_
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L_pthread_self$stub:
	.indirect_symbol _pthread_self
	mflr r0
	bcl 20,31,"L00000000033$spb"
"L00000000033$spb":
	mflr r11
	addis r11,r11,ha16(L_pthread_self$lazy_ptr-"L00000000033$spb")
	mtlr r0
	lwzu r12,lo16(L_pthread_self$lazy_ptr-"L00000000033$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L_pthread_self$lazy_ptr:
	.indirect_symbol _pthread_self
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__Z20atomicCompareAndSet2jjjjPj$stub:
	.indirect_symbol __Z20atomicCompareAndSet2jjjjPj
	mflr r0
	bcl 20,31,"L00000000034$spb"
"L00000000034$spb":
	mflr r11
	addis r11,r11,ha16(L__Z20atomicCompareAndSet2jjjjPj$lazy_ptr-"L00000000034$spb")
	mtlr r0
	lwzu r12,lo16(L__Z20atomicCompareAndSet2jjjjPj$lazy_ptr-"L00000000034$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__Z20atomicCompareAndSet2jjjjPj$lazy_ptr:
	.indirect_symbol __Z20atomicCompareAndSet2jjjjPj
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__ZN17atomic_shared_ptrI1AE3RefC1IS0_EEPT_$stub:
	.indirect_symbol __ZN17atomic_shared_ptrI1AE3RefC1IS0_EEPT_
	mflr r0
	bcl 20,31,"L00000000035$spb"
"L00000000035$spb":
	mflr r11
	addis r11,r11,ha16(L__ZN17atomic_shared_ptrI1AE3RefC1IS0_EEPT_$lazy_ptr-"L00000000035$spb")
	mtlr r0
	lwzu r12,lo16(L__ZN17atomic_shared_ptrI1AE3RefC1IS0_EEPT_$lazy_ptr-"L00000000035$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__ZN17atomic_shared_ptrI1AE3RefC1IS0_EEPT_$lazy_ptr:
	.indirect_symbol __ZN17atomic_shared_ptrI1AE3RefC1IS0_EEPT_
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__ZN17atomic_shared_ptrI1AEC1IS0_EEPT_$stub:
	.indirect_symbol __ZN17atomic_shared_ptrI1AEC1IS0_EEPT_
	mflr r0
	bcl 20,31,"L00000000036$spb"
"L00000000036$spb":
	mflr r11
	addis r11,r11,ha16(L__ZN17atomic_shared_ptrI1AEC1IS0_EEPT_$lazy_ptr-"L00000000036$spb")
	mtlr r0
	lwzu r12,lo16(L__ZN17atomic_shared_ptrI1AEC1IS0_EEPT_$lazy_ptr-"L00000000036$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__ZN17atomic_shared_ptrI1AEC1IS0_EEPT_$lazy_ptr:
	.indirect_symbol __ZN17atomic_shared_ptrI1AEC1IS0_EEPT_
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__ZNK17atomic_shared_ptrI1AE6_scan_Ev$stub:
	.indirect_symbol __ZNK17atomic_shared_ptrI1AE6_scan_Ev
	mflr r0
	bcl 20,31,"L00000000037$spb"
"L00000000037$spb":
	mflr r11
	addis r11,r11,ha16(L__ZNK17atomic_shared_ptrI1AE6_scan_Ev$lazy_ptr-"L00000000037$spb")
	mtlr r0
	lwzu r12,lo16(L__ZNK17atomic_shared_ptrI1AE6_scan_Ev$lazy_ptr-"L00000000037$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__ZNK17atomic_shared_ptrI1AE6_scan_Ev$lazy_ptr:
	.indirect_symbol __ZNK17atomic_shared_ptrI1AE6_scan_Ev
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__Z16atomicDecAndTestIiEbPT_$stub:
	.indirect_symbol __Z16atomicDecAndTestIiEbPT_
	mflr r0
	bcl 20,31,"L00000000038$spb"
"L00000000038$spb":
	mflr r11
	addis r11,r11,ha16(L__Z16atomicDecAndTestIiEbPT_$lazy_ptr-"L00000000038$spb")
	mtlr r0
	lwzu r12,lo16(L__Z16atomicDecAndTestIiEbPT_$lazy_ptr-"L00000000038$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__Z16atomicDecAndTestIiEbPT_$lazy_ptr:
	.indirect_symbol __Z16atomicDecAndTestIiEbPT_
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__ZN17atomic_shared_ptrI1AE9_RefLocalC1Ev$stub:
	.indirect_symbol __ZN17atomic_shared_ptrI1AE9_RefLocalC1Ev
	mflr r0
	bcl 20,31,"L00000000039$spb"
"L00000000039$spb":
	mflr r11
	addis r11,r11,ha16(L__ZN17atomic_shared_ptrI1AE9_RefLocalC1Ev$lazy_ptr-"L00000000039$spb")
	mtlr r0
	lwzu r12,lo16(L__ZN17atomic_shared_ptrI1AE9_RefLocalC1Ev$lazy_ptr-"L00000000039$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__ZN17atomic_shared_ptrI1AE9_RefLocalC1Ev$lazy_ptr:
	.indirect_symbol __ZN17atomic_shared_ptrI1AE9_RefLocalC1Ev
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L_pthread_mutexattr_init$stub:
	.indirect_symbol _pthread_mutexattr_init
	mflr r0
	bcl 20,31,"L00000000040$spb"
"L00000000040$spb":
	mflr r11
	addis r11,r11,ha16(L_pthread_mutexattr_init$lazy_ptr-"L00000000040$spb")
	mtlr r0
	lwzu r12,lo16(L_pthread_mutexattr_init$lazy_ptr-"L00000000040$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L_pthread_mutexattr_init$lazy_ptr:
	.indirect_symbol _pthread_mutexattr_init
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__Z9atomicAddIiEvPT_S0_$stub:
	.indirect_symbol __Z9atomicAddIiEvPT_S0_
	mflr r0
	bcl 20,31,"L00000000041$spb"
"L00000000041$spb":
	mflr r11
	addis r11,r11,ha16(L__Z9atomicAddIiEvPT_S0_$lazy_ptr-"L00000000041$spb")
	mtlr r0
	lwzu r12,lo16(L__Z9atomicAddIiEvPT_S0_$lazy_ptr-"L00000000041$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__Z9atomicAddIiEvPT_S0_$lazy_ptr:
	.indirect_symbol __Z9atomicAddIiEvPT_S0_
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__ZN17atomic_shared_ptrI1AE3RefC1I1BEEPT_$stub:
	.indirect_symbol __ZN17atomic_shared_ptrI1AE3RefC1I1BEEPT_
	mflr r0
	bcl 20,31,"L00000000042$spb"
"L00000000042$spb":
	mflr r11
	addis r11,r11,ha16(L__ZN17atomic_shared_ptrI1AE3RefC1I1BEEPT_$lazy_ptr-"L00000000042$spb")
	mtlr r0
	lwzu r12,lo16(L__ZN17atomic_shared_ptrI1AE3RefC1I1BEEPT_$lazy_ptr-"L00000000042$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__ZN17atomic_shared_ptrI1AE3RefC1I1BEEPT_$lazy_ptr:
	.indirect_symbol __ZN17atomic_shared_ptrI1AE3RefC1I1BEEPT_
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__ZN1AC2Ei$stub:
	.indirect_symbol __ZN1AC2Ei
	mflr r0
	bcl 20,31,"L00000000043$spb"
"L00000000043$spb":
	mflr r11
	addis r11,r11,ha16(L__ZN1AC2Ei$lazy_ptr-"L00000000043$spb")
	mtlr r0
	lwzu r12,lo16(L__ZN1AC2Ei$lazy_ptr-"L00000000043$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__ZN1AC2Ei$lazy_ptr:
	.indirect_symbol __ZN1AC2Ei
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__Z10atomicSwapIP1AET_S2_PS2_$stub:
	.indirect_symbol __Z10atomicSwapIP1AET_S2_PS2_
	mflr r0
	bcl 20,31,"L00000000044$spb"
"L00000000044$spb":
	mflr r11
	addis r11,r11,ha16(L__Z10atomicSwapIP1AET_S2_PS2_$lazy_ptr-"L00000000044$spb")
	mtlr r0
	lwzu r12,lo16(L__Z10atomicSwapIP1AET_S2_PS2_$lazy_ptr-"L00000000044$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__Z10atomicSwapIP1AET_S2_PS2_$lazy_ptr:
	.indirect_symbol __Z10atomicSwapIP1AET_S2_PS2_
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L_pthread_mutexattr_destroy$stub:
	.indirect_symbol _pthread_mutexattr_destroy
	mflr r0
	bcl 20,31,"L00000000045$spb"
"L00000000045$spb":
	mflr r11
	addis r11,r11,ha16(L_pthread_mutexattr_destroy$lazy_ptr-"L00000000045$spb")
	mtlr r0
	lwzu r12,lo16(L_pthread_mutexattr_destroy$lazy_ptr-"L00000000045$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L_pthread_mutexattr_destroy$lazy_ptr:
	.indirect_symbol _pthread_mutexattr_destroy
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L_pthread_equal$stub:
	.indirect_symbol _pthread_equal
	mflr r0
	bcl 20,31,"L00000000046$spb"
"L00000000046$spb":
	mflr r11
	addis r11,r11,ha16(L_pthread_equal$lazy_ptr-"L00000000046$spb")
	mtlr r0
	lwzu r12,lo16(L_pthread_equal$lazy_ptr-"L00000000046$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L_pthread_equal$lazy_ptr:
	.indirect_symbol _pthread_equal
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L_pthread_cond_destroy$stub:
	.indirect_symbol _pthread_cond_destroy
	mflr r0
	bcl 20,31,"L00000000047$spb"
"L00000000047$spb":
	mflr r11
	addis r11,r11,ha16(L_pthread_cond_destroy$lazy_ptr-"L00000000047$spb")
	mtlr r0
	lwzu r12,lo16(L_pthread_cond_destroy$lazy_ptr-"L00000000047$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L_pthread_cond_destroy$lazy_ptr:
	.indirect_symbol _pthread_cond_destroy
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L_pthread_cond_broadcast$stub:
	.indirect_symbol _pthread_cond_broadcast
	mflr r0
	bcl 20,31,"L00000000048$spb"
"L00000000048$spb":
	mflr r11
	addis r11,r11,ha16(L_pthread_cond_broadcast$lazy_ptr-"L00000000048$spb")
	mtlr r0
	lwzu r12,lo16(L_pthread_cond_broadcast$lazy_ptr-"L00000000048$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L_pthread_cond_broadcast$lazy_ptr:
	.indirect_symbol _pthread_cond_broadcast
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__Z10_my_assertbPKci$stub:
	.indirect_symbol __Z10_my_assertbPKci
	mflr r0
	bcl 20,31,"L00000000049$spb"
"L00000000049$spb":
	mflr r11
	addis r11,r11,ha16(L__Z10_my_assertbPKci$lazy_ptr-"L00000000049$spb")
	mtlr r0
	lwzu r12,lo16(L__Z10_my_assertbPKci$lazy_ptr-"L00000000049$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__Z10_my_assertbPKci$lazy_ptr:
	.indirect_symbol __Z10_my_assertbPKci
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L_srand$stub:
	.indirect_symbol _srand
	mflr r0
	bcl 20,31,"L00000000050$spb"
"L00000000050$spb":
	mflr r11
	addis r11,r11,ha16(L_srand$lazy_ptr-"L00000000050$spb")
	mtlr r0
	lwzu r12,lo16(L_srand$lazy_ptr-"L00000000050$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L_srand$lazy_ptr:
	.indirect_symbol _srand
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L_abort$stub:
	.indirect_symbol _abort
	mflr r0
	bcl 20,31,"L00000000051$spb"
"L00000000051$spb":
	mflr r11
	addis r11,r11,ha16(L_abort$lazy_ptr-"L00000000051$spb")
	mtlr r0
	lwzu r12,lo16(L_abort$lazy_ptr-"L00000000051$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L_abort$lazy_ptr:
	.indirect_symbol _abort
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__ZdlPv$stub:
	.indirect_symbol __ZdlPv
	mflr r0
	bcl 20,31,"L00000000052$spb"
"L00000000052$spb":
	mflr r11
	addis r11,r11,ha16(L__ZdlPv$lazy_ptr-"L00000000052$spb")
	mtlr r0
	lwzu r12,lo16(L__ZdlPv$lazy_ptr-"L00000000052$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__ZdlPv$lazy_ptr:
	.indirect_symbol __ZdlPv
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__ZN17atomic_shared_ptrI1AE5resetIS0_EEvPT_$stub:
	.indirect_symbol __ZN17atomic_shared_ptrI1AE5resetIS0_EEvPT_
	mflr r0
	bcl 20,31,"L00000000053$spb"
"L00000000053$spb":
	mflr r11
	addis r11,r11,ha16(L__ZN17atomic_shared_ptrI1AE5resetIS0_EEvPT_$lazy_ptr-"L00000000053$spb")
	mtlr r0
	lwzu r12,lo16(L__ZN17atomic_shared_ptrI1AE5resetIS0_EEvPT_$lazy_ptr-"L00000000053$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__ZN17atomic_shared_ptrI1AE5resetIS0_EEvPT_$lazy_ptr:
	.indirect_symbol __ZN17atomic_shared_ptrI1AE5resetIS0_EEvPT_
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L_pthread_mutex_init$stub:
	.indirect_symbol _pthread_mutex_init
	mflr r0
	bcl 20,31,"L00000000054$spb"
"L00000000054$spb":
	mflr r11
	addis r11,r11,ha16(L_pthread_mutex_init$lazy_ptr-"L00000000054$spb")
	mtlr r0
	lwzu r12,lo16(L_pthread_mutex_init$lazy_ptr-"L00000000054$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L_pthread_mutex_init$lazy_ptr:
	.indirect_symbol _pthread_mutex_init
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__ZN1BC1Ei$stub:
	.indirect_symbol __ZN1BC1Ei
	mflr r0
	bcl 20,31,"L00000000055$spb"
"L00000000055$spb":
	mflr r11
	addis r11,r11,ha16(L__ZN1BC1Ei$lazy_ptr-"L00000000055$spb")
	mtlr r0
	lwzu r12,lo16(L__ZN1BC1Ei$lazy_ptr-"L00000000055$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__ZN1BC1Ei$lazy_ptr:
	.indirect_symbol __ZN1BC1Ei
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L_gettimeofday$stub:
	.indirect_symbol _gettimeofday
	mflr r0
	bcl 20,31,"L00000000056$spb"
"L00000000056$spb":
	mflr r11
	addis r11,r11,ha16(L_gettimeofday$lazy_ptr-"L00000000056$spb")
	mtlr r0
	lwzu r12,lo16(L_gettimeofday$lazy_ptr-"L00000000056$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L_gettimeofday$lazy_ptr:
	.indirect_symbol _gettimeofday
	.long	dyld_stub_binding_helper
	.non_lazy_symbol_pointer
L__ZTV1A$non_lazy_ptr:
	.indirect_symbol __ZTV1A
	.long	0
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__ZN17atomic_shared_ptrI1AE5resetEv$stub:
	.indirect_symbol __ZN17atomic_shared_ptrI1AE5resetEv
	mflr r0
	bcl 20,31,"L00000000057$spb"
"L00000000057$spb":
	mflr r11
	addis r11,r11,ha16(L__ZN17atomic_shared_ptrI1AE5resetEv$lazy_ptr-"L00000000057$spb")
	mtlr r0
	lwzu r12,lo16(L__ZN17atomic_shared_ptrI1AE5resetEv$lazy_ptr-"L00000000057$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__ZN17atomic_shared_ptrI1AE5resetEv$lazy_ptr:
	.indirect_symbol __ZN17atomic_shared_ptrI1AE5resetEv
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__ZN6atomicIiED1Ev$stub:
	.indirect_symbol __ZN6atomicIiED1Ev
	mflr r0
	bcl 20,31,"L00000000058$spb"
"L00000000058$spb":
	mflr r11
	addis r11,r11,ha16(L__ZN6atomicIiED1Ev$lazy_ptr-"L00000000058$spb")
	mtlr r0
	lwzu r12,lo16(L__ZN6atomicIiED1Ev$lazy_ptr-"L00000000058$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__ZN6atomicIiED1Ev$lazy_ptr:
	.indirect_symbol __ZN6atomicIiED1Ev
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__ZN17atomic_shared_ptrI1AED1Ev$stub:
	.indirect_symbol __ZN17atomic_shared_ptrI1AED1Ev
	mflr r0
	bcl 20,31,"L00000000059$spb"
"L00000000059$spb":
	mflr r11
	addis r11,r11,ha16(L__ZN17atomic_shared_ptrI1AED1Ev$lazy_ptr-"L00000000059$spb")
	mtlr r0
	lwzu r12,lo16(L__ZN17atomic_shared_ptrI1AED1Ev$lazy_ptr-"L00000000059$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__ZN17atomic_shared_ptrI1AED1Ev$lazy_ptr:
	.indirect_symbol __ZN17atomic_shared_ptrI1AED1Ev
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__ZN6atomicIiEpLEi$stub:
	.indirect_symbol __ZN6atomicIiEpLEi
	mflr r0
	bcl 20,31,"L00000000060$spb"
"L00000000060$spb":
	mflr r11
	addis r11,r11,ha16(L__ZN6atomicIiEpLEi$lazy_ptr-"L00000000060$spb")
	mtlr r0
	lwzu r12,lo16(L__ZN6atomicIiEpLEi$lazy_ptr-"L00000000060$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__ZN6atomicIiEpLEi$lazy_ptr:
	.indirect_symbol __ZN6atomicIiEpLEi
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__Znwm$stub:
	.indirect_symbol __Znwm
	mflr r0
	bcl 20,31,"L00000000061$spb"
"L00000000061$spb":
	mflr r11
	addis r11,r11,ha16(L__Znwm$lazy_ptr-"L00000000061$spb")
	mtlr r0
	lwzu r12,lo16(L__Znwm$lazy_ptr-"L00000000061$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__Znwm$lazy_ptr:
	.indirect_symbol __Znwm
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L_pthread_mutex_unlock$stub:
	.indirect_symbol _pthread_mutex_unlock
	mflr r0
	bcl 20,31,"L00000000062$spb"
"L00000000062$spb":
	mflr r11
	addis r11,r11,ha16(L_pthread_mutex_unlock$lazy_ptr-"L00000000062$spb")
	mtlr r0
	lwzu r12,lo16(L_pthread_mutex_unlock$lazy_ptr-"L00000000062$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L_pthread_mutex_unlock$lazy_ptr:
	.indirect_symbol _pthread_mutex_unlock
	.long	dyld_stub_binding_helper
	.non_lazy_symbol_pointer
L___sF$non_lazy_ptr:
	.indirect_symbol ___sF
	.long	0
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__ZN1AC1Ei$stub:
	.indirect_symbol __ZN1AC1Ei
	mflr r0
	bcl 20,31,"L00000000063$spb"
"L00000000063$spb":
	mflr r11
	addis r11,r11,ha16(L__ZN1AC1Ei$lazy_ptr-"L00000000063$spb")
	mtlr r0
	lwzu r12,lo16(L__ZN1AC1Ei$lazy_ptr-"L00000000063$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__ZN1AC1Ei$lazy_ptr:
	.indirect_symbol __ZN1AC1Ei
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L_rand$stub:
	.indirect_symbol _rand
	mflr r0
	bcl 20,31,"L00000000064$spb"
"L00000000064$spb":
	mflr r11
	addis r11,r11,ha16(L_rand$lazy_ptr-"L00000000064$spb")
	mtlr r0
	lwzu r12,lo16(L_rand$lazy_ptr-"L00000000064$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L_rand$lazy_ptr:
	.indirect_symbol _rand
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L_pthread_cond_init$stub:
	.indirect_symbol _pthread_cond_init
	mflr r0
	bcl 20,31,"L00000000065$spb"
"L00000000065$spb":
	mflr r11
	addis r11,r11,ha16(L_pthread_cond_init$lazy_ptr-"L00000000065$spb")
	mtlr r0
	lwzu r12,lo16(L_pthread_cond_init$lazy_ptr-"L00000000065$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L_pthread_cond_init$lazy_ptr:
	.indirect_symbol _pthread_cond_init
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L__ZN17atomic_scoped_ptrI1AED1Ev$stub:
	.indirect_symbol __ZN17atomic_scoped_ptrI1AED1Ev
	mflr r0
	bcl 20,31,"L00000000066$spb"
"L00000000066$spb":
	mflr r11
	addis r11,r11,ha16(L__ZN17atomic_scoped_ptrI1AED1Ev$lazy_ptr-"L00000000066$spb")
	mtlr r0
	lwzu r12,lo16(L__ZN17atomic_scoped_ptrI1AED1Ev$lazy_ptr-"L00000000066$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L__ZN17atomic_scoped_ptrI1AED1Ev$lazy_ptr:
	.indirect_symbol __ZN17atomic_scoped_ptrI1AED1Ev
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L_pthread_mutex_destroy$stub:
	.indirect_symbol _pthread_mutex_destroy
	mflr r0
	bcl 20,31,"L00000000067$spb"
"L00000000067$spb":
	mflr r11
	addis r11,r11,ha16(L_pthread_mutex_destroy$lazy_ptr-"L00000000067$spb")
	mtlr r0
	lwzu r12,lo16(L_pthread_mutex_destroy$lazy_ptr-"L00000000067$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L_pthread_mutex_destroy$lazy_ptr:
	.indirect_symbol _pthread_mutex_destroy
	.long	dyld_stub_binding_helper
	.section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align 5
L_atoi$stub:
	.indirect_symbol _atoi
	mflr r0
	bcl 20,31,"L00000000068$spb"
"L00000000068$spb":
	mflr r11
	addis r11,r11,ha16(L_atoi$lazy_ptr-"L00000000068$spb")
	mtlr r0
	lwzu r12,lo16(L_atoi$lazy_ptr-"L00000000068$spb")(r11)
	mtctr r12
	bctr
	.lazy_symbol_pointer
L_atoi$lazy_ptr:
	.indirect_symbol _atoi
	.long	dyld_stub_binding_helper
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
