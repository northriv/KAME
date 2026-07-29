/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
***************************************************************************/
#include "charinterface.h"
#include "userrelay.h"
#include "motor.h"
#include "measure.h"
#include "ui_relayform.h"
#include <QComboBox>
#include <QLabel>

REGISTER_TYPE(XDriverList, LCUS1, "LCTech LCUS-1 USB 1ch relay module");
REGISTER_TYPE(XDriverList, LCUS2, "LCTech LCUS-2 USB 2ch relay module");
REGISTER_TYPE(XDriverList, LCUS4, "LCTech LCUS-4 USB 4ch relay module");
REGISTER_TYPE(XDriverList, LCUS8, "LCTech LCUS-8 USB 8ch relay module");
REGISTER_TYPE(XDriverList, RelayViaSTM, "Relays using AUX bits of motor driver");

XLCUSRelay::XLCUSRelay(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas,
    unsigned int num_channels) :
    XCharDeviceDriver<XRelayDriver>(name, runtime, ref(tr_meas), meas, num_channels) {
    interface()->setSerialBaudRate(9600);
    interface()->setSerialStopBits(1);
    interface()->setSerialParity(XCharInterface::PARITY_NONE);
    iterate_commit([=](Transaction &tr){
        tr[ *interface()->device()] = "SERIAL";
    });
}
void
XLCUSRelay::changeOutput(unsigned int ch, bool on) {
    XScopedLock<XInterface> lock( *interface());
    unsigned char buf[4];
    buf[0] = 0xa0u; //frame header
    buf[1] = (unsigned char)(ch + 1); //channel, counted from 1
    buf[2] = on ? 1u : 0u;
    buf[3] = (unsigned char)(buf[0] + buf[1] + buf[2]); //checksum
    interface()->write(reinterpret_cast<const char*>(buf), sizeof(buf));
}

XRelayViaSTM::XRelayViaSTM(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XDummyDriver<XRelayDriver>(name, runtime, ref(tr_meas), meas),
    m_stm(create<XItemNode<XDriverList, XMotorDriver>>(
        "STM", false, ref(tr_meas), meas->drivers(), true)) {
    form()->m_lblMaster->setEnabled(true);
    form()->m_cmbMaster->setEnabled(true);
    m_conUIs = {
        xqcon_create<XQComboBoxConnector>(m_stm, form()->m_cmbMaster, ref(tr_meas)),
    };
}
void
XRelayViaSTM::changeOutput(unsigned int ch, bool on) {
    shared_ptr<XMotorDriver> stm__ = Snapshot( *this)[ *stm()];
    if( !stm__)
        throw XInterface::XInterfaceError(i18n("No valid STM setting."), __FILE__, __LINE__);
    //Writes the AUX bits node; the motor driver's own listener sends it to the hardware.
    stm__->iterate_commit([=](Transaction &tr){
        unsigned long bits = tr[ *stm__->auxBits()];
        if(on)
            bits |= 1uL << ch;
        else
            bits &= ~(1uL << ch);
        tr[ *stm__->auxBits()] = bits;
    });
}
