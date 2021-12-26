/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitagawa@phys.s.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef tempmanagerH
#define tempmanagerH
//---------------------------------------------------------------------------
#include "thermometer.h"
#include "dcsource.h"
#include "flowcontroller.h"
#include "secondarydriver.h"
#include "xnodeconnector.h"
#include "tempcontrol.h"

class QTableWidget;
class XScalarEntry;
class Ui_FrmTempManager;
typedef QForm<QMainWindow, Ui_FrmTempManager> FrmTempManager;

class XTempManager : public XSecondaryDriver {
public:
    XTempManager(const char *name, bool runtime, Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XTempManager();
	//! show all forms belonging to driver
    virtual void showForms() override;
  
    static constexpr unsigned int maxNumOfAUXDevices = 6;

    class XZone : public XNode {
	public:
        XZone(const char *name, bool runtime,
            Transaction &tr_list, const shared_ptr<XThermometerList> &list);
        const shared_ptr<XDoubleNode> &upperTemp() const {return m_upperTemp;}
        const shared_ptr<XDoubleNode> &maxRampRate() const {return m_maxRampRate;}
        const shared_ptr<XComboNode> &channel() const {return m_channel;}
        const shared_ptr<XComboNode> &excitation() const {return m_excitation;}
        using XItemThermometer = XItemNode<XThermometerList, XThermometer>;
        const shared_ptr<XItemThermometer> &thermometer() const {return m_thermometer;}
        const shared_ptr<XComboNode> &loop() const {return m_loop;}
        const shared_ptr<XComboNode> &powerRange() const {return m_powerRange;}
        const shared_ptr<XDoubleNode> &prop() const {return m_prop;}
        const shared_ptr<XDoubleNode> &interv() const {return m_interv;}
        const shared_ptr<XDoubleNode> &deriv() const {return m_deriv;}
        const shared_ptr<XDoubleNode> &auxDeviceValues(unsigned int i) const {return m_auxDeviceValues.at(i);}

	private:
        const shared_ptr<XDoubleNode> m_upperTemp, m_maxRampRate;
        const shared_ptr<XComboNode> m_channel;
        const shared_ptr<XComboNode> m_excitation;
        const shared_ptr<XItemThermometer> m_thermometer;
        const shared_ptr<XComboNode> m_loop, m_powerRange;
        const shared_ptr<XDoubleNode> m_prop, m_interv, m_deriv;
        std::deque<shared_ptr<XDoubleNode>> m_auxDeviceValues;

        const shared_ptr<XThermometerList> m_thermometers;
    };
  
    class XZoneList : public XCustomTypeListNode<XZone> {
    public:
        XZoneList(const char *name, bool runtime,
            const shared_ptr<XThermometerList> &list)
            :  XCustomTypeListNode(name, runtime),
            m_thermometers(list) {}

        virtual bool isThreadSafeDuringCreationByTypename() const override {return false;}
        virtual shared_ptr<XNode> createByTypename(
            const XString &, const XString &name) override {
            shared_ptr<XZone> node;
            iterate_commit_if([=,&node](Transaction &tr){
                Transaction tr_th( *m_thermometers);
                node = create<XZone>(tr, "", false, tr_th, m_thermometers); //nameless
                return tr_th.commit();
            });
            return node;
        }
        const shared_ptr<XThermometerList> &thermometers() const {return m_thermometers;}
    private:
        const shared_ptr<XThermometerList> m_thermometers;
    };
  
    const shared_ptr<XBoolNode> &isActivated() const {return m_isActivated;}

    const shared_ptr<XDoubleNode> &targetTemp() const {return m_targetTemp;}
    const shared_ptr<XDoubleNode> &rampRate() const {return m_rampRate;}

    const shared_ptr<XTouchableNode> &dupZone() const {return m_dupZone;}
    const shared_ptr<XTouchableNode> &delZone() const {return m_delZone;}

    const shared_ptr<XZoneList> &zones() const {return m_zones;}

    //! PID control of an external device.
    using XItemExtDevice = XItemNode<XDriverList, XDCSource, XFlowControllerDriver> ;
    const shared_ptr<XItemExtDevice> &extDevice() const {return m_extDevice;}
    const shared_ptr<XBoolNode> &extIsPositive() const {return m_extIsPositive;}

    const shared_ptr<XDoubleNode> &hysteresisOnZoneTr() const { return m_hysteresisOnZoneTr;}
    const shared_ptr<XBoolNode> &doesMixTemp() const { return m_doesMixTemp;}
    using XItemMainDevice = XItemNode<XDriverList, XTempControl>;
    const shared_ptr<XItemMainDevice> &mainDevice() const {return m_mainDevice;}
    using XItemAUXDevice = XItemNode<XDriverList, XTempControl, XDCSource, XFlowControllerDriver>;
    const shared_ptr<XItemAUXDevice> &auxDevice(unsigned int i) const {return m_auxDevices.at(i);}
    const shared_ptr<XComboNode> &auxDevCh(unsigned int i) const {return m_auxDevChs.at(i);}
    const shared_ptr<XComboNode> &auxDevMode(unsigned int i) const {return m_auxDevModes.at(i);}

    const shared_ptr<XStringNode> &statusStr() const {return m_statusStr;}

protected:
    //! This function is called when a connected driver emit a signal
    virtual void analyze(Transaction &tr, const Snapshot &shot_emitter,
        const Snapshot &shot_others, XDriver *emitter) override;
    //! This function is called after committing XPrimaryDriver::analyzeRaw() or XSecondaryDriver::analyze().
    //! This might be called even if the record is invalid (time() == false).
    virtual void visualize(const Snapshot &shot) override;
    //! Checks if the connected drivers have valid time stamps.
    //! \return true if dependency is resolved.
    //! This function must be reentrant unlike analyze().
    virtual bool checkDependency(const Snapshot &shot_this,
        const Snapshot &shot_emitter, const Snapshot &shot_others,
        XDriver *emitter) const override;

private:    
    const shared_ptr<XBoolNode> m_isActivated;
    const shared_ptr<XDoubleNode> m_targetTemp, m_rampRate;
    const shared_ptr<XTouchableNode> m_dupZone, m_delZone;
    const shared_ptr<XItemExtDevice> m_extDevice;
    const shared_ptr<XBoolNode> m_extIsPositive;

    const shared_ptr<XDoubleNode> m_hysteresisOnZoneTr;
    const shared_ptr<XBoolNode> m_doesMixTemp;

    const shared_ptr<XItemMainDevice> m_mainDevice;
    std::deque<shared_ptr<XItemAUXDevice> > m_auxDevices;
    std::deque<shared_ptr<XComboNode> > m_auxDevChs, m_auxDevModes;

    const shared_ptr<XZoneList> m_zones;

    const shared_ptr<XStringNode> m_statusStr;
    const shared_ptr<XStringNode> m_tempStatusStr, m_heaterStatusStr;

    shared_ptr<Listener> m_lsnOnActivateChanged,
        m_lsnOnAUXDeviceChanged, m_lsnOnExtDeviceChanged,
        m_lsnOnMainDeviceChanged, m_lsnOnDupTouched, m_lsnOnDelTouched,
        m_lsnOnTargetChanged,
        m_lsnMainDevOnListChanged,
        m_lsnAUXDevOnListChanged[maxNumOfAUXDevices],
        m_lsnExtDevOnListChanged;

    void onActivateChanged(const Snapshot &shot, XValueNodeBase *);
    void onMainDeviceChanged(const Snapshot &shot, XValueNodeBase *);
    void onExtDeviceChanged(const Snapshot &shot, XValueNodeBase *);
    void onAUXDeviceChanged(const Snapshot &shot, XValueNodeBase *);
    void onTargetChanged(const Snapshot &shot, XValueNodeBase *);
    void onDupTouched(const Snapshot &shot, XTouchableNode *);
    void onDeleteTouched(const Snapshot &shot, XTouchableNode *);
    void onChListChanged(const Snapshot &shot, XItemNodeBase::Payload::ListChangeEvent e);

    shared_ptr<XScalarEntry> m_entryTemp, m_entryPow, m_entryStability;
 
    const qshared_ptr<FrmTempManager> m_form;

    void refreshZoneUIs();
    struct tconZonUIs {
        std::deque<xqcon_ptr> conUIs;
    };
    std::deque<tconZonUIs> m_conZoneUIs;

    std::deque<xqcon_ptr> m_conUIs;

    shared_ptr<XZone> currentZone(const Snapshot &shot) const {
        if((currentZoneNo() >= 0) && (currentZoneNo() < shot.size(zones())))
            return dynamic_pointer_cast<XZone>(shot.list(zones())->at(currentZoneNo()));
        else
            return {};
    }

    int firstMatchingZone(const Snapshot &shot, double temp, double signed_ramprate,
        bool update_missinginfo = false);
    int currentZoneNo() const {return m_currZoneNo;}
    void sanityCheckOfZones(const Snapshot &shot);

    int m_currZoneNo;
    int m_currLoopNo;
    int m_currCh;
    int m_currExcitaion;
    shared_ptr<XThermometer> m_currThermometer;

    double pid(const Snapshot &shot, XTime time, double sv, double pv);

    double m_pidAccum;
    double m_pidLastTemp;
    XTime m_pidLastTime;

    double m_tempAvg;
    double m_tempErrAvg;
    XTime m_lasttime;

    double m_tempStarted;
    XTime m_timeStarted;
    double m_setpointTemp;
};

#endif
