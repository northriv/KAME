#ifndef TWOAXIS_H
#define TWOAXIS_H
//---------------------------------------------------------------------------
#include "secondarydriver.h"
#include "motor.h"
#include "xnodeconnector.h"
//---------------------------------------------------------------------------
class XScalarEntry;
class Ui_FrmTwoaxis;
typedef QForm<QMainWindow, Ui_FrmTwoaxis> FrmTwoaxis;

//! Control TwoAxis rotator
class XTwoAxis : public XSecondaryDriver {
public:
    XTwoAxis(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XTwoAxis ();

    //! Shows all forms belonging to driver
    virtual void showForms();
protected:
    //! This function is called when a connected driver emit a signal
    virtual void analyze(Transaction &tr, const Snapshot &shot_emitter, const Snapshot &shot_others,
        XDriver *emitter) throw (XRecordError&);
    //! This function is called after committing XPrimaryDriver::analyzeRaw() or XSecondaryDriver::analyze().
    //! This might be called even if the record is invalid (time() == false).
    virtual void visualize(const Snapshot &shot);
    //! Checks if the connected drivers have valid time stamps.
    //! \return true if dependency is resolved.
    //! This function must be reentrant unlike analyze().
    virtual bool checkDependency(const Snapshot &shot_this,
        const Snapshot &shot_emitter, const Snapshot &shot_others,
        XDriver *emitter) const;

public:
    const shared_ptr<XItemNode<XDriverList, XMotorDriver> > &rot1() const {return m_rot1;}
    const shared_ptr<XItemNode<XDriverList, XMotorDriver> > &rot2() const {return m_rot2;}

    struct Payload : public XSecondaryDriver::Payload {
    public:
        int currentStep;
        bool isTheta;
        bool isWaitStable;
        bool isRecorded;
        double startAngle; //! for theta or phi
        std::array<double, 2> currentROT;
        std::array<double, 2> targetROT;
        std::array<double, 2> deltaROT;
        std::array<double, 2> startROT;
        XTime timeROTChanged;
        XTime timeRecorded;
        XTime timeStarted;
    };

    const shared_ptr<XScalarEntry> &theta() const {return m_theta;}
    const shared_ptr<XScalarEntry> &phi() const {return m_phi;}
    const shared_ptr<XScalarEntry> &record_step() const {return m_record_step;}

    const shared_ptr<XDoubleNode> &target_theta() const {return m_target_theta;}
    const shared_ptr<XDoubleNode> &target_phi() const {return m_target_phi;}
    const shared_ptr<XDoubleNode> &offset_theta() const {return m_offset_theta;}
    const shared_ptr<XDoubleNode> &offset_phi() const {return m_offset_phi;}
    const shared_ptr<XDoubleNode> &max_theta() const {return m_max_theta;}
    const shared_ptr<XDoubleNode> &min_theta() const {return m_min_theta;}
    const shared_ptr<XDoubleNode> &max_phi() const {return m_max_phi;}
    const shared_ptr<XDoubleNode> &min_phi() const {return m_min_phi;}
    const shared_ptr<XIntNode> &step() const {return m_step;}
    const shared_ptr<XDoubleNode> &wait() const {return m_wait;}
    const shared_ptr<XDoubleNode> &rot1_per_theta() const {return m_rot1_per_theta;}
    const shared_ptr<XDoubleNode> &rot1_per_phi() const {return m_rot1_per_phi;}
    const shared_ptr<XDoubleNode> &rot2_per_theta() const {return m_rot2_per_theta;}
    const shared_ptr<XDoubleNode> &rot2_per_phi() const {return m_rot2_per_phi;}
    const shared_ptr<XTouchableNode> &abort() const {return m_abort;}
    const shared_ptr<XBoolNode> &ready() const {return  m_ready;}
    const shared_ptr<XBoolNode> &slipping() const {return  m_slipping;}
    const shared_ptr<XBoolNode> &running() const {return  m_running;}
    const shared_ptr<XDoubleNode> &timeout() const {return m_timeout;}


private:
    const shared_ptr<XItemNode<XDriverList, XMotorDriver> > m_rot1, m_rot2;
    const shared_ptr<XScalarEntry> m_theta, m_phi;
    const shared_ptr<XScalarEntry> m_record_step;

    const shared_ptr<XDoubleNode> m_target_theta, m_target_phi;
    const shared_ptr<XDoubleNode> m_offset_theta, m_offset_phi;
    const shared_ptr<XDoubleNode> m_max_theta, m_min_theta, m_max_phi, m_min_phi;
    const shared_ptr<XIntNode> m_step;
    const shared_ptr<XDoubleNode> m_wait; //! sec.
    const shared_ptr<XDoubleNode> m_rot1_per_theta, m_rot1_per_phi; //! moter deg. per theta/phi deg. for rot1
    const shared_ptr<XDoubleNode> m_rot2_per_theta, m_rot2_per_phi; //! moter deg. per theta/phi deg. for rot2
    const shared_ptr<XTouchableNode> m_abort;
    const shared_ptr<XBoolNode> m_ready;
    const shared_ptr<XBoolNode> m_slipping;
    const shared_ptr<XBoolNode> m_running;
    const shared_ptr<XDoubleNode> m_timeout;

    std::deque<xqcon_ptr> m_conUIs;

    shared_ptr<Listener> m_lsnOnTargetThetaChanged, m_lsnOnTargetPhiChanged, m_lsnOnAbortTouched;

    const qshared_ptr<FrmTwoaxis> m_form;
    const shared_ptr<XStatusPrinter> m_statusPrinter;

    void onTargetThetaChanged(const Snapshot &shot, XValueNodeBase *);
    void onTargetPhiChanged(const Snapshot &shot, XValueNodeBase *);
    void onAbortTouched(const Snapshot &shot, XTouchableNode *);

    void initSweep(const Snapshot &shot);
    void endSweep(Transaction &tr);
    void enableUIs(Transaction &tr, bool flag);

};

#endif // TWOAXIS_H
