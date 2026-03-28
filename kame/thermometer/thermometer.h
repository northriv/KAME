/***************************************************************************
        Copyright (C) 2002-2025 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU Library General
		Public License and a list of authors along with this program;
		see the files COPYING and AUTHORS.
***************************************************************************/
//---------------------------------------------------------------------------

#ifndef thermometerH
#define thermometerH

#include "xnode.h"
#include "xlistnode.h"
#include "cspline.h"
#include <map>

//! Base class for all calibration curves (raw sensor value → physical output).
class DECLSPEC_KAME XCalibrationCurve : public XNode {
public:
    XCalibrationCurve(const char *name, bool runtime);
    virtual ~XCalibrationCurve() {}
    virtual double getOutput(double raw) const = 0;
    virtual double getRaw(double output) const = 0;
    const shared_ptr<XDoubleNode> &outMin() const { return m_outMin; }
    const shared_ptr<XDoubleNode> &outMax() const { return m_outMax; }
    //! UI label helpers — override in subclasses for context-specific strings.
    virtual XString outputLabel() const { return "Output"; }
    virtual XString outputUnit()  const { return ""; }
    virtual XString rawLabel()    const { return "Raw"; }
    virtual XString rawUnit()     const { return ""; }
private:
    const shared_ptr<XDoubleNode> m_outMin, m_outMax;
};

//! Sub-abstract: thermometers (output = temperature in K).
class DECLSPEC_KAME XThermometer : public XCalibrationCurve {
public:
    XThermometer(const char *name, bool runtime) : XCalibrationCurve(name, runtime) {}
    virtual ~XThermometer() {}
    //! Backward-compat wrappers.
    double getTemp(double res) const { return getOutput(res); }
    double getRawValue(double temp) const { return getRaw(temp); }
    const shared_ptr<XDoubleNode> &tempMin() const { return outMin(); }
    const shared_ptr<XDoubleNode> &tempMax() const { return outMax(); }
    XString outputLabel() const override { return "Temp."; }
    XString outputUnit()  const override { return "K"; }
};

//! Sub-abstract: resistance thermometers.
class DECLSPEC_KAME XResistanceThermometer : public XThermometer {
public:
    XResistanceThermometer(const char *name, bool runtime) : XThermometer(name, runtime) {}
    virtual ~XResistanceThermometer() {}
    XString rawLabel() const override { return "Resistance"; }
    XString rawUnit()  const override { return "\xce\xa9"; } // Ω (U+03A9)
};

//! List of all calibration curves.
class DECLSPEC_KAME XCalibrationCurveList : public XCustomTypeListNode<XCalibrationCurve> {
public:
    XCalibrationCurveList(const char *name, bool runtime);
    virtual ~XCalibrationCurveList() {}
    DEFINE_TYPE_HOLDER()
    shared_ptr<XCalibrationCurve> createCalibration(const XString &type, const XString &name) {
        return static_pointer_cast<XCalibrationCurve>(createByTypename(type, name));
    }
protected:
    virtual shared_ptr<XNode> createByTypename(const XString &type, const XString &name) {
        shared_ptr<XNode> ptr = (creator(type))(name.c_str(), false);
        if(ptr) insert(ptr);
        return ptr;
    }
};
//! Backward-compat alias.
using XThermometerList = XCalibrationCurveList;

//! Non-template interface for CSpline-editable calibrations (used by caltable).
class DECLSPEC_KAME XCSplineCalibrationIF {
public:
    typedef XListNode<XDoubleNode> XDoubleListNode;
    virtual const shared_ptr<XDoubleListNode> &rawList() const = 0;
    virtual const shared_ptr<XDoubleListNode> &outputList() const = 0;
    virtual void invalidateCache() = 0;
    virtual ~XCSplineCalibrationIF() {}
};

//! CSpline calibration template. Base must inherit XCalibrationCurve.
//! Works in log-log space (suitable for positive-valued monotonic calibrations).
template<class Base>
class XCSplineCalibrationX : public Base, public XCSplineCalibrationIF {
public:
    XCSplineCalibrationX(const char *name, bool runtime)
        : XCSplineCalibrationX(name, runtime, "RawList", "OutputList") {}
    double getOutput(double raw) const override;
    double getRaw(double output) const override;
    const shared_ptr<XDoubleListNode> &rawList() const override { return m_rawList; }
    const shared_ptr<XDoubleListNode> &outputList() const override { return m_outputList; }
    void invalidateCache() override;
protected:
    XCSplineCalibrationX(const char *name, bool runtime,
        const char *rawListName, const char *outputListName);
private:
    const shared_ptr<XDoubleListNode> m_rawList, m_outputList;
    mutable atomic_shared_ptr<CSplineInterp> m_approx, m_approx_inv;
};

template<class Base>
XCSplineCalibrationX<Base>::XCSplineCalibrationX(const char *name, bool runtime,
    const char *rawListName, const char *outputListName)
    : Base(name, runtime),
      m_rawList(this->template create<XDoubleListNode>(rawListName, false)),
      m_outputList(this->template create<XDoubleListNode>(outputListName, false)) {}

template<class Base>
void XCSplineCalibrationX<Base>::invalidateCache() {
    m_approx = local_shared_ptr<CSplineInterp>();
    m_approx_inv = local_shared_ptr<CSplineInterp>();
}

template<class Base>
double XCSplineCalibrationX<Base>::getOutput(double raw) const {
    local_shared_ptr<CSplineInterp> approx(m_approx);
    Snapshot shot( *this);
    if( !approx) {
        std::map<double, double> pts;
        if( !shot.size(m_rawList)) return 0;
        const auto &raw_list( *shot.list(m_rawList));
        if( !shot.size(m_outputList)) return 0;
        const auto &out_list( *shot.list(m_outputList));
        for(unsigned int i = 0; i < std::min(raw_list.size(), out_list.size()); i++) {
            double r = shot[ *static_pointer_cast<XDoubleNode>(raw_list.at(i))];
            double o = shot[ *static_pointer_cast<XDoubleNode>(out_list.at(i))];
            pts.insert({log(r), log(o)});
        }
        if(pts.size() < 4)
            throw XKameError(i18n("XCSplineCalibration, Too small number of points"),
                __FILE__, __LINE__);
        approx.reset(new CSplineInterp(pts));
        m_approx = approx;
    }
    return exp(approx->approx(log(raw)));
}

template<class Base>
double XCSplineCalibrationX<Base>::getRaw(double output) const {
    local_shared_ptr<CSplineInterp> approx(m_approx_inv);
    Snapshot shot( *this);
    if( !approx) {
        std::map<double, double> pts;
        if( !shot.size(m_rawList)) return 0;
        const auto &raw_list( *shot.list(m_rawList));
        if( !shot.size(m_outputList)) return 0;
        const auto &out_list( *shot.list(m_outputList));
        for(unsigned int i = 0; i < std::min(raw_list.size(), out_list.size()); i++) {
            double r = shot[ *static_pointer_cast<XDoubleNode>(raw_list.at(i))];
            double o = shot[ *static_pointer_cast<XDoubleNode>(out_list.at(i))];
            pts.insert({log(o), log(r)});
        }
        if(pts.size() < 4)
            throw XKameError(i18n("XCSplineCalibration, Too small number of points"),
                __FILE__, __LINE__);
        approx.reset(new CSplineInterp(pts));
        m_approx_inv = approx;
    }
    return exp(approx->approx(log(output)));
}

//! LakeShore Chebyshev polynomial thermometer.
class XLakeShore : public XResistanceThermometer {
public:
    XLakeShore(const char *name, bool runtime);
    virtual ~XLakeShore() {}
    double getOutput(double res) const override;
    double getRaw(double temp) const override;
    const shared_ptr<XDoubleNode> &resMin() const {return m_resMin;}
    const shared_ptr<XDoubleNode> &resMax() const {return m_resMax;}
    typedef XListNode<XDoubleNode> XDoubleListNode;
    const shared_ptr<XDoubleListNode> &zu() const {return m_zu;}
    const shared_ptr<XDoubleListNode> &zl() const {return m_zl;}
    typedef XListNode<XDoubleListNode> XDouble2DNode;
    const shared_ptr<XDouble2DNode> &ai() const {return m_ai;}
private:
    const shared_ptr<XDoubleNode> m_resMin, m_resMax;
    const shared_ptr<XDoubleListNode> m_zu, m_zl;
    const shared_ptr<XDouble2DNode> m_ai;
};

class DECLSPEC_KAME XScientificInstruments : public XResistanceThermometer {
public:
    XScientificInstruments(const char *name, bool runtime);
    virtual ~XScientificInstruments() {}
    double getOutput(double res) const override;
    double getRaw(double temp) const override;
    const shared_ptr<XDoubleNode> &resMin() const {return m_resMin;}
    const shared_ptr<XDoubleNode> &resMax() const {return m_resMax;}
    typedef XListNode<XDoubleNode> XDoubleListNode;
    const shared_ptr<XDoubleListNode> &abcde() const {return m_abcde;}
    const shared_ptr<XDoubleListNode> &abc() const {return m_abc;}
    const shared_ptr<XDoubleNode> &rCrossover() const {return m_rCrossover;}
private:
    const shared_ptr<XDoubleNode> m_resMin, m_resMax;
    const shared_ptr<XDoubleListNode> m_abcde, m_abc;
    const shared_ptr<XDoubleNode> m_rCrossover;
};

//! Cubic spline resistance thermometer.
//! .kam compat: child nodes are named "ResList"/"TempList".
class DECLSPEC_KAME XApproxThermometer : public XCSplineCalibrationX<XResistanceThermometer> {
public:
    XApproxThermometer(const char *name, bool runtime)
        : XCSplineCalibrationX<XResistanceThermometer>(name, runtime, "ResList", "TempList") {}
    XString getTypename() const override { return "ApproxThermometer"; }
    //! Backward-compat aliases.
    const shared_ptr<XDoubleListNode> &resList() const { return rawList(); }
    const shared_ptr<XDoubleListNode> &tempList() const { return outputList(); }
};

//! Generic cubic spline calibration curve with user-configurable labels.
class DECLSPEC_KAME XGenericCalibration : public XCSplineCalibrationX<XCalibrationCurve> {
public:
    XGenericCalibration(const char *name, bool runtime);
    XString getTypename() const override { return "GenericCalibration"; }
    XString outputLabel() const override { return ***m_outputLabel; }
    XString outputUnit()  const override { return ***m_outputUnit; }
    XString rawLabel()    const override { return ***m_rawLabel; }
    XString rawUnit()     const override { return ***m_rawUnit; }
    const shared_ptr<XStringNode> &outputLabelNode() const { return m_outputLabel; }
    const shared_ptr<XStringNode> &outputUnitNode()  const { return m_outputUnit; }
    const shared_ptr<XStringNode> &rawLabelNode()    const { return m_rawLabel; }
    const shared_ptr<XStringNode> &rawUnitNode()     const { return m_rawUnit; }
private:
    const shared_ptr<XStringNode> m_outputLabel, m_outputUnit;
    const shared_ptr<XStringNode> m_rawLabel, m_rawUnit;
};

//---------------------------------------------------------------------------
#endif
