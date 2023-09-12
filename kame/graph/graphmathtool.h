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
//---------------------------------------------------------------------------

#ifndef graphmathtoolH
#define graphmathtoolH
//---------------------------------------------------------------------------

#include "graph.h"
#include "analyzer.h"

class DECLSPEC_KAME XGraph1DMathTool: public XNode {
public:
    XGraph1DMathTool(const char *name, bool runtime, const shared_ptr<XScalarEntryList> &entries);
    virtual ~XGraph1DMathTool() {}

//    enum class MathTool1D {Sum, Average, CoG, Moment2nd, MaxValue, MinValue};
//    enum class MathTool2D {Sum, Average, CoG, Moment2nd, MaxValue, MinValue};
    using cv_iterator = std::vector<XGraph::VFloat>::const_iterator;
    virtual void update(Transaction &tr, cv_iterator xbegin, cv_iterator xend, cv_iterator ybegin, cv_iterator yend) = 0;

    const shared_ptr<XDoubleNode> &begin() const {return m_begin;}
    const shared_ptr<XDoubleNode> &end() const {return m_end;}

    virtual void insertEntries(Transaction &tr_meas, const shared_ptr<XDriver> &driver) {}
    virtual void releaseEntries(Transaction &tr) {}
protected:
    shared_ptr<XScalarEntryList> entries() const {return m_entries.lock();}
private:
    const shared_ptr<XDoubleNode> m_begin, m_end;
    const weak_ptr<XScalarEntryList> m_entries;
};

class DECLSPEC_KAME XGraph2DMathTool: public XNode {
public:
    XGraph2DMathTool(const char *name, bool runtime, const shared_ptr<XScalarEntryList> &entries);
    virtual ~XGraph2DMathTool() {}

    virtual void update(Transaction &tr, const uint32_t *leftupper, unsigned int width,
        unsigned int stride, unsigned int numlines, double coefficient) = 0;

    const shared_ptr<XDoubleNode> &beginX() const {return m_beginX;}
    const shared_ptr<XDoubleNode> &beginY() const {return m_beginY;}
    const shared_ptr<XDoubleNode> &endX() const {return m_endX;}
    const shared_ptr<XDoubleNode> &endY() const {return m_endY;}

    virtual void insertEntries(Transaction &tr_meas, const shared_ptr<XDriver> &driver) {}
    virtual void releaseEntries(Transaction &tr) {}
protected:
    shared_ptr<XScalarEntryList> entries() const {return m_entries.lock();}
private:
    const shared_ptr<XDoubleNode> m_beginX, m_beginY, m_endX, m_endY;
    const weak_ptr<XScalarEntryList> m_entries;
};

template <class F>
class XGraph1DMathToolX: public XGraph1DMathTool {
public:
    XGraph1DMathToolX(const char *name, bool runtime, const shared_ptr<XScalarEntryList> &entries) :
        XGraph1DMathTool(name, runtime, entries) {}
    virtual ~XGraph1DMathToolX() {}
    virtual void update(Transaction &tr, cv_iterator xbegin, cv_iterator xend, cv_iterator ybegin, cv_iterator yend) override {
        double v = F()(xbegin, xend, ybegin, yend);
        m_entry->value(tr, v);
    }
    virtual void insertEntries(Transaction &tr_meas, const shared_ptr<XDriver> &driver) {
        m_entry = create<XScalarEntry>(tr_meas, getName().c_str(), false, driver);
        entries()->insert(tr_meas, m_entry);
    }
    virtual void releaseEntries(Transaction &tr) override {entries()->release(tr, m_entry);}
private:
    shared_ptr<XScalarEntry> m_entry;
};

template <class F>
class XGraph2DMathToolX: public XGraph2DMathTool {
public:
    XGraph2DMathToolX(const char *name, bool runtime, const shared_ptr<XScalarEntryList> &entries) :
        XGraph2DMathTool(name, runtime, entries) {}
    virtual ~XGraph2DMathToolX() {}
    virtual void update(Transaction &tr, const uint32_t *leftupper, unsigned int width,
        unsigned int stride, unsigned int numlines, double coefficient) override {
        double v = F()(leftupper, width, stride, numlines, coefficient);
        m_entry->value(tr, v);
    }
    virtual void insertEntries(Transaction &tr_meas, const shared_ptr<XDriver> &driver) override {
        m_entry = create<XScalarEntry>(tr_meas, getName().c_str(), false, driver);
        entries()->insert(tr_meas, m_entry);
    }
    virtual void releaseEntries(Transaction &tr) override {entries()->release(tr, m_entry);}
private:
    shared_ptr<XScalarEntry> m_entry;
};

struct FuncGraph1DMathToolSum{
    using cv_iterator = std::vector<XGraph::VFloat>::const_iterator;
    double operator()(cv_iterator xbegin, cv_iterator xend, cv_iterator ybegin, cv_iterator yend){
        double v = 0.0;
        for(auto yit = ybegin; yit != yend; ++yit)
            v += *yit;
        return v;
    }
};
using XGraph1DMathToolSum = XGraph1DMathToolX<FuncGraph1DMathToolSum>;

struct FuncGraph1DMathToolAverage{
    using cv_iterator = std::vector<XGraph::VFloat>::const_iterator;
    double operator()(cv_iterator xbegin, cv_iterator xend, cv_iterator ybegin, cv_iterator yend){
        double v = 0.0;
        for(auto yit = ybegin; yit != yend; ++yit)
            v += *yit;
        return v / (yend - ybegin);
    }
};
using XGraph1DMathToolAverage = XGraph1DMathToolX<FuncGraph1DMathToolAverage>;
struct FuncGraph1DMathToolMaxValue{
    using cv_iterator = std::vector<XGraph::VFloat>::const_iterator;
    double operator()(cv_iterator xbegin, cv_iterator xend, cv_iterator ybegin, cv_iterator yend){
        double vmax = -1e10;
        for(auto yit = ybegin; yit != yend; ++yit) {
            if(*yit > vmax)
                vmax = *yit;
        }
        return vmax;
    }
};
using XGraph1DMathToolMaxValue = XGraph1DMathToolX<FuncGraph1DMathToolMaxValue>;
struct FuncGraph1DMathToolMinValue{
    using cv_iterator = std::vector<XGraph::VFloat>::const_iterator;
    double operator()(cv_iterator xbegin, cv_iterator xend, cv_iterator ybegin, cv_iterator yend){
        double vmin = 1e10;
        for(auto yit = ybegin; yit != yend; ++yit) {
            if(*yit < vmin)
                vmin = *yit;
        }
        return vmin;
    }
};
using XGraph1DMathToolMinValue = XGraph1DMathToolX<FuncGraph1DMathToolMinValue>;
struct FuncGraph1DMathToolMaxPosition{
    using cv_iterator = std::vector<XGraph::VFloat>::const_iterator;
    double operator()(cv_iterator xbegin, cv_iterator xend, cv_iterator ybegin, cv_iterator yend){
        double vmax = -1e10;
        double x = *xbegin;
        for(auto yit = ybegin; yit != yend; ++yit) {
            if(*yit >= vmax) {
                vmax = *yit;
                x = *(xbegin + (yit - ybegin));
            }
        }
        return x;
    }
};
using XGraph1DMathToolMaxPosition = XGraph1DMathToolX<FuncGraph1DMathToolMaxPosition>;
struct FuncGraph1DMathToolMinPosition{
    using cv_iterator = std::vector<XGraph::VFloat>::const_iterator;
    double operator()(cv_iterator xbegin, cv_iterator xend, cv_iterator ybegin, cv_iterator yend){
        double vmin = 1e10;
        double x = *xbegin;
        for(auto yit = ybegin; yit != yend; ++yit) {
            if(*yit <= vmin) {
                vmin = *yit;
                x = *(xbegin + (yit - ybegin));
            }
        }
        return x;
    }
};
using XGraph1DMathToolMinPosition = XGraph1DMathToolX<FuncGraph1DMathToolMinPosition>;
struct FuncGraph1DMathToolCoG{
    using cv_iterator = std::vector<XGraph::VFloat>::const_iterator;
    double operator()(cv_iterator xit, cv_iterator xend, cv_iterator ybegin, cv_iterator yend){
        double v = 0;
        double ysum = 0;
        for(auto yit = ybegin; yit != yend; ++yit) {
            ysum += *yit;
            v += *xit * *yit;
            *xit++;
        }
        return v / ysum;
    }
};
using XGraph1DMathToolCoG = XGraph1DMathToolX<FuncGraph1DMathToolCoG>;




struct FuncGraph2DMathToolSum{
    using cv_iterator = std::vector<XGraph::VFloat>::const_iterator;
    double operator()(const uint32_t *leftupper, unsigned int width, unsigned int stride, unsigned int numlines, double coefficient){
        double v = 0.0;
        for(unsigned int y = 0; y < numlines; ++y) {
            for(const uint32_t *p = leftupper; p < leftupper + width; ++p)
                v += *p;
            leftupper += stride;
        }
        return v * coefficient;
    }
};
using XGraph2DMathToolSum = XGraph2DMathToolX<FuncGraph2DMathToolSum>;
struct FuncGraph2DMathToolAverage{
    using cv_iterator = std::vector<XGraph::VFloat>::const_iterator;
    double operator()(const uint32_t *leftupper, unsigned int width, unsigned int stride, unsigned int numlines, double coefficient){
        double v = 0.0;
        for(unsigned int y = 0; y < numlines; ++y) {
            for(const uint32_t *p = leftupper; p < leftupper + width; ++p)
                v += *p;
            leftupper += stride;
        }
        return v * coefficient / (width * numlines);
    }
};
using XGraph2DMathToolAverage = XGraph2DMathToolX<FuncGraph2DMathToolAverage>;

class XMeasure;
class XGraph1DMathToolList : public XCustomTypeListNode<XGraph1DMathTool> {
public:
    XGraph1DMathToolList(const char *name, bool runtime,
        const shared_ptr<XMeasure> &meas, const shared_ptr<XDriver> &driver);

    using cv_iterator = XGraph1DMathTool::cv_iterator;
    virtual void update(Transaction &tr,
        cv_iterator xbegin, cv_iterator xend, cv_iterator ybegin, cv_iterator yend);

    DEFINE_TYPE_HOLDER(
        const shared_ptr<XScalarEntryList> &
        )
    virtual shared_ptr<XNode> createByTypename(const XString &, const XString& name);
private:
    const weak_ptr<XMeasure> m_measure;
    const weak_ptr<XScalarEntryList> m_entries;
    const weak_ptr<XDriver> m_driver;
    friend class XQGraph1DMathToolConnector;
    void onAxisSelectedByTool(const Snapshot &shot, const std::tuple<XString, XGraph::VFloat, XGraph::VFloat>&);
};

class XQGraph2DMathToolConnector;
class XGraph2DMathToolList : public XCustomTypeListNode<XGraph2DMathTool> {
public:
    XGraph2DMathToolList(const char *name, bool runtime,
        const shared_ptr<XMeasure> &meas, const shared_ptr<XDriver> &driver);

    virtual void update(Transaction &tr, const uint32_t *leftupper,
        unsigned int width, unsigned int stride, unsigned int numlines, double coefficient);

    DEFINE_TYPE_HOLDER(
        const shared_ptr<XScalarEntryList> &
        )
    virtual shared_ptr<XNode> createByTypename(const XString &, const XString& name);
private:
    friend class XQGraph2DMathToolConnector;
    const weak_ptr<XMeasure> m_measure;
    const weak_ptr<XDriver> m_driver;
    void onPlaneSelectedByTool(const Snapshot &shot, const std::tuple<XString, XGraph::ValPoint, XGraph::ValPoint>&);
};

#endif
