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

#ifndef graphmathtoolH
#define graphmathtoolH
//---------------------------------------------------------------------------

#include "graph.h"
#include "analyzer.h"

class QPainter;
class XQGraph;
class OnScreenObjectWithMarker;

class DECLSPEC_KAME XGraphMathTool: public XNode {
public:
    XGraphMathTool(const char *name, bool runtime, Transaction &tr_meas,
        const shared_ptr<XScalarEntryList> &entries, const shared_ptr<XDriver> &driver,
        const shared_ptr<XPlot> &plot);
    virtual ~XGraphMathTool() {}

    using cv_iterator = std::vector<XGraph::VFloat>::const_iterator;

    const shared_ptr<XHexNode> &baseColor() const {return m_baseColor;}
    virtual XString getMenuLabel() const {return getLabel();}

    virtual bool releaseEntries(Transaction &tr) {return true;}

    void highlight(bool state, XQGraph *graphwidget);

    void updateOnScreenObjects(const Snapshot &shot, XQGraph *graphwidget, XString msg);
protected:
    shared_ptr<XScalarEntryList> entries() const {return m_entries.lock();}
    const weak_ptr<XPlot> m_plot;
    bool isHighLighted() const {return m_highlight;}

    virtual void updateAdditionalOnScreenObjects(const Snapshot &shot, XQGraph *graphwidget, const XString &msg) = 0;
    virtual std::deque<shared_ptr<OnScreenObject>> createAdditionalOnScreenObjects(const shared_ptr<XQGraphPainter> &painter) = 0;
    weak_ptr<OnScreenObjectWithMarker> m_osoHighlight;
private:
    const shared_ptr<XDoubleNode> m_begin, m_end;
    const weak_ptr<XScalarEntryList> m_entries;
    const shared_ptr<XHexNode> m_baseColor;
    bool m_highlight = false;
    std::deque<shared_ptr<OnScreenObject>> m_osos;
};

class DECLSPEC_KAME XGraph1DMathTool: public XGraphMathTool {
public:
    XGraph1DMathTool(const char *name, bool runtime, Transaction &tr_meas,
        const shared_ptr<XScalarEntryList> &entries, const shared_ptr<XDriver> &driver,
        const shared_ptr<XPlot> &plot);
    virtual ~XGraph1DMathTool();

    virtual void update(Transaction &tr, XQGraph *graphwidget, cv_iterator xbegin, cv_iterator xend, cv_iterator ybegin, cv_iterator yend) = 0;

    const shared_ptr<XDoubleNode> &begin() const {return m_begin;}
    const shared_ptr<XDoubleNode> &end() const {return m_end;}

    virtual XString getMenuLabel() const override;
protected:
    virtual void updateAdditionalOnScreenObjects(const Snapshot &shot, XQGraph *graphwidget, const XString &msg) override;
    virtual std::deque<shared_ptr<OnScreenObject>> createAdditionalOnScreenObjects(const shared_ptr<XQGraphPainter> &painter) override;
private:
    const shared_ptr<XDoubleNode> m_begin, m_end;
    weak_ptr<OnScreenObjectWithMarker> m_osoRect, m_osoLabel;
};

class DECLSPEC_KAME XGraph2DMathTool: public XGraphMathTool {
public:
    XGraph2DMathTool(const char *name, bool runtime, Transaction &tr_meas,
                     const shared_ptr<XScalarEntryList> &entries, const shared_ptr<XDriver> &driver,
                     const shared_ptr<XPlot> &plot);
    virtual ~XGraph2DMathTool() {}

    virtual void update(Transaction &tr, XQGraph *graphwidget, const uint32_t *leftupper, unsigned int width,
        unsigned int stride, unsigned int numlines, double coefficient, double offset) = 0;

    const shared_ptr<XDoubleNode> &beginX() const {return m_beginX;}
    const shared_ptr<XDoubleNode> &beginY() const {return m_beginY;}
    const shared_ptr<XDoubleNode> &endX() const {return m_endX;}
    const shared_ptr<XDoubleNode> &endY() const {return m_endY;}
    unsigned int pixels(const Snapshot &shot) const {
        return std::abs((shot[ *endX()] - shot[ *beginX()]) * (shot[ *endY()] - shot[ *beginY()]));
    }

    virtual XString getMenuLabel() const override;
protected:
    virtual void updateAdditionalOnScreenObjects(const Snapshot &shot, XQGraph *graphwidget, const XString &msg) override;
    virtual std::deque<shared_ptr<OnScreenObject>> createAdditionalOnScreenObjects(const shared_ptr<XQGraphPainter> &painter) override;
private:
    const shared_ptr<XDoubleNode> m_beginX, m_beginY, m_endX, m_endY;
    weak_ptr<OnScreenObjectWithMarker> m_osoRect, m_osoLabel;
};

//! entrynames semi colon-sparated entry names.
template <class F, class Base>
class XGraphMathToolX: public Base {
public:
    XGraphMathToolX(const char *name, bool runtime, Transaction &tr_meas,
                      const shared_ptr<XScalarEntryList> &entries, const shared_ptr<XDriver> &driver,
                      const shared_ptr<XPlot> &plot, const std::vector<std::string> &entrynames) :
        Base(name, runtime, ref(tr_meas), entries, driver, plot) {
        for(size_t i = 0; i < entrynames.size(); ++i) {
             this->m_entries.push_back(XNode::create<XScalarEntry>(
                entrynames[i].c_str(), false, driver));
             entries->insert(tr_meas, m_entries.back());
        }
    }
    virtual ~XGraphMathToolX() {}
//    const shared_ptr<XScalarEntry> entry(unsigned int i = 0) const {return m_entries.at(i);}
    virtual bool releaseEntries(Transaction &tr) override {
        for(auto &x: m_entries) {
            if( !this->entries()->release(tr, x))
                return false;
        }
        return true;
    }
    unsigned int numEntries() const {return m_entries.size();}
    const shared_ptr<XScalarEntry> entry(unsigned int i = 0) const {return m_entries.at(i);}
private:
    std::deque<shared_ptr<XScalarEntry>> m_entries;
};

template <class F, bool HasSingleEntry = true>
class XGraph1DMathToolX: public XGraphMathToolX<F, XGraph1DMathTool> {
public:
    using XGraphMathToolX<F, XGraph1DMathTool>::XGraphMathToolX;
    using cv_iterator = typename XGraphMathToolX<F, XGraph1DMathTool>::cv_iterator;
    using ret_type = typename std::conditional<HasSingleEntry, double, std::vector<double>>::type;

    virtual void update(Transaction &tr, XQGraph *graphwidget, cv_iterator xbegin, cv_iterator xend, cv_iterator ybegin, cv_iterator yend) override {
        XString msg;
        if constexpr(HasSingleEntry) {
            double v = tr[ *this].functor(xbegin, xend, ybegin, yend);
            this->entry()->value(tr, v);
            msg += tr[ *this->entry()->value()].to_str();
        }
        else {
            try {
                std::vector<double> v = tr[ *this].functor(xbegin, xend, ybegin, yend);
                for(unsigned int i = 0; i < this->numEntries(); ++i) {
                    this->entry(i)->value(tr, v.at(i));
                    msg += tr[ *this->entry(i)->value()].to_str() + " ";
                }
            }
            catch(std::out_of_range&) {
            }
        }
        this->updateOnScreenObjects(tr, graphwidget, msg);
    }
    struct Payload : public XGraph1DMathTool::Payload {
        F functor;
    };
};

template <class F, bool HasSingleEntry = true>
class XGraph2DMathToolX: public XGraphMathToolX<F, XGraph2DMathTool> {
public:
    using XGraphMathToolX<F, XGraph2DMathTool>::XGraphMathToolX;
    using ret_type = typename std::conditional<HasSingleEntry, double, std::vector<double>>::type;

    virtual void update(Transaction &tr, XQGraph *graphwidget, const uint32_t *leftupper, unsigned int width,
        unsigned int stride, unsigned int numlines, double coefficient, double offset) override {
//        using namespace Eigen;
//        using RMatrixXu32 = Matrix<uint32_t, Dynamic, Dynamic, RowMajor>;
//        auto cmatrix = Map<const RMatrixXu32, 0, Stride<Dynamic, 1>>(
//            leftupper, numlines, width, Stride<Dynamic, 1>(stride, 1));
        XString msg;
        if constexpr(HasSingleEntry) {
            double v = tr[ *this].functor(leftupper, width, stride, numlines, coefficient, offset);
            this->entry()->value(tr, v);
            msg += tr[ *this->entry()->value()].to_str();
        }
        else {
            try {
                std::vector<double> v = tr[ *this].functor(leftupper, width, stride, numlines, coefficient, offset);
                for(unsigned int i = 0; i < this->numEntries(); ++i) {
                    this->entry(i)->value(tr, v.at(i));
                    msg += tr[ *this->entry(i)->value()].to_str() + " ";
                }
            }
            catch(std::out_of_range&) {
            }
        }
        this->updateOnScreenObjects(tr, graphwidget, msg);
    }
    struct Payload : public XGraph2DMathTool::Payload {
        F functor;
    };
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
    double operator()(const uint32_t *leftupper, unsigned int width, unsigned int stride, unsigned int numlines, double coefficient, double offset){
        double v = 0.0;
        for(unsigned int y = 0; y < numlines; ++y) {
            for(const uint32_t *p = leftupper; p < leftupper + width; ++p)
                v += *p;
            leftupper += stride;
        }
        return v * coefficient + offset * numlines * width;
    }
};
using XGraph2DMathToolSum = XGraph2DMathToolX<FuncGraph2DMathToolSum>;
struct FuncGraph2DMathToolAverage{
    using cv_iterator = std::vector<XGraph::VFloat>::const_iterator;
    double operator()(const uint32_t *leftupper, unsigned int width, unsigned int stride, unsigned int numlines, double coefficient, double offset){
        double v = 0.0;
        for(unsigned int y = 0; y < numlines; ++y) {
            for(const uint32_t *p = leftupper; p < leftupper + width; ++p)
                v += *p;
            leftupper += stride;
        }
        return v * coefficient / (width * numlines) + offset;
    }
};
using XGraph2DMathToolAverage = XGraph2DMathToolX<FuncGraph2DMathToolAverage>;

class XMeasure;

template <class X, class XQC>
class XGraphMathToolList : public XCustomTypeListNode<X> {
public:
    XGraphMathToolList(const char *name, bool runtime,
        const shared_ptr<XMeasure> &meas, const shared_ptr<XDriver> &driver,
        const shared_ptr<XPlot> &plot);
    virtual ~XGraphMathToolList();

    using cv_iterator = typename X::cv_iterator;

    void setBaseColor(unsigned int color) {m_basecolor = color;}
protected:
    const weak_ptr<XMeasure> m_measure;
    const weak_ptr<XScalarEntryList> m_entries;
    const weak_ptr<XDriver> m_driver;
    const weak_ptr<XPlot> m_plot;
    unsigned int m_basecolor = 0xffa070u;

    friend XQC;

    void onRelease(const Snapshot &shot, const XListNodeBase::Payload::ReleaseEvent &e);

    shared_ptr<Listener> m_lsnRelease;
};

class XQGraph1DMathToolConnector;
class XGraph1DMathToolList : public XGraphMathToolList<XGraph1DMathTool, XQGraph1DMathToolConnector> {
public:
    XGraph1DMathToolList(const char *name, bool runtime,
        const shared_ptr<XMeasure> &meas, const shared_ptr<XDriver> &driver,
        const shared_ptr<XPlot> &plot);
    virtual ~XGraph1DMathToolList() {}

    DEFINE_TYPE_HOLDER(
        std::reference_wrapper<Transaction>, const shared_ptr<XScalarEntryList> &,
        const shared_ptr<XDriver> &, const shared_ptr<XPlot> &, const std::vector<std::string> &
        )
    virtual shared_ptr<XNode> createByTypename(const XString &, const XString& name);

    virtual void update(Transaction &tr, XQGraph *graphwidget,
        cv_iterator xbegin, cv_iterator xend, cv_iterator ybegin, cv_iterator yend);

    void onAxisSelectedByToolForCreate(const Snapshot &shot, const std::tuple<XString, XGraph::VFloat, XGraph::VFloat, XQGraph*>&);
    void onAxisSelectedByToolForReselect(const Snapshot &shot, const std::tuple<XString, XGraph::VFloat, XGraph::VFloat, XQGraph*>&);
protected:
};

class XQGraph2DMathToolConnector;
class XGraph2DMathToolList : public XGraphMathToolList<XGraph2DMathTool, XQGraph2DMathToolConnector> {
public:
    XGraph2DMathToolList(const char *name, bool runtime,
        const shared_ptr<XMeasure> &meas, const shared_ptr<XDriver> &driver,
        const shared_ptr<XPlot> &plot);
    virtual ~XGraph2DMathToolList() {}

    DEFINE_TYPE_HOLDER(
        std::reference_wrapper<Transaction>, const shared_ptr<XScalarEntryList> &,
        const shared_ptr<XDriver> &, const shared_ptr<XPlot> &, const std::vector<std::string> &
        )
    virtual shared_ptr<XNode> createByTypename(const XString &, const XString& name);

    virtual void update(Transaction &tr, XQGraph *graphwidget,
        const uint32_t *leftupper,
        unsigned int width, unsigned int stride, unsigned int numlines, double coefficient, double offset = 0.0);

    void onPlaneSelectedByToolForCreate(const Snapshot &shot,
        const std::tuple<XString, XGraph::ValPoint, XGraph::ValPoint, XQGraph*>&);
    void onPlaneSelectedByToolForReselect(const Snapshot &shot,
        const std::tuple<XString, XGraph::ValPoint, XGraph::ValPoint, XQGraph*>&);
protected:
};

#endif
