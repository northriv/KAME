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
//---------------------------------------------------------------------------

#ifndef graphmathtoolH
#define graphmathtoolH
//---------------------------------------------------------------------------

#include "graph.h"
#include "analyzer.h"
#include <cmath>

class QPainter;
class XQGraph;
class OnScreenPickableObject;
class OnPlotMaskObject;

enum class MaskShape : int {
    Rectangle = 0,
    Ellipse = 1,
    Arbitrary = 2,
};

class DECLSPEC_KAME XGraphMathTool: public XNode {
public:
    XGraphMathTool(const char *name, bool runtime, Transaction &tr_meas,
        const shared_ptr<XScalarEntryList> &entries, const shared_ptr<XDriver> &driver,
        const shared_ptr<XPlot> &plot, const shared_ptr<XNode> &parentList);
    virtual ~XGraphMathTool() {}

    using cv_iterator = std::vector<XGraph::VFloat>::const_iterator;

    const shared_ptr<XHexNode> &baseColor() const {return m_baseColor;}
    virtual XString getMenuLabel() const {return getLabel();}

    virtual bool releaseEntries(Transaction &tr) {return true;}

    void highlight(bool state, const shared_ptr<XQGraphPainter> &painter);

    //! True if OSOs exist and are all valid for the given painter.
    bool hasValidOSOs(XQGraphPainter *painter) const;

    //! Clears all on-screen objects, e.g. when the tool is released.
    //! Invalidates each OSO first so the painter skips them even if
    //! a temporary shared_ptr from weak_ptr::lock() extends their lifetime.
    void clearOnScreenObjects();

    virtual XString getTypename() const override {
        return m_storedTypename.empty() ? XNode::getTypename() : m_storedTypename;
    }
    void setStoredTypename(const XString &t) { m_storedTypename = t; }

    void updateOnScreenObjects(const Snapshot &shot, const shared_ptr<XQGraphPainter> &painter, const XString &msg);

    shared_ptr<XNode> parentList() {return m_parentList.lock();}
protected:
    shared_ptr<XScalarEntryList> entries() const {return m_entries.lock();}
    const weak_ptr<XPlot> m_plot;
    bool isHighLighted() const {return m_highlight;}

    virtual void updateAdditionalOnScreenObjects(const Snapshot &shot, const shared_ptr<XQGraphPainter> &painter, const XString &msg) = 0;
    virtual std::deque<shared_ptr<OnScreenObject>> createAdditionalOnScreenObjects(const shared_ptr<XQGraphPainter> &painter) = 0;
    weak_ptr<OnScreenPickableObject> m_osoHighlight;
private:
    const shared_ptr<XDoubleNode> m_begin, m_end;
    const weak_ptr<XScalarEntryList> m_entries;
    const shared_ptr<XHexNode> m_baseColor;
    const weak_ptr<XNode> m_parentList;
    bool m_highlight = false;
    XString m_storedTypename;
    std::deque<shared_ptr<OnScreenObject>> m_osos;
};

class DECLSPEC_KAME XGraph1DMathTool: public XGraphMathTool {
public:
    XGraph1DMathTool(const char *name, bool runtime, Transaction &tr_meas,
        const shared_ptr<XScalarEntryList> &entries, const shared_ptr<XDriver> &driver,
        const shared_ptr<XPlot> &plot, const shared_ptr<XNode> &parentList);
    virtual ~XGraph1DMathTool();

    virtual void update(Transaction &tr, const shared_ptr<XQGraphPainter> &painter, cv_iterator xbegin, cv_iterator xend, cv_iterator ybegin, cv_iterator yend) = 0;

    const shared_ptr<XDoubleNode> &first() const {return m_first;}
    const shared_ptr<XDoubleNode> &last() const {return m_last;}

    virtual XString getMenuLabel() const override;
protected:
    virtual void updateAdditionalOnScreenObjects(const Snapshot &shot, const shared_ptr<XQGraphPainter> &painter, const XString &msg) override;
    virtual std::deque<shared_ptr<OnScreenObject>> createAdditionalOnScreenObjects(const shared_ptr<XQGraphPainter> &painter) override;
private:
    const shared_ptr<XDoubleNode> m_first, m_last;
    weak_ptr<OnScreenPickableObject> m_osoRect, m_osoLabel;
};

class DECLSPEC_KAME XGraph2DMathTool: public XGraphMathTool {
public:
    XGraph2DMathTool(const char *name, bool runtime, Transaction &tr_meas,
                     const shared_ptr<XScalarEntryList> &entries, const shared_ptr<XDriver> &driver,
                     const shared_ptr<XPlot> &plot, const shared_ptr<XNode> &parentList);
    virtual ~XGraph2DMathTool() {}

    virtual void update(Transaction &tr, const shared_ptr<XQGraphPainter> &painter, const uint32_t *leftupper, unsigned int width,
        unsigned int stride, unsigned int numlines, double coefficient, double offset) = 0;

    const shared_ptr<XDoubleNode> &firstX() const {return m_firstX;}
    const shared_ptr<XDoubleNode> &firstY() const {return m_firstY;}
    const shared_ptr<XDoubleNode> &lastX() const {return m_lastX;}
    const shared_ptr<XDoubleNode> &lastY() const {return m_lastY;}
    const shared_ptr<XComboNode> &maskType() const {return m_maskType;}

    //! Returns the number of unmasked pixels, counting from the stored mask.
    unsigned int pixels(const Snapshot &shot) const {
        auto m = shot[ *this].m_mask;
        if( !m || m->empty()) {
            ssize_t w = lrint(std::abs(shot[ *lastX()] - shot[ *firstX()]));
            ssize_t h = lrint(std::abs(shot[ *lastY()] - shot[ *firstY()]));
            return (w > 0 && h > 0) ? w * h : 0;
        }
        unsigned int count = 0;
        for(auto v: *m) count += v;
        return count ? count : 1;
    }

    //! Generates a mask for the given shape and dimensions.
    //! Returns empty vector for Rectangle (no mask needed), otherwise width*numlines elements (1=included, 0=excluded).
    static std::vector<uint8_t> generateMask(MaskShape shape, unsigned int width, unsigned int numlines);

    //! (Re)generates the mask from the current selection coordinates and mask type, storing it in \a tr.
    //! For Arbitrary, does nothing (mask is set externally).
    void regenerateMask(Transaction &tr);

    //! Atomically sets MaskType to Arbitrary and writes the mask bitmap.
    void setArbitraryMask(const std::vector<uint8_t> &mask);

    struct DECLSPEC_KAME Payload : public XNode::Payload {
        shared_ptr<std::vector<uint8_t>> m_mask; //!< stored mask bitmap; null or empty = all included (Rectangle).
    };

    virtual XString getMenuLabel() const override;
protected:
    virtual void updateAdditionalOnScreenObjects(const Snapshot &shot, const shared_ptr<XQGraphPainter> &painter, const XString &msg) override;
    virtual std::deque<shared_ptr<OnScreenObject>> createAdditionalOnScreenObjects(const shared_ptr<XQGraphPainter> &painter) override;
private:
    const shared_ptr<XDoubleNode> m_firstX, m_firstY, m_lastX, m_lastY;
    const shared_ptr<XComboNode> m_maskType;
    weak_ptr<OnScreenPickableObject> m_osoRect, m_osoLabel;
    weak_ptr<OnPlotMaskObject> m_osoMaskHighlight;
};

//! entrynames semi colon-sparated entry names.
template <class F, class Base>
class DECLSPEC_KAME XGraphMathToolX: public Base {
public:
    XGraphMathToolX(const char *name, bool runtime, Transaction &tr_meas,
                      const shared_ptr<XScalarEntryList> &entries, const shared_ptr<XDriver> &driver,
                      const shared_ptr<XPlot> &plot, const shared_ptr<XNode> &parentList, const std::vector<std::string> &entrynames) :
        Base(name, runtime, ref(tr_meas), entries, driver, plot, parentList) {
        for(size_t i = 0; i < entrynames.size(); ++i) {
             this->m_entries.push_back(XNode::create<XScalarEntry>(
                entrynames[i].c_str(), true, driver));
             if(entries) entries->insert(tr_meas, m_entries.back());
        }
    }
    virtual ~XGraphMathToolX() {}
//    const shared_ptr<XScalarEntry> entry(unsigned int i = 0) const {return m_entries.at(i);}
    virtual bool releaseEntries(Transaction &tr) override {
        auto elist = this->entries();
        if( !elist) return true;
        for(auto &x: m_entries) {
            if( !elist->release(tr, x))
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
class DECLSPEC_KAME XGraph1DMathToolX: public XGraphMathToolX<F, XGraph1DMathTool> {
public:
    using XGraphMathToolX<F, XGraph1DMathTool>::XGraphMathToolX;
    using cv_iterator = typename XGraphMathToolX<F, XGraph1DMathTool>::cv_iterator;
    using ret_type = typename std::conditional<HasSingleEntry, double, std::vector<double>>::type;

    virtual void update(Transaction &tr, const shared_ptr<XQGraphPainter> &painter, cv_iterator xbegin, cv_iterator xend, cv_iterator ybegin, cv_iterator yend) override {
        Snapshot shot( *this); //read-only access to the tool's payload; avoids CoW on the tool node.
        auto func = shot[ *this].functor; //copy functor locally so non-const operator() can be called.
        XString msg;
        if constexpr(HasSingleEntry) {
            double v = func(xbegin, xend, ybegin, yend);
            this->entry()->value(tr, v);
            msg += tr[ *this->entry()->value()].to_str();
        }
        else {
            try {
                std::vector<double> v = func(xbegin, xend, ybegin, yend);
                for(unsigned int i = 0; i < this->numEntries(); ++i) {
                    this->entry(i)->value(tr, v.at(i));
                    msg += tr[ *this->entry(i)->value()].to_str() + " ";
                }
            }
            catch(std::out_of_range&) {
            }
        }
        this->updateOnScreenObjects(shot, painter, msg);
    }
    struct Payload : public XGraph1DMathTool::Payload {
        F functor;
    };
};

template <class F, bool HasSingleEntry = true>
class DECLSPEC_KAME XGraph2DMathToolX: public XGraphMathToolX<F, XGraph2DMathTool> {
public:
    using XGraphMathToolX<F, XGraph2DMathTool>::XGraphMathToolX;
    using ret_type = typename std::conditional<HasSingleEntry, double, std::vector<double>>::type;

    virtual void update(Transaction &tr, const shared_ptr<XQGraphPainter> &painter, const uint32_t *leftupper, unsigned int width,
        unsigned int stride, unsigned int numlines, double coefficient, double offset) override {
        // Ensure mask is generated if MaskType requires one but m_mask is empty.
        {
            Snapshot pre( *this);
            auto shape = (MaskShape)(int)pre[ *this->maskType()];
            if(shape != MaskShape::Rectangle && shape != MaskShape::Arbitrary && !pre[ *this].m_mask) {
                this->iterate_commit([&](Transaction &mtr){
                    this->regenerateMask(mtr);
                });
            }
        }
        Snapshot shot( *this); //read-only access to the tool's payload; avoids CoW on the tool node.
        auto maskptr = shot[ *this].m_mask;
        auto func = shot[ *this].functor; //copy functor locally so non-const operator() can be called.
        static const std::vector<uint8_t> s_empty;
        //discard mask if dimensions mismatch (e.g. selection clipped by image boundary).
        const auto &mask = (maskptr && (maskptr->size() == (size_t)width * numlines))
            ? *maskptr : s_empty;
        XString msg;
        if constexpr(HasSingleEntry) {
            double v = func(leftupper, width, stride, numlines, coefficient, offset, mask);
            this->entry()->value(tr, v);
            msg += tr[ *this->entry()->value()].to_str();
        }
        else {
            try {
                std::vector<double> v = func(leftupper, width, stride, numlines, coefficient, offset, mask);
                for(unsigned int i = 0; i < this->numEntries(); ++i) {
                    this->entry(i)->value(tr, v.at(i));
                    msg += tr[ *this->entry(i)->value()].to_str() + " ";
                }
            }
            catch(std::out_of_range&) {
            }
        }
        this->updateOnScreenObjects(shot, painter, msg);
    }
    struct Payload : public XGraph2DMathTool::Payload {
        F functor;
    };
};

struct DECLSPEC_KAME FuncGraph1DMathToolSum{
    using cv_iterator = std::vector<XGraph::VFloat>::const_iterator;
    double operator()(cv_iterator xbegin, cv_iterator xend, cv_iterator ybegin, cv_iterator yend){
        double v = 0.0;
        for(auto yit = ybegin; yit != yend; ++yit)
            v += *yit;
        return v;
    }
};
using XGraph1DMathToolSum = XGraph1DMathToolX<FuncGraph1DMathToolSum>;

struct DECLSPEC_KAME FuncGraph1DMathToolAverage{
    using cv_iterator = std::vector<XGraph::VFloat>::const_iterator;
    double operator()(cv_iterator xbegin, cv_iterator xend, cv_iterator ybegin, cv_iterator yend){
        double v = 0.0;
        for(auto yit = ybegin; yit != yend; ++yit)
            v += *yit;
        return v / (yend - ybegin);
    }
};
using XGraph1DMathToolAverage = XGraph1DMathToolX<FuncGraph1DMathToolAverage>;
struct DECLSPEC_KAME FuncGraph1DMathToolMaxValue{
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
struct DECLSPEC_KAME FuncGraph1DMathToolMinValue{
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
struct DECLSPEC_KAME FuncGraph1DMathToolMaxPosition{
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
struct DECLSPEC_KAME FuncGraph1DMathToolMinPosition{
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
struct DECLSPEC_KAME FuncGraph1DMathToolCoG{
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




struct DECLSPEC_KAME FuncGraph2DMathToolSum{
    using cv_iterator = std::vector<XGraph::VFloat>::const_iterator;
    double operator()(const uint32_t *leftupper, unsigned int width, unsigned int stride, unsigned int numlines,
        double coefficient, double offset, const std::vector<uint8_t> &mask){
        double v = 0.0;
        unsigned int count = 0;
        if(mask.empty()) {
            count = width * numlines;
            for(unsigned int y = 0; y < numlines; ++y) {
                for(unsigned int x = 0; x < width; ++x)
                    v += leftupper[x];
                leftupper += stride;
            }
        }
        else {
            for(unsigned int y = 0; y < numlines; ++y) {
                for(unsigned int x = 0; x < width; ++x) {
                    if(mask[y * width + x]) {
                        v += leftupper[x];
                        ++count;
                    }
                }
                leftupper += stride;
            }
        }
        return v * coefficient + offset * count;
    }
};
using XGraph2DMathToolSum = XGraph2DMathToolX<FuncGraph2DMathToolSum>;
struct DECLSPEC_KAME FuncGraph2DMathToolAverage{
    using cv_iterator = std::vector<XGraph::VFloat>::const_iterator;
    double operator()(const uint32_t *leftupper, unsigned int width, unsigned int stride, unsigned int numlines,
        double coefficient, double offset, const std::vector<uint8_t> &mask){
        double v = 0.0;
        unsigned int count = 0;
        if(mask.empty()) {
            count = width * numlines;
            for(unsigned int y = 0; y < numlines; ++y) {
                for(unsigned int x = 0; x < width; ++x)
                    v += leftupper[x];
                leftupper += stride;
            }
        }
        else {
            for(unsigned int y = 0; y < numlines; ++y) {
                for(unsigned int x = 0; x < width; ++x) {
                    if(mask[y * width + x]) {
                        v += leftupper[x];
                        ++count;
                    }
                }
                leftupper += stride;
            }
        }
        if( !count) return offset;
        return v * coefficient / count + offset;
    }
};
using XGraph2DMathToolAverage = XGraph2DMathToolX<FuncGraph2DMathToolAverage>;

class XMeasure;

template <class X, class XQC>
class DECLSPEC_KAME XGraphMathToolList : public XCustomTypeListNode<X> {
public:
    XGraphMathToolList(const char *name, bool runtime,
        const shared_ptr<XMeasure> &meas, const shared_ptr<XDriver> &driver,
        const shared_ptr<XPlot> &plot);
    virtual ~XGraphMathToolList();

    using cv_iterator = typename X::cv_iterator;

    void setBaseColor(unsigned int color) {m_basecolor = color;}

    //! Refresh OSOs for all tools without recomputing values (e.g. for lists not in the active sequence).
    void refreshOSOs(const shared_ptr<XQGraphPainter> &painter);

    struct DECLSPEC_KAME Payload : public XCustomTypeListNode<X>::Payload {
        //requests popup Menu if XQGraph1/2DMathToolConnector is connected.
        Talker<int, int, XGraphMathTool *> &popupMenu() {return m_tlkOnPopupMenu;}
    protected:
        Talker<int, int, XGraphMathTool *> m_tlkOnPopupMenu;
    };
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
class DECLSPEC_KAME XGraph1DMathToolList : public XGraphMathToolList<XGraph1DMathTool, XQGraph1DMathToolConnector> {
public:
    XGraph1DMathToolList(const char *name, bool runtime,
        const shared_ptr<XMeasure> &meas, const shared_ptr<XDriver> &driver,
        const shared_ptr<XPlot> &plot);
    virtual ~XGraph1DMathToolList() {}

    DEFINE_TYPE_HOLDER(
        std::reference_wrapper<Transaction>, const shared_ptr<XScalarEntryList> &,
        const shared_ptr<XDriver> &, const shared_ptr<XPlot> &, const shared_ptr<XNode> &, const std::vector<std::string> &
        )
    virtual shared_ptr<XNode> createByTypename(const XString &, const XString& name);

    virtual void update(Transaction &tr, const shared_ptr<XQGraphPainter> &painter,
        cv_iterator xbegin, cv_iterator xend, cv_iterator ybegin, cv_iterator yend);

    void onAxisSelectedByToolForCreate(const Snapshot &shot, const std::tuple<XString, XGraph::VFloat, XGraph::VFloat, XQGraph*>&);
    void onAxisSelectedByToolForReselect(const Snapshot &shot, const std::tuple<XString, XGraph::VFloat, XGraph::VFloat, XQGraph*>&);
protected:
};

class XQGraph2DMathToolConnector;
class DECLSPEC_KAME XGraph2DMathToolList : public XGraphMathToolList<XGraph2DMathTool, XQGraph2DMathToolConnector> {
public:
    XGraph2DMathToolList(const char *name, bool runtime,
        const shared_ptr<XMeasure> &meas, const shared_ptr<XDriver> &driver,
        const shared_ptr<XPlot> &plot);
    virtual ~XGraph2DMathToolList() {}

    DEFINE_TYPE_HOLDER(
        std::reference_wrapper<Transaction>, const shared_ptr<XScalarEntryList> &,
        const shared_ptr<XDriver> &, const shared_ptr<XPlot> &, const shared_ptr<XNode> &, const std::vector<std::string> &
        )
    virtual shared_ptr<XNode> createByTypename(const XString &, const XString& name);

    virtual void update(Transaction &tr, const shared_ptr<XQGraphPainter> &painter,
        const uint32_t *leftupper,
        unsigned int width, unsigned int stride, unsigned int numlines, double coefficient, double offset = 0.0);

    void onPlaneSelectedByToolForCreate(const Snapshot &shot,
        const std::tuple<XString, XGraph::ValPoint, XGraph::ValPoint, XQGraph*>&);
    void onPlaneSelectedByToolForReselect(const Snapshot &shot,
        const std::tuple<XString, XGraph::ValPoint, XGraph::ValPoint, XQGraph*>&);
protected:
};

#endif
