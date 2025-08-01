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
#ifdef USE_PYBIND11
    #include <pybind11/pybind11.h>
#endif

#include "graphmathtool.h"
#include "graphmathfittool.h"
#include "measure.h"
#include "graphpainter.h"
//---------------------------------------------------------------------------
DECLARE_TYPE_HOLDER(XGraph1DMathToolList)
DECLARE_TYPE_HOLDER(XGraph2DMathToolList)


REGISTER_TYPE(XGraph1DMathToolList, Graph1DMathToolSum, "Sum");
REGISTER_TYPE(XGraph1DMathToolList, Graph1DMathToolAverage, "Average");
REGISTER_TYPE(XGraph1DMathToolList, Graph1DMathToolCoG, "CoG");
REGISTER_TYPE(XGraph1DMathToolList, Graph1DMathToolMaxValue, "MaxValue");
REGISTER_TYPE(XGraph1DMathToolList, Graph1DMathToolMinValue, "MinValue");
REGISTER_TYPE(XGraph1DMathToolList, Graph1DMathToolMaxPosition, "MaxPosition");
REGISTER_TYPE(XGraph1DMathToolList, Graph1DMathToolMinPosition, "MinPosition");

REGISTER_TYPE(XGraph1DMathToolList, Graph1DMathGaussianPositionTool, "GaussianCenter");
REGISTER_TYPE(XGraph1DMathToolList, Graph1DMathGaussianFWHMTool, "GaussianFWHM");
REGISTER_TYPE(XGraph1DMathToolList, Graph1DMathGaussianHeightTool, "GaussianHeight");

REGISTER_TYPE(XGraph1DMathToolList, Graph1DMathLorenzianPositionTool, "LorenzianCenter");
REGISTER_TYPE(XGraph1DMathToolList, Graph1DMathLorenzianFWHMTool, "LorenzianFWHM");
REGISTER_TYPE(XGraph1DMathToolList, Graph1DMathLorenzianHeightTool, "LorenzianHeight");

REGISTER_TYPE(XGraph2DMathToolList, Graph2DMathToolSum, "Sum");
REGISTER_TYPE(XGraph2DMathToolList, Graph2DMathToolAverage, "Average");

XGraphMathTool::XGraphMathTool(const char *name, bool runtime, Transaction &tr_meas,
    const shared_ptr<XScalarEntryList> &entries, const shared_ptr<XDriver> &driver,
    const shared_ptr<XPlot> &plot) :
    XNode(name, runtime),
    m_plot(plot),
    m_entries(entries),
    m_baseColor(create<XHexNode>("BaseColor", false)) {
    trans( *baseColor()) = 0x4080ffu;
}
void
XGraphMathTool::highlight(bool state, XQGraph *graphwidget) {
    m_highlight = state;
    Snapshot shot( *this);
    updateOnScreenObjects(shot, graphwidget, {});
}

XGraph1DMathTool::XGraph1DMathTool(const char *name, bool runtime, Transaction &tr_meas,
    const shared_ptr<XScalarEntryList> &entries, const shared_ptr<XDriver> &driver,
    const shared_ptr<XPlot> &plot) :
    XGraphMathTool(name, runtime, ref(tr_meas), entries, driver, plot),
    m_begin(create<XDoubleNode>("Begin", false)),
    m_end(create<XDoubleNode>("End", false)) {

}
XGraph1DMathTool::~XGraph1DMathTool() {
}
XGraph2DMathTool::XGraph2DMathTool(const char *name, bool runtime, Transaction &tr_meas,
    const shared_ptr<XScalarEntryList> &entries, const shared_ptr<XDriver> &driver,
    const shared_ptr<XPlot> &plot) :
    XGraphMathTool(name, runtime, ref(tr_meas), entries, driver, plot),
    m_beginX(create<XDoubleNode>("BeginX", false)),
    m_beginY(create<XDoubleNode>("BeginY", false)),
    m_endX(create<XDoubleNode>("EndX", false)),
    m_endY(create<XDoubleNode>("EndY", false)) {

}

void
XGraphMathTool::updateOnScreenObjects(const Snapshot &shot, XQGraph *graphwidget, XString msg) {
    if( !shot[ *this].isUIEnabled())
        return;
    auto painter = graphwidget->painter().lock();
    if( !painter) {
        m_osos.clear();
        return;
    }
    for(auto &&oso: m_osos) {
        if(!oso->isValid(painter.get())) {
            m_osos.clear();
            break;
        }
        //painter unchanged unless the same address is recycled.
    }
    if(isHighLighted() != (bool)m_osoHighlight.lock()) {
        m_osos.clear();
    }
    if(m_osos.empty()) {
        m_osos = createAdditionalOnScreenObjects(painter);
    }
    updateAdditionalOnScreenObjects(shot, graphwidget, std::move(msg));
    graphwidget->update();
}
std::deque<shared_ptr<OnScreenObject>>
XGraph1DMathTool::createAdditionalOnScreenObjects(const shared_ptr<XQGraphPainter> &painter) {
    auto oso_rect = painter->createOnScreenObjectWeakly<OnXAxisRectObject>(OnScreenRectObject::Type::BorderLines);
    m_osoRect = oso_rect;
    auto oso_lbl = painter->createOnScreenObjectWeakly<OnXAxisTextObject>();
    m_osoLabel = oso_lbl;
    if(isHighLighted()) {
        auto oso_rect2 = painter->createOnScreenObjectWeakly<OnXAxisRectObject>(OnScreenRectObject::Type::Selection);
        m_osoHighlight = oso_rect2;
        return {oso_rect, oso_rect2, oso_lbl};
    }
    return {oso_rect, oso_lbl};
}
void
XGraph1DMathTool::updateAdditionalOnScreenObjects(const Snapshot &shot, XQGraph *graphwidget, XString msg) {
    if(auto plot = m_plot.lock()) {
        double bgx = shot[ *begin()];
        double edx = shot[ *end()];
        double bgy = 0.0;
        double edy = 1.0;
        if(auto oso = static_pointer_cast<OnXAxisRectObject>(m_osoRect.lock())) {
            oso->setBaseColor(shot[ *baseColor()]);
            oso->placeObject(plot, bgx, edx, bgy, edy, {0.0, 0.0, 0.01});
        }
        if(auto oso_rect = m_osoHighlight.lock()) {
            auto oso = static_pointer_cast<OnXAxisRectObject>(oso_rect);
            QColor c = (unsigned long)***graphwidget->graph()->titleColor();
            c.setAlphaF(0.25);
            oso->setBaseColor(c.rgba());
            oso->placeObject(plot, bgx, edx, bgy, edy, {0.0, 0.0, 0.02});
        }
        if(auto oso = static_pointer_cast<OnXAxisTextObject>(m_osoLabel.lock())) {
            oso->setBaseColor(shot[ *baseColor()]);
            oso->placeObject(plot, bgx, edx, bgy, edy, {0.01, 0.01, 0.01});
            oso->setAlignment(Qt::AlignTop | Qt::AlignLeft);
            oso->defaultFont();
            oso->drawTextAtPlacedPosition(getLabel()
                + (isHighLighted() ? "" : " " + msg), isHighLighted() ? +2 : -4);
        }
    }
}

XString
XGraph1DMathTool::getMenuLabel() const {
    Snapshot shot( *this);
    double bgx = shot[ *begin()];
    double edx = shot[ *end()];
    return getLabel() + formatString(" (%.4g)-(%.4g)", bgx, edx);
}
XString
XGraph2DMathTool::getMenuLabel() const {
    Snapshot shot( *this);
    double bgx = shot[ *beginX()];
    double bgy = shot[ *beginY()];
    double edx = shot[ *endX()];
    double edy = shot[ *endY()];
    return getLabel() + formatString(" (%.0f,%.0f)-(%.0f,%.0f)",bgx, bgy, edx, edy);
}
std::deque<shared_ptr<OnScreenObject>>
XGraph2DMathTool::createAdditionalOnScreenObjects(const shared_ptr<XQGraphPainter> &painter) {
    auto oso_rect = painter->createOnScreenObjectWeakly<OnPlotRectObject>(OnScreenRectObject::Type::AreaTool);
    m_osoRect = oso_rect;
    auto oso_lbl = painter->createOnScreenObjectWeakly<OnPlotTextObject>();
    m_osoLabel = oso_lbl;
    if(isHighLighted()) {
        auto oso_rect2 = painter->createOnScreenObjectWeakly<OnPlotRectObject>(OnScreenRectObject::Type::Selection);
        m_osoHighlight = oso_rect2;
        return {oso_rect, oso_rect2, oso_lbl};
    }
    return {oso_rect, oso_lbl};
}
void
XGraph2DMathTool::updateAdditionalOnScreenObjects(const Snapshot &shot, XQGraph *graphwidget, XString msg) {
    if(auto plot = m_plot.lock()) {
        double bgx = shot[ *beginX()];
        double bgy = shot[ *beginY()];
        double edx = shot[ *endX()];
        double edy = shot[ *endY()];
        XGraph::ValPoint corners[4] = {{bgx, bgy}, {edx, bgy}, {edx, edy}, {bgx, edy}};
        if(auto oso = static_pointer_cast<OnPlotRectObject>(m_osoRect.lock())) {
            oso->setBaseColor(shot[ *baseColor()]);
            oso->placeObject(plot, corners, {0.0, 0.0, 0.01});
        }
        if(auto oso_rect = m_osoHighlight.lock()) {
            auto oso = static_pointer_cast<OnPlotRectObject>(oso_rect);
            QColor c = (unsigned long)***graphwidget->graph()->titleColor();
            c.setAlphaF(0.25);
            oso->setBaseColor(c.rgba());
            oso->placeObject(plot, corners, {0.0, 0.0, 0.02});
        }
        if(auto oso = static_pointer_cast<OnPlotTextObject>(m_osoLabel.lock())) {
            oso->setBaseColor(shot[ *baseColor()]);
            oso->placeObject(plot, corners, {0.01, 0.01, 0.01});
            oso->setAlignment(Qt::AlignTop | Qt::AlignLeft);
            oso->defaultFont();
            oso->drawTextAtPlacedPosition(getLabel()
                + (isHighLighted() ? "" : " " + msg), isHighLighted() ? +5 : +1);
        }
    }
}

template <class X, class XQC>
XGraphMathToolList<X, XQC>::XGraphMathToolList(const char *name, bool runtime,
        const shared_ptr<XMeasure> &meas, const shared_ptr<XDriver> &driver, const shared_ptr<XPlot> &plot) :
      XCustomTypeListNode<X>(name, runtime),
      m_measure(meas), m_driver(driver), m_plot(plot) {
      this->iterate_commit([=](Transaction &tr){
          m_lsnRelease = tr[ *this].onRelease().connectWeakly(this->shared_from_this(), &XGraphMathToolList::onRelease);
      });
}
template <class X, class XQC>
XGraphMathToolList<X, XQC>::~XGraphMathToolList() {
    this->releaseAll();
}
template <class X, class XQC>
void
XGraphMathToolList<X, XQC>::onRelease(const Snapshot &, const XListNodeBase::Payload::ReleaseEvent &e) {
    m_measure.lock()->iterate_commit([&](Transaction &tr){
        if( !static_pointer_cast<X>(e.released)->releaseEntries(tr))
            return;
    });
}

template class XGraphMathToolList<XGraph1DMathTool, XQGraph1DMathToolConnector>;
template class XGraphMathToolList<XGraph2DMathTool, XQGraph2DMathToolConnector>;

shared_ptr<XNode>
XGraph1DMathToolList::createByTypename(const XString &type, const XString& name) {
    shared_ptr<XMeasure> meas(m_measure.lock());
    shared_ptr<XNode> ptr;
    auto plot = m_plot.lock();

    std::vector<std::string> name_split;
    ssize_t pos = 0;
    for(;;) {
        auto npos = name.find_first_of(";", pos);
        name_split.push_back(this->getName() + "-" + name.substr(pos, npos - pos));
        if(npos == std::string::npos)
            break;
        pos = npos + 1;
    }
    meas->iterate_commit_if([=, &ptr](Transaction &tr)->bool{
        ptr = creator(type)
            (name.c_str(), false, ref(tr), meas->scalarEntries(), m_driver.lock(), plot,
            name_split);
        if(ptr)
            if( !this->insert(tr, ptr))
                return false;
        return true;
    });
    return ptr;
}

shared_ptr<XNode>
XGraph2DMathToolList::createByTypename(const XString &type, const XString& name) {
    shared_ptr<XMeasure> meas(m_measure.lock());
    shared_ptr<XNode> ptr;
    auto plot = m_plot.lock();

    std::vector<std::string> name_split;
    ssize_t pos = 0;
    for(;;) {
        auto npos = name.find_first_of(";", pos);
        name_split.push_back(this->getName() + "-" + name.substr(pos, npos - pos));
        if(npos == std::string::npos)
            break;
        pos = npos + 1;
    }
    meas->iterate_commit_if([=, &ptr](Transaction &tr)->bool{
        ptr = creator(type)
            (name.c_str(), false, ref(tr), meas->scalarEntries(), m_driver.lock(), plot,
            name_split);
        if(ptr)
            if( !this->insert(tr, ptr))
                return false;
        return true;
    });
    return ptr;
}

XGraph1DMathToolList::XGraph1DMathToolList(const char *name, bool runtime,
                         const shared_ptr<XMeasure> &meas, const shared_ptr<XDriver> &driver, const shared_ptr<XPlot> &plot) :
    XGraphMathToolList<XGraph1DMathTool, XQGraph1DMathToolConnector>(name, runtime, meas, driver, plot) {
}

void
XGraph1DMathToolList::update(Transaction &tr, XQGraph *graphwidget,
    cv_iterator xbegin, cv_iterator xend, cv_iterator ybegin, cv_iterator yend) {
    if(tr.size(shared_from_this()) &&
            (std::distance(xbegin, xend) > 0) &&
            (std::distance(ybegin, yend) > 0)) {
        for(auto &x: *tr.list(shared_from_this())) {
            auto tool = static_pointer_cast<XGraph1DMathTool>(x);
            //limits to selected region. xmin <= x <= xmax.
            double xmin = tr[ *tool->begin()];
            double xmax = tr[ *tool->end()];
            cv_iterator xbegin_lim = xbegin;
            for(; xbegin_lim != xend; ++xbegin_lim) {
                if( *xbegin_lim >= xmin)
                    break;
            }
            cv_iterator xend_lim = xbegin;
            for(; xend_lim != xend; ++xend_lim) {
                if( *xend_lim > xmax)
                    break;
            }
            cv_iterator ybegin_lim = ybegin + (xbegin_lim - xbegin);
            cv_iterator yend_lim = ybegin + (xend_lim - xbegin);
            tool->update(tr, graphwidget, xbegin_lim, xend_lim, ybegin_lim, yend_lim);
        }
    }
}
XGraph2DMathToolList::XGraph2DMathToolList(const char *name, bool runtime,
    const shared_ptr<XMeasure> &meas, const shared_ptr<XDriver> &driver,
    const shared_ptr<XPlot> &plot) :
    XGraphMathToolList<XGraph2DMathTool, XQGraph2DMathToolConnector>(name, runtime, meas, driver, plot) {
}
void
XGraph2DMathToolList::update(Transaction &tr, XQGraph *graphwidget,
    const uint32_t *leftupper, unsigned int width,
    unsigned int stride, unsigned int numlines, double coefficient, double offset) {
    if(tr.size(shared_from_this())) {
        for(auto &x: *tr.list(shared_from_this())) {
            auto tool = static_pointer_cast<XGraph2DMathTool>(x);
            //limits to selected region.
            double xmin = tr[ *tool->beginX()];
            double xmax = tr[ *tool->endX()];
            double ymin = tr[ *tool->beginY()];
            double ymax = tr[ *tool->endY()];
            ssize_t x0 = lrint(xmin);
            ssize_t y0 = lrint(ymin); //do not mirror y
//            ssize_t y0 = lrint(numlines - 1 - ymax); //mirror y
            ssize_t x1 = std::min((long)width - 1, lrint(xmax));
            ssize_t y1 = lrint(ymax); //do not mirror y
//            ssize_t y1 = lrint(numlines - 1 - ymin); //mirror y
            y1 = std::min(y1, (ssize_t)numlines - 1); //limits y
            if((x0 >= 0) && (y0 >= 0) && (x0 < stride) && (y0 < numlines) && (x1 >= x0)) {
                tool->update(tr, graphwidget, leftupper + x0 + y0 * stride, x1 - x0 + 1,
                    stride, y1 - y0 + 1, coefficient, offset);
            }
        }
    }
}

void
XGraph1DMathToolList::onAxisSelectedByToolForCreate(const Snapshot &shot,
    const std::tuple<XString, XGraph::VFloat, XGraph::VFloat, XQGraph*>& res) {
    auto label = std::get<0>(res);
    auto src = std::get<1>(res);
    auto dst = std::get<2>(res);
    auto widget = std::get<3>(res);
    unsigned int idx = 0;
    for(auto &&tlabel: typelabels()) {
        if(tlabel == label)
            break;
        idx++;
    }
    Snapshot shot_this( *this);
    shared_ptr<XNode> node;
    try {
        node = createByTypename(typenames().at(idx), formatString("%s%u", label.c_str(), shot_this.size()));
    }
#ifdef USE_PYBIND11
    catch (pybind11::error_already_set& e) {
        pybind11::gil_scoped_acquire guard;
        gErrPrint(i18n("Python error: ") + e.what());
    }
#endif
    catch (std::runtime_error &e) {
        gErrPrint(std::string("Python KAME binding error: ") + e.what());
    }
    catch (...) {
        gErrPrint(std::string("Unknown python error."));
    }
    if( !node) return;
    auto tool = static_pointer_cast<XGraph1DMathTool>(node);
    Snapshot shot_tool = tool->iterate_commit([&](Transaction &tr){
        if(src > dst)
            std::swap(src, dst);
        tr[ *tool->begin()] = src;
        tr[ *tool->end()] = dst;
        tr[ *tool->baseColor()] = m_basecolor;
        tr[ *tool].setUIEnabled(shot_this[ *this].isUIEnabled());
    });
    tool->highlight(false, widget);
}

void
XGraph2DMathToolList::onPlaneSelectedByToolForCreate(const Snapshot &shot,
    const std::tuple<XString, XGraph::ValPoint, XGraph::ValPoint, XQGraph* > &res) {
    auto label = std::get<0>(res);
    auto src = std::get<1>(res);
    auto dst = std::get<2>(res);
    auto widget = std::get<3>(res);
    unsigned int idx = 0;
    for(auto &&tlabel: typelabels()) {
        if(tlabel == label)
            break;
        idx++;
    }
    Snapshot shot_this( *this);
    shared_ptr<XNode> node;
    try {
        node = createByTypename(typenames().at(idx), formatString("%s%u", label.c_str(), shot_this.size()));
    }
#ifdef USE_PYBIND11
    catch (pybind11::error_already_set& e) {
        pybind11::gil_scoped_acquire guard;
        gErrPrint(i18n("Python error: ") + e.what());
    }
#endif
    catch (std::runtime_error &e) {
        gErrPrint(std::string("Python KAME binding error: ") + e.what());
    }
    catch (...) {
        gErrPrint(std::string("Unknown python error."));
    }
    if( !node) return;
    auto tool = static_pointer_cast<XGraph2DMathTool>(node);
    Snapshot shot_tool = tool->iterate_commit([&](Transaction &tr){
        if(src.x > dst.x)
            std::swap(src.x, dst.x);
        if(src.y > dst.y)
            std::swap(src.y, dst.y);
        tr[ *tool->beginX()] = src.x;
        tr[ *tool->endX()] = dst.x;
        tr[ *tool->beginY()] = src.y;
        tr[ *tool->endY()] = dst.y;
        tr[ *tool->baseColor()] = m_basecolor;
        tr[ *tool].setUIEnabled(shot_this[ *this].isUIEnabled());
    });
    tool->highlight(false, widget);
}

void
XGraph1DMathToolList::onAxisSelectedByToolForReselect(const Snapshot &shot,
    const std::tuple<XString, XGraph::VFloat, XGraph::VFloat, XQGraph*>& res) {
    auto label = std::get<0>(res);
    auto src = std::get<1>(res);
    auto dst = std::get<2>(res);
    auto widget = std::get<3>(res);
    unsigned int idx = 0;
    Snapshot shot_this( *this);
    auto list = shot_this.list();
    if(list->size())
        for(auto x: *list) {
            if(x->getLabel() == label)
                break;
            idx++;
        }
    auto tool = static_pointer_cast<XGraph1DMathTool>(list->at(idx));
    Snapshot shot_tool = tool->iterate_commit([&](Transaction &tr){
        if(src > dst)
            std::swap(src, dst);
        tr[ *tool->begin()] = src;
        tr[ *tool->end()] = dst;
    });
    tool->highlight(false, widget);
}

void
XGraph2DMathToolList::onPlaneSelectedByToolForReselect(const Snapshot &shot,
    const std::tuple<XString, XGraph::ValPoint, XGraph::ValPoint, XQGraph* > &res) {
    auto label = std::get<0>(res);
    auto src = std::get<1>(res);
    auto dst = std::get<2>(res);
    auto widget = std::get<3>(res);
    unsigned int idx = 0;
    Snapshot shot_this( *this);
    auto list = shot_this.list();
    if(list->size())
        for(auto x: *list) {
            if(x->getLabel() == label)
                break;
            idx++;
        }
    auto tool = static_pointer_cast<XGraph2DMathTool>(list->at(idx));
    Snapshot shot_tool = tool->iterate_commit([&](Transaction &tr){
        if(src.x > dst.x)
            std::swap(src.x, dst.x);
        if(src.y > dst.y)
            std::swap(src.y, dst.y);
        tr[ *tool->beginX()] = src.x;
        tr[ *tool->endX()] = dst.x;
        tr[ *tool->beginY()] = src.y;
        tr[ *tool->endY()] = dst.y;
    });
    tool->highlight(false, widget);
}

