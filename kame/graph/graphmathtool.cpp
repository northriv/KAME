/***************************************************************************
        Copyright (C) 2002-2023 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
 ***************************************************************************/
#include "graphmathtool.h"
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

REGISTER_TYPE(XGraph2DMathToolList, Graph2DMathToolSum, "Sum");
REGISTER_TYPE(XGraph2DMathToolList, Graph2DMathToolAverage, "Average");

XGraph1DMathTool::XGraph1DMathTool(const char *name, bool runtime, Transaction &tr_meas,
    const shared_ptr<XScalarEntryList> &entries, const shared_ptr<XDriver> &driver,
    const shared_ptr<XPlot> &plot) :
    XNode(name, runtime),
    m_begin(create<XDoubleNode>("Begin", false)),
    m_end(create<XDoubleNode>("End", false)),
    m_entries(entries),
    m_baseColor(create<XHexNode>("BaseColor", false)),
    m_plot(plot) {
}
XGraph2DMathTool::XGraph2DMathTool(const char *name, bool runtime, Transaction &tr_meas,
    const shared_ptr<XScalarEntryList> &entries, const shared_ptr<XDriver> &driver,
    const shared_ptr<XPlot> &plot) :
    XNode(name, runtime),
    m_beginX(create<XDoubleNode>("BeginX", false)),
    m_beginY(create<XDoubleNode>("BeginY", false)),
    m_endX(create<XDoubleNode>("EndX", false)),
    m_endY(create<XDoubleNode>("EndY", false)),
    m_entries(entries),
    m_baseColor(create<XHexNode>("BaseColor", false)),
    m_plot(plot) {
}
XGraph2DMathTool::~XGraph2DMathTool() {
}
void
XGraph1DMathTool::updateOnScreenObjects(const Snapshot &shot, XQGraph *graphwidget) {
    auto painter = graphwidget->painter().lock();
    if( !painter) {
        m_oso.reset();
        return;
    }
     //painter unchanged unless the same address is recycled.
    if( !m_oso || !m_oso->isValid(painter.get())) {
        m_oso = painter->createOnScreenObjectWeakly<OnXAxisRectObject>(OnScreenRectObject::Type::BorderLines);
    }

    if(auto plot = m_plot.lock()) {
        double bgx = shot[ *begin()];
        double edx = shot[ *end()];
        double bgy = 0.0;
        double edy = 1.0;
        auto oso = static_pointer_cast<OnXAxisRectObject>(m_oso);
        oso->setBaseColor(shot[ *m_baseColor]);
        oso->placeObject(plot, bgx, edx, bgy, edy, {0.0, 0.0, 0.01});
    }
}
void
XGraph2DMathTool::updateOnScreenObjects(const Snapshot &shot, XQGraph *graphwidget) {
    auto painter = graphwidget->painter().lock();
    if( !painter) {
        m_oso.reset();
        return;
    }
    //painter unchanged unless the same address is recycled.
    if( !m_oso || !m_oso->isValid(painter.get())) {
        m_oso = painter->createOnScreenObjectWeakly<OnPlotRectObject>(OnScreenRectObject::Type::AreaTool);
    }

    if(auto plot = m_plot.lock()) {
        double bgx = shot[ *beginX()];
        double bgy = shot[ *beginY()];
        double edx = shot[ *endX()];
        double edy = shot[ *endY()];
        XGraph::ValPoint corners[4] = {{bgx, bgy}, {edx, bgy}, {edx, edy}, {bgx, edy}};
        auto oso = static_pointer_cast<OnPlotRectObject>(m_oso);
        oso->setBaseColor(shot[ *m_baseColor]);
        oso->placeObject(plot, corners, {0.0, 0.0, 0.01});
    }
}

XGraph1DMathToolList::XGraph1DMathToolList(const char *name, bool runtime,
                         const shared_ptr<XMeasure> &meas, const shared_ptr<XDriver> &driver, const shared_ptr<XPlot> &plot) :
    XCustomTypeListNode<XGraph1DMathTool>(name, runtime),
    m_measure(meas), m_driver(driver), m_plot(plot) {
}

shared_ptr<XNode>
XGraph1DMathToolList::createByTypename(const XString &type, const XString& name) {
    shared_ptr<XMeasure> meas(m_measure.lock());
    shared_ptr<XNode> ptr;
    auto plot = m_plot.lock();
    meas->iterate_commit_if([=, &ptr](Transaction &tr)->bool{
        ptr = creator(type)
            (name.c_str(), false, ref(tr), meas->scalarEntries(), m_driver.lock(), plot);
        if(ptr)
            if( !insert(tr, ptr))
                return false;
        return true;
    });
    return ptr;
}
void
XGraph1DMathToolList::update(Transaction &tr, XQGraph *graphwidget,
    cv_iterator xbegin, cv_iterator xend, cv_iterator ybegin, cv_iterator yend) {
    if(tr.size(shared_from_this())) {
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
    XCustomTypeListNode<XGraph2DMathTool>(name, runtime),
    m_measure(meas), m_driver(driver), m_plot(plot) {
}

shared_ptr<XNode>
XGraph2DMathToolList::createByTypename(const XString &type, const XString& name) {
    shared_ptr<XMeasure> meas(m_measure.lock());
    shared_ptr<XNode> ptr;
    auto plot = m_plot.lock();
    meas->iterate_commit_if([=, &ptr](Transaction &tr)->bool{
        ptr = creator(type)
            (name.c_str(), false, ref(tr), meas->scalarEntries(), m_driver.lock(), plot);
        if(ptr)
            if( !insert(tr, ptr))
                return false;
        return true;
    });
    return ptr;
}
void
XGraph2DMathToolList::update(Transaction &tr, XQGraph *graphwidget,
    const uint32_t *leftupper, unsigned int width,
    unsigned int stride, unsigned int numlines, double coefficient) {
    if(tr.size(shared_from_this())) {
        for(auto &x: *tr.list(shared_from_this())) {
            auto tool = static_pointer_cast<XGraph2DMathTool>(x);
            //limits to selected region.
            double xmin = tr[ *tool->beginX()];
            double xmax = tr[ *tool->endX()];
            double ymin = tr[ *tool->beginY()];
            double ymax = tr[ *tool->endY()];
            ssize_t x0 = lrint(xmin);
//            ssize_t y0 = lrint(ymin);
            ssize_t y0 = lrint(numlines - 1 - ymax); //mirror y
            ssize_t x1 = std::min((long)width - 1, lrint(xmax));
//            ssize_t y1 = std::min((long)stride - 1, lrint(ymax));
            ssize_t y1 = lrint(numlines - 1 - ymin); //mirror y
            if((x0 >= 0) && (y0 >= 0) && (x0 < stride) && (y0 < numlines) && (x1 >= x0)) {
                tool->update(tr, graphwidget, leftupper + x0 + y0 * stride, x1 - x0 + 1,
                    stride, y1 - y0 + 1, coefficient);
            }
        }
    }
}

void
XGraph1DMathToolList::onAxisSelectedByTool(const Snapshot &shot,
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
    auto node = createByTypename(typenames().at(idx), formatString("%s-%s (%.4g)-(%.4g)", getLabel().c_str(),
        label.c_str(), src, dst));
    auto tool = static_pointer_cast<XGraph1DMathTool>(node);
    Snapshot shot_tool = tool->iterate_commit([&](Transaction &tr){
        if(src > dst)
            std::swap(src, dst);
        tr[ *tool->begin()] = src;
        tr[ *tool->end()] = dst;
        tr[ *tool->baseColor()] = m_basecolor;
    });
    tool->updateOnScreenObjects(shot_tool, widget);
}

void
XGraph2DMathToolList::onPlaneSelectedByTool(const Snapshot &shot,
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
    auto node = createByTypename(typenames().at(idx), formatString("%s-%s (%.0f,%.0f)-(%.0f,%.0f)", getLabel().c_str(),
        label.c_str(), src.x, src.y, dst.x, dst.y));
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
    });
    tool->updateOnScreenObjects(shot_tool, widget);
}
