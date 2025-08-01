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
#ifndef ONSCREENOBJECT_H
#define ONSCREENOBJECT_H

#include "graph.h"
#include "graphwidget.h"

#include <Qt>

class XGraph1DMathTool;
class XGraph2DMathTool;
class X2DImagePlot;
class OnScreenObject {
public:
    OnScreenObject(XQGraphPainter* p) : m_painter(p) {}
    virtual ~OnScreenObject() {}
    //! draws in OpenGL.
    virtual void drawNative() = 0;
    //! draws by QPainter.
    virtual void drawByPainter(QPainter *) = 0;
    virtual void drawOffScreenMarker() {}

    unsigned int baseColor() const { return m_baseColor;}
    void setBaseColor(unsigned int basecolor) {m_baseColor = basecolor;}
    bool isValid(XQGraphPainter *currentPainter) const {return painter() == currentPainter;}
    virtual bool hasTexture() const {return false;}
protected:
    friend class XGraph1DMathTool;
    friend class XGraph2DMathTool;
    friend class X2DImagePlot;
    XQGraphPainter *painter() const {return m_painter;}//valid if painter is alive (visible)
private:
    XQGraphPainter *const m_painter;
    atomic<unsigned int> m_baseColor = 0x4080ffu;
};

template <class OSO>
class OnPlotObject : public OSO {
public:
    template <typename... Args>
    OnPlotObject(XQGraphPainter* p, Args&&... args) : OSO(p, std::forward<Args>(args)...) {}

    void placeObject(const shared_ptr<XPlot> &plot,
                     const XGraph::ValPoint corners[4],
                    XGraph::ScrPoint offset = {}) {
        XScopedLock<XMutex> lock( m_mutex);
        m_plot = plot;
        for(unsigned int i = 0; i < 4; ++i)
            m_corners[i] = corners[i];
        m_offset = offset;
    }

    virtual void drawNative() override;
    virtual void drawByPainter(QPainter *) override;
    virtual void drawOffScreenMarker() override;
private:
    void valToScreen();
    XMutex m_mutex;
    weak_ptr<XPlot> m_plot;
    XGraph::ValPoint m_corners[4];
    XGraph::ScrPoint m_offset;
};
template <class OSO, bool IsXAxis>
class OnAxisObject : public OSO {
public:
    template <typename... Args>
    OnAxisObject(XQGraphPainter* p, Args&&... args) : OSO(p, std::forward<Args>(args)...) {}

    void placeObject(const shared_ptr<XPlot> &plot,
                     const XGraph::VFloat &bg1, const XGraph::VFloat &ed1,
                     const XGraph::GFloat &bg2, const XGraph::GFloat &ed2,
                     XGraph::ScrPoint offset = {}) {
        XScopedLock<XMutex> lock( m_mutex);
        m_plot = plot;
        m_bg1 = bg1; m_ed1 = ed1; m_bg2 = bg2; m_ed2 = ed2;
        m_offset = offset;
    }

    virtual void drawNative() override;
    virtual void drawByPainter(QPainter *) override;
    virtual void drawOffScreenMarker() override;

    std::pair<XGraph::VFloat, XGraph::VFloat> axis1ValueRange() const {return {m_bg1, m_ed1};}
    std::pair<XGraph::GFloat, XGraph::GFloat> axis2GraphRange() const {return {m_bg2, m_ed2};}
    XGraph::ScrPoint offsetInScreen() const {return m_offset;}
protected:
    void toScreen();
    XMutex m_mutex;
    weak_ptr<XPlot> m_plot;
private:
    XGraph::VFloat m_bg1, m_ed1;
    XGraph::GFloat m_bg2, m_ed2;
    XGraph::ScrPoint m_offset;
};


class OnScreenObjectWithMarker : public OnScreenObject {
public:
    OnScreenObjectWithMarker(XQGraphPainter* p) : OnScreenObject(p) {}
    //draws objects/bounding box for GL_SELECT
    virtual void drawOffScreenMarker() override;
    enum class HowToEvade {Never, ByAscent, ByDescent, ToLeft, ToRight, Hide};
    void placeObject(const XGraph::ScrPoint &init_lefttop, const XGraph::ScrPoint &init_righttop,
        const XGraph::ScrPoint &init_rightbottom, const XGraph::ScrPoint &init_leftbottom,
        HowToEvade direction = HowToEvade::Never, XGraph::SFloat space = 0.0);
//    void evadeOnScreenObjects(const std::deque<std::weak_ptr<OnScreenObject>> &list, XGraph::SFloat space);
//    static bool evadeMousePointer(const std::deque<std::weak_ptr<OnScreenObject>> &list);
    XGraph::ScrPoint &leftTop() {return m_leftTop;}
    XGraph::ScrPoint &rightTop() {return m_rightTop;}
    XGraph::ScrPoint &rightBottom() {return m_rightBottom;}
    XGraph::ScrPoint &leftBottom() {return m_leftBottom;}
protected:
    XGraph::ScrPoint m_leftTop, m_rightBottom, m_leftBottom, m_rightTop;
    XGraph::SFloat m_space;
    HowToEvade m_direction;
};

class OnScreenRectObject : public OnScreenObjectWithMarker {
public:
    enum class Type {Selection, AreaTool, BorderLines, Legends};
    OnScreenRectObject(XQGraphPainter* p, Type type) :
        OnScreenObjectWithMarker(p), m_type(type) {}
    //! draws in OpenGL.
    virtual void drawNative() override;
    //! draws by QPainter.
    virtual void drawByPainter(QPainter *) override {}
private:
    Type m_type;
};

template <bool IsXAxis>
class OnAxisFuncObject : public OnAxisObject<OnScreenRectObject, IsXAxis> {
public:
    OnAxisFuncObject(XQGraphPainter* p) :
        OnAxisObject<OnScreenRectObject, IsXAxis>(p, OnScreenRectObject::Type::AreaTool) {}
    //! draws in OpenGL.
    virtual void drawNative() override;
    //! draws by QPainter.
    virtual void drawByPainter(QPainter *) override {}
protected:
    virtual std::vector<XGraph::VFloat> func(const std::vector<XGraph::VFloat> &x,
                                             std::vector<XGraph::VFloat>&& prev_y) = 0;
private:
    std::vector<XGraph::VFloat> m_xvec, m_yvec;
};


using OnXAxisRectObject = OnAxisObject<OnScreenRectObject, true>;
using OnYAxisRectObject = OnAxisObject<OnScreenRectObject, false>;
using OnPlotRectObject = OnPlotObject<OnScreenRectObject>;

#include <QImage>
class OnScreenTexture : public OnScreenObjectWithMarker {
public:
   OnScreenTexture(XQGraphPainter *const item, GLuint tid, const shared_ptr<QImage> &image)
       : OnScreenObjectWithMarker(item), id(tid), qimage(image) {}
   virtual ~OnScreenTexture();
   //! update texture by new image.
   void repaint(const shared_ptr<QImage> &image);
   //! draws in OpenGL.
   virtual void drawNative() override;
   //! draws by QPainter.
   virtual void drawByPainter(QPainter *) override {}
   virtual bool hasTexture() const override {return true;}
private:
   const GLuint id = {};
   shared_ptr<QImage> qimage;
   static XMutex garbagemutex;
   static std::deque<GLuint> unusedIDs;
};

class OnScreenTextObject : public OnScreenObjectWithMarker {
public:
    OnScreenTextObject(XQGraphPainter* p);

    virtual void drawNative() override;
    virtual void drawByPainter(QPainter *) override;
    virtual void drawOffScreenMarker() override;

    void clear();
    //! using OnScreenObjectWithMarker::placeObject().
    void drawTextAtPlacedPosition(const XString &str, int sizehint = 0);
    //! not thread safe, be called within paintGL().
    void drawText(const XGraph::ScrPoint &p, const XString &str);
    void defaultFont();
    void setAlignment(int align) {
        m_curAlign = align;
    }
    //! \param start where text be aligned
    //! \param dir a direction where text be aligned
    //! \param width perp. to \a dir, restricting font size
    //! \return return 0 if succeeded
    int selectFont(const XString &str, const XGraph::ScrPoint &start,
                   const XGraph::ScrPoint &dir, const XGraph::ScrPoint &width, int sizehint = 0);
    virtual bool hasTexture() const override {return true;}

    //! in window coordinate, effective after drawText().
    double minXOfBB() const {return m_minX;}
    double minYOfBB() const {return m_minY;}
    double maxXOfBB() const {return m_maxX;}
    double maxYOfBB() const {return m_maxY;}
private:
    QString m_text;
    int m_curFontSize;
    int m_curAlign;
    double m_minX, m_minY, m_maxX, m_maxY;
    struct Text {
        XGraph::ScrPoint pos, corners[4];
        QRgb rgba;
        ssize_t strpos, length;
        int x, y;
    };
    std::vector<Text> m_textOverpaint; //stores text to be overpainted.
};

using OnXAxisTextObject = OnAxisObject<OnScreenTextObject, true>;
using OnYAxisTextObject = OnAxisObject<OnScreenTextObject, false>;
using OnPlotTextObject = OnPlotObject<OnScreenTextObject>;
#endif
