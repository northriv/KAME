/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
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
class DECLSPEC_KAME OnScreenObject {
public:
    explicit OnScreenObject(XQGraphPainter* p) : m_painter(p) {}
    virtual ~OnScreenObject() {}
    //! draws in OpenGL.
    virtual void drawNative(bool colorpicking) = 0;
    //! draws by QPainter.
    virtual void drawByPainter(QPainter *) = 0;

    unsigned int baseColor() const { return m_baseColor;}
    void setBaseColor(unsigned int basecolor) {m_baseColor = basecolor;}
    bool isValid(XQGraphPainter *currentPainter) const {return !m_invalidated && (painter() == currentPainter);}
    void invalidate() {m_invalidated = true;}
    virtual bool hasTexture() const {return false;}
protected:
    friend class XGraph1DMathTool;
    friend class XGraph2DMathTool;
    friend class X2DImagePlot;
    XQGraphPainter *painter() const {return m_painter;}//valid if painter is alive (visible)
private:
    XQGraphPainter *const m_painter;
    atomic<unsigned int> m_baseColor = 0x4080ffu;
    atomic<bool> m_invalidated = false;
};

template <class OSO>
class DECLSPEC_KAME OnPlotObject : public OSO {
public:
    template <typename... Args>
    explicit OnPlotObject(XQGraphPainter* p, Args&&... args) : OSO(p, std::forward<Args>(args)...) {}

    void placeObject(const shared_ptr<XPlot> &plot,
                     const XGraph::ValPoint corners[4],
                    XGraph::ScrPoint offset = {}) {
        XScopedLock<XMutex> lock( m_mutex);
        m_plot = plot;
        for(unsigned int i = 0; i < 4; ++i)
            m_corners[i] = corners[i];
        m_offset = offset;
    }

    virtual void drawNative(bool colorpicking) override;
    virtual void drawByPainter(QPainter *) override;
private:
    void valToScreen();
    XMutex m_mutex;
    weak_ptr<XPlot> m_plot;
    XGraph::ValPoint m_corners[4];
    XGraph::ScrPoint m_offset;
};
template <class OSO, bool IsXAxis>
class DECLSPEC_KAME OnAxisObject : public OSO {
public:
    template <typename... Args>
    explicit OnAxisObject(XQGraphPainter* p, Args&&... args) : OSO(p, std::forward<Args>(args)...) {}

    void placeObject(const shared_ptr<XPlot> &plot,
                     const XGraph::VFloat &bg1, const XGraph::VFloat &ed1,
                     const XGraph::GFloat &bg2, const XGraph::GFloat &ed2,
                     XGraph::ScrPoint offset = {}) {
        XScopedLock<XMutex> lock( m_mutex);
        m_plot = plot;
        m_bg1 = bg1; m_ed1 = ed1; m_bg2 = bg2; m_ed2 = ed2;
        m_offset = offset;
    }

    virtual void drawNative(bool colorpicking) override;
    virtual void drawByPainter(QPainter *) override;

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


class DECLSPEC_KAME OnScreenPickableObject : public OnScreenObject {
public:
    //! param pickable_node XNode object responsible for OSO, eg. XGraph1DMathTool.
    OnScreenPickableObject(XQGraphPainter* p, const shared_ptr<XNode> &pickable_node) :
        OnScreenObject(p), m_pickableNode(pickable_node) {}
    enum class HowToEvade {Never, ByAscent, ByDescent, ToLeft, ToRight, Hide};
    void placeObject(const XGraph::ScrPoint &init_lefttop, const XGraph::ScrPoint &init_righttop,
        const XGraph::ScrPoint &init_rightbottom, const XGraph::ScrPoint &init_leftbottom,
        HowToEvade direction = HowToEvade::Never, XGraph::SFloat space = 0.0);
    //! Set evade direction for objects placed via drawText() rather than placeObject().
    void setEvadeDirection(HowToEvade dir, XGraph::SFloat space = 0.0) {
        m_direction = dir; m_space = space; }
    HowToEvade evadeDirection() const {return m_direction;}
    XGraph::SFloat evadeSpace() const {return m_space;}
    XGraph::ScrPoint &leftTop() {return m_leftTop;}
    XGraph::ScrPoint &rightTop() {return m_rightTop;}
    XGraph::ScrPoint &rightBottom() {return m_rightBottom;}
    XGraph::ScrPoint &leftBottom() {return m_leftBottom;}

    shared_ptr<XNode> pickableNode() const {return m_pickableNode.lock();}
protected:
    XGraph::ScrPoint m_leftTop, m_rightBottom, m_leftBottom, m_rightTop;
    XGraph::SFloat m_space = 0.0;
    HowToEvade m_direction = HowToEvade::Never;
private:
    weak_ptr<XNode> m_pickableNode;
};

class DECLSPEC_KAME OnScreenRectObject : public OnScreenPickableObject {
public:
    enum class Type {Selection, AreaTool, BorderLines, Legends, EllipseTool};
    OnScreenRectObject(XQGraphPainter* p, Type type, const shared_ptr<XNode> &pickable_node) :
        OnScreenPickableObject(p, pickable_node), m_type(type) {}
    //! draws in OpenGL.
    virtual void drawNative(bool colorpicking) override;
    //! draws by QPainter.
    virtual void drawByPainter(QPainter *) override {}
private:
    Type m_type;
};

template <bool IsXAxis>
class DECLSPEC_KAME OnAxisFuncObject : public OnAxisObject<OnScreenRectObject, IsXAxis> {
public:
    OnAxisFuncObject(XQGraphPainter* p, const shared_ptr<XNode> &pickable_node) :
        OnAxisObject<OnScreenRectObject, IsXAxis>(p, OnScreenRectObject::Type::AreaTool, pickable_node) {}
    //! draws in OpenGL.
    virtual void drawNative(bool colorpicking) override;
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
class DECLSPEC_KAME OnScreenTexture : public OnScreenPickableObject {
public:
   OnScreenTexture(XQGraphPainter *const item, GLuint tid, const shared_ptr<QImage> &image, const shared_ptr<XNode> &pickable_node)
       : OnScreenPickableObject(item, pickable_node), id(tid), qimage(image) {}
   virtual ~OnScreenTexture();
   //! update texture by new image.
   void repaint(const shared_ptr<QImage> &image);
   //! draws in OpenGL.
   virtual void drawNative(bool colorpicking) override;
   //! draws by QPainter.
   virtual void drawByPainter(QPainter *) override {}
   virtual bool hasTexture() const override {return true;}
private:
   const GLuint id = {};
   shared_ptr<QImage> qimage;
   static XMutex garbagemutex;
   static std::deque<GLuint> unusedIDs;
};

class DECLSPEC_KAME OnScreenTextObject : public OnScreenPickableObject {
public:
    OnScreenTextObject(XQGraphPainter* p, const shared_ptr<XNode> &pickable_node);

    virtual void drawNative(bool colorpicking) override;
    virtual void drawByPainter(QPainter *) override;

    void clear();
    //! using OnScreenPickableObject::placeObject().
    void drawTextAtPlacedPosition(const XString &str, int alignment, int sizehint = 0);
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
                   const XGraph::ScrPoint &dir, const XGraph::ScrPoint &width,
                   int sizehint = 0);
    virtual bool hasTexture() const override {return true;}

    //! Full axis-aligned bounding box in window coordinates, valid after drawText().
    double minXOfBB() const {return m_minX;}
    double minYOfBB() const {return m_minY;}
    double maxXOfBB() const {return m_maxX;}
    double maxYOfBB() const {return m_maxY;}
    //! Pixel shift applied by resolveOverlaps() and rendered in drawByPainter().
    void setShift(int dx, int dy) {m_shiftX = dx; m_shiftY = dy;}
    int shiftX() const {return m_shiftX;}
    int shiftY() const {return m_shiftY;}
    //! Hidden flag: set by resolveOverlaps() for HowToEvade::Hide objects that overlap.
    void setHidden(bool h) {m_hidden = h;}
    bool isHidden() const {return m_hidden;}
    //! Per-item access for resolveOverlaps() fine-grained hide (HowToEvade::Hide).
    size_t itemCount() const {return m_textOverpaint.size();}
    //! AABB of item \a idx in window coords (shift already included).
    void itemBB(size_t idx, double &x1, double &y1, double &x2, double &y2) const {
        const auto &t = m_textOverpaint[idx];
        x1 = t.bbx1 + m_shiftX; y1 = t.bby1 + m_shiftY;
        x2 = t.bbx2 + m_shiftX; y2 = t.bby2 + m_shiftY;
    }
    void hideItem(size_t idx) {m_textOverpaint[idx].item_hidden = true;}
private:
    atomic_shared_ptr<std::tuple<int, int, XString>> m_textThreadSafe;
    QString m_text;
    int m_curFontSize;
    int m_curAlign;
    double m_minX, m_minY, m_maxX, m_maxY;
    int m_shiftX = 0, m_shiftY = 0;
    bool m_hidden = false;
    struct Text {
        XGraph::ScrPoint pos, corners[4];
        QRgb rgba;
        ssize_t strpos, length;
        int x, y;
        int bbx1 = 0, bby1 = 0, bbx2 = 0, bby2 = 0; //!< window-coord AABB of this item
        bool item_hidden = false;
    };
    std::vector<Text> m_textOverpaint; //stores text to be overpainted.
};

using OnXAxisTextObject = OnAxisObject<OnScreenTextObject, true>;
using OnYAxisTextObject = OnAxisObject<OnScreenTextObject, false>;
using OnPlotTextObject = OnPlotObject<OnScreenTextObject>;

//! Draws a filled highlight for masked pixels within a bounding rectangle in val-space.
//! Renders horizontal spans from the mask as quads, interpolating screen positions.
class DECLSPEC_KAME OnPlotMaskObject : public OnScreenPickableObject {
public:
    OnPlotMaskObject(XQGraphPainter* p, const shared_ptr<XNode> &pickable_node)
        : OnScreenPickableObject(p, pickable_node) {}

    //! Set the mask and bounding rectangle.
    //! \a corners: 4 val-space corners {(bgx,bgy),(edx,bgy),(edx,edy),(bgx,edy)}.
    //! \a mask: width*height uint8 bitmap (row-major, 1=included). Empty = full rect.
    //! \a width, \a height: mask dimensions.
    void setMask(const shared_ptr<XPlot> &plot,
                 const XGraph::ValPoint corners[4],
                 const shared_ptr<std::vector<uint8_t>> &mask,
                 unsigned int width, unsigned int height,
                 XGraph::ScrPoint offset = {});

    void setHighlighted(bool h) { m_highlighted = h; }

    virtual void drawNative(bool colorpicking) override;
    virtual void drawByPainter(QPainter *) override {}
private:
    void drawContourLines(const XGraph::ScrPoint s[4], bool colorpicking);
    void drawFilledTexture(const XGraph::ScrPoint s[4], bool colorpicking);

    XMutex m_mutex;
    weak_ptr<XPlot> m_plot;
    XGraph::ValPoint m_corners[4];
    XGraph::ScrPoint m_offset;
    shared_ptr<std::vector<uint8_t>> m_mask;
    unsigned int m_width = 0, m_height = 0;
    bool m_highlighted = false;
};
#endif
