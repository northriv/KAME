#ifndef GRAPHWIDGET_H
#define GRAPHWIDGET_H

class XGraph;
class XQGraphPainter;

#include "support.h"
#include "xnodeconnector.h"
#include <qgl.h>

class XQGraph : public QGLWidget
{
  Q_OBJECT

 public:
  XQGraph( QWidget* parent = 0, const char* name = 0, WFlags fl = 0 );
  virtual ~XQGraph();
  //! register XGraph instance just after creating
  void setGraph(const shared_ptr<XGraph> &);

 protected:
  virtual void mousePressEvent ( QMouseEvent*);
  virtual void mouseReleaseEvent ( QMouseEvent*);
  virtual void mouseDoubleClickEvent ( QMouseEvent*);
  virtual void mouseMoveEvent ( QMouseEvent*);
  virtual void wheelEvent ( QWheelEvent *);
  virtual void showEvent ( QShowEvent * );
  virtual void hideEvent ( QHideEvent * );  
  //! openGL stuff
  virtual void initializeGL ();
  virtual void resizeGL ( int width, int height );
  virtual void paintGL ();
 private:  
  friend class XQGraphPainter;
  shared_ptr<XGraph> m_graph;
  shared_ptr<XQGraphPainter> m_painter;
  xqcon_ptr m_conDialog;
};

#endif // GRAPHWIDGET_H
