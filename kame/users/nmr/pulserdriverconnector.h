#ifndef pulserdriverconnectorH
#define pulserdriverconnectorH

#include "xnodeconnector.h"
#include "pulserdriver.h"
//---------------------------------------------------------------------------

class QTable;
class XQGraph;
class XXYPlot;
class XGraph;

class XQPulserDriverConnector : public XQConnector
{
  Q_OBJECT
  XQCON_OBJECT
 protected:
  XQPulserDriverConnector(const shared_ptr<XPulser> &node, QTable *item, XQGraph *graph);
 public:
  virtual ~XQPulserDriverConnector();

 protected slots:
   void clicked ( int row, int col, int button, const QPoint & mousePos );
   void selectionChanged ();
 private:

  void updateGraph(bool checkselection);
  
  shared_ptr<XListener> m_lsnOnPulseChanged;
  void onPulseChanged(const shared_ptr<XDriver> &);
  
  QTable *const m_pTable;
  const weak_ptr<XPulser> m_pulser;

  const shared_ptr<XGraph> m_graph;
  shared_ptr<XXYPlot> m_barPlot;
  std::deque<shared_ptr<XXYPlot> > m_plots;
};

#endif
