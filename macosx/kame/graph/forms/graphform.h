/****************************************************************************
** Form interface generated from reading ui file '../../../../kame/graph/forms/graphform.ui'
**
** Created: 木  3 2 16:25:22 2006
**      by: The User Interface Compiler ($Id: graphform.h,v 1.1.2.1 2006/03/02 09:19:26 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef FRMGRAPH_H
#define FRMGRAPH_H

#include <qvariant.h>
#include <qpixmap.h>
#include <qmainwindow.h>

class QVBoxLayout;
class QHBoxLayout;
class QGridLayout;
class QSpacerItem;
class QAction;
class QActionGroup;
class QToolBar;
class QPopupMenu;
class XQGraph;

class FrmGraph : public QMainWindow
{
    Q_OBJECT

public:
    FrmGraph( QWidget* parent = 0, const char* name = 0, WFlags fl = WType_TopLevel );
    ~FrmGraph();

    XQGraph* m_graphwidget;

protected:
    QGridLayout* FrmGraphLayout;

protected slots:
    virtual void languageChange();

private:
    QPixmap image0;
    QPixmap image1;

};

#endif // FRMGRAPH_H
