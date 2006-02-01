/****************************************************************************
** Form interface generated from reading ui file '../../../../kame/graph/forms/graphform.ui'
**
** Created: 水  2 1 03:43:03 2006
**      by: The User Interface Compiler ($Id: graphform.h,v 1.1 2006/02/01 18:44:05 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef FRMGRAPH_H
#define FRMGRAPH_H

#include <qvariant.h>
#include <qpixmap.h>
#include <qdialog.h>

class QVBoxLayout;
class QHBoxLayout;
class QGridLayout;
class QSpacerItem;
class XQGraph;

class FrmGraph : public QDialog
{
    Q_OBJECT

public:
    FrmGraph( QWidget* parent = 0, const char* name = 0, bool modal = FALSE, WFlags fl = 0 );
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
