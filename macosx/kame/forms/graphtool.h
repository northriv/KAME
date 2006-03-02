/****************************************************************************
** Form interface generated from reading ui file '../../../kame/forms/graphtool.ui'
**
** Created: 木  3 2 16:22:58 2006
**      by: The User Interface Compiler ($Id: graphtool.h,v 1.1.2.1 2006/03/02 09:19:18 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef FRMGRAPHLIST_H
#define FRMGRAPHLIST_H

#include <qvariant.h>
#include <qwidget.h>

class QVBoxLayout;
class QHBoxLayout;
class QGridLayout;
class QSpacerItem;
class QPushButton;
class QTable;

class FrmGraphList : public QWidget
{
    Q_OBJECT

public:
    FrmGraphList( QWidget* parent = 0, const char* name = 0, WFlags fl = 0 );
    ~FrmGraphList();

    QPushButton* btnNewGraph;
    QPushButton* btnDeleteGraph;
    QTable* tblGraphs;

protected:
    QGridLayout* FrmGraphListLayout;
    QSpacerItem* spacer3;

protected slots:
    virtual void languageChange();

};

#endif // FRMGRAPHLIST_H
