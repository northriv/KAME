/****************************************************************************
** Form interface generated from reading ui file '../../../../../kame/users/dcsource/forms/dcsourceform.ui'
**
** Created: 木  3 2 16:39:56 2006
**      by: The User Interface Compiler ($Id: dcsourceform.h,v 1.1.2.1 2006/03/02 09:20:47 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef FRMDCSOURCE_H
#define FRMDCSOURCE_H

#include <qvariant.h>
#include <qmainwindow.h>

class QVBoxLayout;
class QHBoxLayout;
class QGridLayout;
class QSpacerItem;
class QAction;
class QActionGroup;
class QToolBar;
class QPopupMenu;
class QLabel;
class QComboBox;
class QCheckBox;
class QLineEdit;

class FrmDCSource : public QMainWindow
{
    Q_OBJECT

public:
    FrmDCSource( QWidget* parent = 0, const char* name = 0, WFlags fl = WType_TopLevel );
    ~FrmDCSource();

    QLabel* textLabel4;
    QComboBox* m_cmbFunction;
    QCheckBox* m_ckbOutput;
    QLabel* textLabel3;
    QLineEdit* m_edValue;

protected:
    QGridLayout* FrmDCSourceLayout;
    QSpacerItem* spacer3;
    QSpacerItem* spacer2;
    QVBoxLayout* layout4;
    QVBoxLayout* layout3;

protected slots:
    virtual void languageChange();

};

#endif // FRMDCSOURCE_H
