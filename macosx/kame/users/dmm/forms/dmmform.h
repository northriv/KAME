/****************************************************************************
** Form interface generated from reading ui file '../../../../../kame/users/dmm/forms/dmmform.ui'
**
** Created: 木  3 2 16:36:12 2006
**      by: The User Interface Compiler ($Id: dmmform.h,v 1.1.2.1 2006/03/02 09:20:44 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef FRMDMM_H
#define FRMDMM_H

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
class QSpinBox;

class FrmDMM : public QMainWindow
{
    Q_OBJECT

public:
    FrmDMM( QWidget* parent = 0, const char* name = 0, WFlags fl = WType_TopLevel );
    ~FrmDMM();

    QLabel* textLabel4;
    QComboBox* m_cmbFunction;
    QLabel* textLabel3;
    QSpinBox* m_numWait;

protected:
    QGridLayout* FrmDMMLayout;
    QSpacerItem* spacer2;
    QSpacerItem* spacer3;
    QVBoxLayout* layout4;
    QVBoxLayout* layout23;

protected slots:
    virtual void languageChange();

};

#endif // FRMDMM_H
