/****************************************************************************
** Form interface generated from reading ui file '../../../../../kame/users/nmr/forms/nmrpllform.ui'
**
** Created: 土  1 7 03:31:33 2006
**      by: The User Interface Compiler ($Id: nmrpllform.h,v 1.1 2006/02/01 18:43:54 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef FRMNMRPLL_H
#define FRMNMRPLL_H

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
class QCheckBox;
class QLabel;
class QLineEdit;

class frmNMRPLL : public QMainWindow
{
    Q_OBJECT

public:
    frmNMRPLL( QWidget* parent = 0, const char* name = 0, WFlags fl = WType_TopLevel );
    ~frmNMRPLL();

    QCheckBox* ckbControl;
    QLabel* textLabel5_3_2_4;
    QLineEdit* edWait4SG;
    QLabel* textLabel2_3_2_3_2_3;
    QLabel* textLabel5_3_2_4_2;
    QLineEdit* eddphidf;
    QLabel* textLabel2_3_2_3_2_3_2;

protected:
    QGridLayout* frmNMRPLLLayout;
    QHBoxLayout* layout59_3_2_4;
    QHBoxLayout* layout1_3_2_3_2_4;
    QHBoxLayout* layout59_3_2_4_2;
    QHBoxLayout* layout1_3_2_3_2_4_2;

protected slots:
    virtual void languageChange();

};

#endif // FRMNMRPLL_H
