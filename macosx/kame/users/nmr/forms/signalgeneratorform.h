/****************************************************************************
** Form interface generated from reading ui file '../../../../../kame/users/nmr/forms/signalgeneratorform.ui'
**
** Created: 木  3 2 16:40:40 2006
**      by: The User Interface Compiler ($Id: signalgeneratorform.h,v 1.1.2.1 2006/03/02 09:19:17 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef FRMSG_H
#define FRMSG_H

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

class FrmSG : public QMainWindow
{
    Q_OBJECT

public:
    FrmSG( QWidget* parent = 0, const char* name = 0, WFlags fl = WType_TopLevel );
    ~FrmSG();

    QCheckBox* m_ckbAMON;
    QLabel* textLabel5_3;
    QLineEdit* m_edOLevel;
    QLabel* textLabel2_3_2_3;
    QCheckBox* m_ckbFMON;
    QLabel* textLabel5_3_2;
    QLineEdit* m_edFreq;
    QLabel* textLabel2_3_2_3_2;

protected:
    QGridLayout* FrmSGLayout;
    QSpacerItem* spacer20;
    QSpacerItem* spacer21;
    QGridLayout* layout41;
    QHBoxLayout* layout59_3;
    QHBoxLayout* layout1_3_2_3;
    QHBoxLayout* layout59_3_2;
    QHBoxLayout* layout1_3_2_3_2;

protected slots:
    virtual void languageChange();

};

#endif // FRMSG_H
