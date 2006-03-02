/****************************************************************************
** Form interface generated from reading ui file '../../../../../kame/users/tempcontrol/forms/tempcontrolform.ui'
**
** Created: 木  3 2 16:37:24 2006
**      by: The User Interface Compiler ($Id: tempcontrolform.h,v 1.1.2.1 2006/03/02 09:19:46 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef FRMTEMPCONTROL_H
#define FRMTEMPCONTROL_H

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
class QLCDNumber;
class QComboBox;
class QGroupBox;
class QLineEdit;

class FrmTempControl : public QMainWindow
{
    Q_OBJECT

public:
    FrmTempControl( QWidget* parent = 0, const char* name = 0, WFlags fl = WType_TopLevel );
    ~FrmTempControl();

    QLabel* textLabel5_2;
    QLCDNumber* m_lcdSourceTemp;
    QLabel* textLabel9_2;
    QLabel* textLabel17;
    QComboBox* m_cmbHeaterMode;
    QGroupBox* groupBox1;
    QLabel* textLabel19;
    QComboBox* m_cmbSetupChannel;
    QLabel* textLabel20;
    QComboBox* m_cmbExcitation;
    QLabel* textLabel21;
    QComboBox* m_cmbThermometer;
    QLabel* textLabel18_2_2;
    QLineEdit* m_edP;
    QLabel* textLabel18_2_2_2;
    QLineEdit* m_edI;
    QLabel* textLabel18_2_2_3;
    QLineEdit* m_edD;
    QLabel* textLabel16;
    QComboBox* m_cmbSourceChannel;
    QLabel* textLabel5;
    QLCDNumber* m_lcdHeater;
    QLabel* textLabel18;
    QLineEdit* m_edTargetTemp;
    QLabel* textLabel9_2_2;
    QLabel* textLabel18_2;
    QLineEdit* m_edManHeater;
    QLabel* textLabel22;
    QComboBox* m_cmbPowerRange;

protected:
    QGridLayout* FrmTempControlLayout;
    QHBoxLayout* layout7_2;
    QHBoxLayout* layout21;
    QGridLayout* groupBox1Layout;
    QVBoxLayout* layout29;
    QHBoxLayout* layout30;
    QVBoxLayout* layout31;
    QHBoxLayout* layout33;
    QHBoxLayout* layout20;
    QHBoxLayout* layout7;
    QHBoxLayout* layout22;
    QHBoxLayout* layout22_2;
    QHBoxLayout* layout34;

protected slots:
    virtual void languageChange();

};

#endif // FRMTEMPCONTROL_H
