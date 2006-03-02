/****************************************************************************
** Form interface generated from reading ui file '../../../../../kame/users/magnetps/forms/magnetpsform.ui'
**
** Created: 木  3 2 16:39:07 2006
**      by: The User Interface Compiler ($Id: magnetpsform.h,v 1.1.2.1 2006/03/02 09:19:47 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef FRMMAGNETPS_H
#define FRMMAGNETPS_H

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
class KLed;
class QCheckBox;
class QLineEdit;

class FrmMagnetPS : public QMainWindow
{
    Q_OBJECT

public:
    FrmMagnetPS( QWidget* parent = 0, const char* name = 0, WFlags fl = WType_TopLevel );
    ~FrmMagnetPS();

    QLabel* textLabel5;
    QLCDNumber* m_lcdMagnetField;
    QLabel* textLabel9;
    QLabel* textLabel7;
    QLCDNumber* m_lcdOutputField;
    QLabel* textLabel10;
    QLabel* textLabel6;
    QLCDNumber* m_lcdCurrent;
    QLabel* textLabel11;
    QLabel* textLabel8;
    QLCDNumber* m_lcdVoltage;
    QLabel* textLabel12;
    KLed* m_ledSwitchHeater;
    QLabel* textLabel15;
    KLed* m_ledPersistent;
    QLabel* textLabel15_2;
    QCheckBox* m_ckbAllowPersistent;
    QLabel* textLabel13;
    QLineEdit* m_edTargetField;
    QLabel* textLabel14;
    QLabel* textLabel13_2;
    QLineEdit* m_edSweepRate;
    QLabel* textLabel14_2;

protected:
    QGridLayout* FrmMagnetPSLayout;
    QSpacerItem* spacer5;
    QVBoxLayout* layout15;
    QHBoxLayout* layout7;
    QHBoxLayout* layout8;
    QHBoxLayout* layout9;
    QHBoxLayout* layout10;
    QHBoxLayout* layout13;
    QHBoxLayout* layout13_2;
    QHBoxLayout* layout16;
    QHBoxLayout* layout17;

protected slots:
    virtual void languageChange();

};

#endif // FRMMAGNETPS_H
