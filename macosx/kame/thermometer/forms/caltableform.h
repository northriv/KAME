/****************************************************************************
** Form interface generated from reading ui file '../../../../kame/thermometer/forms/caltableform.ui'
**
** Created: 水  2 1 03:42:17 2006
**      by: The User Interface Compiler ($Id: caltableform.h,v 1.1 2006/02/01 18:45:19 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef FRMCALTABLE_H
#define FRMCALTABLE_H

#include <qvariant.h>
#include <qwidget.h>

class QVBoxLayout;
class QHBoxLayout;
class QGridLayout;
class QSpacerItem;
class QPushButton;
class QLabel;
class KComboBox;
class QLineEdit;

class FrmCalTable : public QWidget
{
    Q_OBJECT

public:
    FrmCalTable( QWidget* parent = 0, const char* name = 0, WFlags fl = 0 );
    ~FrmCalTable();

    QPushButton* btnDump;
    QLabel* textLabel1;
    KComboBox* cmbThermometer;
    QLabel* textLabel3;
    QLineEdit* edValue;
    QLabel* textLabel2;
    QLineEdit* edTemp;
    QLabel* textLabel1_2;

public slots:
    virtual void btnOK_clicked();

protected:
    QGridLayout* FrmCalTableLayout;
    QSpacerItem* spacer5_2;
    QSpacerItem* spacer6;
    QGridLayout* layout8;
    QSpacerItem* spacer3;
    QSpacerItem* spacer5;
    QVBoxLayout* layout3;
    QVBoxLayout* layout8_2;
    QVBoxLayout* layout10;
    QHBoxLayout* layout9;

protected slots:
    virtual void languageChange();

};

#endif // FRMCALTABLE_H
