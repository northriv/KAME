/****************************************************************************
** Form interface generated from reading ui file '../../../kame/forms/drivertool.ui'
**
** Created: 木  3 2 16:22:58 2006
**      by: The User Interface Compiler ($Id: drivertool.h,v 1.1.2.1 2006/03/02 09:19:19 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef FRMDRIVER_H
#define FRMDRIVER_H

#include <qvariant.h>
#include <qwidget.h>

class QVBoxLayout;
class QHBoxLayout;
class QGridLayout;
class QSpacerItem;
class QGroupBox;
class KURLRequester;
class QCheckBox;
class QTable;
class QPushButton;

class FrmDriver : public QWidget
{
    Q_OBJECT

public:
    FrmDriver( QWidget* parent = 0, const char* name = 0, WFlags fl = 0 );
    ~FrmDriver();

    QGroupBox* groupBox1;
    KURLRequester* m_urlBinRec;
    QCheckBox* m_ckbBinRecWrite;
    QTable* m_tblDrivers;
    QPushButton* m_btnNew;
    QPushButton* m_btnDelete;

protected:
    QGridLayout* FrmDriverLayout;
    QSpacerItem* spacer2;
    QGridLayout* groupBox1Layout;
    QSpacerItem* spacer4;

protected slots:
    virtual void languageChange();

};

#endif // FRMDRIVER_H
