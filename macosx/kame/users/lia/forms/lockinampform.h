/****************************************************************************
** Form interface generated from reading ui file '../../../../../kame/users/lia/forms/lockinampform.ui'
**
** Created: 水  2 1 03:49:06 2006
**      by: The User Interface Compiler ($Id: lockinampform.h,v 1.1 2006/02/01 18:45:00 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef FRMLIA_H
#define FRMLIA_H

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
class QComboBox;

class FrmLIA : public QMainWindow
{
    Q_OBJECT

public:
    FrmLIA( QWidget* parent = 0, const char* name = 0, WFlags fl = WType_TopLevel );
    ~FrmLIA();

    QCheckBox* m_ckbAutoScaleY;
    QLabel* textLabel2_2;
    QLineEdit* m_edFetchFreq;
    QCheckBox* m_ckbAutoScaleX;
    QLabel* textLabel1;
    QComboBox* m_cmbSens;
    QLabel* textLabel4;
    QComboBox* m_cmbTimeConst;
    QLabel* textLabel2;
    QLineEdit* m_edOutput;
    QLabel* textLabel3;
    QLabel* textLabel2_3;
    QLineEdit* m_edFreq;
    QLabel* textLabel3_2;

protected:
    QGridLayout* FrmLIALayout;
    QHBoxLayout* layout3_2;
    QHBoxLayout* layout2;
    QHBoxLayout* layout1;
    QHBoxLayout* layout3;
    QHBoxLayout* layout3_3;

protected slots:
    virtual void languageChange();

};

#endif // FRMLIA_H
