/****************************************************************************
** Form interface generated from reading ui file '../../../../../kame/users/funcsynth/forms/funcsynthform.ui'
**
** Created: 水  2 1 03:49:53 2006
**      by: The User Interface Compiler ($Id: funcsynthform.h,v 1.1 2006/02/01 18:45:30 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef FRMFUNCSYNTH_H
#define FRMFUNCSYNTH_H

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
class QPushButton;
class QLabel;
class QComboBox;
class QLineEdit;

class FrmFuncSynth : public QMainWindow
{
    Q_OBJECT

public:
    FrmFuncSynth( QWidget* parent = 0, const char* name = 0, WFlags fl = WType_TopLevel );
    ~FrmFuncSynth();

    QCheckBox* m_ckbOutput;
    QPushButton* m_btnTrig;
    QLabel* textLabel1;
    QComboBox* m_cmbMode;
    QLabel* textLabel2;
    QComboBox* m_cmbFunc;
    QLabel* textLabel1_5_2_2;
    QLineEdit* m_edFreq;
    QLabel* textLabel2_2_3;
    QLabel* textLabel1_5_2_2_3;
    QLineEdit* m_edAmp;
    QLabel* textLabel2_2_3_3;
    QLabel* textLabel1_5_2_2_2;
    QLineEdit* m_edPhase;
    QLabel* la;
    QLabel* textLabel1_5_2_2_2_2;
    QLineEdit* m_edOffset;
    QLabel* la_2;

protected:
    QGridLayout* FrmFuncSynthLayout;
    QHBoxLayout* layout23;
    QHBoxLayout* layout1_2_3;
    QHBoxLayout* layout23_3;
    QHBoxLayout* layout1_2_3_3;
    QHBoxLayout* layout23_2;
    QHBoxLayout* layout1_2_3_2;
    QHBoxLayout* layout23_2_2;
    QHBoxLayout* layout1_2_3_2_2;

protected slots:
    virtual void languageChange();

};

#endif // FRMFUNCSYNTH_H
