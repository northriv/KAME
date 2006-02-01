/****************************************************************************
** Form interface generated from reading ui file '../../../../../kame/users/nmr/forms/nmrrelaxform.ui'
**
** Created: 水  2 1 03:50:25 2006
**      by: The User Interface Compiler ($Id: nmrrelaxform.h,v 1.1 2006/02/01 18:43:55 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef FRMNMRT1_H
#define FRMNMRT1_H

#include <qvariant.h>
#include <qpixmap.h>
#include <qmainwindow.h>

class QVBoxLayout;
class QHBoxLayout;
class QGridLayout;
class QSpacerItem;
class QAction;
class QActionGroup;
class QToolBar;
class QPopupMenu;
class XQGraph;
class KURLRequester;
class QPushButton;
class QGroupBox;
class QCheckBox;
class QLabel;
class QLineEdit;
class QComboBox;
class QSpinBox;
class KDoubleNumInput;
class QTextBrowser;

class FrmNMRT1 : public QMainWindow
{
    Q_OBJECT

public:
    FrmNMRT1( QWidget* parent = 0, const char* name = 0, WFlags fl = WType_TopLevel );
    ~FrmNMRT1();

    XQGraph* m_graph;
    KURLRequester* m_urlDump;
    QPushButton* m_btnDump;
    QGroupBox* groupBox4;
    QCheckBox* m_ckbActive;
    QLabel* textLabel1_2;
    QLabel* textLabel2_2_3;
    QLineEdit* m_edP1Min;
    QLabel* textLabel2_2_2;
    QLineEdit* m_edP1Max;
    QCheckBox* m_ckbT2Mode;
    QLabel* textLabel1;
    QComboBox* m_cmbP1Dist;
    QLabel* textLabel5_2_3_3_2;
    QSpinBox* m_numExtraAvg;
    QLabel* textLabel5_2_3_3_2_2;
    QSpinBox* m_numIgnore;
    QLabel* textLabel1_3_2_2;
    QComboBox* m_cmbPulse1;
    QLabel* textLabel1_3_2;
    QComboBox* m_cmbPulse2;
    QLabel* textLabel1_3;
    QComboBox* m_cmbPulser;
    QGroupBox* groupBox5;
    QCheckBox* m_ckbAutoPhase;
    QCheckBox* m_ckbAbsFit;
    QLabel* textLabel4_2;
    QLabel* textLabel5;
    QComboBox* m_cmbFunction;
    QLabel* textLabel3_2;
    QLineEdit* m_edSmoothSamples;
    QLabel* textLabel3_3;
    QLineEdit* m_edBW;
    QLabel* textLabel2_3_2_2;
    QLabel* textLabel4;
    KDoubleNumInput* m_numPhase;
    QLabel* textLabel3;
    QLineEdit* m_edFreq;
    QLabel* textLabel2_3_2;
    QTextBrowser* m_txtFitStatus;
    QPushButton* m_btnClear;
    QPushButton* m_btnResetFit;

protected:
    QGridLayout* FrmNMRT1Layout;
    QVBoxLayout* layout19;
    QHBoxLayout* layout15;
    QGridLayout* groupBox4Layout;
    QVBoxLayout* layout28;
    QHBoxLayout* layout13;
    QHBoxLayout* layout9;
    QHBoxLayout* layout10;
    QHBoxLayout* layout15_2;
    QHBoxLayout* layout21;
    QHBoxLayout* layout22;
    QVBoxLayout* layout23;
    QHBoxLayout* layout18_2_2;
    QHBoxLayout* layout18_2;
    QHBoxLayout* layout18;
    QGridLayout* groupBox5Layout;
    QHBoxLayout* layout25;
    QHBoxLayout* layout61;
    QHBoxLayout* layout56_2;
    QHBoxLayout* layout1_3_2;
    QHBoxLayout* layout9_2;
    QHBoxLayout* layout56;
    QHBoxLayout* layout1_3;

protected slots:
    virtual void languageChange();

private:
    QPixmap image0;

};

#endif // FRMNMRT1_H
