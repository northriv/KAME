/****************************************************************************
** Form interface generated from reading ui file '../../../../../kame/users/nmr/forms/nmrpulseform.ui'
**
** Created: 木  3 2 16:40:39 2006
**      by: The User Interface Compiler ($Id: nmrpulseform.h,v 1.1.2.1 2006/03/02 09:19:12 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef FRMNMRPULSE_H
#define FRMNMRPULSE_H

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
class QGroupBox;
class QLabel;
class QLineEdit;
class QSpinBox;
class QCheckBox;
class QPushButton;
class QComboBox;
class KDoubleNumInput;
class KURLRequester;

class FrmNMRPulse : public QMainWindow
{
    Q_OBJECT

public:
    FrmNMRPulse( QWidget* parent = 0, const char* name = 0, WFlags fl = WType_TopLevel );
    ~FrmNMRPulse();

    QGroupBox* groupBox1_2;
    QLabel* textLabel1_5_2_2;
    QLineEdit* m_edEchoPeriod;
    QLabel* textLabel2_2_3;
    QLabel* textLabel1_5_2_2_2;
    QSpinBox* m_numEcho;
    QLabel* textLabel1_5_2;
    QLineEdit* m_edFFTPos;
    QLabel* textLabel2_3_2;
    QLabel* textLabel1_4;
    QLineEdit* m_edFFTLen;
    QLabel* textLabel2_4;
    QLabel* textLabel1_5_2_2_3;
    QLineEdit* m_edDIFFreq;
    QLabel* textLabel2_2_3_2;
    QCheckBox* m_ckbDNR;
    QLabel* textLabel1_5;
    QLineEdit* m_edBGPos;
    QLabel* textLabel2_3;
    QLabel* textLabel1_2_2;
    QLineEdit* m_edBGWidth;
    QLabel* textLabel2_2_2;
    QPushButton* m_btnFFT;
    QLabel* textLabel1_3;
    QComboBox* m_cmbWindowFunc;
    QLabel* textLabel4;
    KDoubleNumInput* m_numPhaseAdv;
    QLabel* textLabel1;
    QLineEdit* m_edPos;
    QLabel* textLabel2;
    QLabel* textLabel1_2;
    QLineEdit* m_edWidth;
    QLabel* textLabel2_2;
    QLabel* textLabel1_2_3;
    QComboBox* m_cmbDSO;
    KURLRequester* m_urlDump;
    QPushButton* m_btnDump;
    XQGraph* m_graph;
    QGroupBox* groupBox1;
    QCheckBox* m_ckbIncrAvg;
    QSpinBox* m_numExtraAvg;
    QPushButton* m_btnAvgClear;

protected:
    QGridLayout* FrmNMRPulseLayout;
    QGridLayout* groupBox1_2Layout;
    QHBoxLayout* layout23;
    QHBoxLayout* layout1_2_3;
    QHBoxLayout* layout24;
    QHBoxLayout* layout21;
    QVBoxLayout* layout2_3_2;
    QHBoxLayout* layout1_3_2;
    QVBoxLayout* layout2_4;
    QHBoxLayout* layout1_4;
    QHBoxLayout* layout23_2;
    QHBoxLayout* layout1_2_3_2;
    QVBoxLayout* layout23_3;
    QHBoxLayout* layout18;
    QVBoxLayout* layout2_3;
    QHBoxLayout* layout1_3;
    QVBoxLayout* layout2_2_2;
    QHBoxLayout* layout1_2_2;
    QHBoxLayout* layout93;
    QHBoxLayout* layout9;
    QHBoxLayout* layout20;
    QVBoxLayout* layout2;
    QHBoxLayout* layout1;
    QVBoxLayout* layout2_2;
    QHBoxLayout* layout1_2;
    QHBoxLayout* layout26;
    QVBoxLayout* layout30;
    QHBoxLayout* layout15;
    QVBoxLayout* layout29;
    QHBoxLayout* groupBox1Layout;

protected slots:
    virtual void languageChange();

private:
    QPixmap image0;

};

#endif // FRMNMRPULSE_H
