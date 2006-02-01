/****************************************************************************
** Form interface generated from reading ui file '../../../../../kame/users/nmr/forms/nmrfspectrumform.ui'
**
** Created: 水  2 1 03:50:25 2006
**      by: The User Interface Compiler ($Id: nmrfspectrumform.h,v 1.1 2006/02/01 18:43:58 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef FRMNMRFSPECTRUM_H
#define FRMNMRFSPECTRUM_H

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
class QCheckBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QComboBox;
class QGroupBox;
class KURLRequester;

class FrmNMRFSpectrum : public QMainWindow
{
    Q_OBJECT

public:
    FrmNMRFSpectrum( QWidget* parent = 0, const char* name = 0, WFlags fl = WType_TopLevel );
    ~FrmNMRFSpectrum();

    QCheckBox* m_ckbActive;
    QLabel* textLabel1;
    QLineEdit* m_edCenterFreq;
    QLabel* textLabel2;
    QLabel* textLabel1_3;
    QLineEdit* m_edFreqStep;
    QLabel* textLabel2_3;
    QLabel* textLabel1_2_2;
    QLineEdit* m_edFreqSpan;
    QLabel* textLabel2_2_2;
    QLabel* textLabel1_2;
    QLineEdit* m_edBW;
    QLabel* textLabel2_2;
    QPushButton* m_btnClear;
    QLabel* textLabel1_4;
    QComboBox* m_cmbPulse;
    QGroupBox* groupBox1;
    QLineEdit* m_edSG1FreqOffset;
    QLabel* textLabel2_4;
    QComboBox* m_cmbSG1;
    QLabel* textLabel1_5;
    QGroupBox* groupBox1_2;
    QLineEdit* m_edSG2FreqOffset;
    QLabel* textLabel2_4_2;
    QComboBox* m_cmbSG2;
    QLabel* textLabel1_5_2;
    KURLRequester* m_urlDump;
    QPushButton* m_btnDump;
    XQGraph* m_graph;

protected:
    QGridLayout* FrmNMRFSpectrumLayout;
    QVBoxLayout* layout23;
    QVBoxLayout* layout2;
    QHBoxLayout* layout1;
    QVBoxLayout* layout2_3;
    QHBoxLayout* layout1_3;
    QVBoxLayout* layout2_2_2;
    QHBoxLayout* layout1_2_2;
    QVBoxLayout* layout2_2;
    QHBoxLayout* layout1_2;
    QVBoxLayout* layout22;
    QVBoxLayout* layout12;
    QGridLayout* groupBox1Layout;
    QHBoxLayout* layout1_4;
    QGridLayout* groupBox1_2Layout;
    QHBoxLayout* layout1_4_2;
    QHBoxLayout* layout15;

protected slots:
    virtual void languageChange();

private:
    QPixmap image0;

};

#endif // FRMNMRFSPECTRUM_H
