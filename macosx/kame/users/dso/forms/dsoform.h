/****************************************************************************
** Form interface generated from reading ui file '../../../../../kame/users/dso/forms/dsoform.ui'
**
** Created: 木  3 2 16:36:29 2006
**      by: The User Interface Compiler ($Id: dsoform.h,v 1.1.2.1 2006/03/02 09:19:33 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef FRMDSO_H
#define FRMDSO_H

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
class QLabel;
class QComboBox;
class QLineEdit;
class QCheckBox;
class QPushButton;
class QGroupBox;
class KURLRequester;

class FrmDSO : public QMainWindow
{
    Q_OBJECT

public:
    FrmDSO( QWidget* parent = 0, const char* name = 0, WFlags fl = WType_TopLevel );
    ~FrmDSO();

    QLabel* textLabel2_2_2;
    QComboBox* m_cmbRecordLength;
    QLabel* textLabel1_2_5_2;
    QLineEdit* m_edTrigPos;
    QLabel* textLabel1_2_2_4_2;
    QLabel* textLabel1_2_5;
    QLineEdit* m_edTimeWidth;
    QLabel* textLabel1_2_2_4;
    QLabel* textLabel1;
    QLineEdit* m_edAverage;
    QCheckBox* m_ckbFetch;
    QPushButton* m_btnForceTrigger;
    QGroupBox* groupBox4;
    QComboBox* m_cmbTrace1;
    QLabel* textLabel1_2;
    QLineEdit* m_edVFullScale1;
    QLabel* textLabel1_2_2;
    QLabel* textLabel1_2_3;
    QLineEdit* m_edVOffset1;
    QLabel* textLabel1_2_2_2;
    QGroupBox* groupBox5;
    QComboBox* m_cmbTrace2;
    QLabel* textLabel1_2_4;
    QLineEdit* m_edVFullScale2;
    QLabel* textLabel1_2_2_3;
    QLabel* textLabel1_2_3_2;
    QLineEdit* m_edVOffset2;
    QLabel* textLabel1_2_2_2_2;
    QCheckBox* m_ckbSingleSeq;
    QGroupBox* groupBox4_2;
    QComboBox* m_cmbPulser;
    QLabel* textLabel1_3;
    QCheckBox* m_ckb4x;
    QCheckBox* m_ckbEnable;
    KURLRequester* m_urlDump;
    QPushButton* m_btnDump;
    XQGraph* m_graphwidget;
    QGroupBox* groupBox1;
    QCheckBox* m_ckbFIREnabled;
    QLabel* textLabel1_2_5_3_2;
    QLineEdit* m_edFIRSharpness;
    QLabel* textLabel1_2_5_3;
    QLineEdit* m_edFIRBandWidth;
    QLabel* textLabel1_2_2_4_3;
    QLabel* textLabel1_2_5_3_3;
    QLineEdit* m_edFIRCenterFreq;
    QLabel* textLabel1_2_2_4_3_2;

protected:
    QGridLayout* FrmDSOLayout;
    QHBoxLayout* layout2_2_2;
    QHBoxLayout* layout5_4_2;
    QHBoxLayout* layout5_4;
    QHBoxLayout* layout4;
    QGridLayout* groupBox4Layout;
    QVBoxLayout* layout7;
    QHBoxLayout* layout5;
    QHBoxLayout* layout5_2;
    QGridLayout* groupBox5Layout;
    QVBoxLayout* layout7_2;
    QHBoxLayout* layout5_3;
    QHBoxLayout* layout5_2_2;
    QGridLayout* groupBox4_2Layout;
    QVBoxLayout* layout15;
    QHBoxLayout* layout15_2;
    QGridLayout* groupBox1Layout;
    QHBoxLayout* layout5_4_3_2;
    QHBoxLayout* layout5_4_3;
    QHBoxLayout* layout5_4_3_3;

protected slots:
    virtual void languageChange();

private:
    QPixmap image0;

};

#endif // FRMDSO_H
