/****************************************************************************
** Form interface generated from reading ui file '../../../../../kame/users/dso/forms/dsoform.ui'
**
** Created: 水  2 1 03:46:26 2006
**      by: The User Interface Compiler ($Id: dsoform.h,v 1.1 2006/02/01 18:44:57 northriv Exp $)
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
class QGroupBox;
class QComboBox;
class QLabel;
class QLineEdit;
class QCheckBox;
class QPushButton;
class KURLRequester;

class FrmDSO : public QMainWindow
{
    Q_OBJECT

public:
    FrmDSO( QWidget* parent = 0, const char* name = 0, WFlags fl = WType_TopLevel );
    ~FrmDSO();

    QGroupBox* groupBox5;
    QComboBox* m_cmbTrace2;
    QLabel* textLabel1_2_4;
    QLineEdit* m_edVFullScale2;
    QLabel* textLabel1_2_2_3;
    QLabel* textLabel1_2_3_2;
    QLineEdit* m_edVOffset2;
    QLabel* textLabel1_2_2_2_2;
    QGroupBox* groupBox4;
    QComboBox* m_cmbTrace1;
    QLabel* textLabel1_2;
    QLineEdit* m_edVFullScale1;
    QLabel* textLabel1_2_2;
    QLabel* textLabel1_2_3;
    QLineEdit* m_edVOffset1;
    QLabel* textLabel1_2_2_2;
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
    QCheckBox* m_ckbSingleSeq;
    QPushButton* m_btnForceTrigger;
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
    QGridLayout* groupBox5Layout;
    QVBoxLayout* layout7_2;
    QHBoxLayout* layout5_3;
    QHBoxLayout* layout5_2_2;
    QGridLayout* groupBox4Layout;
    QVBoxLayout* layout7;
    QHBoxLayout* layout5;
    QHBoxLayout* layout5_2;
    QHBoxLayout* layout2_2_2;
    QHBoxLayout* layout5_4_2;
    QHBoxLayout* layout5_4;
    QHBoxLayout* layout4;
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
