/****************************************************************************
** Form interface generated from reading ui file '../../../../../kame/users/nmr/forms/nmrspectrumform.ui'
**
** Created: 水  2 1 03:50:25 2006
**      by: The User Interface Compiler ($Id: nmrspectrumform.h,v 1.1 2006/02/01 18:43:55 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef FRMNMRSPECTRUM_H
#define FRMNMRSPECTRUM_H

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
class QLabel;
class QLineEdit;
class QComboBox;

class FrmNMRSpectrum : public QMainWindow
{
    Q_OBJECT

public:
    FrmNMRSpectrum( QWidget* parent = 0, const char* name = 0, WFlags fl = WType_TopLevel );
    ~FrmNMRSpectrum();

    XQGraph* m_graph;
    KURLRequester* m_urlDump;
    QPushButton* m_btnDump;
    QLabel* textLabel1_3_2;
    QLineEdit* m_edFieldFactor;
    QLabel* textLabel2_3_2;
    QLabel* textLabel1_2_2_2;
    QLineEdit* m_edResidual;
    QLabel* textLabel2_2_2_2;
    QPushButton* m_btnClear;
    QLabel* textLabel1_3;
    QLineEdit* m_edHMin;
    QLabel* textLabel2_3;
    QLabel* textLabel1_2_2;
    QLineEdit* m_edHMax;
    QLabel* textLabel2_2_2;
    QLabel* textLabel1;
    QLineEdit* m_edFreq;
    QLabel* textLabel2;
    QLabel* textLabel1_2;
    QLineEdit* m_edBW;
    QLabel* textLabel2_2;
    QLabel* textLabel1_3_3;
    QLineEdit* m_edResolution;
    QLabel* textLabel2_3_3;
    QLabel* textLabel1_4;
    QComboBox* m_cmbFieldEntry;
    QLabel* textLabel1_4_2;
    QComboBox* m_cmbPulse;

protected:
    QGridLayout* FrmNMRSpectrumLayout;
    QVBoxLayout* layout19;
    QHBoxLayout* layout15;
    QHBoxLayout* layout12_2_2;
    QVBoxLayout* layout2_3_2;
    QHBoxLayout* layout1_3_2;
    QVBoxLayout* layout2_2_2_2;
    QHBoxLayout* layout1_2_2_2;
    QHBoxLayout* layout12_2;
    QVBoxLayout* layout2_3;
    QHBoxLayout* layout1_3;
    QVBoxLayout* layout2_2_2;
    QHBoxLayout* layout1_2_2;
    QHBoxLayout* layout12;
    QVBoxLayout* layout2;
    QHBoxLayout* layout1;
    QVBoxLayout* layout2_2;
    QHBoxLayout* layout1_2;
    QVBoxLayout* layout2_3_3;
    QHBoxLayout* layout1_3_3;
    QHBoxLayout* layout49;
    QHBoxLayout* layout49_2;

protected slots:
    virtual void languageChange();

private:
    QPixmap image0;

};

#endif // FRMNMRSPECTRUM_H
