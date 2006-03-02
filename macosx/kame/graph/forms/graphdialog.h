/****************************************************************************
** Form interface generated from reading ui file '../../../../kame/graph/forms/graphdialog.ui'
**
** Created: 木  3 2 16:25:21 2006
**      by: The User Interface Compiler ($Id: graphdialog.h,v 1.1.2.1 2006/03/02 09:19:26 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef DLGGRAPHSETUP_H
#define DLGGRAPHSETUP_H

#include <qvariant.h>
#include <qpixmap.h>
#include <qdialog.h>

class QVBoxLayout;
class QHBoxLayout;
class QGridLayout;
class QSpacerItem;
class QPushButton;
class QTabWidget;
class QWidget;
class QCheckBox;
class KColorCombo;
class QListBox;
class QListBoxItem;
class QLabel;
class QLineEdit;
class KDoubleNumInput;

class DlgGraphSetup : public QDialog
{
    Q_OBJECT

public:
    DlgGraphSetup( QWidget* parent = 0, const char* name = 0, bool modal = FALSE, WFlags fl = 0 );
    ~DlgGraphSetup();

    QPushButton* buttonHelp;
    QPushButton* buttonOk;
    QTabWidget* tab1;
    QWidget* tab;
    QCheckBox* ckbDrawBars;
    KColorCombo* clrBarColor;
    QCheckBox* ckbDrawLines;
    KColorCombo* clrLineColor;
    QCheckBox* ckbDrawPoints;
    KColorCombo* clrPointColor;
    QCheckBox* ckbDisplayMajorGrids;
    KColorCombo* clrMajorGridColor;
    QCheckBox* ckbDisplayMinorGrids;
    KColorCombo* clrMinorGridColor;
    QListBox* lbPlots;
    QLabel* textLabel3;
    QLineEdit* edMaxCount;
    QPushButton* btnClearPoints;
    QLabel* textLabel1_4;
    KDoubleNumInput* dblIntensity;
    QCheckBox* ckbColorPlot;
    KColorCombo* clrColorPlotLow;
    KColorCombo* clrColorPlotHigh;
    QWidget* tab_2;
    QLabel* textLabel1_3;
    QLineEdit* edTicLabelFormat;
    QCheckBox* ckbDisplayTicLabels;
    QCheckBox* ckbDisplayMajorTics;
    QCheckBox* ckbDisplayMinorTics;
    QListBox* lbAxes;
    QCheckBox* ckbAutoScale;
    QLabel* textLabel1;
    QLineEdit* edAxisMax;
    QLabel* textLabel1_2;
    QLineEdit* edAxisMin;
    QCheckBox* ckbLogScale;
    QWidget* tab_3;
    QLabel* textLabel4;
    KColorCombo* clrBackGroundColor;

protected:
    QGridLayout* DlgGraphSetupLayout;
    QHBoxLayout* Layout1;
    QSpacerItem* Horizontal_Spacing2;
    QGridLayout* tabLayout;
    QHBoxLayout* layout20_2_2;
    QSpacerItem* spacer2_2_2;
    QHBoxLayout* layout18_2_2;
    QHBoxLayout* layout20_2;
    QSpacerItem* spacer2_2;
    QHBoxLayout* layout18_2;
    QHBoxLayout* layout20;
    QSpacerItem* spacer2;
    QHBoxLayout* layout18;
    QHBoxLayout* layout28;
    QSpacerItem* spacer7;
    QHBoxLayout* layout29;
    QSpacerItem* spacer8;
    QHBoxLayout* layout21;
    QVBoxLayout* layout19;
    QVBoxLayout* layout16;
    QHBoxLayout* layout22;
    QVBoxLayout* layout26;
    QHBoxLayout* layout25;
    QGridLayout* tabLayout_2;
    QHBoxLayout* layout14;
    QVBoxLayout* layout15;
    QHBoxLayout* layout22_2;
    QVBoxLayout* layout16_2;
    QVBoxLayout* layout11;
    QVBoxLayout* layout10;
    QHBoxLayout* layout2;
    QHBoxLayout* layout2_2;
    QGridLayout* tabLayout_3;
    QSpacerItem* spacer11;
    QSpacerItem* spacer12;
    QVBoxLayout* layout18_3;

protected slots:
    virtual void languageChange();

private:
    QPixmap image0;

};

#endif // DLGGRAPHSETUP_H
