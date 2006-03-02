/****************************************************************************
** Form interface generated from reading ui file '../../../../../kame/users/nmr/forms/pulserdrivermoreform.ui'
**
** Created: 木  3 2 16:40:40 2006
**      by: The User Interface Compiler ($Id: pulserdrivermoreform.h,v 1.1.2.1 2006/03/02 09:19:12 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef FRMPULSERMORE_H
#define FRMPULSERMORE_H

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
class QLabel;
class QComboBox;
class QSpinBox;
class QLineEdit;
class QCheckBox;
class QGroupBox;

class FrmPulserMore : public QMainWindow
{
    Q_OBJECT

public:
    FrmPulserMore( QWidget* parent = 0, const char* name = 0, WFlags fl = WType_TopLevel );
    ~FrmPulserMore();

    QLabel* textLabel2_3_3;
    QComboBox* m_cmbPhaseCycle;
    QLabel* textLabel5_2_3_3_2;
    QSpinBox* m_numEcho;
    QLabel* textLabel5_3_2;
    QLineEdit* m_edG2Setup;
    QLabel* textLabel2_3_2_3_2;
    QCheckBox* m_ckbDrivenEquilibrium;
    QGroupBox* groupBox9;
    QLabel* textLabel5_2_3_2_4_2;
    QLineEdit* m_edPortLevel8;
    QLabel* textLabel2_3_2_2_3_2_4_2;
    QLabel* textLabel5_2_3_2_4_2_2_2;
    QLineEdit* m_edPortLevel10;
    QLabel* textLabel2_3_2_2_3_2_4_2_2_2;
    QLabel* textLabel5_2_3_2_4_2_2;
    QLineEdit* m_edPortLevel9;
    QLabel* textLabel2_3_2_2_3_2_4_2_2;
    QLabel* textLabel5_2_3_2_4_2_2_3;
    QLineEdit* m_edPortLevel11;
    QLabel* textLabel2_3_2_2_3_2_4_2_2_3;
    QLabel* textLabel5_2_3_2_4_2_2_4;
    QLineEdit* m_edPortLevel12;
    QLabel* textLabel2_3_2_2_3_2_4_2_2_4;
    QLabel* textLabel5_2_3_2_4_2_2_5;
    QLineEdit* m_edPortLevel13;
    QLabel* textLabel2_3_2_2_3_2_4_2_2_5;
    QLabel* textLabel5_2_3_2_4_2_2_2_2;
    QLineEdit* m_edPortLevel14;
    QLabel* textLabel2_3_2_2_3_2_4_2_2_2_2;
    QGroupBox* groupBox3_2_3;
    QLabel* textLabel5_2_3_2_4_2_2_2_2_2_3;
    QLineEdit* m_edQAMLevel1;
    QLabel* textLabel5_2_3_2_4_2_2_2_2_2_2_3;
    QLineEdit* m_edQAMLevel2;
    QGroupBox* groupBox3_2;
    QLabel* textLabel5_2_3_2_4_2_2_2_2_2;
    QLineEdit* m_edQAMOffset1;
    QLabel* textLabel2_3_2_2_3_2_4_2_2_2_2_2;
    QLabel* textLabel5_2_3_2_4_2_2_2_2_2_2;
    QLineEdit* m_edQAMOffset2;
    QLabel* textLabel2_3_2_2_3_2_4_2_2_2_2_2_2;
    QGroupBox* groupBox3_2_4;
    QLabel* textLabel5_2_3_2_4_2_2_2_2_2_4;
    QLineEdit* m_edQAMDelay1;
    QLabel* textLabel2_3_2_2_3_2_4_2_2_2_2_2_3;
    QLabel* textLabel5_2_3_2_4_2_2_2_2_2_2_4;
    QLineEdit* m_edQAMDelay2;
    QLabel* textLabel2_3_2_2_3_2_4_2_2_2_2_2_2_3;
    QGroupBox* groupBox3;
    QLabel* textLabel5_2_3_2_4;
    QLineEdit* m_edASWHold;
    QLabel* textLabel2_3_2_2_3_2_4;
    QLabel* textLabel5_2_3_2_5;
    QLineEdit* m_edASWSetup;
    QLabel* textLabel2_3_2_2_3_2_5;
    QLabel* textLabel5_2_3_2_5_2;
    QLineEdit* m_edALTSep;
    QLabel* textLabel2_3_2_2_3_2_5_2;
    QLabel* textLabel2_2;
    QComboBox* m_cmbASWFilter;
    QLabel* textLabel5_3_2_2;
    QLineEdit* m_edDIFFreq;
    QLabel* textLabel2_3_2_3_2_2;

protected:
    QGridLayout* FrmPulserMoreLayout;
    QHBoxLayout* layout85_3_2;
    QHBoxLayout* layout71_2;
    QHBoxLayout* layout59_3_2;
    QHBoxLayout* layout1_3_2_3_2;
    QGridLayout* groupBox9Layout;
    QHBoxLayout* layout59_2_3_2_4_2;
    QHBoxLayout* layout1_3_2_2_3_2_4_2;
    QHBoxLayout* layout59_2_3_2_4_2_2_2;
    QHBoxLayout* layout1_3_2_2_3_2_4_2_2_2;
    QHBoxLayout* layout59_2_3_2_4_2_2;
    QHBoxLayout* layout1_3_2_2_3_2_4_2_2;
    QHBoxLayout* layout59_2_3_2_4_2_2_3;
    QHBoxLayout* layout1_3_2_2_3_2_4_2_2_3;
    QHBoxLayout* layout59_2_3_2_4_2_2_4;
    QHBoxLayout* layout1_3_2_2_3_2_4_2_2_4;
    QHBoxLayout* layout59_2_3_2_4_2_2_5;
    QHBoxLayout* layout1_3_2_2_3_2_4_2_2_5;
    QHBoxLayout* layout59_2_3_2_4_2_2_2_2;
    QHBoxLayout* layout1_3_2_2_3_2_4_2_2_2_2;
    QGridLayout* groupBox3_2_3Layout;
    QHBoxLayout* layout59_2_3_2_4_2_2_2_2_2_3;
    QHBoxLayout* layout1_3_2_2_3_2_4_2_2_2_2_2_3;
    QHBoxLayout* layout59_2_3_2_4_2_2_2_2_2_2_3;
    QHBoxLayout* layout1_3_2_2_3_2_4_2_2_2_2_2_2_3;
    QGridLayout* groupBox3_2Layout;
    QHBoxLayout* layout59_2_3_2_4_2_2_2_2_2;
    QHBoxLayout* layout1_3_2_2_3_2_4_2_2_2_2_2;
    QHBoxLayout* layout59_2_3_2_4_2_2_2_2_2_2;
    QHBoxLayout* layout1_3_2_2_3_2_4_2_2_2_2_2_2;
    QGridLayout* groupBox3_2_4Layout;
    QHBoxLayout* layout59_2_3_2_4_2_2_2_2_2_4;
    QHBoxLayout* layout1_3_2_2_3_2_4_2_2_2_2_2_4;
    QHBoxLayout* layout59_2_3_2_4_2_2_2_2_2_2_4;
    QHBoxLayout* layout1_3_2_2_3_2_4_2_2_2_2_2_2_4;
    QGridLayout* groupBox3Layout;
    QHBoxLayout* layout59_2_3_2_4;
    QHBoxLayout* layout1_3_2_2_3_2_4;
    QHBoxLayout* layout59_2_3_2_5;
    QHBoxLayout* layout1_3_2_2_3_2_5;
    QHBoxLayout* layout59_2_3_2_5_2;
    QHBoxLayout* layout1_3_2_2_3_2_5_2;
    QHBoxLayout* layout85_2;
    QHBoxLayout* layout59_3_2_2;
    QHBoxLayout* layout1_3_2_3_2_2;

protected slots:
    virtual void languageChange();

};

#endif // FRMPULSERMORE_H
