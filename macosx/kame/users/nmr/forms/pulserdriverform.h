/****************************************************************************
** Form interface generated from reading ui file '../../../../../kame/users/nmr/forms/pulserdriverform.ui'
**
** Created: 木  3 2 16:40:40 2006
**      by: The User Interface Compiler ($Id: pulserdriverform.h,v 1.1.2.1 2006/03/02 09:19:16 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef FRMPULSER_H
#define FRMPULSER_H

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
class QComboBox;
class QLineEdit;
class QGroupBox;
class KDoubleNumInput;
class QTable;
class QPushButton;
class QSpinBox;

class FrmPulser : public QMainWindow
{
    Q_OBJECT

public:
    FrmPulser( QWidget* parent = 0, const char* name = 0, WFlags fl = WType_TopLevel );
    ~FrmPulser();

    XQGraph* m_graph;
    QCheckBox* m_ckbOutput;
    QLabel* textLabel2_2;
    QComboBox* m_cmbRTMode;
    QLabel* textLabel5_3;
    QLineEdit* m_edRT;
    QLabel* textLabel2_3_2_3;
    QLabel* textLabel5;
    QLineEdit* m_edTau;
    QLabel* textLabel2_3_2;
    QGroupBox* groupBox3;
    QLabel* textLabel5_2;
    QLineEdit* m_edPW1;
    QLabel* textLabel2_3_2_2;
    QLabel* textLabel2_3_4;
    QComboBox* m_cmbP1Func;
    QLabel* textLabel5_2_2_2_2_2;
    KDoubleNumInput* m_dblP1Level;
    QGroupBox* groupBox3_3;
    QLabel* textLabel5_2_4;
    QLineEdit* m_edPW2;
    QLabel* textLabel2_3_2_2_4;
    QLabel* textLabel5_2_2_2_2_2_2;
    KDoubleNumInput* m_dblP2Level;
    QLabel* textLabel2_3_4_2;
    QComboBox* m_cmbP2Func;
    QGroupBox* groupBox3_2;
    QTable* m_tblPulse;
    QPushButton* m_btnMoreConfig;
    QLabel* textLabel5_2_3_3_2_2;
    KDoubleNumInput* m_dblMasterLevel;
    QLabel* textLabel2_3;
    QComboBox* m_cmbCombMode;
    QGroupBox* groupBox4;
    QLabel* textLabel5_2_3_2_3;
    QLineEdit* m_edCombP1Alt;
    QLabel* textLabel2_3_2_2_3_2_3;
    QLabel* textLabel5_2_3_2_2;
    QLineEdit* m_edCombP1;
    QLabel* textLabel2_3_2_2_3_2_2;
    QLabel* textLabel5_2_3_2;
    QLineEdit* m_edCombPT;
    QLabel* textLabel2_3_2_2_3_2;
    QLabel* textLabel5_2_3_3;
    QSpinBox* m_numCombNum;
    QLabel* textLabel5_2_3_2_3_4;
    QLineEdit* m_edCombOffRes;
    QLabel* textLabel2_3_2_2_3_2_3_4;
    QLabel* textLabel5_2_3;
    QLineEdit* m_edCombPW;
    QLabel* textLabel2_3_2_2_3;
    QLabel* textLabel2_3_4_2_2;
    QComboBox* m_cmbCombFunc;
    QLabel* textLabel5_2_2_2_2;
    KDoubleNumInput* m_dblCombLevel;

protected:
    QGridLayout* FrmPulserLayout;
    QVBoxLayout* layout52;
    QHBoxLayout* layout85_2;
    QHBoxLayout* layout59_3;
    QHBoxLayout* layout1_3_2_3;
    QHBoxLayout* layout59;
    QHBoxLayout* layout1_3_2;
    QGridLayout* groupBox3Layout;
    QHBoxLayout* layout59_2;
    QHBoxLayout* layout1_3_2_2;
    QHBoxLayout* layout85_3_3;
    QHBoxLayout* layout47_2;
    QGridLayout* groupBox3_3Layout;
    QHBoxLayout* layout59_2_4;
    QHBoxLayout* layout1_3_2_2_4;
    QHBoxLayout* layout47_2_2;
    QHBoxLayout* layout85_3_3_2;
    QGridLayout* groupBox3_2Layout;
    QVBoxLayout* layout50;
    QHBoxLayout* layout85_3;
    QGridLayout* groupBox4Layout;
    QHBoxLayout* layout59_2_3_2_3;
    QHBoxLayout* layout1_3_2_2_3_2_3;
    QHBoxLayout* layout59_2_3_2_2;
    QHBoxLayout* layout1_3_2_2_3_2_2;
    QHBoxLayout* layout59_2_3_2;
    QHBoxLayout* layout1_3_2_2_3_2;
    QHBoxLayout* layout71;
    QHBoxLayout* layout59_2_3_2_3_2;
    QHBoxLayout* layout1_3_2_2_3_2_3_4;
    QHBoxLayout* layout59_2_3;
    QHBoxLayout* layout1_3_2_2_3;
    QHBoxLayout* layout85_3_3_2_2;
    QHBoxLayout* layout47;

protected slots:
    virtual void languageChange();

private:
    QPixmap image0;

};

#endif // FRMPULSER_H
