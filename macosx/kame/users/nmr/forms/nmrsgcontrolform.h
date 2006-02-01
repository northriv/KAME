/****************************************************************************
** Form interface generated from reading ui file '../../../../../kame/users/nmr/forms/nmrsgcontrolform.ui'
**
** Created: Fri Jan 6 01:06:51 2006
**      by: The User Interface Compiler ($Id: nmrsgcontrolform.h,v 1.1 2006/02/01 18:43:56 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef FRMNMRSGCONTROL_H
#define FRMNMRSGCONTROL_H

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
class QLineEdit;

class frmNMRSGControl : public QMainWindow
{
    Q_OBJECT

public:
    frmNMRSGControl( QWidget* parent = 0, const char* name = 0, WFlags fl = WType_TopLevel );
    ~frmNMRSGControl();

    QLabel* textLabel5_3_2_2;
    QLineEdit* edSG1FreqOffset;
    QLabel* textLabel2_3_2_3_2_2;
    QLabel* textLabel5_3_2_3;
    QLineEdit* edSG2FreqOffset;
    QLabel* textLabel2_3_2_3_2_3;
    QLabel* textLabel5_3_2;
    QLineEdit* edFreq;
    QLabel* textLabel2_3_2_3_2;

protected:
    QGridLayout* frmNMRSGControlLayout;
    QHBoxLayout* layout59_3_2_2;
    QHBoxLayout* layout1_3_2_3_2_2;
    QHBoxLayout* layout59_3_2_3;
    QHBoxLayout* layout1_3_2_3_2_3;
    QHBoxLayout* layout59_3_2;
    QHBoxLayout* layout1_3_2_3_2;

protected slots:
    virtual void languageChange();

};

#endif // FRMNMRSGCONTROL_H
