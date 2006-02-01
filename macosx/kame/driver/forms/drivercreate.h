/****************************************************************************
** Form interface generated from reading ui file '../../../../kame/driver/forms/drivercreate.ui'
**
** Created: 水  2 1 03:40:11 2006
**      by: The User Interface Compiler ($Id: drivercreate.h,v 1.1 2006/02/01 18:45:15 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef DLGCREATEDRIVER_H
#define DLGCREATEDRIVER_H

#include <qvariant.h>
#include <qdialog.h>

class QVBoxLayout;
class QHBoxLayout;
class QGridLayout;
class QSpacerItem;
class QPushButton;
class QListBox;
class QListBoxItem;
class QLabel;
class QLineEdit;

class DlgCreateDriver : public QDialog
{
    Q_OBJECT

public:
    DlgCreateDriver( QWidget* parent = 0, const char* name = 0, bool modal = FALSE, WFlags fl = 0 );
    ~DlgCreateDriver();

    QPushButton* m_buttonOk;
    QPushButton* m_buttonCancel;
    QListBox* m_lstType;
    QLabel* textLabel1;
    QLineEdit* m_edName;

protected:
    QGridLayout* DlgCreateDriverLayout;
    QHBoxLayout* Layout1;
    QSpacerItem* Horizontal_Spacing2;
    QHBoxLayout* layout3;

protected slots:
    virtual void languageChange();

};

#endif // DLGCREATEDRIVER_H
