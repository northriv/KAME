/****************************************************************************
** Form interface generated from reading ui file '../../../kame/forms/scalarentrytool.ui'
**
** Created: 木  3 2 16:22:58 2006
**      by: The User Interface Compiler ($Id: scalarentrytool.h,v 1.1.2.1 2006/03/02 09:19:19 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef FRMENTRY_H
#define FRMENTRY_H

#include <qvariant.h>
#include <qwidget.h>

class QVBoxLayout;
class QHBoxLayout;
class QGridLayout;
class QSpacerItem;
class QTable;
class QGroupBox;
class KURLRequester;
class QCheckBox;
class QLabel;
class QLineEdit;

class FrmEntry : public QWidget
{
    Q_OBJECT

public:
    FrmEntry( QWidget* parent = 0, const char* name = 0, WFlags fl = 0 );
    ~FrmEntry();

    QTable* m_tblEntries;
    QGroupBox* groupBox1;
    KURLRequester* m_urlTextWriter;
    QCheckBox* m_ckbTextWrite;
    QLabel* textLabel1;
    QLineEdit* m_edLastLine;

protected:
    QGridLayout* FrmEntryLayout;
    QGridLayout* groupBox1Layout;
    QSpacerItem* spacer4;
    QHBoxLayout* layout1;

protected slots:
    virtual void languageChange();

};

#endif // FRMENTRY_H
