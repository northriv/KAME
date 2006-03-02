/****************************************************************************
** Form interface generated from reading ui file '../../../kame/forms/interfacetool.ui'
**
** Created: 木  3 2 16:22:58 2006
**      by: The User Interface Compiler ($Id: interfacetool.h,v 1.1.2.1 2006/03/02 09:19:19 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef FRMINTERFACE_H
#define FRMINTERFACE_H

#include <qvariant.h>
#include <qwidget.h>

class QVBoxLayout;
class QHBoxLayout;
class QGridLayout;
class QSpacerItem;
class QTable;

class FrmInterface : public QWidget
{
    Q_OBJECT

public:
    FrmInterface( QWidget* parent = 0, const char* name = 0, WFlags fl = 0 );
    ~FrmInterface();

    QTable* tblInterfaces;

protected:
    QGridLayout* FrmInterfaceLayout;

protected slots:
    virtual void languageChange();

};

#endif // FRMINTERFACE_H
