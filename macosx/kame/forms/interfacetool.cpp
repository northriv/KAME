#include <klocale.h>
/****************************************************************************
** Form implementation generated from reading ui file '../../../kame/forms/interfacetool.ui'
**
** Created: 水  2 1 03:36:32 2006
**      by: The User Interface Compiler ($Id: interfacetool.cpp,v 1.1 2006/02/01 18:45:13 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "interfacetool.h"

#include <qvariant.h>
#include <qtable.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>

/*
 *  Constructs a FrmInterface as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 */
FrmInterface::FrmInterface( QWidget* parent, const char* name, WFlags fl )
    : QWidget( parent, name, fl )
{
    if ( !name )
	setName( "FrmInterface" );
    FrmInterfaceLayout = new QGridLayout( this, 1, 1, 2, 6, "FrmInterfaceLayout"); 

    tblInterfaces = new QTable( this, "tblInterfaces" );
    tblInterfaces->setEnabled( FALSE );
    tblInterfaces->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)5, 0, 0, tblInterfaces->sizePolicy().hasHeightForWidth() ) );
    tblInterfaces->setMinimumSize( QSize( 300, 300 ) );
    tblInterfaces->setNumRows( 0 );
    tblInterfaces->setNumCols( 0 );
    tblInterfaces->setSorting( FALSE );
    tblInterfaces->setSelectionMode( QTable::Single );

    FrmInterfaceLayout->addWidget( tblInterfaces, 0, 0 );
    languageChange();
    resize( QSize(322, 383).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );
}

/*
 *  Destroys the object and frees any allocated resources
 */
FrmInterface::~FrmInterface()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void FrmInterface::languageChange()
{
    setCaption( tr2i18n( "Interfaces" ) );
}

#include "interfacetool.moc"
