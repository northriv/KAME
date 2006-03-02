#include <klocale.h>
/****************************************************************************
** Form implementation generated from reading ui file '../../../kame/forms/graphtool.ui'
**
** Created: 木  3 2 16:22:59 2006
**      by: The User Interface Compiler ($Id: graphtool.cpp,v 1.1.2.1 2006/03/02 09:19:19 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "graphtool.h"

#include <qvariant.h>
#include <qpushbutton.h>
#include <qtable.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>

/*
 *  Constructs a FrmGraphList as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 */
FrmGraphList::FrmGraphList( QWidget* parent, const char* name, WFlags fl )
    : QWidget( parent, name, fl )
{
    if ( !name )
	setName( "FrmGraphList" );
    FrmGraphListLayout = new QGridLayout( this, 1, 1, 2, 6, "FrmGraphListLayout"); 

    btnNewGraph = new QPushButton( this, "btnNewGraph" );
    btnNewGraph->setEnabled( FALSE );

    FrmGraphListLayout->addWidget( btnNewGraph, 1, 1 );
    spacer3 = new QSpacerItem( 216, 20, QSizePolicy::Expanding, QSizePolicy::Minimum );
    FrmGraphListLayout->addItem( spacer3, 1, 0 );

    btnDeleteGraph = new QPushButton( this, "btnDeleteGraph" );
    btnDeleteGraph->setEnabled( FALSE );

    FrmGraphListLayout->addWidget( btnDeleteGraph, 1, 2 );

    tblGraphs = new QTable( this, "tblGraphs" );
    tblGraphs->setEnabled( FALSE );
    tblGraphs->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)5, 0, 0, tblGraphs->sizePolicy().hasHeightForWidth() ) );
    tblGraphs->setMinimumSize( QSize( 300, 0 ) );
    tblGraphs->setNumRows( 0 );
    tblGraphs->setNumCols( 0 );
    tblGraphs->setSorting( FALSE );
    tblGraphs->setSelectionMode( QTable::Single );

    FrmGraphListLayout->addMultiCellWidget( tblGraphs, 0, 0, 0, 2 );
    languageChange();
    resize( QSize(304, 199).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );
}

/*
 *  Destroys the object and frees any allocated resources
 */
FrmGraphList::~FrmGraphList()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void FrmGraphList::languageChange()
{
    setCaption( tr2i18n( "Graph" ) );
    btnNewGraph->setText( tr2i18n( "New Graph" ) );
    btnDeleteGraph->setText( tr2i18n( "Delete Graph" ) );
}

#include "graphtool.moc"
