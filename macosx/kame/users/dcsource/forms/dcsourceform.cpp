#include <klocale.h>
/****************************************************************************
** Form implementation generated from reading ui file '../../../../../kame/users/dcsource/forms/dcsourceform.ui'
**
** Created: 木  3 2 16:39:57 2006
**      by: The User Interface Compiler ($Id: dcsourceform.cpp,v 1.1.2.1 2006/03/02 09:20:48 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "dcsourceform.h"

#include <qvariant.h>
#include <qpushbutton.h>
#include <qlabel.h>
#include <qcombobox.h>
#include <qcheckbox.h>
#include <qlineedit.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>
#include <qaction.h>
#include <qmenubar.h>
#include <qpopupmenu.h>
#include <qtoolbar.h>

/*
 *  Constructs a FrmDCSource as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 */
FrmDCSource::FrmDCSource( QWidget* parent, const char* name, WFlags fl )
    : QMainWindow( parent, name, fl )
{
    (void)statusBar();
    if ( !name )
	setName( "FrmDCSource" );
    setCentralWidget( new QWidget( this, "qt_central_widget" ) );
    FrmDCSourceLayout = new QGridLayout( centralWidget(), 1, 1, 2, 6, "FrmDCSourceLayout"); 
    spacer3 = new QSpacerItem( 20, 20, QSizePolicy::Minimum, QSizePolicy::Expanding );
    FrmDCSourceLayout->addItem( spacer3, 4, 1 );

    layout4 = new QVBoxLayout( 0, 0, 6, "layout4"); 

    textLabel4 = new QLabel( centralWidget(), "textLabel4" );
    layout4->addWidget( textLabel4 );

    m_cmbFunction = new QComboBox( FALSE, centralWidget(), "m_cmbFunction" );
    layout4->addWidget( m_cmbFunction );

    FrmDCSourceLayout->addMultiCellLayout( layout4, 1, 1, 0, 1 );
    spacer2 = new QSpacerItem( 20, 20, QSizePolicy::Minimum, QSizePolicy::Expanding );
    FrmDCSourceLayout->addItem( spacer2, 2, 0 );

    m_ckbOutput = new QCheckBox( centralWidget(), "m_ckbOutput" );
    m_ckbOutput->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, m_ckbOutput->sizePolicy().hasHeightForWidth() ) );

    FrmDCSourceLayout->addWidget( m_ckbOutput, 0, 0 );

    layout3 = new QVBoxLayout( 0, 0, 6, "layout3"); 

    textLabel3 = new QLabel( centralWidget(), "textLabel3" );
    layout3->addWidget( textLabel3 );

    m_edValue = new QLineEdit( centralWidget(), "m_edValue" );
    layout3->addWidget( m_edValue );

    FrmDCSourceLayout->addMultiCellLayout( layout3, 3, 3, 0, 1 );

    // toolbars

    languageChange();
    resize( QSize(150, 184).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );
}

/*
 *  Destroys the object and frees any allocated resources
 */
FrmDCSource::~FrmDCSource()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void FrmDCSource::languageChange()
{
    setCaption( tr2i18n( "DC Source Settings" ) );
    textLabel4->setText( tr2i18n( "Function" ) );
    m_ckbOutput->setText( tr2i18n( "Output" ) );
    textLabel3->setText( tr2i18n( "Value" ) );
}

#include "dcsourceform.moc"
