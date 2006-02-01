#include <klocale.h>
/****************************************************************************
** Form implementation generated from reading ui file '../../../../kame/driver/forms/drivercreate.ui'
**
** Created: 水  2 1 03:40:12 2006
**      by: The User Interface Compiler ($Id: drivercreate.cpp,v 1.1 2006/02/01 18:45:15 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "drivercreate.h"

#include <qvariant.h>
#include <qpushbutton.h>
#include <qlistbox.h>
#include <qlabel.h>
#include <qlineedit.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>

/*
 *  Constructs a DlgCreateDriver as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 *  The dialog will by default be modeless, unless you set 'modal' to
 *  TRUE to construct a modal dialog.
 */
DlgCreateDriver::DlgCreateDriver( QWidget* parent, const char* name, bool modal, WFlags fl )
    : QDialog( parent, name, modal, fl )
{
    if ( !name )
	setName( "DlgCreateDriver" );
    setSizeGripEnabled( TRUE );
    DlgCreateDriverLayout = new QGridLayout( this, 1, 1, 2, 6, "DlgCreateDriverLayout"); 

    Layout1 = new QHBoxLayout( 0, 0, 6, "Layout1"); 
    Horizontal_Spacing2 = new QSpacerItem( 20, 20, QSizePolicy::Expanding, QSizePolicy::Minimum );
    Layout1->addItem( Horizontal_Spacing2 );

    m_buttonOk = new QPushButton( this, "m_buttonOk" );
    m_buttonOk->setAutoDefault( TRUE );
    m_buttonOk->setDefault( TRUE );
    Layout1->addWidget( m_buttonOk );

    m_buttonCancel = new QPushButton( this, "m_buttonCancel" );
    m_buttonCancel->setAutoDefault( TRUE );
    Layout1->addWidget( m_buttonCancel );

    DlgCreateDriverLayout->addLayout( Layout1, 2, 0 );

    m_lstType = new QListBox( this, "m_lstType" );

    DlgCreateDriverLayout->addWidget( m_lstType, 0, 0 );

    layout3 = new QHBoxLayout( 0, 0, 6, "layout3"); 

    textLabel1 = new QLabel( this, "textLabel1" );
    layout3->addWidget( textLabel1 );

    m_edName = new QLineEdit( this, "m_edName" );
    layout3->addWidget( m_edName );

    DlgCreateDriverLayout->addLayout( layout3, 1, 0 );
    languageChange();
    resize( QSize(213, 323).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );

    // signals and slots connections
    connect( m_buttonOk, SIGNAL( clicked() ), this, SLOT( accept() ) );
    connect( m_buttonCancel, SIGNAL( clicked() ), this, SLOT( reject() ) );

    // tab order
    setTabOrder( m_lstType, m_edName );
    setTabOrder( m_edName, m_buttonOk );
    setTabOrder( m_buttonOk, m_buttonCancel );
}

/*
 *  Destroys the object and frees any allocated resources
 */
DlgCreateDriver::~DlgCreateDriver()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void DlgCreateDriver::languageChange()
{
    setCaption( tr2i18n( "Create New Driver" ) );
    m_buttonOk->setText( tr2i18n( "&OK" ) );
    m_buttonOk->setAccel( QKeySequence( QString::null ) );
    m_buttonCancel->setText( tr2i18n( "&Cancel" ) );
    m_buttonCancel->setAccel( QKeySequence( QString::null ) );
    m_lstType->clear();
    m_lstType->insertItem( tr2i18n( "New Item" ) );
    textLabel1->setText( tr2i18n( "Name" ) );
}

#include "drivercreate.moc"
