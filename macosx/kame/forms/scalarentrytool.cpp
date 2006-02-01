#include <klocale.h>
/****************************************************************************
** Form implementation generated from reading ui file '../../../kame/forms/scalarentrytool.ui'
**
** Created: 水  2 1 03:36:15 2006
**      by: The User Interface Compiler ($Id: scalarentrytool.cpp,v 1.1 2006/02/01 18:45:11 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "scalarentrytool.h"

#include <qvariant.h>
#include <qpushbutton.h>
#include <qtable.h>
#include <qgroupbox.h>
#include <kurlrequester.h>
#include <qcheckbox.h>
#include <qlabel.h>
#include <qlineedit.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>

/*
 *  Constructs a FrmEntry as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 */
FrmEntry::FrmEntry( QWidget* parent, const char* name, WFlags fl )
    : QWidget( parent, name, fl )
{
    if ( !name )
	setName( "FrmEntry" );
    FrmEntryLayout = new QGridLayout( this, 1, 1, 2, 6, "FrmEntryLayout"); 

    m_tblEntries = new QTable( this, "m_tblEntries" );
    m_tblEntries->setEnabled( FALSE );
    m_tblEntries->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)5, 0, 0, m_tblEntries->sizePolicy().hasHeightForWidth() ) );
    m_tblEntries->setMinimumSize( QSize( 350, 300 ) );
    m_tblEntries->setNumRows( 0 );
    m_tblEntries->setNumCols( 0 );
    m_tblEntries->setSorting( FALSE );
    m_tblEntries->setSelectionMode( QTable::Single );

    FrmEntryLayout->addWidget( m_tblEntries, 1, 0 );

    groupBox1 = new QGroupBox( this, "groupBox1" );
    groupBox1->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, groupBox1->sizePolicy().hasHeightForWidth() ) );
    groupBox1->setMinimumSize( QSize( 300, 0 ) );
    groupBox1->setFrameShape( QGroupBox::GroupBoxPanel );
    groupBox1->setFrameShadow( QGroupBox::Sunken );
    groupBox1->setColumnLayout(0, Qt::Vertical );
    groupBox1->layout()->setSpacing( 6 );
    groupBox1->layout()->setMargin( 11 );
    groupBox1Layout = new QGridLayout( groupBox1->layout() );
    groupBox1Layout->setAlignment( Qt::AlignTop );

    m_urlTextWriter = new KURLRequester( groupBox1, "m_urlTextWriter" );
    m_urlTextWriter->setEnabled( FALSE );

    groupBox1Layout->addWidget( m_urlTextWriter, 0, 0 );

    m_ckbTextWrite = new QCheckBox( groupBox1, "m_ckbTextWrite" );
    m_ckbTextWrite->setEnabled( FALSE );
    m_ckbTextWrite->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, m_ckbTextWrite->sizePolicy().hasHeightForWidth() ) );

    groupBox1Layout->addWidget( m_ckbTextWrite, 0, 2 );
    spacer4 = new QSpacerItem( 16, 20, QSizePolicy::Fixed, QSizePolicy::Minimum );
    groupBox1Layout->addItem( spacer4, 0, 1 );

    layout1 = new QHBoxLayout( 0, 0, 6, "layout1"); 

    textLabel1 = new QLabel( groupBox1, "textLabel1" );
    QFont textLabel1_font(  textLabel1->font() );
    textLabel1_font.setFamily( "Helvetica" );
    textLabel1_font.setPointSize( 10 );
    textLabel1->setFont( textLabel1_font ); 
    layout1->addWidget( textLabel1 );

    m_edLastLine = new QLineEdit( groupBox1, "m_edLastLine" );
    m_edLastLine->setEnabled( FALSE );
    QFont m_edLastLine_font(  m_edLastLine->font() );
    m_edLastLine_font.setFamily( "Helvetica" );
    m_edLastLine_font.setPointSize( 10 );
    m_edLastLine->setFont( m_edLastLine_font ); 
    m_edLastLine->setAlignment( int( QLineEdit::AlignRight ) );
    layout1->addWidget( m_edLastLine );

    groupBox1Layout->addMultiCellLayout( layout1, 1, 1, 0, 2 );

    FrmEntryLayout->addWidget( groupBox1, 0, 0 );
    languageChange();
    resize( QSize(397, 393).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );

    // tab order
    setTabOrder( m_urlTextWriter, m_ckbTextWrite );
    setTabOrder( m_ckbTextWrite, m_edLastLine );
    setTabOrder( m_edLastLine, m_tblEntries );
}

/*
 *  Destroys the object and frees any allocated resources
 */
FrmEntry::~FrmEntry()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void FrmEntry::languageChange()
{
    setCaption( tr2i18n( "Scalar Entries" ) );
    groupBox1->setTitle( tr2i18n( "Text Writer" ) );
    m_ckbTextWrite->setText( tr2i18n( "Write" ) );
    textLabel1->setText( tr2i18n( "LastLine" ) );
}

#include "scalarentrytool.moc"
