#include <klocale.h>
/****************************************************************************
** Form implementation generated from reading ui file '../../../../kame/script/forms/rubythreadtool.ui'
**
** Created: 木  3 2 16:23:05 2006
**      by: The User Interface Compiler ($Id: rubythreadtool.cpp,v 1.1.2.1 2006/03/02 09:19:54 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "rubythreadtool.h"

#include <qvariant.h>
#include <qpushbutton.h>
#include <qtextbrowser.h>
#include <qlabel.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>

/*
 *  Constructs a FrmRubyThread as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 */
FrmRubyThread::FrmRubyThread( QWidget* parent, const char* name, WFlags fl )
    : QWidget( parent, name, fl )
{
    if ( !name )
	setName( "FrmRubyThread" );
    FrmRubyThreadLayout = new QGridLayout( this, 1, 1, 11, 6, "FrmRubyThreadLayout"); 

    m_pbtnKill = new QPushButton( this, "m_pbtnKill" );

    FrmRubyThreadLayout->addWidget( m_pbtnKill, 3, 3 );

    m_pbtnResume = new QPushButton( this, "m_pbtnResume" );

    FrmRubyThreadLayout->addMultiCellWidget( m_pbtnResume, 3, 3, 1, 2 );
    spacer1 = new QSpacerItem( 41, 20, QSizePolicy::Expanding, QSizePolicy::Minimum );
    FrmRubyThreadLayout->addItem( spacer1, 3, 0 );

    m_ptxtDefout = new QTextBrowser( this, "m_ptxtDefout" );
    m_ptxtDefout->setEnabled( TRUE );
    m_ptxtDefout->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)7, 0, 0, m_ptxtDefout->sizePolicy().hasHeightForWidth() ) );
    m_ptxtDefout->setMinimumSize( QSize( 250, 200 ) );
    QFont m_ptxtDefout_font(  m_ptxtDefout->font() );
    m_ptxtDefout->setFont( m_ptxtDefout_font ); 
    m_ptxtDefout->setHScrollBarMode( QTextBrowser::AlwaysOff );
    m_ptxtDefout->setTextFormat( QTextBrowser::PlainText );
    m_ptxtDefout->setWrapPolicy( QTextBrowser::Anywhere );
    m_ptxtDefout->setAutoFormatting( int( QTextBrowser::AutoAll ) );

    FrmRubyThreadLayout->addMultiCellWidget( m_ptxtDefout, 2, 2, 0, 3 );

    m_plblFilename = new QLabel( this, "m_plblFilename" );
    m_plblFilename->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)2, (QSizePolicy::SizeType)4, 0, 0, m_plblFilename->sizePolicy().hasHeightForWidth() ) );
    m_plblFilename->setFocusPolicy( QLabel::ClickFocus );
    m_plblFilename->setFrameShape( QLabel::TabWidgetPanel );
    m_plblFilename->setScaledContents( TRUE );
    m_plblFilename->setAlignment( int( QLabel::AlignVCenter | QLabel::AlignRight ) );

    FrmRubyThreadLayout->addMultiCellWidget( m_plblFilename, 0, 0, 0, 3 );

    m_plblStatus = new QLabel( this, "m_plblStatus" );
    m_plblStatus->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)7, (QSizePolicy::SizeType)5, 0, 0, m_plblStatus->sizePolicy().hasHeightForWidth() ) );

    FrmRubyThreadLayout->addMultiCellWidget( m_plblStatus, 1, 1, 2, 3 );

    m_plblFilename_2_2 = new QLabel( this, "m_plblFilename_2_2" );
    m_plblFilename_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)5, 0, 0, m_plblFilename_2_2->sizePolicy().hasHeightForWidth() ) );
    m_plblFilename_2_2->setAlignment( int( QLabel::AlignVCenter | QLabel::AlignRight ) );

    FrmRubyThreadLayout->addMultiCellWidget( m_plblFilename_2_2, 1, 1, 0, 1 );
    languageChange();
    resize( QSize(334, 407).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );
}

/*
 *  Destroys the object and frees any allocated resources
 */
FrmRubyThread::~FrmRubyThread()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void FrmRubyThread::languageChange()
{
    setCaption( tr2i18n( "Script" ) );
    m_pbtnKill->setText( tr2i18n( "KILL" ) );
    m_pbtnResume->setText( tr2i18n( "RESUME" ) );
    m_plblFilename->setText( tr2i18n( "tesssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssst" ) );
    m_plblStatus->setText( QString::null );
    m_plblFilename_2_2->setText( tr2i18n( "Status:" ) );
}

#include "rubythreadtool.moc"
