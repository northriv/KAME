#include <klocale.h>
/****************************************************************************
** Form implementation generated from reading ui file '../../../../../kame/users/magnetps/forms/magnetpsform.ui'
**
** Created: 木  3 2 16:39:08 2006
**      by: The User Interface Compiler ($Id: magnetpsform.cpp,v 1.1.2.1 2006/03/02 09:19:47 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "magnetpsform.h"

#include <qvariant.h>
#include <qpushbutton.h>
#include <qlabel.h>
#include <qlcdnumber.h>
#include <kled.h>
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
 *  Constructs a FrmMagnetPS as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 */
FrmMagnetPS::FrmMagnetPS( QWidget* parent, const char* name, WFlags fl )
    : QMainWindow( parent, name, fl )
{
    (void)statusBar();
    if ( !name )
	setName( "FrmMagnetPS" );
    setCentralWidget( new QWidget( this, "qt_central_widget" ) );
    FrmMagnetPSLayout = new QGridLayout( centralWidget(), 1, 1, 2, 6, "FrmMagnetPSLayout"); 

    layout15 = new QVBoxLayout( 0, 0, 6, "layout15"); 

    layout7 = new QHBoxLayout( 0, 0, 6, "layout7"); 

    textLabel5 = new QLabel( centralWidget(), "textLabel5" );
    textLabel5->setFrameShape( QLabel::NoFrame );
    textLabel5->setFrameShadow( QLabel::Plain );
    layout7->addWidget( textLabel5 );

    m_lcdMagnetField = new QLCDNumber( centralWidget(), "m_lcdMagnetField" );
    m_lcdMagnetField->setPaletteBackgroundColor( QColor( 255, 255, 255 ) );
    m_lcdMagnetField->setBackgroundOrigin( QLCDNumber::ParentOrigin );
    m_lcdMagnetField->setFrameShape( QLCDNumber::Box );
    m_lcdMagnetField->setFrameShadow( QLCDNumber::Raised );
    m_lcdMagnetField->setSmallDecimalPoint( FALSE );
    m_lcdMagnetField->setNumDigits( 8 );
    m_lcdMagnetField->setMode( QLCDNumber::Dec );
    m_lcdMagnetField->setSegmentStyle( QLCDNumber::Flat );
    m_lcdMagnetField->setProperty( "value", 3 );
    m_lcdMagnetField->setProperty( "intValue", 3 );
    layout7->addWidget( m_lcdMagnetField );

    textLabel9 = new QLabel( centralWidget(), "textLabel9" );
    textLabel9->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel9->sizePolicy().hasHeightForWidth() ) );
    layout7->addWidget( textLabel9 );
    layout15->addLayout( layout7 );

    layout8 = new QHBoxLayout( 0, 0, 6, "layout8"); 

    textLabel7 = new QLabel( centralWidget(), "textLabel7" );
    layout8->addWidget( textLabel7 );

    m_lcdOutputField = new QLCDNumber( centralWidget(), "m_lcdOutputField" );
    m_lcdOutputField->setPaletteBackgroundColor( QColor( 255, 255, 255 ) );
    m_lcdOutputField->setFrameShape( QLCDNumber::Box );
    m_lcdOutputField->setFrameShadow( QLCDNumber::Raised );
    m_lcdOutputField->setSmallDecimalPoint( FALSE );
    m_lcdOutputField->setNumDigits( 8 );
    m_lcdOutputField->setMode( QLCDNumber::Dec );
    m_lcdOutputField->setSegmentStyle( QLCDNumber::Flat );
    m_lcdOutputField->setProperty( "value", 3 );
    m_lcdOutputField->setProperty( "intValue", 3 );
    layout8->addWidget( m_lcdOutputField );

    textLabel10 = new QLabel( centralWidget(), "textLabel10" );
    textLabel10->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel10->sizePolicy().hasHeightForWidth() ) );
    layout8->addWidget( textLabel10 );
    layout15->addLayout( layout8 );

    layout9 = new QHBoxLayout( 0, 0, 6, "layout9"); 

    textLabel6 = new QLabel( centralWidget(), "textLabel6" );
    layout9->addWidget( textLabel6 );

    m_lcdCurrent = new QLCDNumber( centralWidget(), "m_lcdCurrent" );
    m_lcdCurrent->setPaletteBackgroundColor( QColor( 255, 255, 255 ) );
    m_lcdCurrent->setFrameShape( QLCDNumber::Box );
    m_lcdCurrent->setFrameShadow( QLCDNumber::Raised );
    m_lcdCurrent->setSmallDecimalPoint( FALSE );
    m_lcdCurrent->setNumDigits( 8 );
    m_lcdCurrent->setMode( QLCDNumber::Dec );
    m_lcdCurrent->setSegmentStyle( QLCDNumber::Flat );
    m_lcdCurrent->setProperty( "value", 3 );
    m_lcdCurrent->setProperty( "intValue", 3 );
    layout9->addWidget( m_lcdCurrent );

    textLabel11 = new QLabel( centralWidget(), "textLabel11" );
    textLabel11->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel11->sizePolicy().hasHeightForWidth() ) );
    layout9->addWidget( textLabel11 );
    layout15->addLayout( layout9 );

    layout10 = new QHBoxLayout( 0, 0, 6, "layout10"); 

    textLabel8 = new QLabel( centralWidget(), "textLabel8" );
    layout10->addWidget( textLabel8 );

    m_lcdVoltage = new QLCDNumber( centralWidget(), "m_lcdVoltage" );
    m_lcdVoltage->setPaletteBackgroundColor( QColor( 255, 255, 255 ) );
    m_lcdVoltage->setFrameShape( QLCDNumber::Box );
    m_lcdVoltage->setFrameShadow( QLCDNumber::Raised );
    m_lcdVoltage->setSmallDecimalPoint( FALSE );
    m_lcdVoltage->setNumDigits( 7 );
    m_lcdVoltage->setMode( QLCDNumber::Dec );
    m_lcdVoltage->setSegmentStyle( QLCDNumber::Flat );
    m_lcdVoltage->setProperty( "value", 3 );
    m_lcdVoltage->setProperty( "intValue", 3 );
    layout10->addWidget( m_lcdVoltage );

    textLabel12 = new QLabel( centralWidget(), "textLabel12" );
    textLabel12->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel12->sizePolicy().hasHeightForWidth() ) );
    layout10->addWidget( textLabel12 );
    layout15->addLayout( layout10 );

    FrmMagnetPSLayout->addMultiCellLayout( layout15, 0, 2, 0, 0 );

    layout13 = new QHBoxLayout( 0, 0, 6, "layout13"); 

    m_ledSwitchHeater = new KLed( centralWidget(), "m_ledSwitchHeater" );
    m_ledSwitchHeater->setShape( KLed::Circular );
    m_ledSwitchHeater->setLook( KLed::Raised );
    layout13->addWidget( m_ledSwitchHeater );

    textLabel15 = new QLabel( centralWidget(), "textLabel15" );
    layout13->addWidget( textLabel15 );

    FrmMagnetPSLayout->addLayout( layout13, 0, 1 );

    layout13_2 = new QHBoxLayout( 0, 0, 6, "layout13_2"); 

    m_ledPersistent = new KLed( centralWidget(), "m_ledPersistent" );
    m_ledPersistent->setShape( KLed::Circular );
    m_ledPersistent->setLook( KLed::Raised );
    layout13_2->addWidget( m_ledPersistent );

    textLabel15_2 = new QLabel( centralWidget(), "textLabel15_2" );
    layout13_2->addWidget( textLabel15_2 );

    FrmMagnetPSLayout->addLayout( layout13_2, 1, 1 );
    spacer5 = new QSpacerItem( 20, 31, QSizePolicy::Minimum, QSizePolicy::Expanding );
    FrmMagnetPSLayout->addItem( spacer5, 2, 1 );

    m_ckbAllowPersistent = new QCheckBox( centralWidget(), "m_ckbAllowPersistent" );
    m_ckbAllowPersistent->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, m_ckbAllowPersistent->sizePolicy().hasHeightForWidth() ) );

    FrmMagnetPSLayout->addWidget( m_ckbAllowPersistent, 3, 1 );

    layout16 = new QHBoxLayout( 0, 0, 6, "layout16"); 

    textLabel13 = new QLabel( centralWidget(), "textLabel13" );
    textLabel13->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)5, 0, 0, textLabel13->sizePolicy().hasHeightForWidth() ) );
    layout16->addWidget( textLabel13 );

    m_edTargetField = new QLineEdit( centralWidget(), "m_edTargetField" );
    m_edTargetField->setMaximumSize( QSize( 80, 32767 ) );
    layout16->addWidget( m_edTargetField );

    textLabel14 = new QLabel( centralWidget(), "textLabel14" );
    textLabel14->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel14->sizePolicy().hasHeightForWidth() ) );
    layout16->addWidget( textLabel14 );

    FrmMagnetPSLayout->addLayout( layout16, 3, 0 );

    layout17 = new QHBoxLayout( 0, 0, 6, "layout17"); 

    textLabel13_2 = new QLabel( centralWidget(), "textLabel13_2" );
    layout17->addWidget( textLabel13_2 );

    m_edSweepRate = new QLineEdit( centralWidget(), "m_edSweepRate" );
    m_edSweepRate->setMaximumSize( QSize( 80, 32767 ) );
    layout17->addWidget( m_edSweepRate );

    textLabel14_2 = new QLabel( centralWidget(), "textLabel14_2" );
    textLabel14_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel14_2->sizePolicy().hasHeightForWidth() ) );
    layout17->addWidget( textLabel14_2 );

    FrmMagnetPSLayout->addLayout( layout17, 4, 0 );

    // toolbars

    languageChange();
    resize( QSize(339, 205).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );

    // tab order
    setTabOrder( m_edTargetField, m_edSweepRate );
    setTabOrder( m_edSweepRate, m_ckbAllowPersistent );
}

/*
 *  Destroys the object and frees any allocated resources
 */
FrmMagnetPS::~FrmMagnetPS()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void FrmMagnetPS::languageChange()
{
    setCaption( tr2i18n( "Magnet Power Supply Settings" ) );
    textLabel5->setText( tr2i18n( "Magnet Field" ) );
    textLabel9->setText( tr2i18n( "T" ) );
    textLabel7->setText( tr2i18n( "Output Field" ) );
    textLabel10->setText( tr2i18n( "T" ) );
    textLabel6->setText( tr2i18n( "Output Current" ) );
    textLabel11->setText( tr2i18n( "A" ) );
    textLabel8->setText( tr2i18n( "Output Voltage" ) );
    textLabel12->setText( tr2i18n( "V" ) );
    textLabel15->setText( tr2i18n( "Switch Heater" ) );
    textLabel15_2->setText( tr2i18n( "Persistent" ) );
    m_ckbAllowPersistent->setText( tr2i18n( "Allow Persistent" ) );
    textLabel13->setText( tr2i18n( "Field" ) );
    textLabel14->setText( tr2i18n( "T" ) );
    textLabel13_2->setText( tr2i18n( "Sweep Rate" ) );
    textLabel14_2->setText( tr2i18n( "T/min" ) );
}

#include "magnetpsform.moc"
