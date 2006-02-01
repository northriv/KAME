#include <klocale.h>
/****************************************************************************
** Form implementation generated from reading ui file '../../../../../kame/users/tempcontrol/forms/tempcontrolform.ui'
**
** Created: 水  2 1 03:47:28 2006
**      by: The User Interface Compiler ($Id: tempcontrolform.cpp,v 1.1 2006/02/01 18:45:44 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "tempcontrolform.h"

#include <qvariant.h>
#include <qpushbutton.h>
#include <qlabel.h>
#include <qlcdnumber.h>
#include <qcombobox.h>
#include <qgroupbox.h>
#include <qlineedit.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>
#include <qaction.h>
#include <qmenubar.h>
#include <qpopupmenu.h>
#include <qtoolbar.h>

/*
 *  Constructs a FrmTempControl as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 */
FrmTempControl::FrmTempControl( QWidget* parent, const char* name, WFlags fl )
    : QMainWindow( parent, name, fl )
{
    (void)statusBar();
    if ( !name )
	setName( "FrmTempControl" );
    setCentralWidget( new QWidget( this, "qt_central_widget" ) );
    FrmTempControlLayout = new QGridLayout( centralWidget(), 1, 1, 11, 6, "FrmTempControlLayout"); 

    layout7_2 = new QHBoxLayout( 0, 0, 6, "layout7_2"); 

    textLabel5_2 = new QLabel( centralWidget(), "textLabel5_2" );
    layout7_2->addWidget( textLabel5_2 );

    m_lcdSourceTemp = new QLCDNumber( centralWidget(), "m_lcdSourceTemp" );
    m_lcdSourceTemp->setPaletteBackgroundColor( QColor( 255, 255, 255 ) );
    m_lcdSourceTemp->setBackgroundOrigin( QLCDNumber::ParentOrigin );
    m_lcdSourceTemp->setFrameShape( QLCDNumber::Box );
    m_lcdSourceTemp->setFrameShadow( QLCDNumber::Raised );
    m_lcdSourceTemp->setSmallDecimalPoint( FALSE );
    m_lcdSourceTemp->setNumDigits( 7 );
    m_lcdSourceTemp->setMode( QLCDNumber::Dec );
    m_lcdSourceTemp->setSegmentStyle( QLCDNumber::Flat );
    m_lcdSourceTemp->setProperty( "value", 3 );
    m_lcdSourceTemp->setProperty( "intValue", 3 );
    layout7_2->addWidget( m_lcdSourceTemp );

    textLabel9_2 = new QLabel( centralWidget(), "textLabel9_2" );
    textLabel9_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel9_2->sizePolicy().hasHeightForWidth() ) );
    layout7_2->addWidget( textLabel9_2 );

    FrmTempControlLayout->addLayout( layout7_2, 3, 0 );

    layout21 = new QHBoxLayout( 0, 0, 6, "layout21"); 

    textLabel17 = new QLabel( centralWidget(), "textLabel17" );
    layout21->addWidget( textLabel17 );

    m_cmbHeaterMode = new QComboBox( FALSE, centralWidget(), "m_cmbHeaterMode" );
    layout21->addWidget( m_cmbHeaterMode );

    FrmTempControlLayout->addLayout( layout21, 0, 0 );

    groupBox1 = new QGroupBox( centralWidget(), "groupBox1" );
    groupBox1->setColumnLayout(0, Qt::Vertical );
    groupBox1->layout()->setSpacing( 6 );
    groupBox1->layout()->setMargin( 2 );
    groupBox1Layout = new QGridLayout( groupBox1->layout() );
    groupBox1Layout->setAlignment( Qt::AlignTop );

    layout29 = new QVBoxLayout( 0, 0, 6, "layout29"); 

    textLabel19 = new QLabel( groupBox1, "textLabel19" );
    layout29->addWidget( textLabel19 );

    m_cmbSetupChannel = new QComboBox( FALSE, groupBox1, "m_cmbSetupChannel" );
    layout29->addWidget( m_cmbSetupChannel );

    groupBox1Layout->addLayout( layout29, 0, 0 );

    layout30 = new QHBoxLayout( 0, 0, 6, "layout30"); 

    textLabel20 = new QLabel( groupBox1, "textLabel20" );
    layout30->addWidget( textLabel20 );

    m_cmbExcitation = new QComboBox( FALSE, groupBox1, "m_cmbExcitation" );
    layout30->addWidget( m_cmbExcitation );

    groupBox1Layout->addLayout( layout30, 1, 0 );

    layout31 = new QVBoxLayout( 0, 0, 6, "layout31"); 

    textLabel21 = new QLabel( groupBox1, "textLabel21" );
    layout31->addWidget( textLabel21 );

    m_cmbThermometer = new QComboBox( FALSE, groupBox1, "m_cmbThermometer" );
    layout31->addWidget( m_cmbThermometer );

    groupBox1Layout->addLayout( layout31, 2, 0 );

    FrmTempControlLayout->addWidget( groupBox1, 8, 0 );

    layout33 = new QHBoxLayout( 0, 0, 6, "layout33"); 

    textLabel18_2_2 = new QLabel( centralWidget(), "textLabel18_2_2" );
    layout33->addWidget( textLabel18_2_2 );

    m_edP = new QLineEdit( centralWidget(), "m_edP" );
    m_edP->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edP->sizePolicy().hasHeightForWidth() ) );
    layout33->addWidget( m_edP );

    textLabel18_2_2_2 = new QLabel( centralWidget(), "textLabel18_2_2_2" );
    layout33->addWidget( textLabel18_2_2_2 );

    m_edI = new QLineEdit( centralWidget(), "m_edI" );
    m_edI->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edI->sizePolicy().hasHeightForWidth() ) );
    layout33->addWidget( m_edI );

    textLabel18_2_2_3 = new QLabel( centralWidget(), "textLabel18_2_2_3" );
    layout33->addWidget( textLabel18_2_2_3 );

    m_edD = new QLineEdit( centralWidget(), "m_edD" );
    m_edD->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, m_edD->sizePolicy().hasHeightForWidth() ) );
    layout33->addWidget( m_edD );

    FrmTempControlLayout->addLayout( layout33, 7, 0 );

    layout20 = new QHBoxLayout( 0, 0, 6, "layout20"); 

    textLabel16 = new QLabel( centralWidget(), "textLabel16" );
    layout20->addWidget( textLabel16 );

    m_cmbSourceChannel = new QComboBox( FALSE, centralWidget(), "m_cmbSourceChannel" );
    layout20->addWidget( m_cmbSourceChannel );

    FrmTempControlLayout->addLayout( layout20, 2, 0 );

    layout7 = new QHBoxLayout( 0, 0, 6, "layout7"); 

    textLabel5 = new QLabel( centralWidget(), "textLabel5" );
    layout7->addWidget( textLabel5 );

    m_lcdHeater = new QLCDNumber( centralWidget(), "m_lcdHeater" );
    m_lcdHeater->setPaletteBackgroundColor( QColor( 255, 255, 255 ) );
    m_lcdHeater->setBackgroundOrigin( QLCDNumber::ParentOrigin );
    m_lcdHeater->setFrameShape( QLCDNumber::Box );
    m_lcdHeater->setFrameShadow( QLCDNumber::Raised );
    m_lcdHeater->setSmallDecimalPoint( FALSE );
    m_lcdHeater->setNumDigits( 7 );
    m_lcdHeater->setMode( QLCDNumber::Dec );
    m_lcdHeater->setSegmentStyle( QLCDNumber::Flat );
    m_lcdHeater->setProperty( "value", 3 );
    m_lcdHeater->setProperty( "intValue", 3 );
    layout7->addWidget( m_lcdHeater );

    FrmTempControlLayout->addLayout( layout7, 1, 0 );

    layout22 = new QHBoxLayout( 0, 0, 6, "layout22"); 

    textLabel18 = new QLabel( centralWidget(), "textLabel18" );
    layout22->addWidget( textLabel18 );

    m_edTargetTemp = new QLineEdit( centralWidget(), "m_edTargetTemp" );
    layout22->addWidget( m_edTargetTemp );

    textLabel9_2_2 = new QLabel( centralWidget(), "textLabel9_2_2" );
    textLabel9_2_2->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, textLabel9_2_2->sizePolicy().hasHeightForWidth() ) );
    layout22->addWidget( textLabel9_2_2 );

    FrmTempControlLayout->addLayout( layout22, 4, 0 );

    layout22_2 = new QHBoxLayout( 0, 0, 6, "layout22_2"); 

    textLabel18_2 = new QLabel( centralWidget(), "textLabel18_2" );
    layout22_2->addWidget( textLabel18_2 );

    m_edManHeater = new QLineEdit( centralWidget(), "m_edManHeater" );
    layout22_2->addWidget( m_edManHeater );

    FrmTempControlLayout->addLayout( layout22_2, 5, 0 );

    layout34 = new QHBoxLayout( 0, 0, 6, "layout34"); 

    textLabel22 = new QLabel( centralWidget(), "textLabel22" );
    layout34->addWidget( textLabel22 );

    m_cmbPowerRange = new QComboBox( FALSE, centralWidget(), "m_cmbPowerRange" );
    layout34->addWidget( m_cmbPowerRange );

    FrmTempControlLayout->addLayout( layout34, 6, 0 );

    // toolbars

    languageChange();
    resize( QSize(192, 439).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );

    // tab order
    setTabOrder( m_cmbHeaterMode, m_cmbSourceChannel );
    setTabOrder( m_cmbSourceChannel, m_edTargetTemp );
    setTabOrder( m_edTargetTemp, m_edManHeater );
    setTabOrder( m_edManHeater, m_cmbPowerRange );
    setTabOrder( m_cmbPowerRange, m_edP );
    setTabOrder( m_edP, m_edI );
    setTabOrder( m_edI, m_edD );
    setTabOrder( m_edD, m_cmbSetupChannel );
    setTabOrder( m_cmbSetupChannel, m_cmbExcitation );
    setTabOrder( m_cmbExcitation, m_cmbThermometer );
}

/*
 *  Destroys the object and frees any allocated resources
 */
FrmTempControl::~FrmTempControl()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void FrmTempControl::languageChange()
{
    setCaption( tr2i18n( "Temperature Control" ) );
    textLabel5_2->setText( tr2i18n( "Temp." ) );
    textLabel9_2->setText( tr2i18n( "K" ) );
    textLabel17->setText( tr2i18n( "Heater Mode" ) );
    groupBox1->setTitle( tr2i18n( "Channel Setup" ) );
    textLabel19->setText( tr2i18n( "Setup Channel" ) );
    textLabel20->setText( tr2i18n( "Excitation" ) );
    textLabel21->setText( tr2i18n( "Thermometer" ) );
    textLabel18_2_2->setText( tr2i18n( "P" ) );
    textLabel18_2_2_2->setText( tr2i18n( "I" ) );
    textLabel18_2_2_3->setText( tr2i18n( "D" ) );
    textLabel16->setText( tr2i18n( "Source CH" ) );
    textLabel5->setText( tr2i18n( "Heater Power" ) );
    textLabel18->setText( tr2i18n( "Target Temp" ) );
    textLabel9_2_2->setText( tr2i18n( "K" ) );
    textLabel18_2->setText( tr2i18n( "Manual Heater" ) );
    textLabel22->setText( tr2i18n( "Power Range" ) );
}

#include "tempcontrolform.moc"
