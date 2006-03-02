#include <qtimer.h>
#include <qmessagebox.h>
#include <kmessagebox.h>
#include <qaction.h>
#include <kapp.h>
#include <kmdimainfrm.h>
#include <kfiledialog.h>
#include <qgroupbox.h>
#include <kurlrequester.h>
#include <qpushbutton.h>
#include <qcheckbox.h>
#include <qsplitter.h>
#include <qtable.h>
#include <qtextbrowser.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>
#include <qaction.h>
#include <qmenubar.h>
#include <qpopupmenu.h>
#include <qtoolbar.h>
#include <qimage.h>
#include <kiconloader.h>
#include <klocale.h>
#include <kapplication.h>

#include "kame.h"
#include "xsignal.h"
#include "measure.h"
#include "interface.h"
#include "xrubysupport.h"
#include "xrubywriter.h"
#include "xdotwriter.h"
#include "xrubythreadconnector.h"
#include "script/forms/rubythreadtool.h"
#include "thermometer/forms/caltableform.h"
#include "analyzer/forms/recordreaderform.h"
#include "forms/interfacetool.h"
#include "forms/graphtool.h"
#include "forms/drivertool.h"
#include "forms/scalarentrytool.h"
#include "icons/icon.h"

QWidget *g_pFrmMain = 0L;

FrmKameMain::FrmKameMain()
 :KMdiMainFrm(NULL, "kame", KMdi::IDEAlMode)
 {    
 KApplication *app = KApplication::kApplication();   
    
    g_pFrmMain = this;
    
    setCentralWidget( new QWidget( this, "qt_central_widget" ) );
    setSysButtonsAtMenuPosition();
//    setTabWidgetVisibility(KMdi::AlwaysShowTabs);
    
    g_signalBuffer.reset(new XSignalBuffer());
    
    m_pFrmDriver = new FrmDriver(this, "Drivers");
    m_pFrmDriver->setIcon(*g_pIconDriver);
    KMdiToolViewAccessor *accessor = addToolWindow( m_pFrmDriver, KDockWidget::DockLeft, 
         getMainDockWidget(), 50, m_pFrmDriver->caption() );
    accessor->show();
    
    m_pFrmInterface = new FrmInterface(this, "Interfaces");
    m_pFrmInterface->setIcon(*g_pIconInterface);
    accessor = addToolWindow( m_pFrmInterface, KDockWidget::DockRight,
         getMainDockWidget(), 50, m_pFrmInterface->caption() );
    accessor->show();
    
    m_pFrmScalarEntry = new FrmEntry(this, "Scalar Entries");
    m_pFrmScalarEntry->setIcon(*g_pIconScalar);
    accessor = addToolWindow( m_pFrmScalarEntry, KDockWidget::DockRight, 
         getMainDockWidget(), 50, m_pFrmScalarEntry->caption() );
    
    m_pFrmGraphList = new FrmGraphList(this, "Graph");
    m_pFrmGraphList->setIcon(*g_pIconGraph);
    accessor = addToolWindow( m_pFrmGraphList, KDockWidget::DockLeft,
         getMainDockWidget(), 30, m_pFrmGraphList->caption() );
        
    m_pFrmRecordReader = new FrmRecordReader(this, "RawStreamReader");
    m_pFrmRecordReader->setIcon(*g_pIconReader);
    accessor = addToolWindow( m_pFrmRecordReader, KDockWidget::DockRight,
         getMainDockWidget(), 20,m_pFrmRecordReader->caption() );
    
    m_pFrmCalTable = new FrmCalTable(this, "Thermometers");
//     frmCalTable->setIcon(*IconKame48x48);
    m_pFrmCalTable->setIcon(app->iconLoader()->loadIcon("contents", KIcon::Toolbar, 0, KIcon::DefaultState, 0, false ) );
    accessor = addToolWindow( m_pFrmCalTable, KDockWidget::DockLeft, 
         getMainDockWidget(), 20, m_pFrmCalTable->caption() );    
        
    // actions
    m_pFileOpenAction = new QAction( this, "fileOpenAction" );
//     fileOpenAction->setIconSet( QIconSet( *IconKame48x48 ) );
    m_pFileOpenAction->setIconSet( app->iconLoader()->loadIconSet("fileopen", 
            KIcon::Toolbar, 0, false ) );
    m_pFileSaveAction = new QAction( this, "fileSaveAction" );
    m_pFileSaveAction->setEnabled( TRUE );
    m_pFileSaveAction->setIconSet( app->iconLoader()->loadIconSet("filesave", 
            KIcon::Toolbar, 0, false ) );
    m_pFileCloseAction = new QAction( this, "fileCloseAction" );
    m_pFileCloseAction->setEnabled( TRUE );
//     fileCloseAction->setIconSet( QIconSet( *IconClose48x48 ) );
    m_pFileCloseAction->setIconSet( app->iconLoader()->loadIconSet("fileclose", 
            KIcon::Toolbar, 0, false ) );
    m_pFileExitAction = new QAction( this, "fileExitAction" );
//     fileExitAction->setIconSet( QIconSet( *IconStop48x48 ) );
    m_pFileExitAction->setIconSet( app->iconLoader()->loadIconSet("exit", 
            KIcon::Toolbar, 0, false ) );
    m_pHelpContentsAction = new QAction( this, "helpContentsAction" );
    m_pHelpIndexAction = new QAction( this, "helpIndexAction" );
    m_pHelpAboutAction = new QAction( this, "helpAboutAction" );
    m_pHelpAboutAction->setIconSet( app->iconLoader()->loadIconSet("info", 
            KIcon::Toolbar, 0, false ) );
    m_pFileLogAction = new QAction( this, "fileLogAction" );
    m_pFileLogAction->setToggleAction( true );
    m_pFileLogAction->setOn( g_bLogDbgPrint );
    m_pFileLogAction->setIconSet( app->iconLoader()->loadIconSet("toggle_log", 
            KIcon::Toolbar, 0, false ) );
//    m_pMesRunAction = new QAction( this, "mesRunAction" );
//    m_pMesRunAction->setEnabled( TRUE );
 //   m_pMesRunAction->setIconSet( QIconSet( *g_pIconDriver) );
    m_pMesStopAction = new QAction( this, "mesStopAction" );
    m_pMesStopAction->setEnabled( TRUE );
    m_pMesStopAction->setIconSet( QIconSet( *g_pIconStop) );
    m_pScriptRunAction = new QAction( this, "scriptRunAction" );
    m_pScriptRunAction->setEnabled( TRUE );
    m_pScriptRunAction->setIconSet( QIconSet( *g_pIconScript) );
    m_pScriptDotSaveAction = new QAction( this, "scriptDotSaveAction" );
    m_pScriptDotSaveAction->setEnabled( TRUE );
    m_pScriptDotSaveAction->setIconSet( app->iconLoader()->loadIconSet("filesave", 
            KIcon::Toolbar, 0, false ) );

    // toolbars
    m_pToolbar = new QToolBar( QString(""), this, DockTop ); 

    // menubar
    m_pMenubar = new QMenuBar( this, "menubar" );


    m_pFileMenu = new QPopupMenu( this );
    m_pFileOpenAction->addTo( m_pFileMenu );
    m_pFileSaveAction->addTo( m_pFileMenu );
    m_pFileCloseAction->addTo( m_pFileMenu );
    m_pFileMenu->insertSeparator();
    m_pFileLogAction->addTo( m_pFileMenu );
    m_pFileMenu->insertSeparator();
    m_pFileExitAction->addTo( m_pFileMenu );
    m_pMenubar->insertItem( QString(""), m_pFileMenu, 1 );

    m_pMeasureMenu = new QPopupMenu( this );
//    m_pMesRunAction->addTo( m_pMeasureMenu );
    m_pMesStopAction->addTo( m_pMeasureMenu );
    m_pMenubar->insertItem( QString(""), m_pMeasureMenu, 2 );

    m_pScriptMenu = new QPopupMenu( this );
    m_pScriptRunAction->addTo( m_pScriptMenu );
    m_pScriptMenu->insertSeparator();
    m_pScriptDotSaveAction->addTo( m_pScriptMenu );
    m_pMenubar->insertItem( QString(""), m_pScriptMenu, 3 );

    m_pMenubar->insertItem( QString(""), dockHideShowMenu(), 4 );
    m_pMenubar->insertItem( QString(""), windowMenu(), 5 );    

    m_pHelpMenu = new QPopupMenu( this );
    m_pHelpContentsAction->addTo( m_pHelpMenu );
    m_pHelpIndexAction->addTo( m_pHelpMenu );
    m_pHelpMenu->insertSeparator();
    m_pHelpAboutAction->addTo( m_pHelpMenu );
    m_pMenubar->insertItem( QString(""), m_pHelpMenu, 6 );

//     setCaption( i18n( "KAME 1.8.4" ) );
    
    m_pFileOpenAction->setText( i18n( "Open" ) );
    m_pFileOpenAction->setMenuText( i18n( "&Open..." ) );
    m_pFileOpenAction->setAccel( i18n( "Ctrl+O" ) );
    m_pFileSaveAction->setText( i18n( "Save" ) );
    m_pFileSaveAction->setMenuText( tr( "&Save..." ) );
    m_pFileSaveAction->setAccel( i18n( "Ctrl+S" ) );
    m_pFileExitAction->setText( i18n( "Exit" ) );
    m_pFileExitAction->setMenuText( i18n( "E&xit" ) );
    m_pFileExitAction->setAccel( QString::null );
    m_pHelpContentsAction->setText( i18n( "Contents" ) );
    m_pHelpContentsAction->setMenuText( i18n( "&Contents..." ) );
    m_pHelpContentsAction->setAccel( QString::null );
    m_pHelpIndexAction->setText( i18n( "Index" ) );
    m_pHelpIndexAction->setMenuText( i18n( "&Index..." ) );
    m_pHelpIndexAction->setAccel( QString::null );
    m_pHelpAboutAction->setText( i18n( "About" ) );
    m_pHelpAboutAction->setMenuText( i18n( "&About" ) );
    m_pHelpAboutAction->setAccel( QString::null );
    m_pFileLogAction->setMenuText( i18n( "&Log Debugging Info" ) );
//    m_pMesRunAction->setMenuText( i18n( "&Run" ) );
    m_pMesStopAction->setMenuText( i18n( "&Stop" ) );
    m_pScriptRunAction->setMenuText( i18n( "&Run..." ) );
    m_pScriptDotSaveAction->setMenuText( i18n( "&Graphviz Save .dot..." ) );
    m_pFileCloseAction->setText( i18n( "Close" ) );
    m_pFileCloseAction->setMenuText( i18n( "&Close" ) );
//     Toolbar->setLabel( i18n( "Toolbar" ) );
    if (m_pMenubar->findItem(1))
        m_pMenubar->findItem(1)->setText( i18n( "&File" ) );
    if (m_pMenubar->findItem(2))
        m_pMenubar->findItem(2)->setText( i18n( "&Measure" ) );
    if (m_pMenubar->findItem(3))
        m_pMenubar->findItem(3)->setText( i18n( "&Script" ) );
    if (m_pMenubar->findItem(4))
        m_pMenubar->findItem(4)->setText( i18n( "&View" ) );
    if (m_pMenubar->findItem(5))
        m_pMenubar->findItem(5)->setText( i18n( "&Window" ) );
    if (m_pMenubar->findItem(6))
        m_pMenubar->findItem(6)->setText( i18n( "&Help" ) );
    m_pMenubar->show();

//      clearWState( WState_Polished );   
   
    m_measure = createOrphan<XMeasure>("Measurement", false);
      
    // signals and slots connections
    connect( m_pFileCloseAction, SIGNAL( activated() ), this, SLOT( fileCloseAction_activated() ) );
    connect( m_pFileExitAction, SIGNAL( activated() ), this, SLOT( fileExitAction_activated() ) );
    connect( m_pFileOpenAction, SIGNAL( activated() ), this, SLOT( fileOpenAction_activated() ) );
    connect( m_pFileSaveAction, SIGNAL( activated() ), this, SLOT( fileSaveAction_activated() ) );
    connect( m_pHelpAboutAction, SIGNAL( activated() ), this, SLOT( helpAboutAction_activated() ) );
    connect( m_pHelpContentsAction, SIGNAL( activated() ), this, SLOT( helpContentsAction_activated() ) );
    connect( m_pHelpIndexAction, SIGNAL( activated() ), this, SLOT( helpIndexAction_activated() ) );
//    connect( m_pMesRunAction, SIGNAL( activated() ), this, SLOT( mesRunAction_activated() ) );
    connect( m_pMesStopAction, SIGNAL( activated() ), this, SLOT( mesStopAction_activated() ) );
    connect( m_pScriptRunAction, SIGNAL( activated() ), this, SLOT( scriptRunAction_activated() ) );
    connect( m_pScriptDotSaveAction, SIGNAL( activated() ), this, SLOT( scriptDotSaveAction_activated() ) );
    connect( m_pFileLogAction, SIGNAL( toggled(bool) ), this, SLOT( fileLogAction_toggled(bool) ) );
    
   connect(app, SIGNAL(aboutToQuit()), this, SLOT(aboutToQuit()));
   connect(app, SIGNAL( lastWindowClosed() ), app, SLOT( quit() ) );
   
   m_pTimer = new QTimer(this);
   connect(m_pTimer, SIGNAL (timeout() ), this, SLOT(processSignals()));
   m_pTimer->start(1);
}

FrmKameMain::~FrmKameMain()
{
  m_pTimer->stop();

  m_measure.reset();
  g_signalBuffer.reset();
}

void
FrmKameMain::aboutToQuit() {
}

void
FrmKameMain::processSignals() {
  bool idle = g_signalBuffer->synchronize();
  if(idle) {
#ifdef HAVE_LIBGCCPP
static XTime last = XTime::now();
    if(XTime::now() - last > 3) {
        GC_gcollect();
        last = XTime::now();
    }
#endif    
  }
}

void
FrmKameMain::closeEvent( QCloseEvent* ce )
{
   bool opened = false;
   { XScopedReadLock<XRecursiveRWLock> lock(m_measure->interfaceList()->childMutex());
       for(unsigned int i = 0; i < m_measure->interfaceList()->count(); i++) {
            shared_ptr<XInterface> interface = (*m_measure->interfaceList())[i];
            if(interface->isOpened()) opened = true;
       }
   }
   if(opened) {
       QMessageBox::warning( this, "KAME", i18n("Stop running first.") );
       ce->ignore();
   }
   else {
    ce->accept();
    printf("quit\n");
    m_conMeasRubyThread.reset();
    m_measure->terminate();
   
    m_measure.reset();
   }
}

void FrmKameMain::fileCloseAction_activated()
{
  m_conMeasRubyThread.reset();
  m_measure->terminate();
}


void FrmKameMain::fileExitAction_activated()
{
  close();
}

int
FrmKameMain::openMes(const QString &filename)
{
  if(!filename.isEmpty())
    {
          shared_ptr<XRubyThread> rbthread = m_measure->ruby()->
            create<XRubyThread>("Open Measurement", true, filename );
          FrmRubyThread* form = new FrmRubyThread(this);
          m_conMeasRubyThread = xqcon_create<XRubyThreadConnector>(
                rbthread, form, m_measure->ruby());
          KMdiChildView *view = createWrapper(form, form->caption(), form->caption());
          addWindow(view);
          while(rbthread->isAlive()) {
            KApplication::kApplication()->processEvents();
            g_signalBuffer->synchronize();
          }
//          closeWindow(view);

//        m_pMesRunAction->setEnabled(true);
//        m_pMesStopAction->setEnabled(false);
//        m_pFileSaveAction->setEnabled(true);
//        m_pFileOpenAction->setEnabled(false);
//        m_pFileCloseAction->setEnabled(true);
        return 0;
    }
    return -1;
}

void FrmKameMain::fileOpenAction_activated()
{
  QString filename = KFileDialog::getOpenFileName (
					   "",
                        "*.kam|KAME2 Measurement files (*.kam)\n"
                        "*.mes|KAME1 Measurement files (*.mes)\n"
                        "*.*|All files (*.*)",
					   this,
					   i18n("Open Measurement File"));
  openMes(filename);
}


void FrmKameMain::fileSaveAction_activated()
{
  QString filename = KFileDialog::getSaveFileName (
					   "",
                        "*.kam|KAME2 Measurement files (*.kam)\n"
					   "*.mes|KAME1 Measurement files (*.mes)\n"
                        "*.*|All files (*.*)",
					   this,
					   i18n("Save Measurement File") );
  if(!filename.isEmpty())
    {
        std::ofstream ofs(filename.local8Bit(), std::ios::out);
        if(ofs.good()) {
            XRubyWriter writer(m_measure, ofs);
            writer.write();
        }
    }
}


void FrmKameMain::helpAboutAction_activated()
{
  KMessageBox::about( this,
		      i18n("K's Adaptive Measurement Engine."), "KAME");
}

void FrmKameMain::helpContentsAction_activated()
{
}


void FrmKameMain::helpIndexAction_activated()
{

}

/*
void FrmKameMain::mesRunAction_activated()
{
  m_pMesRunAction->setEnabled(false);
  m_pMesStopAction->setEnabled(true);
  m_pFileCloseAction->setEnabled(false);
  m_pFileExitAction->setEnabled(false);
  m_measure->start();
}
*/

void FrmKameMain::mesStopAction_activated()
{
  m_measure->stop();
/*
 *   m_pMesRunAction->setEnabled(true);
  m_pMesStopAction->setEnabled(false);
  m_pFileCloseAction->setEnabled(true);
  m_pFileExitAction->setEnabled(true);
  */
}


void FrmKameMain::scriptRunAction_activated()
{
  QString filename = KFileDialog::getOpenFileName (
					   "",
					   "*.seq|KAME Script files (*.seq)",
					   this,
					   i18n("Open Script File") );
  if(!filename.isEmpty())
    {
      static unsigned int thread_no = 1;
      shared_ptr<XRubyThread> rbthread = m_measure->ruby()->
            create<XRubyThread>(QString().sprintf("Thread%d", thread_no).latin1(), true, filename );
      thread_no++;
      FrmRubyThread* form = new FrmRubyThread(this);
      m_conRubyThreadList.push_back(xqcon_create<XRubyThreadConnector>(
            rbthread, form, m_measure->ruby()));
      addWindow(createWrapper(form, form->caption(), form->caption()));
      // erase unused xqcon_ptr
      for(std::deque<xqcon_ptr>::iterator it = m_conRubyThreadList.begin();
            it != m_conRubyThreadList.end(); ) {
            if((*it)->isAlive()) {
                it++;
            }
            else {
                it = m_conRubyThreadList.erase(it);
            }
       }
    }
}

void FrmKameMain::scriptDotSaveAction_activated()
{
  QString filename = KFileDialog::getSaveFileName (
                       "",
                        "*.dot|Graphviz dot files (*.dot)\n"
                        "*.*|All files (*.*)",
                       this,
                       i18n("Save Graphviz dot File") );
  if(!filename.isEmpty())
    {
        std::ofstream ofs(filename.local8Bit(), std::ios::out);
        if(ofs.good()) {
            XDotWriter writer(m_measure, ofs);
            writer.write();
        }
    }
}

void FrmKameMain::fileLogAction_toggled( bool var)
{
  g_bLogDbgPrint = var;
}

