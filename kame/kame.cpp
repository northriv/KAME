/***************************************************************************
		Copyright (C) 2002-2010 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include <QTimer>
#include <kmessagebox.h>
#include <QAction>
#include <kfiledialog.h>
#include <kurlrequester.h>
#include <kmenubar.h>
#include <QMenu>
#include <kiconloader.h>
#include <kstandarddirs.h>
#include <QDesktopWidget>
#include <QApplication>
#include <QDockWidget>
#include <QCloseEvent>
#include <QMdiArea>
#include <QMdiSubWindow>

#include "kame.h"
#include "xsignal.h"
#include "xscheduler.h"
#include "measure.h"
#include "interface.h"
#include "xrubysupport.h"
#include "xrubywriter.h"
#include "xdotwriter.h"
#include "xrubythreadconnector.h"
#include "ui_rubythreadtool.h"
#include "ui_caltableform.h"
#include "ui_recordreaderform.h"
#include "ui_nodebrowserform.h"
#include "ui_interfacetool.h"
#include "ui_graphtool.h"
#include "ui_drivertool.h"
#include "ui_scalarentrytool.h"
#include "icon.h"

QWidget *g_pFrmMain = 0L;

FrmKameMain::FrmKameMain()
	:KMainWindow(NULL) {
	resize(QSize(QApplication::desktop()->width(), height()).expandedTo(sizeHint()) );

	setToolButtonStyle(Qt::ToolButtonTextUnderIcon);

	show();

	g_pFrmMain = this;

	createActions();
	createMenus();

    KIconLoader *loader = KIconLoader::global();

	//Central MDI area.
	m_pMdiCentral = new QMdiArea( this );
    setCentralWidget( m_pMdiCentral );
    m_pMdiCentral->setViewMode(QMdiArea::TabbedView);
    m_pMdiCentral->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

//    setDockOptions(QMainWindow::ForceTabbedDocks | QMainWindow::VerticalTabs);
    //Left MDI area.
    QDockWidget* dock = new QDockWidget(i18n("KAME Toolbox West"), this);
    dock->setFeatures(QDockWidget::DockWidgetFloatable);
    dock->setWindowIcon(*g_pIconDriver);
    m_pMdiLeft = new QMdiArea( this );
    m_pMdiLeft->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    m_pMdiLeft->setViewMode(QMdiArea::TabbedView);
    m_pMdiLeft->setTabPosition(QTabWidget::West);
   dock->setWidget(m_pMdiLeft);
    addDockWidget(Qt::LeftDockWidgetArea, dock);

    //Right MDI area.
    dock = new QDockWidget(i18n("KAME Toolbox East"), this);
    dock->setFeatures(QDockWidget::DockWidgetFloatable);
    dock->setWindowIcon(*g_pIconInterface);
    m_pMdiRight= new QMdiArea( this );
    m_pMdiRight->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    m_pMdiRight->setViewMode(QMdiArea::TabbedView);
    m_pMdiRight->setTabPosition(QTabWidget::East);
   dock->setWidget(m_pMdiRight);
    addDockWidget(Qt::RightDockWidgetArea, dock);

    g_signalBuffer.reset(new XSignalBuffer());

    m_pFrmDriver = new FrmDriver(this);
    m_pFrmDriver->setWindowIcon(*g_pIconDriver);
    QMdiSubWindow* swnd = addDockableWindow(m_pMdiLeft, m_pFrmDriver, false);

    m_pFrmGraphList = new FrmGraphList(this);
    m_pFrmGraphList->setWindowIcon(*g_pIconGraph);
    addDockableWindow(m_pMdiLeft, m_pFrmGraphList, false);

    m_pFrmCalTable = new FrmCalTable(this);
    m_pFrmCalTable->setWindowIcon(loader->loadIcon("contents", KIconLoader::Toolbar, 0, false));
    addDockableWindow(m_pMdiLeft, m_pFrmCalTable, false);

    m_pFrmNodeBrowser = new FrmNodeBrowser(this);
    m_pFrmNodeBrowser->setWindowIcon(loader->loadIcon("find", KIconLoader::Toolbar, 0, false ) );
    addDockableWindow(m_pMdiLeft, m_pFrmNodeBrowser, false);

    m_pMdiLeft->activatePreviousSubWindow();
    m_pMdiLeft->activatePreviousSubWindow();
    m_pMdiLeft->activatePreviousSubWindow();

    m_pFrmInterface = new FrmInterface(this);
    m_pFrmInterface ->setWindowIcon(*g_pIconInterface);
    swnd = addDockableWindow(m_pMdiRight, m_pFrmInterface, false);

    m_pFrmScalarEntry = new FrmEntry(this);
    m_pFrmScalarEntry->setWindowIcon(*g_pIconScalar);
    addDockableWindow(m_pMdiRight, m_pFrmScalarEntry, false);

    m_pFrmRecordReader = new FrmRecordReader(this);
    m_pFrmRecordReader->setWindowIcon(*g_pIconReader);
    addDockableWindow(m_pMdiRight, m_pFrmRecordReader, false);

    m_pMdiRight->activatePreviousSubWindow();
    m_pMdiRight->activatePreviousSubWindow();

//	resize(QSize(width(), 480 ));
   
    // The root for all nodes.
    m_measure = XNode::createOrphan<XMeasure>("Measurement", false);
      
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
    connect( m_pScriptLineShellAction, SIGNAL( activated() ), this, SLOT( scriptLineShellAction_activated() ) );
    connect( m_pScriptDotSaveAction, SIGNAL( activated() ), this, SLOT( scriptDotSaveAction_activated() ) );
    connect( m_pFileLogAction, SIGNAL( toggled(bool) ), this, SLOT( fileLogAction_toggled(bool) ) );
    
	connect(qApp, SIGNAL(aboutToQuit()), this, SLOT(aboutToQuit()));
	connect(qApp, SIGNAL( lastWindowClosed() ), qApp, SLOT( quit() ) );
   
	m_pTimer = new QTimer(this);
	connect(m_pTimer, SIGNAL (timeout() ), this, SLOT(processSignals()));
	m_pTimer->start(0);
	
	scriptLineShellAction_activated();
}

struct MySubWindow : public QMdiSubWindow {
	void closeEvent(QCloseEvent *e) {
		e->ignore();
	}
};
QMdiSubWindow *
FrmKameMain::addDockableWindow(QMdiArea *area, QWidget *widget, bool closable) {
	QMdiSubWindow *wnd;
	if(closable) {
		 wnd = new QMdiSubWindow();
		 wnd->setAttribute(Qt::WA_DeleteOnClose);
	}
	else {
		 wnd = new MySubWindow(); //delegated class, which ignores closing events.
		 QAction *act = new QAction(widget->windowIcon(), widget->windowTitle(), this);
	     connect(act, SIGNAL(triggered()), wnd, SLOT(showMaximized()));
	     m_pViewMenu->addAction(act);
	}
	wnd->setWidget(widget);
	area->addSubWindow(wnd);
	wnd->setWindowIcon(widget->windowIcon());
	wnd->setWindowTitle(widget->windowTitle());
	wnd->showMaximized();
	return wnd;
}

FrmKameMain::~FrmKameMain() {
	m_pTimer->stop();

	m_measure.reset();
	g_signalBuffer.reset();
}

void
FrmKameMain::aboutToQuit() {
}

void
FrmKameMain::createActions() {
    KIconLoader *loader = KIconLoader::global();

    // actions
    m_pFileOpenAction = new QAction( this );
//     fileOpenAction->setIcon( QIconSet( *IconKame48x48 ) );
    m_pFileOpenAction->setIcon( loader->loadIcon("fileopen",
																  KIconLoader::Toolbar, 0, false ) );
    m_pFileSaveAction = new QAction( this );
    m_pFileSaveAction->setEnabled( TRUE );
    m_pFileSaveAction->setIcon( loader->loadIcon("filesave",
																  KIconLoader::Toolbar, 0, false ) );
    m_pFileCloseAction = new QAction( this );
    m_pFileCloseAction->setEnabled( TRUE );
//     fileCloseAction->setIcon( QIconSet( *IconClose48x48 ) );
    m_pFileCloseAction->setIcon( loader->loadIcon("fileclose",
																   KIconLoader::Toolbar, 0, false ) );
    m_pFileExitAction = new QAction( this );
//     fileExitAction->setIcon( QIconSet( *IconStop48x48 ) );
    m_pFileExitAction->setIcon( loader->loadIcon("exit",
																  KIconLoader::Toolbar, 0, false ) );
    m_pHelpContentsAction = new QAction( this );
    m_pHelpIndexAction = new QAction( this, "helpIndexAction" );
    m_pHelpAboutAction = new QAction( this, "helpAboutAction" );
    m_pHelpAboutAction->setIcon( loader->loadIcon("info",
																   KIconLoader::Toolbar, 0, false ) );
    m_pFileLogAction = new QAction( this );
    m_pFileLogAction->setCheckable( true );
    m_pFileLogAction->setChecked( g_bLogDbgPrint );
    m_pFileLogAction->setIcon( loader->loadIcon("toggle_log",
																 KIconLoader::Toolbar, 0, false ) );
//    m_pMesRunAction = new QAction( this, "mesRunAction" );
//    m_pMesRunAction->setEnabled( TRUE );
	//   m_pMesRunAction->setIcon( QIconSet( *g_pIconDriver) );
    m_pMesStopAction = new QAction( this );
    m_pMesStopAction->setEnabled( TRUE );
    m_pMesStopAction->setIcon( QIcon( *g_pIconStop) );
    m_pScriptRunAction = new QAction( this );
    m_pScriptRunAction->setEnabled( TRUE );
    m_pScriptRunAction->setIcon( QIcon( *g_pIconScript) );
    m_pScriptLineShellAction = new QAction( this );
    m_pScriptLineShellAction->setEnabled( TRUE );
    m_pScriptLineShellAction->setIcon( loader->loadIcon("openterm",
		   KIconLoader::Toolbar, 0, false ) );
    m_pScriptDotSaveAction = new QAction( this );
    m_pScriptDotSaveAction->setEnabled( TRUE );
    m_pScriptDotSaveAction->setIcon( loader->loadIcon("filesave",
																	   KIconLoader::Toolbar, 0, false ) );

    m_pFileOpenAction->setText( i18n( "&Open..." ) );
    m_pFileOpenAction->setShortcut( i18n( "Ctrl+O" ) );
    m_pFileSaveAction->setText( tr( "&Save..." ) );
    m_pFileSaveAction->setShortcut( i18n( "Ctrl+S" ) );
    m_pFileExitAction->setText( i18n( "E&xit" ) );
    m_pHelpContentsAction->setText( i18n( "&Contents..." ) );
    m_pHelpIndexAction->setText( i18n( "&Index..." ) );
    m_pHelpAboutAction->setText( i18n( "&About" ) );
    m_pFileLogAction->setText( i18n( "&Log Debugging Info" ) );
    m_pMesStopAction->setText( i18n( "&Stop" ) );
    m_pScriptRunAction->setText( i18n( "&Run..." ) );
    m_pScriptLineShellAction->setText( i18n( "&New Line Shell" ) );
    m_pScriptDotSaveAction->setText( i18n( "&Graphviz Save .dot..." ) );
    m_pFileCloseAction->setText( i18n( "&Close" ) );
}
void
FrmKameMain::createMenus() {

    // menubar
    m_pFileMenu = menuBar()->addMenu(i18n( "&File" ) );
    m_pFileMenu->addAction(m_pFileOpenAction);
    m_pFileMenu->addAction(m_pFileSaveAction);
    m_pFileMenu->addAction(m_pFileCloseAction);
    m_pFileMenu->addSeparator();
    m_pFileMenu->addAction(m_pFileLogAction);
    m_pFileMenu->addSeparator();
    m_pFileMenu->addAction(m_pFileExitAction);

    m_pMeasureMenu = menuBar()->addMenu(i18n( "&Measure" ));
    m_pMeasureMenu->addAction(m_pMesStopAction);

    m_pScriptMenu = menuBar()->addMenu( i18n( "&Script" ) );
    m_pScriptMenu->addAction(m_pScriptRunAction);
    m_pScriptMenu->addAction(m_pScriptLineShellAction);
    m_pScriptMenu->addSeparator();
    m_pScriptMenu->addAction(m_pScriptDotSaveAction);

    m_pViewMenu = menuBar()->addMenu(i18n( "&View" ) );

    m_pHelpMenu = menuBar()->addMenu(i18n( "&Help" ) );
    m_pHelpMenu->addAction(m_pHelpContentsAction);
    m_pHelpMenu->addAction(m_pHelpIndexAction );
    m_pHelpMenu->addSeparator();
    m_pHelpMenu->addAction(m_pHelpAboutAction);
}

void
FrmKameMain::processSignals() {
	bool idle = g_signalBuffer->synchronize();
	if(idle) {
		usleep(5000);
	#ifdef HAVE_LIBGCCPP
		static XTime last = XTime::now();
		if(XTime::now() - last > 3) {
			GC_gcollect();
			last = XTime::now();
		}
	#endif    
	}
	usleep(500);
}

void
FrmKameMain::closeEvent( QCloseEvent* ce )
{
	bool opened = false;
    Snapshot shot(*m_measure->interfaces());
    if(shot.size()) {
    	const XNode::NodeList &list(*shot.list());
		for(XNode::const_iterator it = list.begin(); it != list.end(); it++) {
			shared_ptr<XInterface> _interface = dynamic_pointer_cast<XInterface>(*it);
			if(_interface->isOpened()) opened = true;
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

void FrmKameMain::fileOpenAction_activated()
{
	QString filename = KFileDialog::getOpenFileName (
		KUrl(),
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
		KUrl(),
		"*.kam|KAME2 Measurement files (*.kam)\n"
		"*.mes|KAME1 Measurement files (*.mes)\n"
		"*.*|All files (*.*)",
		this,
		i18n("Save Measurement File") );
	if(!filename.isEmpty())
	{
		std::ofstream ofs(filename.toLocal8Bit().data(), std::ios::out);
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

int
FrmKameMain::openMes(const XString &filename)
{
	if(!filename.empty())
	{
		runNewScript("Open Measurement", filename );
//		while(rbthread->isAlive()) {
//			KApplication::kApplication()->processEvents();
//			g_signalBuffer->synchronize();
//		}
//          closeWindow(view);
		return 0;
	}
    return -1;
}

shared_ptr<XRubyThread>
FrmKameMain::runNewScript(const XString &label, const XString &filename) {
	shared_ptr<XRubyThread> rbthread = m_measure->ruby()->
		create<XRubyThread>(label.c_str(), true, filename );
	FrmRubyThread* form = new FrmRubyThread(this);
	m_conRubyThreadList.push_back(xqcon_create<XRubyThreadConnector>(
									  rbthread, form, m_measure->ruby()));
	addDockableWindow(m_pMdiCentral, form, true);

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
	return rbthread;
}
void FrmKameMain::scriptRunAction_activated()
{
	QString filename = KFileDialog::getOpenFileName (
		KUrl(),
		"*.seq|KAME Script files (*.seq)",
		this,
		i18n("Open Script File") );
	if(!filename.isEmpty()) {
		static unsigned int thread_no = 1;
		runNewScript(formatString("Thread%d", thread_no), filename );
		thread_no++;
	}
}

void FrmKameMain::scriptLineShellAction_activated()
{
    QString filename = KStandardDirs::locate("appdata", "rubylineshell.rb");
    if(filename.isEmpty()) {
        g_statusPrinter->printError("No KAME ruby support file installed.");
    }
    else {
		static unsigned int int_no = 1;
		runNewScript(formatString("Line Shell%d", int_no), filename );
		int_no++;
	}
}

void FrmKameMain::scriptDotSaveAction_activated()
{
	QString filename = KFileDialog::getSaveFileName (
		KUrl(),
		"*.dot|Graphviz dot files (*.dot)\n"
		"*.*|All files (*.*)",
		this,
		i18n("Save Graphviz dot File") );
	if(!filename.isEmpty())
	{
		std::ofstream ofs(filename.toLocal8Bit().data(), std::ios::out);
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

