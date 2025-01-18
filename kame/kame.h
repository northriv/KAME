/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
 ***************************************************************************/
#ifndef KAME_H
#define KAME_H

#include "support.h"
#include "xnodeconnector.h"
#include <QMainWindow>

class Ui_FrmRecordReader;
typedef QForm<QWidget, Ui_FrmRecordReader> FrmRecordReader;
class Ui_FrmGraphList;
typedef QForm<QWidget, Ui_FrmGraphList> FrmGraphList;
class Ui_FrmCalTable;
typedef QForm<QWidget, Ui_FrmCalTable> FrmCalTable;
class Ui_FrmInterface;
typedef QForm<QWidget, Ui_FrmInterface> FrmInterface;
class Ui_FrmDriver;
typedef QForm<QWidget, Ui_FrmDriver> FrmDriver;
class Ui_FrmEntry;
typedef QForm<QWidget, Ui_FrmEntry> FrmEntry;
class Ui_FrmSequence;
typedef QForm<QWidget, Ui_FrmSequence> FrmSequence;
class Ui_FrmNodeBrowser;
typedef QForm<QWidget, Ui_FrmNodeBrowser> FrmNodeBrowser;
class QTimer;
class QAction;
class QActionGroup;
class QMenu;
class XMeasure;
class XScriptingThread;
class QMdiArea;
class QMdiSubWindow;

/*! Main window widget of KAME.
 * use \a g_pFrmMain to access this.
 * \sa g_pFrmMain
 */
class FrmKameMain : public QMainWindow {
	Q_OBJECT
public:
	FrmKameMain();
	~FrmKameMain();

	QMenu *m_pFileMenu;
	QMenu *m_pMeasureMenu;
	QMenu *m_pScriptMenu;
    QMenu* m_pJupyterConsoleMenu;
    QMenu* m_pJupyterQtConsoleMenu;
    QMenu* m_pJupyterNotebookMenu;
    QMenu *m_pViewMenu;
    QMenu *m_pGraphThemeMenu;
	QMenu *m_pHelpMenu;
	QAction* m_pFileOpenAction;
	QAction* m_pFileSaveAction;
	QAction* m_pFileExitAction;
	QAction* m_pHelpContentsAction;
	QAction* m_pHelpIndexAction;
	QAction* m_pHelpAboutAction;
	QAction* m_pFileLogAction;
	//    QAction* m_pMesRunAction;
	QAction* m_pMesStopAction;
	QAction* m_pScriptRunAction;
    QAction* m_pRubyLineShellAction;
    QAction* m_pPythonLineShellAction;
	QAction* m_pFileCloseAction;
    QAction* m_pGraphThemeNightAction;
    QAction* m_pGraphThemeDaylightAction;
    QActionGroup *m_pGraphThemeActionGroup;

	FrmRecordReader *m_pFrmRecordReader;
	FrmGraphList *m_pFrmGraphList;
	FrmCalTable *m_pFrmCalTable;
	FrmInterface *m_pFrmInterface;
	FrmDriver *m_pFrmDriver;
	FrmEntry *m_pFrmScalarEntry;
	FrmNodeBrowser *m_pFrmNodeBrowser;

	int openMes(const XString &filename);

    bool running() const {return !!m_measure;}
public slots:
    virtual void fileCloseAction_activated();
    virtual void fileExitAction_activated();
    virtual void fileOpenAction_activated();
    virtual void fileSaveAction_activated();
    virtual void helpAboutAction_activated();
    virtual void helpContentsAction_activated();
    virtual void helpIndexAction_activated();
    //    virtual void mesRunAction_activated();
    virtual void mesStopAction_activated();
    virtual void scriptMenu_activated();
    virtual void scriptRunAction_activated();
    virtual void rubyLineShellAction_activated();
    virtual void pythonLineShellAction_activated();
    virtual void jupyterConsoleAction_activated( QAction *act );
    virtual void jupyterQtConsoleAction_activated( QAction *act );
    virtual void jupyterNotebookAction_activated( QAction *act );
    virtual void fileLogAction_toggled( bool var );
    virtual void graphThemeNightAction_toggled( bool var );
//    virtual void graphThemeDayightAction_toggled( bool var );
protected slots:
    virtual void aboutToQuit();
    virtual void processSignals();
private:
    void scriptLineShellAction_activated(const char *filename);
    void createActions();
	void createMenus();
	QMdiSubWindow* addDockableWindow(QMdiArea *area, QWidget *widget, bool closable);
	QMdiArea *m_pMdiCentral, *m_pMdiLeft, *m_pMdiRight;
	void closeEvent( QCloseEvent* ce );
	shared_ptr<XScriptingThread> runNewScript(const XString &label, const XString &filename);
	QTimer *m_pTimer;
	shared_ptr<XMeasure> m_measure;
	std::deque<xqcon_ptr> m_conScriptThreadList;
};

#endif /*KAME_H*/
