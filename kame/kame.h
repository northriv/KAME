/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU General 
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
class QDockWidget;
class QToolBar;
class QPropertyAnimation;
class QUrl;

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
    void signalAllModulesLoaded(); //!< Call after all driver modules are loaded.

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
    //! Declared unconditionally even though only a USE_RUBY build can reach
    //! it: a virtual behind a build-file macro moves every later vtable slot
    //! for the targets that lack the macro.  Same failure as 8bb86a9b6, one
    //! table over.  The body is what is gated, in kame.cpp.
    virtual void rubyLineShellAction_activated();
    virtual void pythonLineShellAction_activated();
    virtual void jupyterConsoleAction_activated( QAction *act );
    virtual void jupyterQtConsoleAction_activated( QAction *act );
    virtual void jupyterNotebookAction_activated( QAction *act );
    //! Handle clicks on hyperlinks in a script / IPython output pane.
    void onScriptLinkClicked(const QUrl &url);
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
	bool eventFilter(QObject *obj, QEvent *event) override;
	void placeNewWindow(QWidget *w);
	QMdiSubWindow* addDockableWindow(QMdiArea *area, QWidget *widget, bool closable);
	QMdiArea *m_pMdiCentral, *m_pMdiLeft, *m_pMdiRight;
	QDockWidget *m_pDockLeft, *m_pDockRight;
	//! Thin always-visible bars at the window edges, one button per toolbox
	//! pane: click to reveal that pane, click the revealed one to hide the
	//! toolbox again (auto-hide, VS/Dock style but click-driven — a
	//! hover-driven panel would pop open while the pointer travels to a graph).
	QToolBar *m_pStripLeft, *m_pStripRight;
	//! One entry per toolbox pane, tying its strip/View-menu action to the
	//! subwindow it reveals.  The action lives in both the strip and the View
	//! menu, so both routes go through toggleToolboxPane().
	struct ToolboxPane {
		QAction *action;
		QDockWidget *dock;
		QMdiArea *area;
		QMdiSubWindow *wnd;
	};
	std::deque<ToolboxPane> m_toolboxPanes;
	void toggleToolboxPane(QMdiSubWindow *wnd);
	//! Syncs the check marks with what is actually on screen.
	void updateToolboxStrips();

	//! Dock-style auto-hide for a toolbox floating at a screen edge: it rests
	//! shrunk to a narrow bar there — just its MDI tab column — and grows back
	//! to full width under the pointer, shrinking again once the pointer has
	//! been elsewhere for a moment.  Only where windows can actually be placed
	//! (not Wayland), and only while the toolbox is floating.
	struct EdgeSlider {
		QDockWidget *dock;
		QMdiArea *area;             //!< the toolbox's pane stack, for its minimum
		QPropertyAnimation *anim;   //!< animates dock->geometry()
		QRect expanded;             //!< full size; follows the user's own moves
		int collapsedWidth;
		bool left;                  //!< which screen edge it clings to
		bool collapsed;
		int idleTicks;
		bool autoHide;              //!< per-toolbox switch, from the View menu
	};
	std::deque<EdgeSlider> m_edgeSliders;
	QTimer *m_pEdgeHoverTimer = nullptr;
	void setupEdgeAutoHide(const QRect &screen);
	//! Trims the toolboxes against the message window once their frames exist.
	void fitToolboxHeights();
	//! Reveals a toolbox and hands it the keyboard: west at startup, east once
	//! a .kam has finished loading.
	void focusToolbox(bool left);
	void pollEdgeAutoHide();
	void setToolboxCollapsed(EdgeSlider &slider, bool collapse);
	//! nullptr where a dock has no edge slider (docked layout, or Wayland).
	EdgeSlider *edgeSliderFor(QDockWidget *dock);
	int m_cascadeIndex = 0;
	void closeEvent( QCloseEvent* ce );
	shared_ptr<XScriptingThread> runNewScript(const XString &label, const XString &filename);
	QTimer *m_pTimer;
	shared_ptr<XMeasure> m_measure;
	std::deque<xqcon_ptr> m_conScriptThreadList;
};

#endif /*KAME_H*/
