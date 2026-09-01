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
#include <QPointer>

//! What the command line asked the appearance to be, so the View menu can say
//! which one is in force: QStyleHints::colorScheme() reports the EFFECTIVE
//! scheme and cannot tell a forced Dark from a system that happens to be dark.
//! Qt::ColorScheme::Unknown means nothing was forced.
extern Qt::ColorScheme g_kameColorSchemeRequested;
//! Puts one of the three choices into effect, everywhere it has to be said.
void kameApplyColorScheme(Qt::ColorScheme scheme);
//! What the View menu was last set to, Dark if it never was.  The
//! --appearance option overrides it for one run without replacing it.
Qt::ColorScheme kameStoredColorScheme();
void kameStoreColorScheme(Qt::ColorScheme scheme);

class Ui_FrmJournalReader;
typedef QForm<QWidget, Ui_FrmJournalReader> FrmJournalReader;
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
class XJournalWriter;

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

	FrmJournalReader *m_pFrmJournalReader;
	FrmGraphList *m_pFrmGraphList;
	FrmCalTable *m_pFrmCalTable;
	FrmInterface *m_pFrmInterface;
	FrmDriver *m_pFrmDriver;
	FrmEntry *m_pFrmScalarEntry;
	FrmNodeBrowser *m_pFrmNodeBrowser;

	//! Brings the Interface pane to the front and gives it the keyboard.
	//! Called after creating a driver that came with an interface, whose port
	//! has to be set before it can be started.
	void revealInterfacePane();

	int openMes(const XString &filename);
    void signalAllModulesLoaded(); //!< Call after all driver modules are loaded.
    //! Folds every auto-hiding toolbox, and keeps it folded until the pointer
    //! has left it.  For the moment a pane opens a window of its own: the
    //! toolbox has just done its job and is now standing in front of the
    //! result.  \sa XDriverListConnector, XInterfaceListConnector
    void foldToolboxes();
    //! Holds every auto-hiding toolbox open.  For loading a measurement: what
    //! follows is a stretch of work across several drivers and their
    //! interfaces, and a toolbox that folds between each one is in the way.
    void pinToolboxes();

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
	void revealToolboxPane(ToolboxPane &pane);
	//! Syncs the check marks with what is actually on screen.
	void updateToolboxStrips();

	//! Dock-style auto-hide.  A toolbox floating at a screen edge rests shrunk
	//! to a narrow bar there — just its MDI tab column — and grows back to full
	//! width under the pointer, shrinking again once the pointer has been
	//! elsewhere for a moment.  The main window does the same downwards: it
	//! keeps its top edge and rests at half height.  Only where windows can
	//! actually be placed (not Wayland), and a toolbox only while it floats.
	struct EdgeSlider {
		QWidget *win;               //!< a floating toolbox, or the main window
		QMdiArea *area;             //!< its pane stack, for the layout minimum
		QPropertyAnimation *anim;   //!< animates win->geometry()
		QRect expanded;             //!< full size; follows the user's own moves
		int collapsedWidth;         //!< resting width of a toolbox; unused when vertical
		//! Fold downwards to half height (the main window) instead of sideways
		//! to a tab column.
		bool vertical;
		bool left;                  //!< which screen edge it clings to
		bool collapsed;
		int idleTicks;
		bool autoHide;              //!< per-window switch, from the View menu
		QAction *autoHideAction;    //!< the View-menu entry, kept in sync
		//! Text is being typed into this window: the one thing that keeps it
		//! open with the pointer elsewhere.  \sa pollEdgeAutoHide()
		bool wasFocused;
		//! Folded on purpose, and not to be reopened by the pointer that is
		//! still sitting on it -- until that pointer leaves and comes back.
		bool dismissed = false;
	};
	std::deque<EdgeSlider> m_edgeSliders;
	QTimer *m_pEdgeHoverTimer = nullptr;
	//! Magnifying the tab under the pointer.  The icon is redrawn larger, not
	//! the tab: the icon rect is fixed, so the strip never re-lays out and
	//! nothing jumps.  Style sheets cannot animate, and a tab that changed
	//! size would move its neighbours on every frame.
	class QVariantAnimation *m_pTabMagnify = nullptr;
	QPointer<class QTabBar> m_tabMagnifyBar;
	int m_tabMagnifyIdx = -1;
	//! Auto-hide waits for the end of startup and stops at the start of
	//! shutdown.  Loading the driver modules takes seconds, during which the
	//! pointer is wherever the user left it and nothing on screen is theirs to
	//! keep open yet -- a toolbox that folds itself away then looks like a
	//! failure rather than a feature.  \sa pollEdgeAutoHide()
	bool m_edgeAutoHideArmed = false;
	void setupEdgeAutoHide(const QRect &screen);
	//! Fixes the icon rect a tab bar draws into, so magnifying inside it moves
	//! nothing.  Idempotent: the poll calls it for bars that appear later.
	void setupTabMagnify(class QTabBar *tabs, class QMdiArea *area);
	//! Puts the pinned state in the window's own title bar, where a docking UI
	//! conventionally keeps it.
	void markPinned(EdgeSlider &s);
	//! The window's own title: what is loaded, the version, and the pin mark.
	//! Nothing set one at all, so the main window carried no title of any
	//! kind, and the pinned mark had nowhere to go.
	void updateWindowTitle();
	//! Grows the window if the layout wants more height than it has.
	void ensureMinimumHeight();
	//! Base name of the measurement file in the tree, for the title bar.
	QString m_titleDoc;
	//! Starts the pointer's tab growing and lets the one it left shrink back.
	void magnifyTab(class QTabBar *tabs, int idx);
	//! Trims the toolboxes against the message window once their frames exist.
	void fitToolboxHeights();
	//! Reveals a toolbox and hands it the keyboard: west at startup, east once
	//! a .kam has finished loading.
	void pollEdgeAutoHide();
	void setToolboxCollapsed(EdgeSlider &slider, bool collapse);
	//! nullptr where a window has no edge slider (docked layout, or Wayland).
	EdgeSlider *edgeSliderFor(QWidget *win);
	int m_cascadeIndex = 0;
	void closeEvent( QCloseEvent* ce ) override;
    //! Writes the four hand-placed windows' geometries out, on the way to a
    //! clean exit.  Restoring them is done in the constructor's layout pass.
    void saveWindowLayout();
    //! The geometry \a win would have open, which for a folded toolbox is not
    //! the one it has.
    QRect layoutGeometryOf(QWidget *win) const;
    //! Where folding puts a toolbox: its resting bar, at the edge it clings to.
    QRect collapsedGeometryOf(const struct EdgeSlider &s) const;
    //! Re-applies that after something else has re-laid the toolboxes out.
    void reassertToolboxFolds();
	shared_ptr<XScriptingThread> runNewScript(const XString &label, const XString &filename);
	QTimer *m_pTimer;
	shared_ptr<XMeasure> m_measure;
	//! Provenance capture, off unless KAME_JOURNAL is set.
	//! \sa doc/design/PROVENANCE.md
	shared_ptr<XJournalWriter> m_journalWriter;
	std::deque<xqcon_ptr> m_conScriptThreadList;
};

#endif /*KAME_H*/
