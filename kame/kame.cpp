/***************************************************************************
        Copyright (C) 2002-2025 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU General
		Public License and a list of authors along with this program;
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "xpythonsupport.h"
#ifdef USE_RUBY
    #include "xrubysupport.h"
#endif
#include <QTimer>
#include <QAction>
#include <QMenu>
#include <QMenuBar>
#include <QApplication>
#include <QScreen>
#include <QDockWidget>
#include <QToolBar>
#include <QToolButton>
#include <QVBoxLayout>
#include <QPropertyAnimation>
#include <QCursor>
#include <QTabBar>
#include <QProxyStyle>
#include <QMouseEvent>
#include <QCloseEvent>
#include <QMdiArea>
#include <QMdiSubWindow>
#include <QMainWindow>
#include <QWindow>
#include <QEvent>
#include <QMessageBox>
#include <QFileDialog>
#include <QDir>
#include <QFile>
#include <QTextBrowser>
#include <QDesktopServices>
#include <QUrl>
#ifdef WITH_KDE
	#include <kstandarddirs.h>
#else
	#include <QStandardPaths>
#endif

#if QT_VERSION >= QT_VERSION_CHECK(6,0,0)
    #include <QActionGroup>
#endif

#include "kame.h"
#include "xscheduler.h"
#include "measure.h"
#include "xjournal.h"
#include "interface.h"
#include "xrubywriter.h"
#include "xdotwriter.h"
#include "xscriptingthreadconnector.h"
#include "ui_caltableform.h"
#include "ui_recordreaderform.h"
#include "ui_nodebrowserform.h"
#include "ui_interfacetool.h"
#include "ui_graphtool.h"
#include "ui_drivertool.h"
#include "ui_scalarentrytool.h"
#include "icon.h"
#include "messagebox.h"
#include "graph.h"

QWidget *g_pFrmMain = nullptr;
static std::unique_ptr<XMessageBox> s_pMessageBox;

FrmKameMain::FrmKameMain()
    :QMainWindow(NULL) {
    resize(0,0);

    setToolButtonStyle(Qt::ToolButtonTextUnderIcon);

    s_pMessageBox.reset(new XMessageBox(this));

    show();

    g_pFrmMain = this;

	createActions();
	createMenus();

	//Central MDI area.
	m_pMdiCentral = new QMdiArea( this );
    setCentralWidget( m_pMdiCentral );
    m_pMdiCentral->setViewMode(QMdiArea::TabbedView);
    m_pMdiCentral->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    m_pMdiCentral->setTabsClosable(true);

//    setDockOptions(QMainWindow::ForceTabbedDocks | QMainWindow::VerticalTabs);
    //Left MDI area.
    QDockWidget* dockLeft = new QDockWidget(i18n("KAME Toolbox West"), this);
    m_pDockLeft = dockLeft;
    dockLeft->setFeatures(QDockWidget::DockWidgetFloatable);
    dockLeft->setWindowIcon(*g_pIconDriver);
    m_pMdiLeft = new QMdiArea( this );
    m_pMdiLeft->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    m_pMdiLeft->setViewMode(QMdiArea::TabbedView);
    m_pMdiLeft->setTabPosition(QTabWidget::West);
//    m_pMdiLeft->setTabPosition(QTabWidget::North);
    dockLeft->setWidget(m_pMdiLeft);
    addDockWidget(Qt::LeftDockWidgetArea, dockLeft);

    //Right MDI area.
    QDockWidget* dockRight = new QDockWidget(i18n("KAME Toolbox East"), this);
    m_pDockRight = dockRight;
    dockRight->setFeatures(QDockWidget::DockWidgetFloatable);
    dockRight->setWindowIcon(*g_pIconInterface);
    m_pMdiRight= new QMdiArea( this );
    m_pMdiRight->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    m_pMdiRight->setViewMode(QMdiArea::TabbedView);
    m_pMdiRight->setTabPosition(QTabWidget::East);
//    m_pMdiRight->setTabPosition(QTabWidget::North);
    dockRight->setWidget(m_pMdiRight);
    addDockWidget(Qt::RightDockWidgetArea, dockRight);
//    addDockWidget(Qt::TopDockWidgetArea, dockRight);

    //Auto-hide strips: a thin bar of pane icons pinned to each window edge.
    //They stay put whether the toolbox is docked or floating, so a hidden
    //toolbox is always one click away — the reason a toolbox may be hidden at
    //all (the docks are not Closable, and a hidden dock swallows the plain
    //showMaximized() the View menu used to do).
    for(auto &&s: {std::make_pair( &m_pStripLeft, Qt::LeftToolBarArea),
                   std::make_pair( &m_pStripRight, Qt::RightToolBarArea)}) {
        QToolBar *strip = new QToolBar(
            (s.second == Qt::LeftToolBarArea) ? i18n("West Toolbox Bar") : i18n("East Toolbox Bar"), this);
        strip->setObjectName((s.second == Qt::LeftToolBarArea) ? "stripWest" : "stripEast");
        strip->setMovable(false);
        strip->setFloatable(false);
        strip->setToolButtonStyle(Qt::ToolButtonIconOnly);
        strip->setIconSize(QSize(20, 20));
        addToolBar(s.second, strip);
        *s.first = strip;
    }
    connect(dockLeft, &QDockWidget::visibilityChanged, this, [this](bool){updateToolboxStrips();});
    connect(dockRight, &QDockWidget::visibilityChanged, this, [this](bool){updateToolboxStrips();});
    connect(m_pMdiLeft, &QMdiArea::subWindowActivated, this, [this](QMdiSubWindow *){updateToolboxStrips();});
    connect(m_pMdiRight, &QMdiArea::subWindowActivated, this, [this](QMdiSubWindow *){updateToolboxStrips();});

    Transactional::SignalBuffer::initialize();

    m_pFrmDriver = new FrmDriver(this);
    m_pFrmDriver->setWindowIcon(*g_pIconDriver);
    addDockableWindow(m_pMdiLeft, m_pFrmDriver, false);

    m_pFrmGraphList = new FrmGraphList(this);
    m_pFrmGraphList->setWindowIcon(*g_pIconGraph);
    addDockableWindow(m_pMdiLeft, m_pFrmGraphList, false);

    m_pFrmCalTable = new FrmCalTable(this);
    m_pFrmCalTable->setWindowIcon( *g_pIconRoverT);
    addDockableWindow(m_pMdiLeft, m_pFrmCalTable, false);

    m_pFrmNodeBrowser = new FrmNodeBrowser(this);
    m_pFrmNodeBrowser->setWindowIcon(QApplication::style()->standardIcon(QStyle::SP_FileDialogContentsView));
    addDockableWindow(m_pMdiLeft, m_pFrmNodeBrowser, false);

    m_pMdiLeft->activatePreviousSubWindow();
    m_pMdiLeft->activatePreviousSubWindow();
    m_pMdiLeft->activatePreviousSubWindow();

    m_pFrmInterface = new FrmInterface(this);
    m_pFrmInterface ->setWindowIcon(*g_pIconInterface);
    addDockableWindow(m_pMdiRight, m_pFrmInterface, false);

    m_pFrmScalarEntry = new FrmEntry(this);
    m_pFrmScalarEntry->setWindowIcon(*g_pIconScalar);
    addDockableWindow(m_pMdiRight, m_pFrmScalarEntry, false);

    m_pFrmRecordReader = new FrmRecordReader(this);
    m_pFrmRecordReader->setWindowIcon(*g_pIconReader);
    addDockableWindow(m_pMdiRight, m_pFrmRecordReader, false);

    m_pMdiRight->activatePreviousSubWindow();
    m_pMdiRight->activatePreviousSubWindow();

    m_pViewMenu->addSeparator();
    m_pGraphThemeMenu = m_pViewMenu->addMenu(i18n( "Theme Color of &Graph" ) );
    m_pGraphThemeMenu->setIcon( QIcon( *g_pIconGraph));
    m_pGraphThemeMenu->addAction(m_pGraphThemeNightAction);
    m_pGraphThemeMenu->addAction(m_pGraphThemeDaylightAction);
    m_pViewMenu->addSeparator();
    QAction *act = new QAction( *g_pIconInfo, XMessageBox::form()->windowTitle(), this);
    connect(act, SIGNAL(triggered()), XMessageBox::form(), SLOT(showNormal()));
    m_pViewMenu->addAction(act);

    //rearranges window positions, sizes.
    QRect rect = dockLeft->window()->windowHandle()->screen()->availableGeometry();
    // The three-pane layout below detaches both toolboxes into free-floating
    // always-on-top tool windows and positions them by hand.  Wayland's
    // xdg-shell has no client-side positioning: move(), raise() and
    // WindowStaysOnTopHint are all silently ignored by Qt's wayland plugin, so
    // the two toolboxes would be dumped wherever the compositor felt like —
    // typically stacked over the main window.  Keep them docked there; the
    // result is a conventional single-window layout instead of a broken
    // floating one.  X11/XWayland, macOS and Windows are unaffected.
    bool can_place_windows = !QGuiApplication::platformName().startsWith("wayland");
    if(can_place_windows) {
        dockLeft->setFloating(true);
        //No minimize button: shrinking to the edge bar is what getting a
        //toolbox out of the way means now, and it leaves something on screen
        //to bring it back with.  A minimized toolbox leaves nothing.
        dockLeft->setWindowFlags(Qt::Tool | Qt::WindowStaysOnTopHint |
            Qt::CustomizeWindowHint | Qt::WindowTitleHint);
        //Both toolboxes run from the top of the screen down to just above the
        //message window, which parks itself at the bottom-left corner and
        //keeps its top edge there (it grows downwards when a popup appears).
        int msg_top = XMessageBox::form()->frameGeometry().top();
        int deco = std::max(0, dockLeft->frameSize().height() - dockLeft->height());
        int toolbox_h = std::max(msg_top - 6 - rect.top() - deco, 360);
        dockLeft->resize(std::max(rect.width() / 5, XMessageBox::form()->width() + 80),
            toolbox_h);
        dockLeft->move(0, rect.top());
        //Only a first guess: see fitToolboxHeights(), which trims both once
        //their frames exist and the window server has placed them.
        dockRight->setFloating(true);
        dockRight->setWindowFlags(Qt::Tool | Qt::WindowStaysOnTopHint |
            Qt::CustomizeWindowHint | Qt::WindowTitleHint);
        dockRight->resize(std::max(rect.width() / 5, 450), dockLeft->height());
        dockRight->move(rect.right() - dockRight->frameSize().width() - 6, rect.top());
        setupEdgeAutoHide(rect);
    }
    //The following 2 lines should be after setting up docks. Otherwise, crashes in windows.
    //A third wider than it used to be, both terms alike (screen/4 -> 13/40,
    //500 -> 650), now that the toolboxes fold themselves away and the space
    //between them is the main window's to use.
    resize(QSize(std::max(rect.width() * 13 / 40, 650), minimumHeight()));
    if(can_place_windows)
        move((rect.width() - frameSize().width()) / 2, rect.top());

    updateToolboxStrips(); //initial check marks, after the panes are laid out.

#if defined __MACOSX__ || defined __APPLE__
    //Bringing the application forward from the Dock does NOT bring this window
    //forward: AppKit orders the app's windows front keeping their order among
    //themselves, so the main window stays wherever it was in the stack —
    //underneath a driver form, and always underneath the toolboxes, which are
    //Qt::Tool panels marked stays-on-top.  Asking for it is the way out.
    //
    //Which window was activated cannot decide this: measured, AppKit hands the
    //key back to whichever plain window held it last — a driver form, say —
    //whether the app was raised from the Dock or by clicking that form.  (And
    //asking `flags & Qt::Tool` cannot tell them apart either: Qt::Tool is a
    //composite of Popup|Dialog|Window, so that test is true of every ordinary
    //window as well.)
    //
    //Where the pointer is does decide it.  Clicking the Dock leaves it over
    //the Dock; clicking a window of ours leaves it over that window, and then
    //the choice of window is the user's and not ours to overrule.
    connect(qApp, &QGuiApplication::applicationStateChanged, this,
        [this](Qt::ApplicationState state) {
            if(state != Qt::ApplicationActive) return;
            QTimer::singleShot(0, this, [this]{
                if(QApplication::widgetAt(QCursor::pos())) return;
                raise();
                activateWindow();
            });
        });
#endif

    // The root for all nodes.
    //Say what this thread is before anything is created on it: a write's
    //class -- request or report -- is read off the committing thread.
    XJournal::declareThisThread(XJournal::ThreadClass::UI);

    m_measure = XNode::createOrphan<XMeasure>("Measurement", false);

    // signals and slots connections
    connect( m_pFileCloseAction, SIGNAL( triggered() ), this, SLOT( fileCloseAction_activated() ) );
    connect( m_pFileExitAction, SIGNAL( triggered() ), this, SLOT( fileExitAction_activated() ) );
    connect( m_pFileOpenAction, SIGNAL( triggered() ), this, SLOT( fileOpenAction_activated() ) );
    connect( m_pFileSaveAction, SIGNAL( triggered() ), this, SLOT( fileSaveAction_activated() ) );
    connect( m_pHelpAboutAction, SIGNAL( triggered() ), this, SLOT( helpAboutAction_activated() ) );
    connect( m_pHelpContentsAction, SIGNAL( triggered() ), this, SLOT( helpContentsAction_activated() ) );
    connect( m_pHelpIndexAction, SIGNAL( triggered() ), this, SLOT( helpIndexAction_activated() ) );
//    connect( m_pMesRunAction, SIGNAL( triggered() ), this, SLOT( mesRunAction_activated() ) );
    connect( m_pMesStopAction, SIGNAL( triggered() ), this, SLOT( mesStopAction_activated() ) );
    connect( m_pScriptMenu, SIGNAL( aboutToShow() ), this, SLOT( scriptMenu_activated() ) );
    connect( m_pJupyterConsoleMenu, SIGNAL( triggered( QAction *) ), this, SLOT( jupyterConsoleAction_activated(QAction *) ) );
    connect( m_pJupyterQtConsoleMenu, SIGNAL( triggered(QAction *) ), this, SLOT( jupyterQtConsoleAction_activated(QAction *) ) );
    connect( m_pJupyterNotebookMenu, SIGNAL( triggered(QAction *) ), this, SLOT( jupyterNotebookAction_activated(QAction *) ) );
    connect( m_pScriptRunAction, SIGNAL( triggered() ), this, SLOT( scriptRunAction_activated() ) );
#ifdef USE_RUBY
    connect( m_pRubyLineShellAction, SIGNAL( triggered() ), this, SLOT( rubyLineShellAction_activated() ) );
#endif
    connect( m_pPythonLineShellAction, SIGNAL( triggered() ), this, SLOT( pythonLineShellAction_activated() ) );
    connect( m_pFileLogAction, SIGNAL( toggled(bool) ), this, SLOT( fileLogAction_toggled(bool) ) );
    connect( m_pGraphThemeNightAction, SIGNAL( toggled(bool) ), this, SLOT( graphThemeNightAction_toggled(bool) ) );
//    connect( m_pGraphThemeDaylightAction, SIGNAL( toggled(bool) ), this, SLOT( graphThemeDaylightAction_toggled(bool) ) );

	connect(qApp, SIGNAL(aboutToQuit()), this, SLOT(aboutToQuit()));
	connect(qApp, SIGNAL( lastWindowClosed() ), qApp, SLOT( quit() ) );

	m_pTimer = new QTimer(this);
    connect(m_pTimer, SIGNAL (timeout() ), this, SLOT(processSignals()));
    m_pTimer->start(1); //never 0 -- see processSignals().

#ifdef USE_PYBIND11
    pythonLineShellAction_activated();
#elif defined(USE_RUBY)
    rubyLineShellAction_activated();
#endif
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
         act->setCheckable(true);
         //The same action drives the View menu and the edge strip: showing a
         //pane has to reveal its toolbox first, which a bare showMaximized()
         //on the subwindow cannot do once the toolbox is hidden.
         connect(act, &QAction::triggered, this, [this, wnd](bool){toggleToolboxPane(wnd);});
	     m_pViewMenu->addAction(act);
         QDockWidget *dock = (area == m_pMdiLeft) ? m_pDockLeft :
             ((area == m_pMdiRight) ? m_pDockRight : nullptr);
         QToolBar *strip = (area == m_pMdiLeft) ? m_pStripLeft :
             ((area == m_pMdiRight) ? m_pStripRight : nullptr);
         if(dock && strip) {
             strip->addAction(act);
             m_toolboxPanes.push_back({act, dock, area, wnd});
         }
	}
    widget->setAutoFillBackground(true);
	wnd->setWidget(widget);
    area->addSubWindow(wnd);
	wnd->setWindowIcon(widget->windowIcon());
    wnd->setWindowTitle(widget->windowTitle());
    wnd->showMaximized();
//    auto sub = area->addSubWindow(wnd,Qt::Window);
//    area->setActiveSubWindow(sub);
    return wnd;
}
void
FrmKameMain::toggleToolboxPane(QMdiSubWindow *wnd) {
    for(auto &&pane: m_toolboxPanes) {
        if(pane.wnd != wnd) continue;
        EdgeSlider *slider = edgeSliderFor(pane.dock);
        bool folded = !pane.dock->isVisible() || pane.dock->isMinimized() ||
            (slider && slider->collapsed);
        if( !folded && (pane.area->activeSubWindow() == pane.wnd)) {
            //The pane on screen was clicked: fold the toolbox away — shrink it
            //to its edge bar where it has one, else hide the dock outright.
            if(slider) setToolboxCollapsed( *slider, true);
            else pane.dock->hide();
        }
        else
            revealToolboxPane(pane);
        break;
    }
    updateToolboxStrips();
}
void
FrmKameMain::revealToolboxPane(ToolboxPane &pane) {
    //A toolbox that got minimized is out of reach of its hover bar, so this is
    //its only way back — and it cannot trust Qt's own answer about the state:
    //a Qt::Tool window sent to the macOS Dock still reports isVisible() ==
    //true and isMinimized() == false (measured), which is why nothing here
    //used to bring it back.  So clear the state where the platform does report
    //it, and otherwise rely on raise() + activateWindow(), which
    //deminiaturizes.
    EdgeSlider *slider = edgeSliderFor(pane.dock);
    if(pane.dock->isMinimized())
        pane.dock->setWindowState(pane.dock->windowState() & ~Qt::WindowMinimized);
    pane.dock->showNormal();
    if(slider && slider->collapsed) setToolboxCollapsed( *slider, false);
    pane.area->setActiveSubWindow(pane.wnd);
    pane.wnd->showMaximized();
    pane.dock->raise();
    pane.dock->activateWindow(); //asked for explicitly, unlike a hover
    updateToolboxStrips();
}
void
FrmKameMain::revealInterfacePane() {
    for(auto &&pane: m_toolboxPanes)
        if(pane.wnd->widget() == m_pFrmInterface) {
            revealToolboxPane(pane);
            break;
        }
}
void
FrmKameMain::updateToolboxStrips() {
    //A check mark means "this pane is the one you can see right now".  Driven
    //from the real widget state, so tab clicks and dock closes stay in sync.
    for(auto &&pane: m_toolboxPanes) {
        EdgeSlider *slider = edgeSliderFor(pane.dock);
        bool shown = pane.dock->isVisible() && !(slider && slider->collapsed) &&
            (pane.area->activeSubWindow() == pane.wnd);
        if(pane.action->isChecked() != shown)
            pane.action->setChecked(shown);
    }
}
namespace {
//! Flat styling for a pane tab column: no frames, room to breathe, and the
//! pane in front marked by an accent line against the edge the column sits at.
//! The old look — the platform's boxed, shaded tabs with rotated labels — is
//! what dated the resting toolbox more than anything else about it.
//!
//! Every colour is taken from the palette rather than written down, so this
//! follows the platform's light/dark setting; literal colours would be wrong
//! in one of the two.  The width here is also what sets the column's
//! thickness, which is why no style proxy is needed for that any more.
QString flatTabStyleSheet(Qt::Edge accent) {
    const bool vertical = (accent == Qt::LeftEdge) || (accent == Qt::RightEdge);
    const char *side = (accent == Qt::LeftEdge) ? "left" :
        ((accent == Qt::RightEdge) ? "right" : "bottom");
    //`width` means different things to the two orientations, and getting that
    //wrong is not subtle: for a column of rotated labels it is the column's
    //thickness, but for tabs running along the top it is each tab's LENGTH, so
    //fixing it there squeezes every title out of existence.  Only the columns
    //get a width; a row is left to size itself around its titles.
    const QString metrics = vertical
        ? "width:26px;padding:10px 4px;margin:2px 3px;"
        : "padding:6px 14px;margin:3px 2px;";
    return QString(
        "QTabBar{background:transparent;border:none;}"
        "QTabBar::tab{background:transparent;border:none;color:palette(text);"
        "  %1border-radius:6px;}"
        "QTabBar::tab:hover{background:palette(midlight);}"
        "QTabBar::tab:selected{background:palette(alternate-base);"
        "  border-%2:3px solid palette(highlight);}").arg(metrics).arg(side);
}
} // namespace

FrmKameMain::EdgeSlider *
FrmKameMain::edgeSliderFor(QWidget *win) {
    for(auto &&s: m_edgeSliders)
        if(s.win == win) return &s;
    return nullptr;
}
void
FrmKameMain::setupEdgeAutoHide(const QRect &screen) {
    //Each floating toolbox becomes its own edge bar: it shrinks against the
    //screen edge it was placed at, leaving its MDI tab column visible, and
    //grows back under the pointer.  Nothing else is added to the screen — the
    //bar IS the toolbox, so its tabs keep working while it is narrow.
    m_pViewMenu->addSeparator();
    for(auto &&side: {std::make_pair(m_pDockLeft, true), std::make_pair(m_pDockRight, false)}) {
        QDockWidget *dock = side.first;
        bool left = side.second;
        QMdiArea *area = left ? m_pMdiLeft : m_pMdiRight;
        //A QMdiArea's minimum size hint (~196 px, inherited from its
        //subwindows) would clamp the shrink.  Lifting it on the area sticks;
        //lifting it on the dock does NOT — QDockWidget re-derives its own
        //minimum from that hint on every layout pass — so the dock's half is
        //re-applied in setToolboxCollapsed() each time.  The panes keep their
        //own minimums and are merely clipped while the toolbox is narrow.
        area->setMinimumWidth(0);
        //Growing must not steal the keyboard from whatever is being typed
        //into, and an activated toolbox would also pin itself open below.
        dock->setAttribute(Qt::WA_ShowWithoutActivating);
        int tabw = 24;
        if(QTabBar *tabs = area->findChild<QTabBar *>()) {
            //Clicking the tab already in front pins/unpins this toolbox; see
            //eventFilter().  (The poll installs this for a tab bar that does
            //not exist yet at this point.)
            tabs->installEventFilter(this);
            tabs->setProperty("kame_pin_filter", true);
            tabs->setStyleSheet(flatTabStyleSheet(left ? Qt::LeftEdge : Qt::RightEdge));
            tabs->updateGeometry();
            tabw = std::max(tabw, tabs->sizeHint().width());
        }
        auto *anim = new QPropertyAnimation(dock, "geometry", this);
        anim->setDuration(170);
        anim->setEasingCurve(QEasingCurve::OutQuint);
        QAction *autohide = new QAction(left ? i18n("Auto-hide &West Toolbox")
                                            : i18n("Auto-hide &East Toolbox"), this);
        autohide->setCheckable(true);
        autohide->setChecked(true);
        m_pViewMenu->addAction(autohide);
        m_edgeSliders.push_back({dock, area, anim, dock->geometry(), tabw + 6,
            false, left, false, 0, true, autohide, false});
        //Pointers into a deque stay valid across push_back.
        EdgeSlider *s = &m_edgeSliders.back();
        connect(autohide, &QAction::toggled, this, [this, s](bool on){
            s->autoHide = on;
            s->idleTicks = 0;
            //Switching it off has to undo it: a toolbox left sitting as a bar
            //with nothing watching the pointer could not be opened by hover.
            if( !on && s->collapsed) setToolboxCollapsed( *s, false);
        });
        connect(anim, &QPropertyAnimation::finished, this, [this, s]{
            //Belt and braces for the edge-clinging side: should the width ever
            //come out wider than asked (a layout minimum reasserting itself),
            //the window would have grown past the screen edge it clings to and
            //pushed its tab column off-screen — exactly what the user sees as
            //"the wrong side is showing".  Re-anchor instead.
            if(s->collapsed && !s->left)
                s->win->move(s->expanded.right() - s->win->width() + 1, s->win->y());
            updateToolboxStrips();
        });
    }
    //The main window folds too, but downwards and only half way: it is worked
    //IN rather than glanced at, so what it gives back is the lower half of the
    //screen while it waits, not all of itself.  Its top edge stays put, so the
    //menu bar and the pane tabs never move.
    {
        QAction *autohide = new QAction(i18n("Auto-hide &Main Window"), this);
        autohide->setCheckable(true);
        autohide->setChecked(true);
        m_pViewMenu->addAction(autohide);
        auto *anim = new QPropertyAnimation(this, "geometry", this);
        anim->setDuration(170);
        anim->setEasingCurve(QEasingCurve::OutQuint);
        m_edgeSliders.push_back({this, m_pMdiCentral, anim, geometry(), 0,
            true, true, false, 0, true, autohide, false});
        EdgeSlider *s = &m_edgeSliders.back();
        connect(autohide, &QAction::toggled, this, [this, s](bool on){
            s->autoHide = on;
            s->idleTicks = 0;
            if( !on && s->collapsed) setToolboxCollapsed( *s, false);
        });
        connect(anim, &QPropertyAnimation::finished, this, [this]{updateToolboxStrips();});
    }
    m_pEdgeHoverTimer = new QTimer(this);
    connect(m_pEdgeHoverTimer, &QTimer::timeout, this, &FrmKameMain::pollEdgeAutoHide);
    m_pEdgeHoverTimer->start(150);
    //Deferred to the first turn of the event loop: window frames do not exist
    //yet inside this constructor, so neither the trim nor the activation below
    //would land.
    QTimer::singleShot(0, this, [this]{
        fitToolboxHeights();
        //Start with the west toolbox in hand.  It also holds itself open until
        //the user clicks elsewhere, through the focus guard in the poll.
        focusToolbox(true);
    });
    //The in-window strips would only duplicate what the edge bars now do.
    m_pStripLeft->hide();
    m_pStripRight->hide();
}
void
FrmKameMain::fitToolboxHeights() {
    //A toolbox asked to sit at the top of the screen does not end up with its
    //frame there — the window server places it below the menu bar, and the
    //title bar is not measurable until the frame exists — so a height worked
    //out in the constructor overshoots by that much and the toolbox runs too
    //long.  Measure where each one actually landed and take the excess off.
    int msg_top = XMessageBox::form()->frameGeometry().top();
    for(auto &&s: m_edgeSliders) {
        auto *dock = qobject_cast<QDockWidget *>(s.win);
        if( !dock || !dock->isFloating()) continue; //the main window is not ours to trim
        int over = s.win->frameGeometry().bottom() - (msg_top - 8);
        if(over > 0)
            s.win->resize(s.win->width(), std::max(s.win->height() - over, 360));
        s.expanded = s.win->geometry();
    }
}
void
FrmKameMain::focusToolbox(bool left) {
    QDockWidget *dock = left ? m_pDockLeft : m_pDockRight;
    QMdiArea *area = left ? m_pMdiLeft : m_pMdiRight;
    if(EdgeSlider *s = edgeSliderFor(dock))
        if(s->collapsed) setToolboxCollapsed( *s, false);
    if(dock->isMinimized())
        dock->setWindowState(dock->windowState() & ~Qt::WindowMinimized);
    dock->showNormal();
    dock->raise();
    dock->activateWindow();
    if(QMdiSubWindow *sub = area->activeSubWindow())
        sub->setFocus();
}
void
FrmKameMain::pollEdgeAutoHide() {
    //Polling the pointer beats enter/leave events here: these are separate
    //top-level windows, and a leave event fires for every excursion over a
    //child widget.
    QPoint c = QCursor::pos();
    //Signs that the user is in the middle of something, where shrinking the
    //toolbox would be sabotage: an open popup (a combo list is its own window,
    //outside the toolbox rectangle) or a held mouse button (dragging a
    //scrollbar or a spin box).
    bool busy = QApplication::activePopupWidget() ||
        (QApplication::mouseButtons() != Qt::NoButton);
    //Focus inside a toolbox AND that toolbox being the active window means the
    //user clicked in and is working there, so it stays open until they click
    //elsewhere.  Both halves are needed: a pane holding a line edit can take
    //focus merely by being shown (that alone would pin it open for ever), and
    //what counts as "active" for a Qt::Tool window varies between platforms.
    //Typing always implies both, so nothing can shrink away mid-edit.
    QWidget *focus = QApplication::focusWidget();
    for(auto &&s: m_edgeSliders) {
        //Sampled for every window, pinned or not: it is what tells a tab click
        //whether the user was already working in this one.
        s.wasFocused = focus && s.win->isAncestorOf(focus) && s.win->isActiveWindow();
        //The central pane stack has no tab bar until the first script pane
        //exists, so the pin gesture's filter cannot all be installed at setup.
        if(QTabBar *tabs = s.area->findChild<QTabBar *>())
            if( !tabs->property("kame_pin_filter").toBool()) {
                tabs->installEventFilter(this);
                tabs->setProperty("kame_pin_filter", true);
                //Its tabs run along the top, so the accent goes underneath.
                tabs->setStyleSheet(flatTabStyleSheet(Qt::BottomEdge));
            }
        if( !s.autoHide) continue;
        if(s.anim->state() == QAbstractAnimation::Running) continue;
        if(auto *dock = qobject_cast<QDockWidget *>(s.win))
            if( !dock->isFloating()) continue; //re-docked by the user: not ours to move
        if( !s.win->isVisible()) continue;
        if(s.win->isMinimized()) continue; //out of reach until the View menu restores it
        bool over = s.win->frameGeometry().contains(c);
        //Hovering a tab picks that pane, collapsed or open: the tab strip is
        //all one sees of a resting toolbox, so pointing at a name there should
        //be what brings that pane up as it grows.  The main window's strip
        //runs across its top instead of down a screen edge, so the pointer can
        //cross it on the way in — accepted deliberately: the pane it lands on
        //is the one under the pointer, and picking by hover there is worth
        //more than the occasional pass-through.
        if(over) {
            if(QTabBar *tabs = s.area->findChild<QTabBar *>()) {
                QPoint local = tabs->mapFromGlobal(c);
                if(tabs->isVisible() && tabs->rect().contains(local)) {
                    int idx = tabs->tabAt(local);
                    if((idx >= 0) && (idx != tabs->currentIndex()))
                        tabs->setCurrentIndex(idx);
                }
            }
        }
        if(s.collapsed) {
            if(over) setToolboxCollapsed(s, false);
            continue;
        }
        s.expanded = s.win->geometry(); //follows the user moving or resizing it
        if(over || busy || s.wasFocused)
            s.idleTicks = 0;
        else if(++s.idleTicks >= 4) //~0.6 s with the pointer elsewhere
            setToolboxCollapsed(s, true);
    }
}
void
FrmKameMain::setToolboxCollapsed(EdgeSlider &s, bool collapse) {
    //Lifted for BOTH directions, not just the collapse.  A window re-derives
    //its minimum from the QMdiArea's size hint on every layout pass, so on the
    //way back OUT the early frames were clamped up to that minimum: measured,
    //the first frame jumped straight from the 43 px bar to 196 px, skipping
    //most of the animation.  On the toolbox that keeps its RIGHT edge that is
    //visible as the tab column disappearing for an instant — the width is
    //forced wide while the animation's x is still back at the collapsed
    //position, so the window overhangs the screen edge and takes its own tabs
    //off-screen with it.
    if(s.vertical) {
        s.area->setMinimumHeight(0);
        s.win->setMinimumHeight(0);
    }
    else {
        s.area->setMinimumWidth(0);
        s.win->setMinimumWidth(0);
    }
    QRect to = s.expanded;
    if(collapse) {
        if(s.vertical)
            //Half of whatever it is now, keeping the top edge.
            to.setHeight(std::max(s.expanded.height() / 2, 200));
        //Keep the edge the toolbox clings to; give up the width on the other
        //side, so it grows out of the screen edge rather than sliding along it.
        else if(s.left) to.setWidth(s.collapsedWidth);
        else to.setLeft(s.expanded.right() - s.collapsedWidth + 1);
    }
    s.idleTicks = 0;
    s.collapsed = collapse;
    //No fade on the way in.  It was tried, and a window at 0.75 opacity shows
    //what is behind it: on a bar barely wider than its tabs that reads as the
    //tabs blinking out, not as an entrance.  These windows are opaque.
    s.win->setWindowOpacity(1.0);
    s.anim->stop();
    s.anim->setStartValue(s.win->geometry());
    s.anim->setEndValue(to);
    s.anim->start();
}
bool
FrmKameMain::eventFilter(QObject *obj, QEvent *event) {
    if(event->type() == QEvent::MouseButtonPress) {
        //Pin gesture: while already working in a toolbox, clicking the tab of
        //the pane in front toggles its auto-hide.  Only the pane in front, so
        //clicking any other tab still just switches panes; and only when the
        //toolbox already held the keyboard, which is why the poll's remembered
        //answer is used rather than a fresh one — this very click may have
        //activated the window, and a fresh test would say yes every time.
        for(auto &&s: m_edgeSliders) {
            QTabBar *tabs = s.area->findChild<QTabBar *>();
            if(obj != tabs) continue;
            if( !s.wasFocused) break;
            auto *me = static_cast<QMouseEvent *>(event);
            int idx = tabs->tabAt(me->position().toPoint());
            if((idx < 0) || (idx != tabs->currentIndex())) break;
            s.autoHideAction->setChecked( !s.autoHide); //drives the toggle
            XString what = s.vertical ? i18n("Main window") :
                (s.left ? i18n("West toolbox") : i18n("East toolbox"));
            gMessagePrint(what + (s.autoHide ? i18n(" auto-hides again.")
                                             : i18n(" pinned open.")));
            return true; //the click meant this, not a tab change
        }
    }
    if(event->type() == QEvent::Show) {
        auto w = qobject_cast<QWidget*>(obj);
        if(w && !w->property("kame_placed").toBool()) {
            placeNewWindow(w);
            w->setProperty("kame_placed", true);
        }
    }
    return QMainWindow::eventFilter(obj, event);
}

void
FrmKameMain::placeNewWindow(QWidget *w) {
    auto *screen = this->screen();
    if( !screen) return;
    QRect rect = screen->availableGeometry();
    // Place new windows between left dock and right dock, cascading.
    int x0 = rect.left() + rect.width() / 5;
    int y0 = rect.top();
    int cascadeStep = 30;
    QPoint pos(x0 + m_cascadeIndex * cascadeStep, y0 + m_cascadeIndex * cascadeStep);
    // Wrap if window would go off-screen.
    if(pos.x() + w->width() > rect.right() || pos.y() + w->height() > rect.bottom()) {
        m_cascadeIndex = 0;
        pos = QPoint(x0, y0);
    }
    w->move(pos);
    m_cascadeIndex++;
}

FrmKameMain::~FrmKameMain() {
    m_pTimer->stop();
    if(m_journal) {
        m_journal->stop();
        m_journal.reset();
    }
//	while( !g_signalBuffer->synchronize()) {}
    Transactional::SignalBuffer::cleanup();
    s_pMessageBox.reset();
    m_measure.reset();
}

void
FrmKameMain::aboutToQuit() {
}

void
FrmKameMain::createActions() {
    // actions
    m_pFileOpenAction = new QAction( this );
//     fileOpenAction->setIcon( QIconSet( *IconKame48x48 ) );
    m_pFileOpenAction->setIcon(QApplication::style()->standardIcon(QStyle::SP_DirOpenIcon));
    m_pFileSaveAction = new QAction( this );
    m_pFileSaveAction->setEnabled( true );
    m_pFileSaveAction->setIcon(QApplication::style()->standardIcon(QStyle::SP_DialogSaveButton));
    m_pFileCloseAction = new QAction( this );
    m_pFileCloseAction->setEnabled( true );
//     fileCloseAction->setIcon( QIconSet( *IconClose48x48 ) );
    m_pFileCloseAction->setIcon(QApplication::style()->standardIcon(QStyle::SP_DirClosedIcon));
    m_pFileExitAction = new QAction( this );
//     fileExitAction->setIcon( QIconSet( *IconStop48x48 ) );
    m_pFileExitAction->setIcon(QApplication::style()->standardIcon(QStyle::SP_DialogCloseButton));
    m_pHelpContentsAction = new QAction( this );
    m_pHelpIndexAction = new QAction( this );
    m_pHelpAboutAction = new QAction( this );
    m_pHelpAboutAction->setIcon(QApplication::style()->standardIcon(QStyle::SP_DialogHelpButton));
    m_pFileLogAction = new QAction( this );
    m_pFileLogAction->setCheckable( true );
    m_pFileLogAction->setChecked( g_bLogDbgPrint );
    m_pFileLogAction->setIcon(QApplication::style()->standardIcon(QStyle::SP_DriveCDIcon));
//    m_pMesRunAction = new QAction( this, "mesRunAction" );
//    m_pMesRunAction->setEnabled( TRUE );
	//   m_pMesRunAction->setIcon( QIconSet( *g_pIconDriver) );
    m_pMesStopAction = new QAction( this );
    m_pMesStopAction->setEnabled( true );
    m_pMesStopAction->setIcon( QIcon( *g_pIconStop) );
    m_pScriptRunAction = new QAction( this );
    m_pScriptRunAction->setEnabled( true );
    m_pScriptRunAction->setIcon(QApplication::style()->standardIcon(QStyle::SP_FileDialogDetailedView));
    m_pPythonLineShellAction = new QAction( this );
    m_pPythonLineShellAction->setEnabled( true );
#ifndef USE_PYBIND11
    m_pPythonLineShellAction->setEnabled( false );
#endif
    m_pPythonLineShellAction->setIcon(QIcon( *g_pIconPython));
#ifdef USE_RUBY
    m_pRubyLineShellAction = new QAction( this );
    m_pRubyLineShellAction->setEnabled( true );
    m_pRubyLineShellAction->setIcon(QIcon( *g_pIconScript));
#endif
    m_pJupyterConsoleMenu = new QMenu( this );
    m_pJupyterConsoleMenu->setIcon(QIcon( *g_pIconPython));
    m_pJupyterQtConsoleMenu = new QMenu( this );
    m_pJupyterQtConsoleMenu->setIcon(QApplication::style()->standardIcon(QStyle::SP_TitleBarMenuButton));
    m_pJupyterNotebookMenu = new QMenu( this );
    m_pJupyterNotebookMenu->setIcon(QIcon( *g_pIconJupyter));
    for(QMenu *menu: {m_pJupyterConsoleMenu, m_pJupyterQtConsoleMenu, m_pJupyterNotebookMenu}) {
        menu->setEnabled( true );
    #ifndef USE_PYBIND11
        menu->setEnabled( false );
    #endif
    }
//    m_pJupyterQtConsoleAction->setIcon(QApplication::style()->standardIcon(QStyle::SP_TitleBarMenuButton));
    m_pGraphThemeNightAction = new QAction( this);
    m_pGraphThemeNightAction->setEnabled( true );
    m_pGraphThemeNightAction->setCheckable( true );
    m_pGraphThemeNightAction->setChecked( true );
    m_pGraphThemeDaylightAction = new QAction( this);
    m_pGraphThemeDaylightAction->setEnabled( true );
    m_pGraphThemeDaylightAction->setCheckable( true );
    m_pGraphThemeActionGroup = new QActionGroup(this);
    m_pGraphThemeActionGroup->setExclusive( true );
    m_pGraphThemeActionGroup->addAction(m_pGraphThemeNightAction);
    m_pGraphThemeActionGroup->addAction(m_pGraphThemeDaylightAction);

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
    m_pPythonLineShellAction->setText( i18n( "New &Python Line Shell" ) );
#ifdef USE_RUBY
    m_pRubyLineShellAction->setText( i18n( "&New Ruby Line Shell" ) );
#endif
    m_pJupyterNotebookMenu->setTitle( i18n( "Launch &Jupyter Notebook" ) );
    m_pJupyterConsoleMenu->setTitle( i18n( "Launch Jupyter &Console" ) );
    m_pJupyterQtConsoleMenu->setTitle( i18n( "Launch Jupyter &Qt Console" ) );
    m_pFileCloseAction->setText( i18n( "&Close" ) );    
    m_pGraphThemeNightAction->setText( i18n( "&Night") );
    m_pGraphThemeDaylightAction->setText( i18n( "&Daylight") );
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
#ifdef USE_RUBY
    m_pScriptMenu->addAction(m_pRubyLineShellAction);
#endif
    m_pScriptMenu->addAction(m_pPythonLineShellAction);
    m_pScriptMenu->addSeparator();
    m_pScriptMenu->addMenu(m_pJupyterNotebookMenu);
    m_pScriptMenu->addMenu(m_pJupyterConsoleMenu);
    m_pScriptMenu->addMenu(m_pJupyterQtConsoleMenu);

    m_pViewMenu = menuBar()->addMenu(i18n( "&View" ) );

    m_pHelpMenu = menuBar()->addMenu(i18n( "&Help" ) );
    m_pHelpMenu->addAction(m_pHelpContentsAction);
    m_pHelpMenu->addAction(m_pHelpIndexAction );
    m_pHelpMenu->addSeparator();
    m_pHelpMenu->addAction(m_pHelpAboutAction);
}

void
FrmKameMain::processSignals() {
    bool idle = Transactional::SignalBuffer::synchronize();
    // Never block in here.  This slot is driven by a QTimer, so it runs inside
    // whatever event loop happens to be current -- including a *foreign* GLib
    // loop.  Qt's GTK3 platform theme, which Linux desktops that ship
    // qt6-gtk-platformtheme select by default, implements the native file and
    // colour dialogs with gtk_dialog_run(); that spins g_main_loop_run() on the
    // same GMainContext as Qt's event dispatcher.  Sleeping 5 ms inside the
    // callback of a *zero-interval* timer makes Qt's timer source permanently
    // ready at G_PRIORITY_DEFAULT, and GLib then never reaches GDK's redraw
    // source (GDK_PRIORITY_REDRAW is numerically larger, i.e. lower priority).
    // Result on Linux: QFileDialog and QColorDialog map as an empty frame that
    // never paints and never accepts input, while Qt's own dialogs
    // (QMessageBox) are unaffected because they run Qt's event loop.
    // Express the same pacing as the timer interval instead, so the thread
    // blocks in poll() -- where the other sources get their turn.
    //
    // 0 means "come straight back", and that is what we want while ordinary
    // events are still queued: synchronize() only leaves them there when it
    // hit its 30 ms drain budget, so the next pass has 30 ms of real work
    // waiting and one event-loop round trip costs nothing against it.
    //
    // It is NOT what we want for the other reason synchronize() reports busy:
    // an empty queue with an event parked in the skipped queue, waiting out
    // its listener's delay_ms() (>= ADAPTIVE_DELAY_MIN = 5 ms).  Then it
    // returns immediately, having done nothing and being able to do nothing
    // until the delay expires, and a zero interval turns that wait into a
    // spin -- on macOS 130k event-loop passes per second, all of it inside
    // the Cocoa dispatcher re-arming its CFRunLoop timer.  That, not real
    // listener traffic, is what held KAME at 100% CPU with nothing running;
    // any FLAG_AVOID_DUP listener event is enough to trigger it.  5 ms cannot
    // lose responsiveness there, because the event is undeliverable before
    // then anyway.
    int interval = ( !idle && Transactional::SignalBuffer::hasImmediateEvents()) ? 0 : 5;
    if(m_pTimer->interval() != interval)
        m_pTimer->setInterval(interval); //restarts the running timer.
}

void
FrmKameMain::closeEvent( QCloseEvent* ce ) {
	bool opened = false;
    {
        Snapshot shot( *m_measure->interfaces());
        if(shot.size()) {
            const XNode::NodeList &list(*shot.list());
            for(auto it = list.begin(); it != list.end(); it++) {
                auto intf = dynamic_pointer_cast<XInterface>( *it);
                if(intf->isOpened()) opened = true;
            }
        }
    }
	if(opened) {
        gWarnPrint(i18n("Stop running first.") );
		ce->ignore();
	}
    else {
        //Tear down BEFORE accepting.  The accept is the only thing that
        //authorizes the quit to proceed, and on macOS a Cmd-Q arrives as
        //-[NSApplication terminate:], which calls exit() -- static destructors
        //and module-dylib finalization -- rather than returning through main().
        //Accepting first let that proceed while the scripting threads were
        //still alive: quitting with an MCP client attached aborted twice in one
        //exit (2026-08-20), SIGABRT in cast_to_pyobject on the IPython thread
        //and an instruction abort in ~XTypeHolder on the main thread, both of
        //them the Python thread still calling into KAME mid-teardown.  With the
        //accept last, exit() cannot start until every join has returned.
        printf("quit\n");
        //Before the tree goes: the journal's last drain and report walk it.
        if(m_journal) {
            m_journal->stop();
            m_journal.reset();
        }
        m_measure->terminate_all();
        m_measure.reset();
        ce->accept();
	}
}

void FrmKameMain::fileCloseAction_activated() {
	m_measure->terminate();
}


void FrmKameMain::fileExitAction_activated() {
	close();
}

void FrmKameMain::fileOpenAction_activated() {
    QString filename = QFileDialog::getOpenFileName (
        this, i18n("Open Measurement File"), "",
        //! No trailing ";;": it appends an empty name filter, which shows up as a
        //! blank row in the file-type combo of Qt's own widget dialog.
        //A journal's head IS a settings file, so it opens the same way.  The
        //combined filter is first so that either kind can just be
        //double-clicked, and .kam is named first inside it because that is
        //still what everyone has.
        //\sa doc/design/PROVENANCE.md
        "Measurement files (*.kam *.kamj);;"
        "Settings, saved by hand (*.kam);;"
        "Journals, written as you work (*.kamj);;"
        "KAME1 Measurement files (*.mes);;"
        "All files (*.*)"
        );
	openMes(filename);
}


void FrmKameMain::fileSaveAction_activated() {
    QString filter = "KAME2 Measurement files (*.kam)";
#if QT_VERSION < QT_VERSION_CHECK(5,0,0)
    QString filename = QFileDialog::getSaveFileName (
        this, i18n("Save Measurement File"), "", filter);
#else
    //old qt cannot make native dialog in this mode.
    QFileDialog dialog(this);
    dialog.setWindowTitle(i18n("Save Measurement File"));
    dialog.setViewMode(QFileDialog::Detail);
    dialog.setNameFilter(filter);
    #if QT_VERSION < QT_VERSION_CHECK(5,4,0)
        dialog.setConfirmOverwrite(true);
    #endif
    dialog.setDefaultSuffix("kam");
    dialog.setAcceptMode(QFileDialog::AcceptSave);
    if( !dialog.exec())
        return;
    QString filename = dialog.selectedFiles().at(0);
#endif
    if( !filename.isEmpty()) {
        std::ofstream ofs(filename.toLocal8Bit().data(), std::ios::out);
		if(ofs.good()) {
            XRubyWriter writer(m_measure, ofs);
			writer.write();
        }
	}
}


void FrmKameMain::helpAboutAction_activated() {
    QMessageBox::about( this,
						i18n("K's Adaptive Measurement Engine."), "KAME");
}

void FrmKameMain::helpContentsAction_activated() {
}


void FrmKameMain::helpIndexAction_activated() {
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

void FrmKameMain::signalAllModulesLoaded() {
#ifdef USE_PYBIND11
    if(m_measure && m_measure->python())
        m_measure->python()->signalModulesLoaded();
#endif
    //Provenance capture starts here, with the driver types registered and
    //before any .kam is loaded, so the loading itself is journaled.  Nodes
    //created later are picked up through onListChanged.
    //The capture engine runs whether or not anything is being written: the
    //Journal group's Write switch chooses the file, not whether the tree is
    //being watched.  KAME_JOURNAL only adds the developer survey report.
    if(m_measure)
        m_journal = XJournal::start(m_measure, m_measure->journal());
}

void FrmKameMain::mesStopAction_activated() {
	m_measure->stop();
/*
 *   m_pMesRunAction->setEnabled(true);
 m_pMesStopAction->setEnabled(false);
 m_pFileCloseAction->setEnabled(true);
 m_pFileExitAction->setEnabled(true);
*/
}

int
FrmKameMain::openMes(const XString &filename) {
	if( !filename.empty()) {
		shared_ptr<XScriptingThread> th = runNewScript("Open Measurement", filename );
        //Interfaces and entries are what one turns to after loading a
        //measurement, so hand the east toolbox the keyboard — but only once
        //the loading thread is done, since drivers, graphs and their windows
        //go on appearing until then and would take it back.
        if(th) {
            auto *timer = new QTimer(this);
            auto ticks = std::make_shared<int>(0);
            connect(timer, &QTimer::timeout, this, [this, th, timer, ticks]{
                //Bounded, so a thread that never reports itself finished does
                //not leave a timer polling for the rest of the session.
                if(th->isAlive() && (++( *ticks) < 300)) return;
                timer->stop();
                timer->deleteLater();
                focusToolbox(false);
            });
            timer->start(200);
        }
//		while(rbthread->isAlive()) {
//			KApplication::kApplication()->processEvents();
//			g_signalBuffer->synchronize();
//		}
//          closeWindow(view);
		return 0;
	}
    return -1;
}

shared_ptr<XScriptingThread>
FrmKameMain::runNewScript(const XString &label, const XString &filename) {
    show();
    raise();
    shared_ptr<XScriptingThreadList> threadlist;
    //Spelt out rather than by rfind()-at-the-end: ".kamj" is not ".kam" with
    //something after it, and the old test read it as a Ruby script -- which
    //then met the gzip magic byte and reported an invalid multibyte
    //character.  A file kind that falls through to the wrong interpreter
    //fails in the interpreter's vocabulary, not in KAME's.
    auto endsWith = [&filename](const char *suffix)->bool {
        size_t n = strlen(suffix);
        return (filename.length() >= n)
            && (filename.compare(filename.length() - n, n, suffix) == 0);
    };
    bool for_python = endsWith(".py") || endsWith(".kam")
        || endsWith(".kamj") || endsWith(".kamj.gz");
#ifdef USE_PYBIND11
    if(for_python) {
        threadlist = m_measure->python();
    } else
#endif
    {
        if(for_python) {
            gErrPrint(i18n("Built without pybind11; measurement files and "
                "journals cannot be loaded."));
            return shared_ptr<XScriptingThread>();
        }
#ifdef USE_RUBY
        threadlist = m_measure->ruby();
#else
        gErrPrint(i18n("Built without the Ruby interpreter; only .py and .kam "
            "files can be run."));
        return shared_ptr<XScriptingThread>();
#endif
    }
    shared_ptr<XScriptingThread> scriptthread =
        threadlist->create<XScriptingThread>(label.c_str(), true, filename);
    FrmScriptingThread* form = new FrmScriptingThread(this);
    m_conScriptThreadList.push_back(xqcon_create<XScriptingThreadConnector>(
                                      scriptthread, form, threadlist));
	addDockableWindow(m_pMdiCentral, form, true);
    //Make hyperlinks in the script / IPython output pane actionable.
    //The pane has setOpenLinks(false), so clicks emit anchorClicked:
    //"kame:" links route to the Python dispatcher (Jupyter / Claude
    //launch); anything else (http/file, e.g. the log or notebook
    //links) opens in the system default handler.
    connect(form->m_ptxtDefout, &QTextBrowser::anchorClicked,
            this, &FrmKameMain::onScriptLinkClicked);

	// erase unused xqcon_ptr
    for(auto it = m_conScriptThreadList.begin(); it != m_conScriptThreadList.end(); ) {
		if((*it)->isAlive()) {
			it++;
		}
		else {
            it = m_conScriptThreadList.erase(it);
		}
	}
    return scriptthread;
}
void FrmKameMain::onScriptLinkClicked(const QUrl &url) {
    if(url.scheme() == "kame") {
#ifdef USE_PYBIND11
        if( !m_measure->python())
            return;
        QString action = url.path();
        if(action == "notebook") {
            //Prompt for the workspace dir (same as the Script menu) so the
            //notebook is never rooted in the application binary's folder.
            //Not simply the first jupyter on PATH: which of them carries the
            //notebook package differs per installation, and picking blindly is
            //how this quick launch ends in a missing-module traceback.  The
            //probe runs the candidates once and is cached, so the wait is a
            //one-off — hence the notice before it.
            gMessagePrint(i18n("Looking for a usable Jupyter..."));
            std::string prog = m_measure->python()->jupyterProgramFor("notebook");
            if(prog.empty()) {
                gMessagePrint(i18n("No usable Jupyter found. A jupyter is skipped "
                    "when the notebook package is missing, and also when its "
                    "JupyterLab data files are (that one serves a page showing "
                    "nothing but the logo). Try: pip install notebook jupyterlab"));
                return;
            }
            gMessagePrint(i18n("Choose root directory of notebook."));
            QString dir = QFileDialog::getExistingDirectory(
                this, i18n("Open Notebook Workspace"));
            if(dir.length())
                m_measure->python()->launchJupyterConsole(
                    prog, ("notebook " + dir).toUtf8().data());
        }
        else if(action == "pyai-agent") {
            //Choosing your own agent is a file, not an environment variable:
            //a GUI application should not require one to be exported before
            //launch.  Cancel clears the choice and returns to the agent KAME
            //ships, which is the only way back that needs no explanation.
            gMessagePrint(i18n("Choose your Pydantic AI agent module (Cancel = use the one KAME ships)."));
            QString file = QFileDialog::getOpenFileName(
                this, i18n("Choose Pydantic AI Agent"), QString(),
                "Agent module or spec (*.py *.yml *.yaml *.json);;All files (*.*)");
            m_measure->python()->handleLink(
                (action + "?file=" + file).toUtf8().constData());
        }
        else if(action.startsWith("pyai-")) {
            //Pydantic AI normally lives in a venv, which no PATH probe can
            //see. First use asks for the venv folder (the same gesture as the
            //notebook workspace dialog); the Python side validates it,
            //remembers it in ~/.kame_pyai_python, and forgets it when it goes
            //stale — which makes this dialog reappear on the next click.
            //Cancel falls through with an empty dir = search common places.
            if( !QFile::exists(QDir::homePath() + "/.kame_pyai_python")
                    && !qEnvironmentVariableIsSet("KAME_PYAI_PYTHON")) {
                gMessagePrint(i18n("Choose the venv folder with pydantic-ai installed (Cancel = search common locations)."));
                QString dir = QFileDialog::getExistingDirectory(
                    this, i18n("Choose Pydantic AI virtualenv"));
                m_measure->python()->handleLink(
                    (action + "?venv=" + dir).toUtf8().constData());
            }
            else
                m_measure->python()->handleLink(action.toUtf8().constData());
        }
        else {
            m_measure->python()->handleLink(action.toUtf8().constData());
        }
#endif
    }
    else {
        QDesktopServices::openUrl(url);
    }
}
void FrmKameMain::scriptRunAction_activated() {
    QString filename = QFileDialog::getOpenFileName (
        this, i18n("Open Script File"), "",
#ifdef USE_PYBIND11
        "Python Script files (*.py);;"
#endif
        "KAME Script files (*.seq);;"
        "Ruby Script files (*.rb);;"
        "All files (*.*)"
    );
	if( !filename.isEmpty()) {
		static unsigned int thread_no = 1;
		runNewScript(formatString("Thread%d", thread_no), filename );
		thread_no++;
	}
}

#define PY_LINESHELL_FILE "pythonlineshell.py"
#define RB_LINESHELL_FILE "rubylineshell.rb"

void FrmKameMain::pythonLineShellAction_activated() {
    scriptLineShellAction_activated(PY_LINESHELL_FILE);
}
void FrmKameMain::rubyLineShellAction_activated() {
#ifdef USE_RUBY
    scriptLineShellAction_activated(RB_LINESHELL_FILE);
#else
    //Unreachable: the menu action itself is only created under USE_RUBY.
    //The FUNCTION still exists in every build so that the vtable does not
    //depend on the macro -- see the note on its declaration.
    gErrPrint(i18n("Built without the Ruby interpreter."));
#endif
}


void FrmKameMain::scriptLineShellAction_activated(const char *name) {
    QString filename =
#ifdef WITH_KDE
        KStandardDirs::locate("appdata", LINESHELL_FILE);
#else
        #if QT_VERSION >= QT_VERSION_CHECK(5,4,0)
            QStandardPaths::locate(QStandardPaths::AppDataLocation, name);
        #else
            QStandardPaths::locate(QStandardPaths::DataLocation, name);
        #endif
    if(filename.isEmpty()) {
        //for macosx/win
        QDir dir(QApplication::applicationDirPath());
#if defined __MACOSX__ || defined __APPLE__
        //For macosx application bundle.
        dir.cdUp();
#endif
        QString path = QString(LINESHELL_DIR) + name;
        dir.filePath(path);
        if(dir.exists())
            filename = dir.absoluteFilePath(path);
    }
#endif
    if(filename.isEmpty()) {
        g_statusPrinter->printError("No KAME script support file installed.");
    }
    else {
        static unsigned int int_no = 1;
        XString f = filename;
        runNewScript(formatString("Line Shell%d", int_no), f );
        int_no++;
    }
}

void FrmKameMain::scriptMenu_activated() {
#ifdef USE_PYBIND11
    auto progs = m_measure->python()->listOfJupyterPrograms();
    for(QMenu *menu: {m_pJupyterConsoleMenu, m_pJupyterQtConsoleMenu, m_pJupyterNotebookMenu}) {
        menu->clear();
        for(auto &s: progs) {
            QAction *act = new QAction(s.c_str(), menu);
            menu->addAction(act);
        }
    }
#endif
}
void FrmKameMain::jupyterConsoleAction_activated( QAction *act ) {
#ifdef USE_PYBIND11
    m_measure->python()->launchJupyterConsole(act->text().toUtf8().data(), "console");
#endif
}
void FrmKameMain::jupyterQtConsoleAction_activated( QAction *act ) {
#ifdef USE_PYBIND11
    m_measure->python()->launchJupyterConsole(act->text().toUtf8().data(), "qtconsole");
#endif
}
void FrmKameMain::jupyterNotebookAction_activated( QAction *act ) {
#ifdef USE_PYBIND11
    gMessagePrint(i18n("Choose root directory of notebook."));
    QString dir = QFileDialog::getExistingDirectory (
        this, i18n("Open Notebook Workspace"));
    if(dir.length())
        m_measure->python()->launchJupyterConsole(act->text().toUtf8().data(),
            ("notebook " + dir).toUtf8().data());
#endif
}

void FrmKameMain::fileLogAction_toggled( bool var) {
	g_bLogDbgPrint = var;
}

static void
applyGraphThemeToAll(const Snapshot &shot, const shared_ptr<XNode> &parent, XGraph::Theme theme) {
    if(shot.size(parent)) {
        auto list = shot.list(parent);
        for(auto &&node: *list) {
            if(auto graph = dynamic_pointer_cast<XGraph>(node)) {
                graph->iterate_commit([=](Transaction &tr){
                    graph->applyTheme(tr, false, theme);
                });
            }
            else
                applyGraphThemeToAll(shot, node, theme);
        }
    }
};

void FrmKameMain::graphThemeNightAction_toggled( bool var ) {
    auto theme = var ? XGraph::Theme::Night : XGraph::Theme::DayLight;
    applyGraphThemeToAll(Snapshot( *m_measure), m_measure, theme);
    XGraph::setCurrentTheme(theme);
}

