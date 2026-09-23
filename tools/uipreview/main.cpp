#include <QApplication>
#include <QUiLoader>
#include <QFile>
#include <QWidget>
#include <QPixmap>
#include <QTimer>
#include <QStyleHints>
#include <QPalette>
#include <QLCDNumber>
#include <QLayout>
int main(int argc, char **argv) {
    QApplication app(argc, argv);
    if(argc < 3) { fprintf(stderr, "usage: uipreview <in.ui> <out.png> [dark]\n"); return 1; }
    //A third argument renders the form the way KAME now starts: the offscreen
    //platform ignores QStyleHints::setColorScheme() (measured), so the dark
    //palette is built by hand here.
    if((argc > 3) && (QByteArray(argv[3]) == "dark")) {
        QGuiApplication::styleHints()->setColorScheme(Qt::ColorScheme::Dark);
        QPalette d;
        d.setColor(QPalette::Window, QColor("#1e1e1e"));
        d.setColor(QPalette::WindowText, QColor("#e0e0e0"));
        d.setColor(QPalette::Base, QColor("#232323"));
        d.setColor(QPalette::AlternateBase, QColor("#2a2a2a"));
        d.setColor(QPalette::Text, QColor("#e0e0e0"));
        d.setColor(QPalette::Button, QColor("#2c2c2c"));
        d.setColor(QPalette::ButtonText, QColor("#e0e0e0"));
        d.setColor(QPalette::Light, QColor("#3a3a3a"));
        d.setColor(QPalette::Midlight, QColor("#303030"));
        d.setColor(QPalette::Mid, QColor("#4a4a4a"));
        d.setColor(QPalette::Dark, QColor("#141414"));
        d.setColor(QPalette::Shadow, QColor("#000000"));
        d.setColor(QPalette::Highlight, QColor("#2f6fa8"));
        d.setColor(QPalette::HighlightedText, QColor("#ffffff"));
        app.setPalette(d);
    }
    QUiLoader loader;
    QFile f(argv[1]);
    f.open(QFile::ReadOnly);
    QWidget *w = loader.load(&f);
    if(!w) { fprintf(stderr, "load failed: %s\n", loader.errorString().toUtf8().data()); return 1; }
    //Readouts show a real number, not the .ui's placeholder: an LCD holding
    //"3" tells you nothing about whether eight digits can be read.
    for(QLCDNumber *lcd: w->findChildren<QLCDNumber *>()) {
        lcd->display("123.456");
        //"panel" renders the candidate: the readout as a display of its own,
        //dark panel and bright digits, the same in either theme.
        if(qEnvironmentVariableIsSet("KAME_LCD_PANEL")) {
            lcd->setSegmentStyle(QLCDNumber::Flat);
            lcd->setAutoFillBackground(true);
            QPalette pl(lcd->palette());
            pl.setColor(QPalette::Window, QColor("#0e1417"));
            pl.setColor(QPalette::WindowText, QColor("#79e0ff"));
            lcd->setPalette(pl);
        }
    }
    w->show();
    //Two turns and a settled layout before the grab: at singleShot(0) the
    //widgets have been shown but not laid out, and the picture came out with
    //a ghost of the pre-layout positions on top of the real one.
    QTimer::singleShot(120, [&]{
        w->layout() && (w->layout()->activate(), true);
        w->grab().save(argv[2]);
        app.quit();
    });
    return app.exec();
}
