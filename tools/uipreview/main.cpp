#include <QApplication>
#include <QUiLoader>
#include <QFile>
#include <QWidget>
#include <QPixmap>
#include <QTimer>
int main(int argc, char **argv) {
    QApplication app(argc, argv);
    if(argc < 3) { fprintf(stderr, "usage: uipreview <in.ui> <out.png>\n"); return 1; }
    QUiLoader loader;
    QFile f(argv[1]);
    f.open(QFile::ReadOnly);
    QWidget *w = loader.load(&f);
    if(!w) { fprintf(stderr, "load failed: %s\n", loader.errorString().toUtf8().data()); return 1; }
    w->show();
    QTimer::singleShot(0, [&]{
        w->grab().save(argv[2]);
        app.quit();
    });
    return app.exec();
}
