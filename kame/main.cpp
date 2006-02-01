#include "support.h"

#include <kcmdlineargs.h>
#include <kaboutdata.h>
#include <klocale.h>
#include <kapp.h>

#include "kame.h"
#include "xsignal.h"
#include "icons/icon.h"
#include <kiconloader.h>
#include <kstandarddirs.h>
#include <qgl.h>
#include <qfile.h>

static const char *description =
  I18N_NOOP("KAME");
// INSERT A DESCRIPTION FOR YOUR APPLICATION HERE
	
	
static KCmdLineOptions options[] =
  {
    { "logging", I18N_NOOP("log debuging info."), 0 },
    { "nodr", 0, 0 },
    { "nodirectrender", I18N_NOOP("do not use direct rendering"), 0 },
    { "+[File]", I18N_NOOP("measurement file to open"), 0 },
    KCmdLineLastOption
    // INSERT YOUR COMMANDLINE OPTIONS HERE
  };

int main(int argc, char *argv[])
{
#ifdef HAVE_LIBGCCPP
   //initialize GC
   GC_INIT();
  // GC_find_leak = 1;
  //GC_dont_gc
 #endif  
   
  KAboutData aboutData( PACKAGE, I18N_NOOP("KAME"),
			VERSION, description, KAboutData::License_GPL,
			"(c) 2003-2006, ", 0, 0, "");
  aboutData.addAuthor("Kentaro Kitagawa",0, "kitagawa@scphys.kyoto-u.ac.jp");
  KCmdLineArgs::init( argc, argv, &aboutData );
  KCmdLineArgs::addCmdLineOptions( options ); // Add our own options.

  KApplication *app;
  app = new KApplication;
  
  {
    KGlobal::dirs()->addPrefix(".");
    makeIcons(app->iconLoader());
    {
            KCmdLineArgs *args = KCmdLineArgs::parsedArgs();
        
           QGLFormat f;
            f.setDirectRendering( args->isSet("directrender") );
            QGLFormat::setDefaultFormat( f );
            
            g_bLogDbgPrint = args->isSet("logging");
            
        FrmKameMain *form;
            form = new FrmKameMain();
            app->setMainWidget(form);
        //    form->resize(QSize(QApplication::desktop()->width(), QApplication::desktop()->height() - 200 ).expandedTo(form->sizeHint()) );
        //    form->switchToChildframeMode();
    //        form->setToolviewStyle(KMdi::IconOnly);
            form->setToolviewStyle(KMdi::TextAndIcon);
        //    form->setGeometry(0, 0, form->width(), form->height());
            form->show();
            
            if (args->count())
              {
                  form->openMes( QFile::decodeName( args->arg(0)));    
              }
            else
              {
              }
            args->clear();
          }
    }
  /*
  while(!app->closingDown()) {
      bool idle = g_signalBuffer->synchronize();  
      if(idle) app->processEvents();
      app->processEvents(15);
  }
    return 0;
  */

  int ret = app->exec();

  return ret;
}  
