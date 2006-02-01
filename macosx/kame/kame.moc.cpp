/****************************************************************************
** FrmKameMain meta object code from reading C++ file 'kame.h'
**
** Created: Wed Feb 1 04:04:56 2006
**      by: The Qt MOC ($Id: kame.moc.cpp,v 1.1 2006/02/01 18:44:35 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#undef QT_NO_COMPAT
#include "../../kame/kame.h"
#include <qmetaobject.h>
#include <qapplication.h>

#include <private/qucomextra_p.h>
#if !defined(Q_MOC_OUTPUT_REVISION) || (Q_MOC_OUTPUT_REVISION != 26)
#error "This file was generated using the moc from 3.3.5. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

const char *FrmKameMain::className() const
{
    return "FrmKameMain";
}

QMetaObject *FrmKameMain::metaObj = 0;
static QMetaObjectCleanUp cleanUp_FrmKameMain( "FrmKameMain", &FrmKameMain::staticMetaObject );

#ifndef QT_NO_TRANSLATION
QString FrmKameMain::tr( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "FrmKameMain", s, c, QApplication::DefaultCodec );
    else
	return QString::fromLatin1( s );
}
#ifndef QT_NO_TRANSLATION_UTF8
QString FrmKameMain::trUtf8( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "FrmKameMain", s, c, QApplication::UnicodeUTF8 );
    else
	return QString::fromUtf8( s );
}
#endif // QT_NO_TRANSLATION_UTF8

#endif // QT_NO_TRANSLATION

QMetaObject* FrmKameMain::staticMetaObject()
{
    if ( metaObj )
	return metaObj;
    QMetaObject* parentObject = KMdiMainFrm::staticMetaObject();
    static const QUMethod slot_0 = {"fileCloseAction_activated", 0, 0 };
    static const QUMethod slot_1 = {"fileExitAction_activated", 0, 0 };
    static const QUMethod slot_2 = {"fileOpenAction_activated", 0, 0 };
    static const QUMethod slot_3 = {"fileSaveAction_activated", 0, 0 };
    static const QUMethod slot_4 = {"helpAboutAction_activated", 0, 0 };
    static const QUMethod slot_5 = {"helpContentsAction_activated", 0, 0 };
    static const QUMethod slot_6 = {"helpIndexAction_activated", 0, 0 };
    static const QUMethod slot_7 = {"mesStopAction_activated", 0, 0 };
    static const QUMethod slot_8 = {"scriptRunAction_activated", 0, 0 };
    static const QUMethod slot_9 = {"scriptDotSaveAction_activated", 0, 0 };
    static const QUParameter param_slot_10[] = {
	{ "var", &static_QUType_bool, 0, QUParameter::In }
    };
    static const QUMethod slot_10 = {"fileLogAction_toggled", 1, param_slot_10 };
    static const QUMethod slot_11 = {"aboutToQuit", 0, 0 };
    static const QUMethod slot_12 = {"processSignals", 0, 0 };
    static const QMetaData slot_tbl[] = {
	{ "fileCloseAction_activated()", &slot_0, QMetaData::Public },
	{ "fileExitAction_activated()", &slot_1, QMetaData::Public },
	{ "fileOpenAction_activated()", &slot_2, QMetaData::Public },
	{ "fileSaveAction_activated()", &slot_3, QMetaData::Public },
	{ "helpAboutAction_activated()", &slot_4, QMetaData::Public },
	{ "helpContentsAction_activated()", &slot_5, QMetaData::Public },
	{ "helpIndexAction_activated()", &slot_6, QMetaData::Public },
	{ "mesStopAction_activated()", &slot_7, QMetaData::Public },
	{ "scriptRunAction_activated()", &slot_8, QMetaData::Public },
	{ "scriptDotSaveAction_activated()", &slot_9, QMetaData::Public },
	{ "fileLogAction_toggled(bool)", &slot_10, QMetaData::Public },
	{ "aboutToQuit()", &slot_11, QMetaData::Protected },
	{ "processSignals()", &slot_12, QMetaData::Protected }
    };
    metaObj = QMetaObject::new_metaobject(
	"FrmKameMain", parentObject,
	slot_tbl, 13,
	0, 0,
#ifndef QT_NO_PROPERTIES
	0, 0,
	0, 0,
#endif // QT_NO_PROPERTIES
	0, 0 );
    cleanUp_FrmKameMain.setMetaObject( metaObj );
    return metaObj;
}

void* FrmKameMain::qt_cast( const char* clname )
{
    if ( !qstrcmp( clname, "FrmKameMain" ) )
	return this;
    return KMdiMainFrm::qt_cast( clname );
}

bool FrmKameMain::qt_invoke( int _id, QUObject* _o )
{
    switch ( _id - staticMetaObject()->slotOffset() ) {
    case 0: fileCloseAction_activated(); break;
    case 1: fileExitAction_activated(); break;
    case 2: fileOpenAction_activated(); break;
    case 3: fileSaveAction_activated(); break;
    case 4: helpAboutAction_activated(); break;
    case 5: helpContentsAction_activated(); break;
    case 6: helpIndexAction_activated(); break;
    case 7: mesStopAction_activated(); break;
    case 8: scriptRunAction_activated(); break;
    case 9: scriptDotSaveAction_activated(); break;
    case 10: fileLogAction_toggled((bool)static_QUType_bool.get(_o+1)); break;
    case 11: aboutToQuit(); break;
    case 12: processSignals(); break;
    default:
	return KMdiMainFrm::qt_invoke( _id, _o );
    }
    return TRUE;
}

bool FrmKameMain::qt_emit( int _id, QUObject* _o )
{
    return KMdiMainFrm::qt_emit(_id,_o);
}
#ifndef QT_NO_PROPERTIES

bool FrmKameMain::qt_property( int id, int f, QVariant* v)
{
    return KMdiMainFrm::qt_property( id, f, v);
}

bool FrmKameMain::qt_static_property( QObject* , int , int , QVariant* ){ return FALSE; }
#endif // QT_NO_PROPERTIES
