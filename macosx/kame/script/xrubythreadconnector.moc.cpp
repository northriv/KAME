/****************************************************************************
** XRubyThreadConnector meta object code from reading C++ file 'xrubythreadconnector.h'
**
** Created: Wed Feb 1 03:40:04 2006
**      by: The Qt MOC ($Id: xrubythreadconnector.moc.cpp,v 1.1 2006/02/01 18:44:37 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#undef QT_NO_COMPAT
#include "../../../kame/script/xrubythreadconnector.h"
#include <qmetaobject.h>
#include <qapplication.h>

#include <private/qucomextra_p.h>
#if !defined(Q_MOC_OUTPUT_REVISION) || (Q_MOC_OUTPUT_REVISION != 26)
#error "This file was generated using the moc from 3.3.5. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

const char *XRubyThreadConnector::className() const
{
    return "XRubyThreadConnector";
}

QMetaObject *XRubyThreadConnector::metaObj = 0;
static QMetaObjectCleanUp cleanUp_XRubyThreadConnector( "XRubyThreadConnector", &XRubyThreadConnector::staticMetaObject );

#ifndef QT_NO_TRANSLATION
QString XRubyThreadConnector::tr( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XRubyThreadConnector", s, c, QApplication::DefaultCodec );
    else
	return QString::fromLatin1( s );
}
#ifndef QT_NO_TRANSLATION_UTF8
QString XRubyThreadConnector::trUtf8( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XRubyThreadConnector", s, c, QApplication::UnicodeUTF8 );
    else
	return QString::fromUtf8( s );
}
#endif // QT_NO_TRANSLATION_UTF8

#endif // QT_NO_TRANSLATION

QMetaObject* XRubyThreadConnector::staticMetaObject()
{
    if ( metaObj )
	return metaObj;
    QMetaObject* parentObject = XQConnector::staticMetaObject();
    metaObj = QMetaObject::new_metaobject(
	"XRubyThreadConnector", parentObject,
	0, 0,
	0, 0,
#ifndef QT_NO_PROPERTIES
	0, 0,
	0, 0,
#endif // QT_NO_PROPERTIES
	0, 0 );
    cleanUp_XRubyThreadConnector.setMetaObject( metaObj );
    return metaObj;
}

void* XRubyThreadConnector::qt_cast( const char* clname )
{
    if ( !qstrcmp( clname, "XRubyThreadConnector" ) )
	return this;
    return XQConnector::qt_cast( clname );
}

bool XRubyThreadConnector::qt_invoke( int _id, QUObject* _o )
{
    return XQConnector::qt_invoke(_id,_o);
}

bool XRubyThreadConnector::qt_emit( int _id, QUObject* _o )
{
    return XQConnector::qt_emit(_id,_o);
}
#ifndef QT_NO_PROPERTIES

bool XRubyThreadConnector::qt_property( int id, int f, QVariant* v)
{
    return XQConnector::qt_property( id, f, v);
}

bool XRubyThreadConnector::qt_static_property( QObject* , int , int , QVariant* ){ return FALSE; }
#endif // QT_NO_PROPERTIES
