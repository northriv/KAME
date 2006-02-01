/****************************************************************************
** XRawStreamRecordReaderConnector meta object code from reading C++ file 'recordreaderconnector.h'
**
** Created: Wed Feb 1 04:02:22 2006
**      by: The Qt MOC ($Id: recordreaderconnector.moc.cpp,v 1.1 2006/02/01 18:45:27 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#undef QT_NO_COMPAT
#include "../../../kame/analyzer/recordreaderconnector.h"
#include <qmetaobject.h>
#include <qapplication.h>

#include <private/qucomextra_p.h>
#if !defined(Q_MOC_OUTPUT_REVISION) || (Q_MOC_OUTPUT_REVISION != 26)
#error "This file was generated using the moc from 3.3.5. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

const char *XRawStreamRecordReaderConnector::className() const
{
    return "XRawStreamRecordReaderConnector";
}

QMetaObject *XRawStreamRecordReaderConnector::metaObj = 0;
static QMetaObjectCleanUp cleanUp_XRawStreamRecordReaderConnector( "XRawStreamRecordReaderConnector", &XRawStreamRecordReaderConnector::staticMetaObject );

#ifndef QT_NO_TRANSLATION
QString XRawStreamRecordReaderConnector::tr( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XRawStreamRecordReaderConnector", s, c, QApplication::DefaultCodec );
    else
	return QString::fromLatin1( s );
}
#ifndef QT_NO_TRANSLATION_UTF8
QString XRawStreamRecordReaderConnector::trUtf8( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XRawStreamRecordReaderConnector", s, c, QApplication::UnicodeUTF8 );
    else
	return QString::fromUtf8( s );
}
#endif // QT_NO_TRANSLATION_UTF8

#endif // QT_NO_TRANSLATION

QMetaObject* XRawStreamRecordReaderConnector::staticMetaObject()
{
    if ( metaObj )
	return metaObj;
    QMetaObject* parentObject = XQConnector::staticMetaObject();
    metaObj = QMetaObject::new_metaobject(
	"XRawStreamRecordReaderConnector", parentObject,
	0, 0,
	0, 0,
#ifndef QT_NO_PROPERTIES
	0, 0,
	0, 0,
#endif // QT_NO_PROPERTIES
	0, 0 );
    cleanUp_XRawStreamRecordReaderConnector.setMetaObject( metaObj );
    return metaObj;
}

void* XRawStreamRecordReaderConnector::qt_cast( const char* clname )
{
    if ( !qstrcmp( clname, "XRawStreamRecordReaderConnector" ) )
	return this;
    return XQConnector::qt_cast( clname );
}

bool XRawStreamRecordReaderConnector::qt_invoke( int _id, QUObject* _o )
{
    return XQConnector::qt_invoke(_id,_o);
}

bool XRawStreamRecordReaderConnector::qt_emit( int _id, QUObject* _o )
{
    return XQConnector::qt_emit(_id,_o);
}
#ifndef QT_NO_PROPERTIES

bool XRawStreamRecordReaderConnector::qt_property( int id, int f, QVariant* v)
{
    return XQConnector::qt_property( id, f, v);
}

bool XRawStreamRecordReaderConnector::qt_static_property( QObject* , int , int , QVariant* ){ return FALSE; }
#endif // QT_NO_PROPERTIES
