/****************************************************************************
** XQPulserDriverConnector meta object code from reading C++ file 'pulserdriverconnector.h'
**
** Created: Thu Mar 2 16:47:26 2006
**      by: The Qt MOC ($Id: pulserdriverconnector.moc.cpp,v 1.1.2.1 2006/03/02 09:20:44 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#undef QT_NO_COMPAT
#include "../../../../kame/users/nmr/pulserdriverconnector.h"
#include <qmetaobject.h>
#include <qapplication.h>

#include <private/qucomextra_p.h>
#if !defined(Q_MOC_OUTPUT_REVISION) || (Q_MOC_OUTPUT_REVISION != 26)
#error "This file was generated using the moc from 3.3.5. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

const char *XQPulserDriverConnector::className() const
{
    return "XQPulserDriverConnector";
}

QMetaObject *XQPulserDriverConnector::metaObj = 0;
static QMetaObjectCleanUp cleanUp_XQPulserDriverConnector( "XQPulserDriverConnector", &XQPulserDriverConnector::staticMetaObject );

#ifndef QT_NO_TRANSLATION
QString XQPulserDriverConnector::tr( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XQPulserDriverConnector", s, c, QApplication::DefaultCodec );
    else
	return QString::fromLatin1( s );
}
#ifndef QT_NO_TRANSLATION_UTF8
QString XQPulserDriverConnector::trUtf8( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XQPulserDriverConnector", s, c, QApplication::UnicodeUTF8 );
    else
	return QString::fromUtf8( s );
}
#endif // QT_NO_TRANSLATION_UTF8

#endif // QT_NO_TRANSLATION

QMetaObject* XQPulserDriverConnector::staticMetaObject()
{
    if ( metaObj )
	return metaObj;
    QMetaObject* parentObject = XQConnector::staticMetaObject();
    static const QUParameter param_slot_0[] = {
	{ "row", &static_QUType_int, 0, QUParameter::In },
	{ "col", &static_QUType_int, 0, QUParameter::In },
	{ "button", &static_QUType_int, 0, QUParameter::In },
	{ "mousePos", &static_QUType_varptr, "\x0e", QUParameter::In }
    };
    static const QUMethod slot_0 = {"clicked", 4, param_slot_0 };
    static const QUMethod slot_1 = {"selectionChanged", 0, 0 };
    static const QMetaData slot_tbl[] = {
	{ "clicked(int,int,int,const QPoint&)", &slot_0, QMetaData::Protected },
	{ "selectionChanged()", &slot_1, QMetaData::Protected }
    };
    metaObj = QMetaObject::new_metaobject(
	"XQPulserDriverConnector", parentObject,
	slot_tbl, 2,
	0, 0,
#ifndef QT_NO_PROPERTIES
	0, 0,
	0, 0,
#endif // QT_NO_PROPERTIES
	0, 0 );
    cleanUp_XQPulserDriverConnector.setMetaObject( metaObj );
    return metaObj;
}

void* XQPulserDriverConnector::qt_cast( const char* clname )
{
    if ( !qstrcmp( clname, "XQPulserDriverConnector" ) )
	return this;
    return XQConnector::qt_cast( clname );
}

bool XQPulserDriverConnector::qt_invoke( int _id, QUObject* _o )
{
    switch ( _id - staticMetaObject()->slotOffset() ) {
    case 0: clicked((int)static_QUType_int.get(_o+1),(int)static_QUType_int.get(_o+2),(int)static_QUType_int.get(_o+3),(const QPoint&)*((const QPoint*)static_QUType_ptr.get(_o+4))); break;
    case 1: selectionChanged(); break;
    default:
	return XQConnector::qt_invoke( _id, _o );
    }
    return TRUE;
}

bool XQPulserDriverConnector::qt_emit( int _id, QUObject* _o )
{
    return XQConnector::qt_emit(_id,_o);
}
#ifndef QT_NO_PROPERTIES

bool XQPulserDriverConnector::qt_property( int id, int f, QVariant* v)
{
    return XQConnector::qt_property( id, f, v);
}

bool XQPulserDriverConnector::qt_static_property( QObject* , int , int , QVariant* ){ return FALSE; }
#endif // QT_NO_PROPERTIES
