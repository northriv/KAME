/****************************************************************************
** XQConnector meta object code from reading C++ file 'xnodeconnector.h'
**
** Created: Wed Feb 1 04:04:48 2006
**      by: The Qt MOC ($Id: xnodeconnector.moc.cpp,v 1.1 2006/02/01 18:44:36 northriv Exp $)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#undef QT_NO_COMPAT
#include "../../kame/xnodeconnector.h"
#include <qmetaobject.h>
#include <qapplication.h>

#include <private/qucomextra_p.h>
#if !defined(Q_MOC_OUTPUT_REVISION) || (Q_MOC_OUTPUT_REVISION != 26)
#error "This file was generated using the moc from 3.3.5. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

const char *XQConnector::className() const
{
    return "XQConnector";
}

QMetaObject *XQConnector::metaObj = 0;
static QMetaObjectCleanUp cleanUp_XQConnector( "XQConnector", &XQConnector::staticMetaObject );

#ifndef QT_NO_TRANSLATION
QString XQConnector::tr( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XQConnector", s, c, QApplication::DefaultCodec );
    else
	return QString::fromLatin1( s );
}
#ifndef QT_NO_TRANSLATION_UTF8
QString XQConnector::trUtf8( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XQConnector", s, c, QApplication::UnicodeUTF8 );
    else
	return QString::fromUtf8( s );
}
#endif // QT_NO_TRANSLATION_UTF8

#endif // QT_NO_TRANSLATION

QMetaObject* XQConnector::staticMetaObject()
{
    if ( metaObj )
	return metaObj;
    QMetaObject* parentObject = QObject::staticMetaObject();
    metaObj = QMetaObject::new_metaobject(
	"XQConnector", parentObject,
	0, 0,
	0, 0,
#ifndef QT_NO_PROPERTIES
	0, 0,
	0, 0,
#endif // QT_NO_PROPERTIES
	0, 0 );
    cleanUp_XQConnector.setMetaObject( metaObj );
    return metaObj;
}

void* XQConnector::qt_cast( const char* clname )
{
    if ( !qstrcmp( clname, "XQConnector" ) )
	return this;
    if ( !qstrcmp( clname, "enable_shared_from_this<XQConnector>" ) )
	return (enable_shared_from_this<XQConnector>*)this;
    return QObject::qt_cast( clname );
}

bool XQConnector::qt_invoke( int _id, QUObject* _o )
{
    return QObject::qt_invoke(_id,_o);
}

bool XQConnector::qt_emit( int _id, QUObject* _o )
{
    return QObject::qt_emit(_id,_o);
}
#ifndef QT_NO_PROPERTIES

bool XQConnector::qt_property( int id, int f, QVariant* v)
{
    return QObject::qt_property( id, f, v);
}

bool XQConnector::qt_static_property( QObject* , int , int , QVariant* ){ return FALSE; }
#endif // QT_NO_PROPERTIES


const char *XQButtonConnector::className() const
{
    return "XQButtonConnector";
}

QMetaObject *XQButtonConnector::metaObj = 0;
static QMetaObjectCleanUp cleanUp_XQButtonConnector( "XQButtonConnector", &XQButtonConnector::staticMetaObject );

#ifndef QT_NO_TRANSLATION
QString XQButtonConnector::tr( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XQButtonConnector", s, c, QApplication::DefaultCodec );
    else
	return QString::fromLatin1( s );
}
#ifndef QT_NO_TRANSLATION_UTF8
QString XQButtonConnector::trUtf8( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XQButtonConnector", s, c, QApplication::UnicodeUTF8 );
    else
	return QString::fromUtf8( s );
}
#endif // QT_NO_TRANSLATION_UTF8

#endif // QT_NO_TRANSLATION

QMetaObject* XQButtonConnector::staticMetaObject()
{
    if ( metaObj )
	return metaObj;
    QMetaObject* parentObject = XQConnector::staticMetaObject();
    static const QUMethod slot_0 = {"onClick", 0, 0 };
    static const QMetaData slot_tbl[] = {
	{ "onClick()", &slot_0, QMetaData::Protected }
    };
    metaObj = QMetaObject::new_metaobject(
	"XQButtonConnector", parentObject,
	slot_tbl, 1,
	0, 0,
#ifndef QT_NO_PROPERTIES
	0, 0,
	0, 0,
#endif // QT_NO_PROPERTIES
	0, 0 );
    cleanUp_XQButtonConnector.setMetaObject( metaObj );
    return metaObj;
}

void* XQButtonConnector::qt_cast( const char* clname )
{
    if ( !qstrcmp( clname, "XQButtonConnector" ) )
	return this;
    return XQConnector::qt_cast( clname );
}

bool XQButtonConnector::qt_invoke( int _id, QUObject* _o )
{
    switch ( _id - staticMetaObject()->slotOffset() ) {
    case 0: onClick(); break;
    default:
	return XQConnector::qt_invoke( _id, _o );
    }
    return TRUE;
}

bool XQButtonConnector::qt_emit( int _id, QUObject* _o )
{
    return XQConnector::qt_emit(_id,_o);
}
#ifndef QT_NO_PROPERTIES

bool XQButtonConnector::qt_property( int id, int f, QVariant* v)
{
    return XQConnector::qt_property( id, f, v);
}

bool XQButtonConnector::qt_static_property( QObject* , int , int , QVariant* ){ return FALSE; }
#endif // QT_NO_PROPERTIES


const char *XValueQConnector::className() const
{
    return "XValueQConnector";
}

QMetaObject *XValueQConnector::metaObj = 0;
static QMetaObjectCleanUp cleanUp_XValueQConnector( "XValueQConnector", &XValueQConnector::staticMetaObject );

#ifndef QT_NO_TRANSLATION
QString XValueQConnector::tr( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XValueQConnector", s, c, QApplication::DefaultCodec );
    else
	return QString::fromLatin1( s );
}
#ifndef QT_NO_TRANSLATION_UTF8
QString XValueQConnector::trUtf8( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XValueQConnector", s, c, QApplication::UnicodeUTF8 );
    else
	return QString::fromUtf8( s );
}
#endif // QT_NO_TRANSLATION_UTF8

#endif // QT_NO_TRANSLATION

QMetaObject* XValueQConnector::staticMetaObject()
{
    if ( metaObj )
	return metaObj;
    QMetaObject* parentObject = XQConnector::staticMetaObject();
    metaObj = QMetaObject::new_metaobject(
	"XValueQConnector", parentObject,
	0, 0,
	0, 0,
#ifndef QT_NO_PROPERTIES
	0, 0,
	0, 0,
#endif // QT_NO_PROPERTIES
	0, 0 );
    cleanUp_XValueQConnector.setMetaObject( metaObj );
    return metaObj;
}

void* XValueQConnector::qt_cast( const char* clname )
{
    if ( !qstrcmp( clname, "XValueQConnector" ) )
	return this;
    return XQConnector::qt_cast( clname );
}

bool XValueQConnector::qt_invoke( int _id, QUObject* _o )
{
    return XQConnector::qt_invoke(_id,_o);
}

bool XValueQConnector::qt_emit( int _id, QUObject* _o )
{
    return XQConnector::qt_emit(_id,_o);
}
#ifndef QT_NO_PROPERTIES

bool XValueQConnector::qt_property( int id, int f, QVariant* v)
{
    return XQConnector::qt_property( id, f, v);
}

bool XValueQConnector::qt_static_property( QObject* , int , int , QVariant* ){ return FALSE; }
#endif // QT_NO_PROPERTIES


const char *XQLineEditConnector::className() const
{
    return "XQLineEditConnector";
}

QMetaObject *XQLineEditConnector::metaObj = 0;
static QMetaObjectCleanUp cleanUp_XQLineEditConnector( "XQLineEditConnector", &XQLineEditConnector::staticMetaObject );

#ifndef QT_NO_TRANSLATION
QString XQLineEditConnector::tr( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XQLineEditConnector", s, c, QApplication::DefaultCodec );
    else
	return QString::fromLatin1( s );
}
#ifndef QT_NO_TRANSLATION_UTF8
QString XQLineEditConnector::trUtf8( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XQLineEditConnector", s, c, QApplication::UnicodeUTF8 );
    else
	return QString::fromUtf8( s );
}
#endif // QT_NO_TRANSLATION_UTF8

#endif // QT_NO_TRANSLATION

QMetaObject* XQLineEditConnector::staticMetaObject()
{
    if ( metaObj )
	return metaObj;
    QMetaObject* parentObject = XValueQConnector::staticMetaObject();
    static const QUParameter param_slot_0[] = {
	{ 0, &static_QUType_QString, 0, QUParameter::In }
    };
    static const QUMethod slot_0 = {"onTextChanged", 1, param_slot_0 };
    static const QUMethod slot_1 = {"onReturnPressed", 0, 0 };
    static const QUMethod slot_2 = {"onExit", 0, 0 };
    static const QMetaData slot_tbl[] = {
	{ "onTextChanged(const QString&)", &slot_0, QMetaData::Protected },
	{ "onReturnPressed()", &slot_1, QMetaData::Protected },
	{ "onExit()", &slot_2, QMetaData::Protected }
    };
    metaObj = QMetaObject::new_metaobject(
	"XQLineEditConnector", parentObject,
	slot_tbl, 3,
	0, 0,
#ifndef QT_NO_PROPERTIES
	0, 0,
	0, 0,
#endif // QT_NO_PROPERTIES
	0, 0 );
    cleanUp_XQLineEditConnector.setMetaObject( metaObj );
    return metaObj;
}

void* XQLineEditConnector::qt_cast( const char* clname )
{
    if ( !qstrcmp( clname, "XQLineEditConnector" ) )
	return this;
    return XValueQConnector::qt_cast( clname );
}

bool XQLineEditConnector::qt_invoke( int _id, QUObject* _o )
{
    switch ( _id - staticMetaObject()->slotOffset() ) {
    case 0: onTextChanged((const QString&)static_QUType_QString.get(_o+1)); break;
    case 1: onReturnPressed(); break;
    case 2: onExit(); break;
    default:
	return XValueQConnector::qt_invoke( _id, _o );
    }
    return TRUE;
}

bool XQLineEditConnector::qt_emit( int _id, QUObject* _o )
{
    return XValueQConnector::qt_emit(_id,_o);
}
#ifndef QT_NO_PROPERTIES

bool XQLineEditConnector::qt_property( int id, int f, QVariant* v)
{
    return XValueQConnector::qt_property( id, f, v);
}

bool XQLineEditConnector::qt_static_property( QObject* , int , int , QVariant* ){ return FALSE; }
#endif // QT_NO_PROPERTIES


const char *XQTextBrowserConnector::className() const
{
    return "XQTextBrowserConnector";
}

QMetaObject *XQTextBrowserConnector::metaObj = 0;
static QMetaObjectCleanUp cleanUp_XQTextBrowserConnector( "XQTextBrowserConnector", &XQTextBrowserConnector::staticMetaObject );

#ifndef QT_NO_TRANSLATION
QString XQTextBrowserConnector::tr( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XQTextBrowserConnector", s, c, QApplication::DefaultCodec );
    else
	return QString::fromLatin1( s );
}
#ifndef QT_NO_TRANSLATION_UTF8
QString XQTextBrowserConnector::trUtf8( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XQTextBrowserConnector", s, c, QApplication::UnicodeUTF8 );
    else
	return QString::fromUtf8( s );
}
#endif // QT_NO_TRANSLATION_UTF8

#endif // QT_NO_TRANSLATION

QMetaObject* XQTextBrowserConnector::staticMetaObject()
{
    if ( metaObj )
	return metaObj;
    QMetaObject* parentObject = XValueQConnector::staticMetaObject();
    metaObj = QMetaObject::new_metaobject(
	"XQTextBrowserConnector", parentObject,
	0, 0,
	0, 0,
#ifndef QT_NO_PROPERTIES
	0, 0,
	0, 0,
#endif // QT_NO_PROPERTIES
	0, 0 );
    cleanUp_XQTextBrowserConnector.setMetaObject( metaObj );
    return metaObj;
}

void* XQTextBrowserConnector::qt_cast( const char* clname )
{
    if ( !qstrcmp( clname, "XQTextBrowserConnector" ) )
	return this;
    return XValueQConnector::qt_cast( clname );
}

bool XQTextBrowserConnector::qt_invoke( int _id, QUObject* _o )
{
    return XValueQConnector::qt_invoke(_id,_o);
}

bool XQTextBrowserConnector::qt_emit( int _id, QUObject* _o )
{
    return XValueQConnector::qt_emit(_id,_o);
}
#ifndef QT_NO_PROPERTIES

bool XQTextBrowserConnector::qt_property( int id, int f, QVariant* v)
{
    return XValueQConnector::qt_property( id, f, v);
}

bool XQTextBrowserConnector::qt_static_property( QObject* , int , int , QVariant* ){ return FALSE; }
#endif // QT_NO_PROPERTIES


const char *XKIntNumInputConnector::className() const
{
    return "XKIntNumInputConnector";
}

QMetaObject *XKIntNumInputConnector::metaObj = 0;
static QMetaObjectCleanUp cleanUp_XKIntNumInputConnector( "XKIntNumInputConnector", &XKIntNumInputConnector::staticMetaObject );

#ifndef QT_NO_TRANSLATION
QString XKIntNumInputConnector::tr( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XKIntNumInputConnector", s, c, QApplication::DefaultCodec );
    else
	return QString::fromLatin1( s );
}
#ifndef QT_NO_TRANSLATION_UTF8
QString XKIntNumInputConnector::trUtf8( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XKIntNumInputConnector", s, c, QApplication::UnicodeUTF8 );
    else
	return QString::fromUtf8( s );
}
#endif // QT_NO_TRANSLATION_UTF8

#endif // QT_NO_TRANSLATION

QMetaObject* XKIntNumInputConnector::staticMetaObject()
{
    if ( metaObj )
	return metaObj;
    QMetaObject* parentObject = XValueQConnector::staticMetaObject();
    static const QUParameter param_slot_0[] = {
	{ "val", &static_QUType_int, 0, QUParameter::In }
    };
    static const QUMethod slot_0 = {"onChange", 1, param_slot_0 };
    static const QMetaData slot_tbl[] = {
	{ "onChange(int)", &slot_0, QMetaData::Protected }
    };
    metaObj = QMetaObject::new_metaobject(
	"XKIntNumInputConnector", parentObject,
	slot_tbl, 1,
	0, 0,
#ifndef QT_NO_PROPERTIES
	0, 0,
	0, 0,
#endif // QT_NO_PROPERTIES
	0, 0 );
    cleanUp_XKIntNumInputConnector.setMetaObject( metaObj );
    return metaObj;
}

void* XKIntNumInputConnector::qt_cast( const char* clname )
{
    if ( !qstrcmp( clname, "XKIntNumInputConnector" ) )
	return this;
    return XValueQConnector::qt_cast( clname );
}

bool XKIntNumInputConnector::qt_invoke( int _id, QUObject* _o )
{
    switch ( _id - staticMetaObject()->slotOffset() ) {
    case 0: onChange((int)static_QUType_int.get(_o+1)); break;
    default:
	return XValueQConnector::qt_invoke( _id, _o );
    }
    return TRUE;
}

bool XKIntNumInputConnector::qt_emit( int _id, QUObject* _o )
{
    return XValueQConnector::qt_emit(_id,_o);
}
#ifndef QT_NO_PROPERTIES

bool XKIntNumInputConnector::qt_property( int id, int f, QVariant* v)
{
    return XValueQConnector::qt_property( id, f, v);
}

bool XKIntNumInputConnector::qt_static_property( QObject* , int , int , QVariant* ){ return FALSE; }
#endif // QT_NO_PROPERTIES


const char *XQSpinBoxConnector::className() const
{
    return "XQSpinBoxConnector";
}

QMetaObject *XQSpinBoxConnector::metaObj = 0;
static QMetaObjectCleanUp cleanUp_XQSpinBoxConnector( "XQSpinBoxConnector", &XQSpinBoxConnector::staticMetaObject );

#ifndef QT_NO_TRANSLATION
QString XQSpinBoxConnector::tr( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XQSpinBoxConnector", s, c, QApplication::DefaultCodec );
    else
	return QString::fromLatin1( s );
}
#ifndef QT_NO_TRANSLATION_UTF8
QString XQSpinBoxConnector::trUtf8( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XQSpinBoxConnector", s, c, QApplication::UnicodeUTF8 );
    else
	return QString::fromUtf8( s );
}
#endif // QT_NO_TRANSLATION_UTF8

#endif // QT_NO_TRANSLATION

QMetaObject* XQSpinBoxConnector::staticMetaObject()
{
    if ( metaObj )
	return metaObj;
    QMetaObject* parentObject = XValueQConnector::staticMetaObject();
    static const QUParameter param_slot_0[] = {
	{ "val", &static_QUType_int, 0, QUParameter::In }
    };
    static const QUMethod slot_0 = {"onChange", 1, param_slot_0 };
    static const QMetaData slot_tbl[] = {
	{ "onChange(int)", &slot_0, QMetaData::Protected }
    };
    metaObj = QMetaObject::new_metaobject(
	"XQSpinBoxConnector", parentObject,
	slot_tbl, 1,
	0, 0,
#ifndef QT_NO_PROPERTIES
	0, 0,
	0, 0,
#endif // QT_NO_PROPERTIES
	0, 0 );
    cleanUp_XQSpinBoxConnector.setMetaObject( metaObj );
    return metaObj;
}

void* XQSpinBoxConnector::qt_cast( const char* clname )
{
    if ( !qstrcmp( clname, "XQSpinBoxConnector" ) )
	return this;
    return XValueQConnector::qt_cast( clname );
}

bool XQSpinBoxConnector::qt_invoke( int _id, QUObject* _o )
{
    switch ( _id - staticMetaObject()->slotOffset() ) {
    case 0: onChange((int)static_QUType_int.get(_o+1)); break;
    default:
	return XValueQConnector::qt_invoke( _id, _o );
    }
    return TRUE;
}

bool XQSpinBoxConnector::qt_emit( int _id, QUObject* _o )
{
    return XValueQConnector::qt_emit(_id,_o);
}
#ifndef QT_NO_PROPERTIES

bool XQSpinBoxConnector::qt_property( int id, int f, QVariant* v)
{
    return XValueQConnector::qt_property( id, f, v);
}

bool XQSpinBoxConnector::qt_static_property( QObject* , int , int , QVariant* ){ return FALSE; }
#endif // QT_NO_PROPERTIES


const char *XKDoubleNumInputConnector::className() const
{
    return "XKDoubleNumInputConnector";
}

QMetaObject *XKDoubleNumInputConnector::metaObj = 0;
static QMetaObjectCleanUp cleanUp_XKDoubleNumInputConnector( "XKDoubleNumInputConnector", &XKDoubleNumInputConnector::staticMetaObject );

#ifndef QT_NO_TRANSLATION
QString XKDoubleNumInputConnector::tr( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XKDoubleNumInputConnector", s, c, QApplication::DefaultCodec );
    else
	return QString::fromLatin1( s );
}
#ifndef QT_NO_TRANSLATION_UTF8
QString XKDoubleNumInputConnector::trUtf8( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XKDoubleNumInputConnector", s, c, QApplication::UnicodeUTF8 );
    else
	return QString::fromUtf8( s );
}
#endif // QT_NO_TRANSLATION_UTF8

#endif // QT_NO_TRANSLATION

QMetaObject* XKDoubleNumInputConnector::staticMetaObject()
{
    if ( metaObj )
	return metaObj;
    QMetaObject* parentObject = XValueQConnector::staticMetaObject();
    static const QUParameter param_slot_0[] = {
	{ "val", &static_QUType_double, 0, QUParameter::In }
    };
    static const QUMethod slot_0 = {"onChange", 1, param_slot_0 };
    static const QMetaData slot_tbl[] = {
	{ "onChange(double)", &slot_0, QMetaData::Protected }
    };
    metaObj = QMetaObject::new_metaobject(
	"XKDoubleNumInputConnector", parentObject,
	slot_tbl, 1,
	0, 0,
#ifndef QT_NO_PROPERTIES
	0, 0,
	0, 0,
#endif // QT_NO_PROPERTIES
	0, 0 );
    cleanUp_XKDoubleNumInputConnector.setMetaObject( metaObj );
    return metaObj;
}

void* XKDoubleNumInputConnector::qt_cast( const char* clname )
{
    if ( !qstrcmp( clname, "XKDoubleNumInputConnector" ) )
	return this;
    return XValueQConnector::qt_cast( clname );
}

bool XKDoubleNumInputConnector::qt_invoke( int _id, QUObject* _o )
{
    switch ( _id - staticMetaObject()->slotOffset() ) {
    case 0: onChange((double)static_QUType_double.get(_o+1)); break;
    default:
	return XValueQConnector::qt_invoke( _id, _o );
    }
    return TRUE;
}

bool XKDoubleNumInputConnector::qt_emit( int _id, QUObject* _o )
{
    return XValueQConnector::qt_emit(_id,_o);
}
#ifndef QT_NO_PROPERTIES

bool XKDoubleNumInputConnector::qt_property( int id, int f, QVariant* v)
{
    return XValueQConnector::qt_property( id, f, v);
}

bool XKDoubleNumInputConnector::qt_static_property( QObject* , int , int , QVariant* ){ return FALSE; }
#endif // QT_NO_PROPERTIES


const char *XKDoubleSpinBoxConnector::className() const
{
    return "XKDoubleSpinBoxConnector";
}

QMetaObject *XKDoubleSpinBoxConnector::metaObj = 0;
static QMetaObjectCleanUp cleanUp_XKDoubleSpinBoxConnector( "XKDoubleSpinBoxConnector", &XKDoubleSpinBoxConnector::staticMetaObject );

#ifndef QT_NO_TRANSLATION
QString XKDoubleSpinBoxConnector::tr( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XKDoubleSpinBoxConnector", s, c, QApplication::DefaultCodec );
    else
	return QString::fromLatin1( s );
}
#ifndef QT_NO_TRANSLATION_UTF8
QString XKDoubleSpinBoxConnector::trUtf8( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XKDoubleSpinBoxConnector", s, c, QApplication::UnicodeUTF8 );
    else
	return QString::fromUtf8( s );
}
#endif // QT_NO_TRANSLATION_UTF8

#endif // QT_NO_TRANSLATION

QMetaObject* XKDoubleSpinBoxConnector::staticMetaObject()
{
    if ( metaObj )
	return metaObj;
    QMetaObject* parentObject = XValueQConnector::staticMetaObject();
    static const QUParameter param_slot_0[] = {
	{ "val", &static_QUType_double, 0, QUParameter::In }
    };
    static const QUMethod slot_0 = {"onChange", 1, param_slot_0 };
    static const QMetaData slot_tbl[] = {
	{ "onChange(double)", &slot_0, QMetaData::Protected }
    };
    metaObj = QMetaObject::new_metaobject(
	"XKDoubleSpinBoxConnector", parentObject,
	slot_tbl, 1,
	0, 0,
#ifndef QT_NO_PROPERTIES
	0, 0,
	0, 0,
#endif // QT_NO_PROPERTIES
	0, 0 );
    cleanUp_XKDoubleSpinBoxConnector.setMetaObject( metaObj );
    return metaObj;
}

void* XKDoubleSpinBoxConnector::qt_cast( const char* clname )
{
    if ( !qstrcmp( clname, "XKDoubleSpinBoxConnector" ) )
	return this;
    return XValueQConnector::qt_cast( clname );
}

bool XKDoubleSpinBoxConnector::qt_invoke( int _id, QUObject* _o )
{
    switch ( _id - staticMetaObject()->slotOffset() ) {
    case 0: onChange((double)static_QUType_double.get(_o+1)); break;
    default:
	return XValueQConnector::qt_invoke( _id, _o );
    }
    return TRUE;
}

bool XKDoubleSpinBoxConnector::qt_emit( int _id, QUObject* _o )
{
    return XValueQConnector::qt_emit(_id,_o);
}
#ifndef QT_NO_PROPERTIES

bool XKDoubleSpinBoxConnector::qt_property( int id, int f, QVariant* v)
{
    return XValueQConnector::qt_property( id, f, v);
}

bool XKDoubleSpinBoxConnector::qt_static_property( QObject* , int , int , QVariant* ){ return FALSE; }
#endif // QT_NO_PROPERTIES


const char *XKURLReqConnector::className() const
{
    return "XKURLReqConnector";
}

QMetaObject *XKURLReqConnector::metaObj = 0;
static QMetaObjectCleanUp cleanUp_XKURLReqConnector( "XKURLReqConnector", &XKURLReqConnector::staticMetaObject );

#ifndef QT_NO_TRANSLATION
QString XKURLReqConnector::tr( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XKURLReqConnector", s, c, QApplication::DefaultCodec );
    else
	return QString::fromLatin1( s );
}
#ifndef QT_NO_TRANSLATION_UTF8
QString XKURLReqConnector::trUtf8( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XKURLReqConnector", s, c, QApplication::UnicodeUTF8 );
    else
	return QString::fromUtf8( s );
}
#endif // QT_NO_TRANSLATION_UTF8

#endif // QT_NO_TRANSLATION

QMetaObject* XKURLReqConnector::staticMetaObject()
{
    if ( metaObj )
	return metaObj;
    QMetaObject* parentObject = XValueQConnector::staticMetaObject();
    static const QUParameter param_slot_0[] = {
	{ 0, &static_QUType_QString, 0, QUParameter::In }
    };
    static const QUMethod slot_0 = {"onSelect", 1, param_slot_0 };
    static const QMetaData slot_tbl[] = {
	{ "onSelect(const QString&)", &slot_0, QMetaData::Protected }
    };
    metaObj = QMetaObject::new_metaobject(
	"XKURLReqConnector", parentObject,
	slot_tbl, 1,
	0, 0,
#ifndef QT_NO_PROPERTIES
	0, 0,
	0, 0,
#endif // QT_NO_PROPERTIES
	0, 0 );
    cleanUp_XKURLReqConnector.setMetaObject( metaObj );
    return metaObj;
}

void* XKURLReqConnector::qt_cast( const char* clname )
{
    if ( !qstrcmp( clname, "XKURLReqConnector" ) )
	return this;
    return XValueQConnector::qt_cast( clname );
}

bool XKURLReqConnector::qt_invoke( int _id, QUObject* _o )
{
    switch ( _id - staticMetaObject()->slotOffset() ) {
    case 0: onSelect((const QString&)static_QUType_QString.get(_o+1)); break;
    default:
	return XValueQConnector::qt_invoke( _id, _o );
    }
    return TRUE;
}

bool XKURLReqConnector::qt_emit( int _id, QUObject* _o )
{
    return XValueQConnector::qt_emit(_id,_o);
}
#ifndef QT_NO_PROPERTIES

bool XKURLReqConnector::qt_property( int id, int f, QVariant* v)
{
    return XValueQConnector::qt_property( id, f, v);
}

bool XKURLReqConnector::qt_static_property( QObject* , int , int , QVariant* ){ return FALSE; }
#endif // QT_NO_PROPERTIES


const char *XQLabelConnector::className() const
{
    return "XQLabelConnector";
}

QMetaObject *XQLabelConnector::metaObj = 0;
static QMetaObjectCleanUp cleanUp_XQLabelConnector( "XQLabelConnector", &XQLabelConnector::staticMetaObject );

#ifndef QT_NO_TRANSLATION
QString XQLabelConnector::tr( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XQLabelConnector", s, c, QApplication::DefaultCodec );
    else
	return QString::fromLatin1( s );
}
#ifndef QT_NO_TRANSLATION_UTF8
QString XQLabelConnector::trUtf8( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XQLabelConnector", s, c, QApplication::UnicodeUTF8 );
    else
	return QString::fromUtf8( s );
}
#endif // QT_NO_TRANSLATION_UTF8

#endif // QT_NO_TRANSLATION

QMetaObject* XQLabelConnector::staticMetaObject()
{
    if ( metaObj )
	return metaObj;
    QMetaObject* parentObject = XValueQConnector::staticMetaObject();
    metaObj = QMetaObject::new_metaobject(
	"XQLabelConnector", parentObject,
	0, 0,
	0, 0,
#ifndef QT_NO_PROPERTIES
	0, 0,
	0, 0,
#endif // QT_NO_PROPERTIES
	0, 0 );
    cleanUp_XQLabelConnector.setMetaObject( metaObj );
    return metaObj;
}

void* XQLabelConnector::qt_cast( const char* clname )
{
    if ( !qstrcmp( clname, "XQLabelConnector" ) )
	return this;
    return XValueQConnector::qt_cast( clname );
}

bool XQLabelConnector::qt_invoke( int _id, QUObject* _o )
{
    return XValueQConnector::qt_invoke(_id,_o);
}

bool XQLabelConnector::qt_emit( int _id, QUObject* _o )
{
    return XValueQConnector::qt_emit(_id,_o);
}
#ifndef QT_NO_PROPERTIES

bool XQLabelConnector::qt_property( int id, int f, QVariant* v)
{
    return XValueQConnector::qt_property( id, f, v);
}

bool XQLabelConnector::qt_static_property( QObject* , int , int , QVariant* ){ return FALSE; }
#endif // QT_NO_PROPERTIES


const char *XKLedConnector::className() const
{
    return "XKLedConnector";
}

QMetaObject *XKLedConnector::metaObj = 0;
static QMetaObjectCleanUp cleanUp_XKLedConnector( "XKLedConnector", &XKLedConnector::staticMetaObject );

#ifndef QT_NO_TRANSLATION
QString XKLedConnector::tr( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XKLedConnector", s, c, QApplication::DefaultCodec );
    else
	return QString::fromLatin1( s );
}
#ifndef QT_NO_TRANSLATION_UTF8
QString XKLedConnector::trUtf8( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XKLedConnector", s, c, QApplication::UnicodeUTF8 );
    else
	return QString::fromUtf8( s );
}
#endif // QT_NO_TRANSLATION_UTF8

#endif // QT_NO_TRANSLATION

QMetaObject* XKLedConnector::staticMetaObject()
{
    if ( metaObj )
	return metaObj;
    QMetaObject* parentObject = XValueQConnector::staticMetaObject();
    metaObj = QMetaObject::new_metaobject(
	"XKLedConnector", parentObject,
	0, 0,
	0, 0,
#ifndef QT_NO_PROPERTIES
	0, 0,
	0, 0,
#endif // QT_NO_PROPERTIES
	0, 0 );
    cleanUp_XKLedConnector.setMetaObject( metaObj );
    return metaObj;
}

void* XKLedConnector::qt_cast( const char* clname )
{
    if ( !qstrcmp( clname, "XKLedConnector" ) )
	return this;
    return XValueQConnector::qt_cast( clname );
}

bool XKLedConnector::qt_invoke( int _id, QUObject* _o )
{
    return XValueQConnector::qt_invoke(_id,_o);
}

bool XKLedConnector::qt_emit( int _id, QUObject* _o )
{
    return XValueQConnector::qt_emit(_id,_o);
}
#ifndef QT_NO_PROPERTIES

bool XKLedConnector::qt_property( int id, int f, QVariant* v)
{
    return XValueQConnector::qt_property( id, f, v);
}

bool XKLedConnector::qt_static_property( QObject* , int , int , QVariant* ){ return FALSE; }
#endif // QT_NO_PROPERTIES


const char *XQLCDNumberConnector::className() const
{
    return "XQLCDNumberConnector";
}

QMetaObject *XQLCDNumberConnector::metaObj = 0;
static QMetaObjectCleanUp cleanUp_XQLCDNumberConnector( "XQLCDNumberConnector", &XQLCDNumberConnector::staticMetaObject );

#ifndef QT_NO_TRANSLATION
QString XQLCDNumberConnector::tr( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XQLCDNumberConnector", s, c, QApplication::DefaultCodec );
    else
	return QString::fromLatin1( s );
}
#ifndef QT_NO_TRANSLATION_UTF8
QString XQLCDNumberConnector::trUtf8( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XQLCDNumberConnector", s, c, QApplication::UnicodeUTF8 );
    else
	return QString::fromUtf8( s );
}
#endif // QT_NO_TRANSLATION_UTF8

#endif // QT_NO_TRANSLATION

QMetaObject* XQLCDNumberConnector::staticMetaObject()
{
    if ( metaObj )
	return metaObj;
    QMetaObject* parentObject = XValueQConnector::staticMetaObject();
    metaObj = QMetaObject::new_metaobject(
	"XQLCDNumberConnector", parentObject,
	0, 0,
	0, 0,
#ifndef QT_NO_PROPERTIES
	0, 0,
	0, 0,
#endif // QT_NO_PROPERTIES
	0, 0 );
    cleanUp_XQLCDNumberConnector.setMetaObject( metaObj );
    return metaObj;
}

void* XQLCDNumberConnector::qt_cast( const char* clname )
{
    if ( !qstrcmp( clname, "XQLCDNumberConnector" ) )
	return this;
    return XValueQConnector::qt_cast( clname );
}

bool XQLCDNumberConnector::qt_invoke( int _id, QUObject* _o )
{
    return XValueQConnector::qt_invoke(_id,_o);
}

bool XQLCDNumberConnector::qt_emit( int _id, QUObject* _o )
{
    return XValueQConnector::qt_emit(_id,_o);
}
#ifndef QT_NO_PROPERTIES

bool XQLCDNumberConnector::qt_property( int id, int f, QVariant* v)
{
    return XValueQConnector::qt_property( id, f, v);
}

bool XQLCDNumberConnector::qt_static_property( QObject* , int , int , QVariant* ){ return FALSE; }
#endif // QT_NO_PROPERTIES


const char *XQToggleButtonConnector::className() const
{
    return "XQToggleButtonConnector";
}

QMetaObject *XQToggleButtonConnector::metaObj = 0;
static QMetaObjectCleanUp cleanUp_XQToggleButtonConnector( "XQToggleButtonConnector", &XQToggleButtonConnector::staticMetaObject );

#ifndef QT_NO_TRANSLATION
QString XQToggleButtonConnector::tr( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XQToggleButtonConnector", s, c, QApplication::DefaultCodec );
    else
	return QString::fromLatin1( s );
}
#ifndef QT_NO_TRANSLATION_UTF8
QString XQToggleButtonConnector::trUtf8( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XQToggleButtonConnector", s, c, QApplication::UnicodeUTF8 );
    else
	return QString::fromUtf8( s );
}
#endif // QT_NO_TRANSLATION_UTF8

#endif // QT_NO_TRANSLATION

QMetaObject* XQToggleButtonConnector::staticMetaObject()
{
    if ( metaObj )
	return metaObj;
    QMetaObject* parentObject = XValueQConnector::staticMetaObject();
    static const QUMethod slot_0 = {"onClick", 0, 0 };
    static const QMetaData slot_tbl[] = {
	{ "onClick()", &slot_0, QMetaData::Protected }
    };
    metaObj = QMetaObject::new_metaobject(
	"XQToggleButtonConnector", parentObject,
	slot_tbl, 1,
	0, 0,
#ifndef QT_NO_PROPERTIES
	0, 0,
	0, 0,
#endif // QT_NO_PROPERTIES
	0, 0 );
    cleanUp_XQToggleButtonConnector.setMetaObject( metaObj );
    return metaObj;
}

void* XQToggleButtonConnector::qt_cast( const char* clname )
{
    if ( !qstrcmp( clname, "XQToggleButtonConnector" ) )
	return this;
    return XValueQConnector::qt_cast( clname );
}

bool XQToggleButtonConnector::qt_invoke( int _id, QUObject* _o )
{
    switch ( _id - staticMetaObject()->slotOffset() ) {
    case 0: onClick(); break;
    default:
	return XValueQConnector::qt_invoke( _id, _o );
    }
    return TRUE;
}

bool XQToggleButtonConnector::qt_emit( int _id, QUObject* _o )
{
    return XValueQConnector::qt_emit(_id,_o);
}
#ifndef QT_NO_PROPERTIES

bool XQToggleButtonConnector::qt_property( int id, int f, QVariant* v)
{
    return XValueQConnector::qt_property( id, f, v);
}

bool XQToggleButtonConnector::qt_static_property( QObject* , int , int , QVariant* ){ return FALSE; }
#endif // QT_NO_PROPERTIES


const char *XListQConnector::className() const
{
    return "XListQConnector";
}

QMetaObject *XListQConnector::metaObj = 0;
static QMetaObjectCleanUp cleanUp_XListQConnector( "XListQConnector", &XListQConnector::staticMetaObject );

#ifndef QT_NO_TRANSLATION
QString XListQConnector::tr( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XListQConnector", s, c, QApplication::DefaultCodec );
    else
	return QString::fromLatin1( s );
}
#ifndef QT_NO_TRANSLATION_UTF8
QString XListQConnector::trUtf8( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XListQConnector", s, c, QApplication::UnicodeUTF8 );
    else
	return QString::fromUtf8( s );
}
#endif // QT_NO_TRANSLATION_UTF8

#endif // QT_NO_TRANSLATION

QMetaObject* XListQConnector::staticMetaObject()
{
    if ( metaObj )
	return metaObj;
    QMetaObject* parentObject = XQConnector::staticMetaObject();
    metaObj = QMetaObject::new_metaobject(
	"XListQConnector", parentObject,
	0, 0,
	0, 0,
#ifndef QT_NO_PROPERTIES
	0, 0,
	0, 0,
#endif // QT_NO_PROPERTIES
	0, 0 );
    cleanUp_XListQConnector.setMetaObject( metaObj );
    return metaObj;
}

void* XListQConnector::qt_cast( const char* clname )
{
    if ( !qstrcmp( clname, "XListQConnector" ) )
	return this;
    return XQConnector::qt_cast( clname );
}

bool XListQConnector::qt_invoke( int _id, QUObject* _o )
{
    return XQConnector::qt_invoke(_id,_o);
}

bool XListQConnector::qt_emit( int _id, QUObject* _o )
{
    return XQConnector::qt_emit(_id,_o);
}
#ifndef QT_NO_PROPERTIES

bool XListQConnector::qt_property( int id, int f, QVariant* v)
{
    return XQConnector::qt_property( id, f, v);
}

bool XListQConnector::qt_static_property( QObject* , int , int , QVariant* ){ return FALSE; }
#endif // QT_NO_PROPERTIES


const char *XItemQConnector::className() const
{
    return "XItemQConnector";
}

QMetaObject *XItemQConnector::metaObj = 0;
static QMetaObjectCleanUp cleanUp_XItemQConnector( "XItemQConnector", &XItemQConnector::staticMetaObject );

#ifndef QT_NO_TRANSLATION
QString XItemQConnector::tr( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XItemQConnector", s, c, QApplication::DefaultCodec );
    else
	return QString::fromLatin1( s );
}
#ifndef QT_NO_TRANSLATION_UTF8
QString XItemQConnector::trUtf8( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XItemQConnector", s, c, QApplication::UnicodeUTF8 );
    else
	return QString::fromUtf8( s );
}
#endif // QT_NO_TRANSLATION_UTF8

#endif // QT_NO_TRANSLATION

QMetaObject* XItemQConnector::staticMetaObject()
{
    if ( metaObj )
	return metaObj;
    QMetaObject* parentObject = XValueQConnector::staticMetaObject();
    metaObj = QMetaObject::new_metaobject(
	"XItemQConnector", parentObject,
	0, 0,
	0, 0,
#ifndef QT_NO_PROPERTIES
	0, 0,
	0, 0,
#endif // QT_NO_PROPERTIES
	0, 0 );
    cleanUp_XItemQConnector.setMetaObject( metaObj );
    return metaObj;
}

void* XItemQConnector::qt_cast( const char* clname )
{
    if ( !qstrcmp( clname, "XItemQConnector" ) )
	return this;
    return XValueQConnector::qt_cast( clname );
}

bool XItemQConnector::qt_invoke( int _id, QUObject* _o )
{
    return XValueQConnector::qt_invoke(_id,_o);
}

bool XItemQConnector::qt_emit( int _id, QUObject* _o )
{
    return XValueQConnector::qt_emit(_id,_o);
}
#ifndef QT_NO_PROPERTIES

bool XItemQConnector::qt_property( int id, int f, QVariant* v)
{
    return XValueQConnector::qt_property( id, f, v);
}

bool XItemQConnector::qt_static_property( QObject* , int , int , QVariant* ){ return FALSE; }
#endif // QT_NO_PROPERTIES


const char *XQComboBoxConnector::className() const
{
    return "XQComboBoxConnector";
}

QMetaObject *XQComboBoxConnector::metaObj = 0;
static QMetaObjectCleanUp cleanUp_XQComboBoxConnector( "XQComboBoxConnector", &XQComboBoxConnector::staticMetaObject );

#ifndef QT_NO_TRANSLATION
QString XQComboBoxConnector::tr( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XQComboBoxConnector", s, c, QApplication::DefaultCodec );
    else
	return QString::fromLatin1( s );
}
#ifndef QT_NO_TRANSLATION_UTF8
QString XQComboBoxConnector::trUtf8( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XQComboBoxConnector", s, c, QApplication::UnicodeUTF8 );
    else
	return QString::fromUtf8( s );
}
#endif // QT_NO_TRANSLATION_UTF8

#endif // QT_NO_TRANSLATION

QMetaObject* XQComboBoxConnector::staticMetaObject()
{
    if ( metaObj )
	return metaObj;
    QMetaObject* parentObject = XItemQConnector::staticMetaObject();
    static const QUParameter param_slot_0[] = {
	{ "index", &static_QUType_int, 0, QUParameter::In }
    };
    static const QUMethod slot_0 = {"onSelect", 1, param_slot_0 };
    static const QMetaData slot_tbl[] = {
	{ "onSelect(int)", &slot_0, QMetaData::Protected }
    };
    metaObj = QMetaObject::new_metaobject(
	"XQComboBoxConnector", parentObject,
	slot_tbl, 1,
	0, 0,
#ifndef QT_NO_PROPERTIES
	0, 0,
	0, 0,
#endif // QT_NO_PROPERTIES
	0, 0 );
    cleanUp_XQComboBoxConnector.setMetaObject( metaObj );
    return metaObj;
}

void* XQComboBoxConnector::qt_cast( const char* clname )
{
    if ( !qstrcmp( clname, "XQComboBoxConnector" ) )
	return this;
    return XItemQConnector::qt_cast( clname );
}

bool XQComboBoxConnector::qt_invoke( int _id, QUObject* _o )
{
    switch ( _id - staticMetaObject()->slotOffset() ) {
    case 0: onSelect((int)static_QUType_int.get(_o+1)); break;
    default:
	return XItemQConnector::qt_invoke( _id, _o );
    }
    return TRUE;
}

bool XQComboBoxConnector::qt_emit( int _id, QUObject* _o )
{
    return XItemQConnector::qt_emit(_id,_o);
}
#ifndef QT_NO_PROPERTIES

bool XQComboBoxConnector::qt_property( int id, int f, QVariant* v)
{
    return XItemQConnector::qt_property( id, f, v);
}

bool XQComboBoxConnector::qt_static_property( QObject* , int , int , QVariant* ){ return FALSE; }
#endif // QT_NO_PROPERTIES


const char *XQListBoxConnector::className() const
{
    return "XQListBoxConnector";
}

QMetaObject *XQListBoxConnector::metaObj = 0;
static QMetaObjectCleanUp cleanUp_XQListBoxConnector( "XQListBoxConnector", &XQListBoxConnector::staticMetaObject );

#ifndef QT_NO_TRANSLATION
QString XQListBoxConnector::tr( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XQListBoxConnector", s, c, QApplication::DefaultCodec );
    else
	return QString::fromLatin1( s );
}
#ifndef QT_NO_TRANSLATION_UTF8
QString XQListBoxConnector::trUtf8( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XQListBoxConnector", s, c, QApplication::UnicodeUTF8 );
    else
	return QString::fromUtf8( s );
}
#endif // QT_NO_TRANSLATION_UTF8

#endif // QT_NO_TRANSLATION

QMetaObject* XQListBoxConnector::staticMetaObject()
{
    if ( metaObj )
	return metaObj;
    QMetaObject* parentObject = XItemQConnector::staticMetaObject();
    static const QUParameter param_slot_0[] = {
	{ "index", &static_QUType_int, 0, QUParameter::In }
    };
    static const QUMethod slot_0 = {"onSelect", 1, param_slot_0 };
    static const QMetaData slot_tbl[] = {
	{ "onSelect(int)", &slot_0, QMetaData::Protected }
    };
    metaObj = QMetaObject::new_metaobject(
	"XQListBoxConnector", parentObject,
	slot_tbl, 1,
	0, 0,
#ifndef QT_NO_PROPERTIES
	0, 0,
	0, 0,
#endif // QT_NO_PROPERTIES
	0, 0 );
    cleanUp_XQListBoxConnector.setMetaObject( metaObj );
    return metaObj;
}

void* XQListBoxConnector::qt_cast( const char* clname )
{
    if ( !qstrcmp( clname, "XQListBoxConnector" ) )
	return this;
    return XItemQConnector::qt_cast( clname );
}

bool XQListBoxConnector::qt_invoke( int _id, QUObject* _o )
{
    switch ( _id - staticMetaObject()->slotOffset() ) {
    case 0: onSelect((int)static_QUType_int.get(_o+1)); break;
    default:
	return XItemQConnector::qt_invoke( _id, _o );
    }
    return TRUE;
}

bool XQListBoxConnector::qt_emit( int _id, QUObject* _o )
{
    return XItemQConnector::qt_emit(_id,_o);
}
#ifndef QT_NO_PROPERTIES

bool XQListBoxConnector::qt_property( int id, int f, QVariant* v)
{
    return XItemQConnector::qt_property( id, f, v);
}

bool XQListBoxConnector::qt_static_property( QObject* , int , int , QVariant* ){ return FALSE; }
#endif // QT_NO_PROPERTIES


const char *XKColorButtonConnector::className() const
{
    return "XKColorButtonConnector";
}

QMetaObject *XKColorButtonConnector::metaObj = 0;
static QMetaObjectCleanUp cleanUp_XKColorButtonConnector( "XKColorButtonConnector", &XKColorButtonConnector::staticMetaObject );

#ifndef QT_NO_TRANSLATION
QString XKColorButtonConnector::tr( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XKColorButtonConnector", s, c, QApplication::DefaultCodec );
    else
	return QString::fromLatin1( s );
}
#ifndef QT_NO_TRANSLATION_UTF8
QString XKColorButtonConnector::trUtf8( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XKColorButtonConnector", s, c, QApplication::UnicodeUTF8 );
    else
	return QString::fromUtf8( s );
}
#endif // QT_NO_TRANSLATION_UTF8

#endif // QT_NO_TRANSLATION

QMetaObject* XKColorButtonConnector::staticMetaObject()
{
    if ( metaObj )
	return metaObj;
    QMetaObject* parentObject = XValueQConnector::staticMetaObject();
    static const QUParameter param_slot_0[] = {
	{ "newColor", &static_QUType_varptr, "\x0a", QUParameter::In }
    };
    static const QUMethod slot_0 = {"onClick", 1, param_slot_0 };
    static const QMetaData slot_tbl[] = {
	{ "onClick(const QColor&)", &slot_0, QMetaData::Protected }
    };
    metaObj = QMetaObject::new_metaobject(
	"XKColorButtonConnector", parentObject,
	slot_tbl, 1,
	0, 0,
#ifndef QT_NO_PROPERTIES
	0, 0,
	0, 0,
#endif // QT_NO_PROPERTIES
	0, 0 );
    cleanUp_XKColorButtonConnector.setMetaObject( metaObj );
    return metaObj;
}

void* XKColorButtonConnector::qt_cast( const char* clname )
{
    if ( !qstrcmp( clname, "XKColorButtonConnector" ) )
	return this;
    return XValueQConnector::qt_cast( clname );
}

bool XKColorButtonConnector::qt_invoke( int _id, QUObject* _o )
{
    switch ( _id - staticMetaObject()->slotOffset() ) {
    case 0: onClick((const QColor&)*((const QColor*)static_QUType_ptr.get(_o+1))); break;
    default:
	return XValueQConnector::qt_invoke( _id, _o );
    }
    return TRUE;
}

bool XKColorButtonConnector::qt_emit( int _id, QUObject* _o )
{
    return XValueQConnector::qt_emit(_id,_o);
}
#ifndef QT_NO_PROPERTIES

bool XKColorButtonConnector::qt_property( int id, int f, QVariant* v)
{
    return XValueQConnector::qt_property( id, f, v);
}

bool XKColorButtonConnector::qt_static_property( QObject* , int , int , QVariant* ){ return FALSE; }
#endif // QT_NO_PROPERTIES


const char *XKColorComboConnector::className() const
{
    return "XKColorComboConnector";
}

QMetaObject *XKColorComboConnector::metaObj = 0;
static QMetaObjectCleanUp cleanUp_XKColorComboConnector( "XKColorComboConnector", &XKColorComboConnector::staticMetaObject );

#ifndef QT_NO_TRANSLATION
QString XKColorComboConnector::tr( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XKColorComboConnector", s, c, QApplication::DefaultCodec );
    else
	return QString::fromLatin1( s );
}
#ifndef QT_NO_TRANSLATION_UTF8
QString XKColorComboConnector::trUtf8( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "XKColorComboConnector", s, c, QApplication::UnicodeUTF8 );
    else
	return QString::fromUtf8( s );
}
#endif // QT_NO_TRANSLATION_UTF8

#endif // QT_NO_TRANSLATION

QMetaObject* XKColorComboConnector::staticMetaObject()
{
    if ( metaObj )
	return metaObj;
    QMetaObject* parentObject = XValueQConnector::staticMetaObject();
    static const QUParameter param_slot_0[] = {
	{ "newColor", &static_QUType_varptr, "\x0a", QUParameter::In }
    };
    static const QUMethod slot_0 = {"onClick", 1, param_slot_0 };
    static const QMetaData slot_tbl[] = {
	{ "onClick(const QColor&)", &slot_0, QMetaData::Protected }
    };
    metaObj = QMetaObject::new_metaobject(
	"XKColorComboConnector", parentObject,
	slot_tbl, 1,
	0, 0,
#ifndef QT_NO_PROPERTIES
	0, 0,
	0, 0,
#endif // QT_NO_PROPERTIES
	0, 0 );
    cleanUp_XKColorComboConnector.setMetaObject( metaObj );
    return metaObj;
}

void* XKColorComboConnector::qt_cast( const char* clname )
{
    if ( !qstrcmp( clname, "XKColorComboConnector" ) )
	return this;
    return XValueQConnector::qt_cast( clname );
}

bool XKColorComboConnector::qt_invoke( int _id, QUObject* _o )
{
    switch ( _id - staticMetaObject()->slotOffset() ) {
    case 0: onClick((const QColor&)*((const QColor*)static_QUType_ptr.get(_o+1))); break;
    default:
	return XValueQConnector::qt_invoke( _id, _o );
    }
    return TRUE;
}

bool XKColorComboConnector::qt_emit( int _id, QUObject* _o )
{
    return XValueQConnector::qt_emit(_id,_o);
}
#ifndef QT_NO_PROPERTIES

bool XKColorComboConnector::qt_property( int id, int f, QVariant* v)
{
    return XValueQConnector::qt_property( id, f, v);
}

bool XKColorComboConnector::qt_static_property( QObject* , int , int , QVariant* ){ return FALSE; }
#endif // QT_NO_PROPERTIES
