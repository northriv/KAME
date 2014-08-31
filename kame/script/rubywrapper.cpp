#ifndef _MSC_VER
    #include <ruby.h>
#endif
#include "rubywrapper.h"
#ifdef _MSC_VER
    #include <ruby.h>
#endif

static_assert(sizeof(VALUE) == sizeof(Ruby::Value), "Size mismatch for VALUE.");

const int Ruby::Nil = Qnil;
const int Ruby::False = Qfalse;
const int Ruby::True = Qtrue;

Ruby::Ruby(const char *scriptname) {
	ruby_init();
    ruby_script(scriptname);
    ruby_init_loadpath();
}
Ruby::~Ruby() {
    ruby_finalize();
}
int
Ruby::evalProtect(const char* str) {
	int state = 0;
	rb_eval_string_protect(str, &state);
    return state;
}
void
Ruby::defineGlobalConst(const char *rbname, Value obj) {
	rb_define_global_const(rbname, obj);
}
Ruby::Value
Ruby::wrap_obj(Value cl, void *p, void (*f)(void *)) {
	return Data_Wrap_Struct(cl, 0, f, p);
}
void *
Ruby::unwrap_obj(Value self) {
    wrapped_t *ptr;
    Data_Get_Struct(self, wrapped_t, ptr);
	return ptr;
}

void 
Ruby::define_method(Value cl, const char *rbname, Value (*func)(...), int argnum) {
    rb_define_method(cl, rbname, func, argnum);
}
void
Ruby::define_singleton_method(Value obj, const char *rbname, Value (*func)(...), int argnum) {
    rb_define_singleton_method(obj, rbname, func, argnum);
}
Ruby::Value
Ruby::define_class(const char *rbname, Value super) {
    Value c = rb_define_class(rbname, (super != Nil) ? super : rb_cObject);
	rb_global_variable(&c);
	return c;
}
void
Ruby::emit_error(const char *errstr) {
    rb_raise(rb_eRuntimeError, "%s", errstr);
}

template <>
bool Ruby::isConvertible<const char*>(Value v) {
    return TYPE(v) == T_STRING;
}
template <>
bool Ruby::isConvertible<long>(Value v) {
    return FIXNUM_P(v);
}
template <>
bool Ruby::isConvertible<double>(Value v) {
    return (TYPE(v) == T_FLOAT) || FIXNUM_P(v) || (TYPE(v) == T_BIGNUM);
}
template <>
bool Ruby::isConvertible<bool>(Value v) {
    return (TYPE(v) == T_TRUE) || (TYPE(v) == T_FALSE);
}

template <>
const char* Ruby::convert(Value v) {
    if( !isConvertible<const char*>(v))
        throw "Type mismatch to STRING.";
    return RSTRING_PTR(v);
}
template <>
long Ruby::convert(Value v) {
    if( !isConvertible<long>(v))
        throw "Type mismatch to LONG.";
    return FIX2LONG(v);
}
template <>
double Ruby::convert(Value v) {
    if( !isConvertible<double>(v))
        throw "Type mismatch to NUM.";
    return NUM2DBL(v);
}
template <>
bool Ruby::convert(Value v) {
    if( !isConvertible<bool>(v))
        throw "Type mismatch to NUM.";
    return (TYPE(v) == T_TRUE) ? true : false;
}

Ruby::Value Ruby::convertToRuby(const std::string &str) {
    if(str.empty()) return rb_str_new2("");
    return rb_str_new2(str.c_str());
}

template <>
Ruby::Value Ruby::convertToRuby(int v) {
    return INT2NUM(v);
}

template <>
Ruby::Value Ruby::convertToRuby(unsigned int v) {
    return UINT2NUM(v);
}

template <>
Ruby::Value Ruby::convertToRuby(long v) {
    return LONG2NUM(v);
}

template <>
Ruby::Value Ruby::convertToRuby(unsigned long v) {
    return ULONG2NUM(v);
}

template <>
Ruby::Value Ruby::convertToRuby(double v) {
    return rb_float_new(v);
}

template <>
Ruby::Value Ruby::convertToRuby(bool v) {
    return v ? Qtrue : Qfalse;
}
