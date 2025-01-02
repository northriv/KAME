#if !defined _MSC_VER && !defined __MINGW64__
    #include <ruby.h>
#endif
#include "rubywrapper.h"
#if defined _MSC_VER || defined __MINGW64__
    #include <ruby.h>
#endif

static_assert(sizeof(VALUE) == sizeof(Ruby::Value), "Size mismatch for VALUE.");

const Ruby::Value Ruby::Nil = Qnil;
const Ruby::Value Ruby::False = Qfalse;
const Ruby::Value Ruby::True = Qtrue;

Ruby::Ruby(const char *scriptname, void *stack) {
    static char options_array[][16] = {"kame", "-e;"};
    static char *argv[] = {options_array[0], nullptr};
    int argc = sizeof(argv) / sizeof(char*) - 1;
#if defined __WIN32__ || defined WINDOWS || defined _WIN32
//    ruby_sysinit(&argc, (char***)(&argv)); //needed for win32
    NtInitialize(&argc, (char***)(&argv)); //needed for win32
#endif
    ruby_init_stack((VALUE*)stack);
    ruby_init();
    static char *options[] = {options_array[0], options_array[1], nullptr};
    argc = sizeof(options) / sizeof(char*) - 1;
    ruby_process_options(argc, options); //needed for ruby >= 3.
    ruby_script(scriptname);
}
Ruby::~Ruby() {
//    ruby_finalize();
    ruby_cleanup(0);
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

template <int argnum>
void
Ruby::define_method(Value cl, const char *rbname, Value (*func)(...)) {
    rb_define_method(cl, rbname, func, argnum);
}
template <int argnum>
void
Ruby::define_singleton_method(Value obj, const char *rbname, Value (*func)(...)) {
    rb_define_singleton_method(obj, rbname, func, argnum);
}

template
void Ruby::define_method<0>(Value cl, const char *rbname, Value (*func)(...));
template
void Ruby::define_method<1>(Value cl, const char *rbname, Value (*func)(...));
template
void Ruby::define_method<2>(Value cl, const char *rbname, Value (*func)(...));
template
void Ruby::define_singleton_method<0>(Value obj, const char *rbname, Value (*func)(...));
template
void Ruby::define_singleton_method<1>(Value obj, const char *rbname, Value (*func)(...));
template
void Ruby::define_singleton_method<2>(Value obj, const char *rbname, Value (*func)(...));


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
    return StringValueCStr(v);
//    if(RSTRING_PTR(v)[RSTRING_LEN(v) + 1] != 0)
//        throw "Type mismatch to STRING.";
//    return RSTRING_PTR(v);
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

void
Ruby::printErrorInfo() {
    rb_p(rb_errinfo());
}

template <class P, class T>
Ruby::Class<P,T>::Class(std::shared_ptr<P> parent, const char *rbname, Value super) :
    m_parent(parent) {
    auto f = [](void *p){delete (Ptr*)p;};
    auto s = [](const void *)->size_t{return sizeof(std::pair<std::weak_ptr<Ruby>, std::weak_ptr<Ruby>>);};
    rb_data_type_t t1{"XNode",
        {0, f, s},
        0, 0,
        RUBY_TYPED_FREE_IMMEDIATELY};
    s_obj_type = std::make_shared<rb_data_type_struct>(t1);

    m_rbObj = define_class(rbname, super);
}
template <class P, class T>
Ruby::Class<P,T>::Class::~Class() {

}

template <class P, class T>
Ruby::Value
Ruby::Class<P,T>::wrap_obj(Value cl, void *p) {
    rb_undef_alloc_func(cl);
    return TypedData_Wrap_Struct(cl, s_obj_type.get(), p);
}
template <class P, class T>
void *
Ruby::Class<P,T>::unwrap_obj(Value self) {
    wrapped_t *ptr;
    TypedData_Get_Struct(self, wrapped_t, s_obj_type.get(), ptr);
    return ptr;
}

template <class P, class T>
Ruby::Value
Ruby::Class<P,T>::rubyClassObject() const {return m_rbObj;}
template <class P, class T>
Ruby::Value
Ruby::Class<P,T>::rubyObject(const std::shared_ptr<T> &obj) const {
    return wrap_obj(m_rbObj, new Ptr(m_parent, obj));
}

#undef truncate
#include "xrubysupport.h"

template <class P, class T>
std::shared_ptr<rb_data_type_struct> Ruby::Class<P,T>::s_obj_type;

template struct Ruby::Class<XRuby, XNode>;
