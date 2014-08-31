#include <memory>
#include <string>
#ifdef _MSC_VER
    #define snprintf _snprintf
#endif

//! Wraps Ruby C interface and hides mysterious ruby.h from C++ libraries.
class Ruby {
private:
    typedef std::pair<std::weak_ptr<void *>, std::weak_ptr<void *>> wrapped_t;
public:
	Ruby(const char *scriptname);
	~Ruby();

    static const int Nil;
    static const int False;
    static const int True;

    //! \return state.
    int evalProtect(const char* str);

    typedef unsigned long Value;

	void defineGlobalConst(const char *rbname, Value obj);

    //! C++ value to Ruby object
    template <typename X>
    static Value convertToRuby(X var);
    static Value convertToRuby(const std::string &var);
    //! Ruby object to C++ value
    template <typename X>
    static X convert(Value var);

    template <typename X>
    static bool isConvertible(Value var);

	template <class P, class T>
	struct Class {
        Class(std::shared_ptr<P> parent, const char *rbname, Value super = Nil);
        //!\todo MSVC2013 cl dies with multi-definitions.
        template<Value(P::*Func)(const std::shared_ptr<T>&)>
        void defineSingletonMethod(Value obj, const char *rbname);
        template<Value(P::*Func)(const std::shared_ptr<T>&,Value)>
        void defineSingletonMethod1(Value obj, const char *rbname);
        template<Value(P::*Func)(const std::shared_ptr<T>&,Value,Value)>
        void defineSingletonMethod2(Value obj, const char *rbname);
        template<Value(P::*Func)(const std::shared_ptr<T>&)>
        void defineMethod(const char *rbname);
        template<Value(P::*Func)(const std::shared_ptr<T>&,Value)>
        void defineMethod1(const char *rbname);
        template<Value(P::*Func)(const std::shared_ptr<T>&,Value,Value)>
        void defineMethod2(const char *rbname);
        Value rubyClassObject() const;
        Value rubyObject(const std::shared_ptr<T> &obj) const;
        static std::weak_ptr<T> unwrap(Value v) {
            return unwrap_internal<Ptr>(v).second;
        }
    private:
//#define RUBYDECL __cdecl
#ifndef RUBYDECL
    #define RUBYDECL
#endif
        typedef std::pair<std::weak_ptr<P>, std::weak_ptr<T>> Ptr;
        static_assert(sizeof(Ptr) == sizeof(wrapped_t), "");
        template<Value(P::*Func)(const std::shared_ptr<T>&)>
        int create_function(Value(**func)(Value)) {
            struct Func_t {
                static Value RUBYDECL func_internal(Value self) {
//            *func = [](Value self)->Value {
                char errstr[256];
                    try {
                        auto &st = unwrap_internal<Ptr>(self);
                        std::shared_ptr<P> p(st.first);
                        return (p.get()->*Func)(std::shared_ptr<T>(st.second));
					}
                    catch(std::bad_weak_ptr &) {
                        snprintf(errstr, sizeof(errstr) - 1, "C object no longer exists.");}
                    catch(std::string &e) {
                        snprintf(errstr, sizeof(errstr) - 1, "%s", e.c_str());}
                    catch(const char *e) {
                        snprintf(errstr, sizeof(errstr) - 1, "%s", e);}
                    emit_error(errstr); return Nil;
                }
            };
            *func = &Func_t::func_internal;
            return 0;
		}
        template<Value(P::*Func)(const std::shared_ptr<T>&, Value)>
        int create_function(Value(**func)(Value, Value)) {
            struct Func_t {
                static Value RUBYDECL func_internal(Value self, Value x) {
//            *func = [](Value self)->Value {
                char errstr[256];
                    try {
                        auto &st = unwrap_internal<Ptr>(self);
                        std::shared_ptr<P> p(st.first);
                        return (p.get()->*Func)(std::shared_ptr<T>(st.second), x);
                    }
                    catch(std::bad_weak_ptr &) {
                        snprintf(errstr, sizeof(errstr) - 1, "C object no longer exists.");}
                    catch(std::string &e) {
                        snprintf(errstr, sizeof(errstr) - 1, "%s", e.c_str());}
                    catch(const char *e) {
                        snprintf(errstr, sizeof(errstr) - 1, "%s", e);}
                    emit_error(errstr); return Nil;
                }
            };
            *func = &Func_t::func_internal;
            return 1;
        }
        template<Value(P::*Func)(const std::shared_ptr<T>&, Value, Value)>
        int create_function(Value(**func)(Value, Value, Value)) {
            struct Func_t {
                static Value RUBYDECL func_internal(Value self, Value x, Value y) {
//            *func = [](Value self)->Value {
                char errstr[256];
                    try {
                        auto &st = unwrap_internal<Ptr>(self);
                        std::shared_ptr<P> p(st.first);
                        return (p.get()->*Func)(std::shared_ptr<T>(st.second), x, y);
                    }
                    catch(std::bad_weak_ptr &) {
                        snprintf(errstr, sizeof(errstr) - 1, "C object no longer exists.");}
                    catch(std::string &e) {
                        snprintf(errstr, sizeof(errstr) - 1, "%s", e.c_str());}
                    catch(const char *e) {
                        snprintf(errstr, sizeof(errstr) - 1, "%s", e);}
                    emit_error(errstr); return Nil;
                }
            };
            *func = &Func_t::func_internal;
            return 2;
        }
        std::weak_ptr<P> m_parent;
		Value m_rbObj;
	};
private:
    template <class Y>
    static Y &unwrap_internal(Value self) {
        return *static_cast<Y*>(unwrap_obj(self));
    }
    static Value wrap_obj(Value cl, void *p, void (*)(void *));
    static void *unwrap_obj(Value self);

    static void define_method(Value cl, const char *rbname, Value (*func)(...), int argnum);
    static void define_singleton_method(Value obj, const char *rbname, Value (*func)(...), int argnum);
    static Value define_class(const char *rbname, Value super);

    //!C++ objects should be destroyed before.
    static void emit_error(const char *errstr);
};

template <class P, class T>
Ruby::Class<P,T>::Class(std::shared_ptr<P> parent, const char *rbname, Value super) : m_parent(parent) {
    m_rbObj = define_class(rbname, super);
}
template <class P, class T>
template<Ruby::Value(P::*Func)(const std::shared_ptr<T>&)>
void
Ruby::Class<P,T>::defineMethod(const char *rbname) {
    Value (*func)(Value);
    typedef Value(*fp)(...);
    int arg_num = create_function<Func>(&func);
    define_method(m_rbObj, rbname, reinterpret_cast<fp>(func), arg_num);
}
template <class P, class T>
template<Ruby::Value(P::*Func)(const std::shared_ptr<T>&,Ruby::Value)>
void
Ruby::Class<P,T>::defineMethod1(const char *rbname) {
    Value (*func)(Value,Value);
    typedef Value(*fp)(...);
    int arg_num = create_function<Func>(&func);
    define_method(m_rbObj, rbname, reinterpret_cast<fp>(func), arg_num);
}
template <class P, class T>
template<Ruby::Value(P::*Func)(const std::shared_ptr<T>&,Ruby::Value,Ruby::Value)>
void
Ruby::Class<P,T>::defineMethod2(const char *rbname) {
    Value (*func)(Value,Value,Value);
    typedef Value(*fp)(...);
    int arg_num = create_function<Func>(&func);
    define_method(m_rbObj, rbname, reinterpret_cast<fp>(func), arg_num);
}
template <class P, class T>
template<Ruby::Value(P::*Func)(const std::shared_ptr<T>&)>
void
Ruby::Class<P,T>::defineSingletonMethod(Value obj, const char *rbname) {
    Value (*func)(Value);
    typedef Value(*fp)(...);
    int arg_num = create_function<Func>(&func);
    define_singleton_method(obj, rbname, reinterpret_cast<fp>(func), arg_num);
}
template <class P, class T>
template<Ruby::Value(P::*Func)(const std::shared_ptr<T>&,Ruby::Value)>
void
Ruby::Class<P,T>::defineSingletonMethod1(Value obj, const char *rbname) {
    Value (*func)(Value,Value);
    typedef Value(*fp)(...);
    int arg_num = create_function<Func>(&func);
    define_singleton_method(obj, rbname, reinterpret_cast<fp>(func), arg_num);
}
template <class P, class T>
template<Ruby::Value(P::*Func)(const std::shared_ptr<T>&,Ruby::Value,Ruby::Value)>
void
Ruby::Class<P,T>::defineSingletonMethod2(Value obj, const char *rbname) {
    Value (*func)(Value,Value,Value);
    typedef Value(*fp)(...);
    int arg_num = create_function<Func>(&func);
    define_singleton_method(obj, rbname, reinterpret_cast<fp>(func), arg_num);
}
template <class P, class T>
Ruby::Value
Ruby::Class<P,T>::rubyClassObject() const {return m_rbObj;}
template <class P, class T>
Ruby::Value
Ruby::Class<P,T>::rubyObject(const std::shared_ptr<T> &obj) const {
    struct Deleter {
        static void deleter(void *p) { delete (Ptr*)p;}
    };
    return wrap_obj(m_rbObj, new Ptr(m_parent, obj), &Deleter::deleter);
}
