#include <memory>
#include <string>
#ifdef _MSC_VER
    #ifndef snprintf
        #define snprintf _snprintf
    #endif
#endif

#ifndef RUBYWRAPPER_H
#define RUBYWRAPPER_H

#ifndef DECLSPEC_KAME
    #define DECLSPEC_KAME
#endif

struct rb_data_type_struct;

//! Wraps Ruby C interface and hides mysterious ruby.h from C++ libraries.
class DECLSPEC_KAME Ruby {
private:
    typedef std::pair<std::weak_ptr<void *>, std::weak_ptr<void *>> wrapped_t;
public:
    Ruby(const char *scriptname, void *stack);
    ~Ruby();

    //! \return state.
    int evalProtect(const char* str);

    void printErrorInfo();

    using Value = uintptr_t; //has to be identical to VALUE
    static const Value Nil;
    static const Value False;
    static const Value True;

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
    struct DECLSPEC_KAME Class {
        Class(std::shared_ptr<P> parent, const char *rbname, Value super = Nil);
        ~Class();
        template<typename R = void>
        void defineSingletonMethod(Value obj, const char *rbname); //dummy for MSVC
        template<Value(P::*Func)(const std::shared_ptr<T>&)>
        void defineSingletonMethod(Value obj, const char *rbname);
        template<Value(P::*Func)(const std::shared_ptr<T>&,Value)>
        void defineSingletonMethod(Value obj, const char *rbname);
        template<Value(P::*Func)(const std::shared_ptr<T>&,Value,Value)>
        void defineSingletonMethod(Value obj, const char *rbname);
        template<typename R = void>
        void defineMethod(const char *rbname); //dummy for MSVC
        template<Value(P::*Func)(const std::shared_ptr<T>&)>
        void defineMethod(const char *rbname);
        template<Value(P::*Func)(const std::shared_ptr<T>&,Value)>
        void defineMethod(const char *rbname);
        template<Value(P::*Func)(const std::shared_ptr<T>&,Value,Value)>
        void defineMethod(const char *rbname);
        Value rubyClassObject() const;
        Value rubyObject(const std::shared_ptr<T> &obj) const;
        static std::weak_ptr<T> unwrap(Value v) {
            return unwrap_internal<Ptr>(v).second;
        }
        template <class Y>
        static Y &unwrap_internal(Value self) {
            return *static_cast<Y*>(unwrap_obj(self));
        }
        static Value wrap_obj(Value cl, void *p);
        static void *unwrap_obj(Value self);
    private:
#ifdef _MSC_VER
    #define RUBYDECL __cdecl
#else
    #define RUBYDECL
#endif
        typedef std::pair<std::weak_ptr<P>, std::weak_ptr<T>> Ptr;
        static_assert(sizeof(Ptr) == sizeof(wrapped_t), "");
        //! prepares a C-style function pointer to be called from Ruby.
        //! \tparam Func a pointer to a C++ member function.
        template<class tFunc, tFunc Func, typename ...Args>
        static void create_function(Value(**func)(Value, Args...)) {
            *func = [](Value self, Args...args)->Value {
                char errstr[256];
                    try {
                        auto &st = unwrap_internal<Ptr>(self);
                        std::shared_ptr<P> p(st.first);
                        return (p.get()->*Func)(std::shared_ptr<T>(st.second), args...);
                    }
                    catch(std::bad_weak_ptr &) {
                        snprintf(errstr, sizeof(errstr) - 1, "C object no longer exists.");}
                    catch(std::string &e) {
                        snprintf(errstr, sizeof(errstr) - 1, "%s", e.c_str());}
                    catch(const char *e) {
                        snprintf(errstr, sizeof(errstr) - 1, "%s", e);}
                    emit_error(errstr); return Nil;
            };
        }
        //! \return # of arguments in the ruby function.
        template<class tFunc, tFunc Func, typename ...Args>
        static constexpr int argnumofFn(Value(**func)(Value, Args...)) {
            return sizeof...(Args);
        }
        std::weak_ptr<P> m_parent;
        Value m_rbObj;
        static std::shared_ptr<rb_data_type_struct> s_obj_type; //For TypedData_Wrap_Struct
    };
private:
    template <int argnum>
    static void define_method(Value cl, const char *rbname, Value (*func)(...));
    template <int argnum>
    static void define_singleton_method(Value obj, const char *rbname, Value (*func)(...));
    static Value define_class(const char *rbname, Value super);

    //!C++ objects should be destroyed before.
    static void emit_error(const char *errstr);
};

template <class P, class T>
template<Ruby::Value(P::*Func)(const std::shared_ptr<T>&)>
void
Ruby::Class<P,T>::defineMethod(const char *rbname) {
    Value (*func)(Value);
    typedef Value(*fp)(...);
    create_function<decltype(Func), Func>(&func);
    constexpr int arg_num = argnumofFn<decltype(Func), Func>(&func);
    define_method<arg_num>(m_rbObj, rbname, reinterpret_cast<fp>(func));
}
template <class P, class T>
template<Ruby::Value(P::*Func)(const std::shared_ptr<T>&,Ruby::Value)>
void
Ruby::Class<P,T>::defineMethod(const char *rbname) {
    Value (*func)(Value,Value);
    typedef Value(*fp)(...);
    create_function<decltype(Func), Func>(&func);
    constexpr int arg_num = argnumofFn<decltype(Func), Func>(&func);
    define_method<arg_num>(m_rbObj, rbname, reinterpret_cast<fp>(func));
}
template <class P, class T>
template<Ruby::Value(P::*Func)(const std::shared_ptr<T>&,Ruby::Value,Ruby::Value)>
void
Ruby::Class<P,T>::defineMethod(const char *rbname) {
    Value (*func)(Value,Value,Value);
    typedef Value(*fp)(...);
    create_function<decltype(Func), Func>(&func);
    constexpr int arg_num = argnumofFn<decltype(Func), Func>(&func);
    define_method<arg_num>(m_rbObj, rbname, reinterpret_cast<fp>(func));
}
template <class P, class T>
template<Ruby::Value(P::*Func)(const std::shared_ptr<T>&)>
void
Ruby::Class<P,T>::defineSingletonMethod(Value obj, const char *rbname) {
    Value (*func)(Value);
    typedef Value(*fp)(...);
    create_function<decltype(Func), Func>(&func);
    constexpr int arg_num = argnumofFn<decltype(Func), Func>(&func);
    define_singleton_method<arg_num>(obj, rbname, reinterpret_cast<fp>(func));
}
template <class P, class T>
template<Ruby::Value(P::*Func)(const std::shared_ptr<T>&,Ruby::Value)>
void
Ruby::Class<P,T>::defineSingletonMethod(Value obj, const char *rbname) {
    Value (*func)(Value,Value);
    typedef Value(*fp)(...);
    create_function<decltype(Func), Func>(&func);
    constexpr int arg_num = argnumofFn<decltype(Func), Func>(&func);
    define_singleton_method<arg_num>(obj, rbname, reinterpret_cast<fp>(func));
}
template <class P, class T>
template<Ruby::Value(P::*Func)(const std::shared_ptr<T>&,Ruby::Value,Ruby::Value)>
void
Ruby::Class<P,T>::defineSingletonMethod(Value obj, const char *rbname) {
    Value (*func)(Value,Value,Value);
    typedef Value(*fp)(...);
    create_function<decltype(Func), Func>(&func);
    constexpr int arg_num = argnumofFn<decltype(Func), Func>(&func);
    define_singleton_method<arg_num>(obj, rbname, reinterpret_cast<fp>(func));
}
#endif
