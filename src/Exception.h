//
// Created by grady on 7/10/22.
//

#ifndef FUNDAMENTALS_PROCESSOR_EXCEPTION_H
#define FUNDAMENTALS_PROCESSOR_EXCEPTION_H

#include<exception>
#include<string>
#include<sstream>

#ifdef __APPLE__
using namespace std;

class Exception : public exception {
    string msg;
public:

    Exception(string error) {
        stringstream sstr;
        if (error.empty()) {
            sstr << "No error message for Exception\n";
        } else {
            sstr << error << "\n";
        }
        msg = sstr.str();
    }

    Exception()
        : Exception("")
    {
    }

    char const * what() const noexcept override {
        return msg.c_str();
    }

    string const & getString() const {
        return msg;
    }
};

#else
#include<stacktrace>

using namespace std;

class Exception : public exception {
    string msg;
public:

    Exception(string error, basic_stacktrace<allocator<stacktrace_entry>> const st = basic_stacktrace<allocator<stacktrace_entry>>::current()) {
        stringstream sstr;
        if(error.empty()) {
            sstr << "No error message for Exception\n" << st;
        } else {
            sstr << error << "\n" << st;
        }
        msg = sstr.str();
    }

    Exception(basic_stacktrace<allocator<stacktrace_entry>> const st = basic_stacktrace<allocator<stacktrace_entry>>::current())
        : Exception("", st)
    {
    }

    char const * what() const noexcept override {
        return msg.c_str();
    }
};
#endif

#if 0
#define map_at(map, key) \
    ({ \
        try { \
            map.at(key); \
        } catch (std::out_of_range& e) { \
            stringstream sstr;\
            sstr << __FILE__ << "(" << __LINE__ << "): key not found in map, " << key; \
            throw Exception(sstr.str()); \
        } \
    })
#else

template<typename T, typename Key>
auto mapAtFunc(T && t, Key && key, string file, int line) {
    try {
        return t.at(key);
    } catch(out_of_range & e){
        stringstream sstr;
        sstr << file << "(" << line << "): map has no key " << key;
        throw Exception(sstr.str());
    }
}

#define mapAt(map, key) mapAtFunc(map, key, __FILE__, __LINE__)

#endif

#endif //FUNDAMENTALS_PROCESSOR_EXCEPTION_H
