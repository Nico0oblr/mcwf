#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <sstream>
#include <string>

// Terminal colors
// These are specific to bash - like consoles

#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"
#define BOLD  "\033[1m"
#define BOFF  "\033[0m"

enum TLogLevel {logERROR,
		logWARNING,
		logINFO,
		logTIME,
                logDEBUG,
		logDEBUG1, logDEBUG2, logDEBUG3, logDEBUG4};

/*
  Logger, that prints to console upon destruction.
*/
class Log {
public:
  Log();

  /*
    Flush the buffer, when the Logger is destroyed,
    i.e. print everything as late as possible
    --> minimal performance loss
  */
  ~Log();

  /*
    returs a logger to stream into
  */
  std::ostringstream & Get(TLogLevel level);

public:
  static TLogLevel & ReportingLevel();
  static std::string ToString(TLogLevel level);
  static TLogLevel FromString(const std::string & level);
protected:
  std::ostringstream os;
private:
  Log(const Log &);
  Log & operator=(const Log &);
};


/*
  Actual Logging macro, that should be used in code.
  Writes inserted string into std::cout, if the level is higher or equal to
  the set level.

  Available levels are

  logERROR,
  logWARNING,
  logINFO,
  logTIME,
  logDEBUG,
  logDEBUG1, 
  logDEBUG2, 
  logDEBUG3, 
  logDEBUG4

  with decreasing importance
*/
#define LOG(level)                              \
  if (level > Log::ReportingLevel()) ;          \
  else Log().Get(level)

/*
  Prints a separator with some and and some separating string to the logger
  looks like 

  ====================================== string ================================
  and ensures, that the whole string has size n_sep.
  Asserts, that the entered string is shorter than the requested length
*/
void log_sep(TLogLevel log_level, std::string name,
	     int n_sep = 40, char sep = '=');

/*
  LOG_VAR helper
*/
template<typename Var>
void log_var(const Var & var, const char* expression) {
  LOG(logINFO) << expression << "\t:\t"
               << var << std::endl;
}

/*
  LOG_VAR prints the variable name and variable to the info level of the logger
*/
#define LOG_VAR(EXPRESSION) log_var(EXPRESSION, #EXPRESSION)

#endif
