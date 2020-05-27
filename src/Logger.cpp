#include "Logger.hpp"

#include <iomanip>
#include <stdio.h>

std::ostringstream & Log::Get(TLogLevel level) {
  os << ToString(level) << ": ";
  os << std::string(level > logDEBUG ? level - logDEBUG : 0, '\t');
  os << std::boolalpha;
  return os;
}

Log::Log() {}

Log::~Log() {
  fprintf(stderr, "%s", os.str().c_str());
  fflush(stderr);
}

TLogLevel & Log::ReportingLevel() {
  static TLogLevel reporting_level = logINFO;
  return reporting_level;
}

std::string Log::ToString(TLogLevel level) {
  static const char * const buffer[] = {BOLD KRED "ERROR" KNRM BOFF,
                                        BOLD KMAG "WARNING" KNRM BOFF,
                                        BOLD KYEL "INFO" KNRM BOFF,
                                        BOLD KGRN "TIMING" KNRM BOFF,
					BOLD KCYN "DEBUG" KNRM BOFF,
                                        BOLD KCYN "DEBUG1" KNRM BOFF,
					BOLD KCYN "DEBUG2" KNRM BOFF,
					BOLD KCYN "DEBUG3" KNRM BOFF,
					BOLD KCYN "DEBUG4" KNRM BOFF};
  return buffer[level];
}

TLogLevel Log::FromString(const std::string & level) {
  if (level == "DEBUG4")
    return logDEBUG4;
  if (level == "DEBUG3")
    return logDEBUG3;
  if (level == "DEBUG2")
    return logDEBUG2;
  if (level == "DEBUG1")
    return logDEBUG1;
  if (level == "DEBUG")
    return logDEBUG;
  if (level == "INFO")
    return logINFO;
  if (level == "WARNING")
    return logWARNING;
  if (level == "ERROR")
    return logERROR;
  if (level == "TIME")
    return logTIME;
  Log().Get(logWARNING) << "Unknown logging level '"
                        << level << "'. Using INFO level as default.";
  return logINFO;
}

void log_sep(TLogLevel log_level, std::string name,
	     int n_sep, char sep) {
  std::ptrdiff_t new_n_sep = n_sep - name.size() / 2;
  LOG(log_level) << std::endl << std::string(new_n_sep, sep)
                 << name << std::string(new_n_sep, sep)
                 << std::endl;
}
