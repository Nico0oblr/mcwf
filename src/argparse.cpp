#include "argparse.hpp"

#include "yaml-cpp/yaml.h"
// #include "Util/HashCombine.hpp"
#include <cassert>
#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>

bool YamlOrCMD::is_option(const std::string & option) const {
  return m_config[option].IsDefined();
}

size_type YamlOrCMD::get_seed() const {
  return std::hash<std::string>()(get_descriptor());
}

std::string YamlOrCMD::get_descriptor() const {
  std::stringstream ss;
  ss << data();
  return ss.str();
}

const YAML::Node & YamlOrCMD::data() const {
  return m_config;
}

YAML::Node & YamlOrCMD::data() {
  return m_config;
}


std::string YamlOrCMD::filename(const std::string & prefix) const {
  if (is_option("filename")) {
    return parse<std::string>("filename");
  }
  std::string csv = ".csv";
  return prefix + std::to_string(get_seed()) + csv;
}

std::ofstream YamlOrCMD::to_file(const std::string & prefix,
				 bool write_descriptor) const {
  std::ofstream out(filename(prefix), std::ofstream::out);
  if (write_descriptor) {
    out << "#" << get_descriptor() << std::endl;
  }
  return out;
}

void YamlOrCMD::parse_cmd(char ** argv, int argc, char delimiter) {
  for (int i = 1; i < argc; ++i) {
    const char * key = *(argv + i);
    const char * end = key + strlen(key);            
    const char * position = std::find(key, end, delimiter);

    if (position != end) {
      std::istringstream istream(key);
      std::string temp;
      std::vector<std::string> strings;
      while (std::getline(istream, temp, delimiter)) strings.push_back(temp);
      assert(strings.size() == 2);
      m_config[strings[0]] = strings[1];
    } else {
      assert(i + 1 < argc);
      const char * value = *(argv + i + 1);
      m_config[key] = value;
      ++i;
    }
  }
}

YamlOrCMD::YamlOrCMD(char ** argv, int argc, const std::string & config)
  :m_config(YAML::LoadFile(config)) {
  parse_cmd(argv, argc);
}

YamlOrCMD::YamlOrCMD(char ** argv, int argc, const YAML::Node & config)
  :m_config(config) {
  parse_cmd(argv, argc);
}

YamlOrCMD::YamlOrCMD(char ** argv, int argc)
  :m_config() {
  parse_cmd(argv, argc);
}

YamlOrCMD::YamlOrCMD(const std::string & config)
  :m_config(YAML::LoadFile(config)) {}

YamlOrCMD::YamlOrCMD(const YAML::Node & config)
  :m_config(config) {}

YamlOrCMD::private_evalautor
YamlOrCMD::autoparse(const std::string & option) {
  return private_evalautor(this, option);
}

std::ostream & operator<<(std::ostream & os, const YamlOrCMD & parser) {
  os << parser.data();
  return os;
}

YamlOrCMD operator+(const YamlOrCMD & lhs, const YamlOrCMD & rhs) {
  return YamlOrCMD(merge_nodes(lhs.data(), rhs.data()));
}

YamlOrCMD::YamlOrCMD(const YamlOrCMD & other)
  :m_config(YAML::Clone(other.m_config)) {}
