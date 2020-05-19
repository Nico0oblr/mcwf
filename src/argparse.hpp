#ifndef ARGPARSE_H
#define ARGPARSE_H

#include <string>
#include "yaml-cpp/yaml.h"
using size_type = std::size_t;

/*
  Example:
  YamlOrCMD parser(argv, argc, "config/main.yaml");
  size_type lead_index1 = parser.parse<size_type>("lead_index1");

  then either call the program as
  ./program lead_index1 (value)
  or
  ./program lead_index1{delimiter}(value)

  or set the value lead_index1 to (value) in a yaml file, 
  in this case the file config/main.yaml
*/
class YamlOrCMD {
  size_type get_seed() const;

  std::string get_descriptor() const;

  
  struct private_evalautor{
    template<typename T>
    operator T() {
      return m_loc_data->parse<T>(m_loc_option);
    }
    
    private_evalautor(YamlOrCMD * data, const std::string & option)
      :m_loc_data(data), m_loc_option(option) {}

  private:
    YamlOrCMD * m_loc_data;
    std::string m_loc_option;
  };
  
 public:

  const YAML::Node & data() const;

  YAML::Node & data();
  
  bool is_option(const std::string & option) const;

  /*
    Parses the string option given:
    first tries to find it in CMD, then searches YAML config and
    throws exception, if neither are found.
   */
  template<typename type>
  type parse(const std::string & option) const {
    if (is_option(option)) {
      return m_config[option].as<type>();
    } else {
      throw std::runtime_error(option);
    }
  }
  
  /*
    Parses option like member above, but returns specified default in case of
    missing argument.
   */
  template<typename type>
  type parse(const std::string & option,
	     const type & default_value) const {
    if (is_option(option)) {
      return m_config[option].as<type>();
    }
    return default_value;
  }

  private_evalautor autoparse(const std::string & option);

  std::string filename(const std::string & prefix = "../new_tens/") const;

  std::ofstream to_file(const std::string & prefix = "../new_tens/",
                        bool write_descriptor = true) const;

  template<typename type>
  void set_explicitly(const std::string & name, const type & in) {
    m_config[name] = in;
  }

  /*
    Parses command line input into existing YAML node.
    Delimiter between pairs of key-value must be whitespace
    valid delimiter between key and value is always whitespace 
    and specified delimiter
   */
  void parse_cmd(char ** m_argv, int m_argc, char delimiter = ':');
  
  /*
    Simply pass argv and argc directly from main and the name of the YAML config file
   */
  YamlOrCMD(char ** argv, int argc, const std::string & config);

  YamlOrCMD(char ** argv, int argc, const YAML::Node & config);

  YamlOrCMD(char ** argv, int argc);

  YamlOrCMD(const std::string & config);

  YamlOrCMD(const YAML::Node & config);
  
  YamlOrCMD(const YamlOrCMD & other);

 private:
  YAML::Node m_config;
};

std::ostream & operator<<(std::ostream & os, const YamlOrCMD & parser);

inline const YAML::Node & cnode(const YAML::Node &n) {
    return n;
}

inline YAML::Node merge_nodes(YAML::Node a, YAML::Node b) {
  if (!b.IsMap()) {
    // If b is not a map, merge result is b, unless b is null
    return b.IsNull() ? a : b;
  }
  if (!a.IsMap()) {
    // If a is not a map, merge result is b
    return b;
  }
  if (!b.size()) {
    // If a is a map, and b is an empty map, return a
    return a;
  }
  // Create a new map 'c' with the same mappings as a, merged with b
  auto c = YAML::Node(YAML::NodeType::Map);
  for (auto n : a) {
    if (n.first.IsScalar()) {
      const std::string & key = n.first.Scalar();
      auto t = YAML::Node(cnode(b)[key]);
      if (t) {
        c[n.first] = merge_nodes(n.second, t);
        continue;
      }
    }
    c[n.first] = n.second;
  }
  // Add the mappings from 'b' not already in 'c'
  for (auto n : b) {
    if (!n.first.IsScalar() || !cnode(c)[n.first.Scalar()]) {
      c[n.first] = n.second;
    }
  }
  return c;
}

YamlOrCMD operator+(const YamlOrCMD & lhs, const YamlOrCMD & rhs);

#endif /* ARGPARSE_H */
