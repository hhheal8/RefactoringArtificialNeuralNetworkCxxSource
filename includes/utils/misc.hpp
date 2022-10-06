#ifndef MISC_H
#define MISC_H

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <assert.h>

#include "var_type_alias.hpp"

namespace utils {

  class misc {

    public:
    
      static vec2d_dbl fetch_data(std::string path);

  };

}

#endif // MISC_H