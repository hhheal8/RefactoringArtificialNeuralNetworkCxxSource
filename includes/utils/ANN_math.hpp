#ifndef ANN_MATH_H
#define ANN_MATH_H

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <assert.h>

#include "matrix.hpp"

class matrix;

namespace utils {

  class ANN_math {
    
    public:

      static auto multiply_matrix(matrix *a, matrix *b, matrix *c) -> void;

  };

}

#endif // ANN_MATH_H