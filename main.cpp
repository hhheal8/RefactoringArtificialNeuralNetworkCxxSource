#include "dealloc_all.hpp"
#include <fstream>
#include <iostream>
#include <streambuf>
#include <time.h>
#include <vector>

#include "matrix.hpp"
#include "ANN_math.hpp"

auto main() -> int {

  srand(time(nullptr));

  for(size_t i{}; i < 999999; ++i) {

    matrix *a = new matrix(10, 10, true);
    matrix *b = new matrix(10, 10, true);

    matrix *c = new matrix(a->get_col(), b->get_row(), false);

    a->transpose();
    b->transpose();
    c->transpose();

    std::cout << "Multiplying matrix at index " << i << "\n";
    utils::ANN_math::multiply_matrix(a, b, c);

    dealloc_all(a, b, c);

  }
  
}


