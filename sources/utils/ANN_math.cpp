#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <assert.h>

#include "ANN_math.hpp"
#include "matrix.hpp"

auto utils::ANN_math::multiply_matrix(matrix *a, matrix *b, matrix *c) -> void {

  for(size_t i{}; i < a->get_row(); ++i) {
    for(size_t j{}; j < b->get_col(); ++j) {
      for(size_t k{}; k < b->get_row(); ++k) {
        double p = a->get_val(i, k) * b->get_val(k, j);
        double new_val = c->get_val(i, j) + p;
        c->set_val(i, j, new_val);
      }
      c->set_val(i, j, c->get_val(i, j));
    }
  }

}