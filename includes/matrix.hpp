#ifndef MATRIX_H
#define MATRIX_H

#include <iomanip>
#include <iostream>
#include <math.h>
#include <random>
#include <vector>

#include "dealloc_all.hpp"
#include "var_type_alias.hpp"

class matrix {

  private:

    auto gen_random_number() const -> double;

    size_t m_row{}; 
    size_t m_col{};

    // bool m_is_random{};
    
    vec2d_dbl m_values;

  public:

    matrix(size_t row, size_t col, bool is_random);

    matrix *transpose();
    matrix *copy();

    auto set_val(size_t r, size_t c, double val) -> void {
      m_values.at(r).at(c) = val;
    }

    auto get_val(size_t r, size_t c) const -> double { 
      return m_values.at(r).at(c); 
    }

    auto get_values() const -> vec2d_dbl { return m_values; }

    auto get_row() const -> size_t { return m_row; }
    auto get_col() const -> size_t { return m_col; }

    auto display_result() -> void;

};

#endif // MATRIX_H