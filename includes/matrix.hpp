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

    int m_row{}; 
    int m_col{};

    // bool m_is_random{};
    
    vec2d_dbl m_values;

  public:

    matrix(int row, int col, bool is_random);

    matrix *transpose();
    matrix *copy();

    auto set_val(int r, int c, double val) -> void {
      m_values.at(r).at(c) = val;
    }

    auto get_val(int r, int c) const -> double { 
      return m_values.at(r).at(c); 
    }

    auto get_values() const -> vec2d_dbl { return m_values; }

    auto get_row() const -> int { return m_row; }
    auto get_col() const -> int { return m_col; }

    auto display_result() -> void;

};

#endif // MATRIX_H