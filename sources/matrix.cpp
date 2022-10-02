#include <iomanip>
#include <iostream>
#include <math.h>
#include <random>
#include <vector>

#include "matrix.hpp"
#include "dealloc_all.hpp"
#include "var_type_alias.hpp"

matrix::matrix(size_t row, size_t col, bool is_random): 
m_row(row), m_col(col) {
  
  for(size_t i{}; i < m_row; ++i) {
    std::vector<double> col_values;
    for(size_t j{}; j < m_col; ++j) {
      double random = is_random == true ? gen_random_number() : 0.00;
      col_values.emplace_back(random);
    }
    m_values.emplace_back(col_values);
  }

}

matrix *matrix::transpose() {

  matrix *m = new matrix(m_col, m_row, false);

  for(size_t i{}; i < m_row; ++i) {
    for(size_t j{}; j < m_col; ++j) {
      m->set_val(j, i, get_val(i, j));
    }
  }

  // dealloc_all(m); //FIXME: 

  return m;

}

matrix *matrix::copy() {

  matrix *m = new matrix(m_row, m_col, false);

  for(size_t i{}; i < m_row; ++i) {
    for(size_t j{}; j < m_col; ++j) {
      m->set_val(i, j, get_val(i, j));
    }
  }

  // dealloc_all(m); //FIXME: 

  return m;

}

auto matrix::gen_random_number() const -> double {
  std::mt19937 gen(rand());
  std::uniform_real_distribution<> dist(-.0001, .0001);

  return dist(gen);
}

auto matrix::display_result() -> void {
  for(size_t i{}; i < m_row; ++i) {
    for(size_t j{}; j < m_col; ++j) {
      std::cout << m_values.at(i).at(j) << "\t\t";
    }
    std::cout << "\n";
  }
}