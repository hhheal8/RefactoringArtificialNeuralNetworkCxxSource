#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include <math.h>
#include <random>
#include <vector>

#include "dealloc_all.hpp"
#include "neuron.hpp"
#include "matrix.hpp"
#include "var_type_alias.hpp"

class neuron;
class matrix;

class layer {

  private: 

    const_int m_size{};

    using vec1d_neuron_p = std::vector<neuron*>; //REVIEW: neuron_p - neuron*

    vec1d_neuron_p m_neurons;

  public:

    layer(int size);
    layer(int size, int activation_type);

    auto set_val(int i_layer, double val) -> void;
    
    matrix *matrixify_vals();
    matrix *matrixify_activated_vals(); 
    matrix *matrixify_derived_vals(); 

    auto get_activated_vals() const -> vec1d_dbl;

    auto set_neuron(vec1d_neuron_p neurons) -> void { 
      m_neurons = neurons; 
    }

    auto get_neuron() const -> vec1d_neuron_p { return m_neurons; }

};

#endif // LAYER_H