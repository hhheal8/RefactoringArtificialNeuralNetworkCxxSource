#ifndef NEURON_H
#define NEURON_H

#define TANH 1
#define RELU 2
#define SIGM 3

#include <iostream>
#include <math.h>
#include <random>
#include <vector>

class neuron {

  private:

    double m_val{}; 

    double m_activated_val{}; 
    double m_derived_val{};

    size_t m_activation_type{3};

  public:

    neuron(double val);
    neuron(double val, size_t activation_type);

    auto set_val(double val) -> void; 

    //Fast Sigmoid Function
    //f(x) = x / (1 + |x|)
    auto activate() -> void; 

    //Derivative for Fast Sigmoid Function
    //f(x) = f(x) * (1 - f(x))
    auto derive() -> void; 

    auto get_val() const -> double { return m_val; } 
    auto get_activated_val() const -> double { return m_activated_val; } 
    auto get_derived_val() const -> double { return m_derived_val; } 

};

#endif // NEURON_H