//
// Created by Grady Schofield on 9/29/23.
//

#include "Bf16.h"

Bf16::Bf16()
    : x(0)
{
}

Bf16::Bf16(float x) {
    union {
        float x;
        uint32_t i;
    } t;
    t.x = x;
    t.i >>= 16;
    this->x = t.x;
}

float Bf16::toFloat() const {
    union {
        float x;
        uint32_t i;
    } t;
    t.i = x;
    t.i <<= 16;
    return t.x;
}

Bf16 Bf16::operator+(Bf16 const & t) {
    return Bf16(this->toFloat() + t.toFloat());
}

Bf16 Bf16::operator-(Bf16 const & t) {
    return Bf16(this->toFloat() - t.toFloat());
}

Bf16 Bf16::operator*(Bf16 const & t) {
    return Bf16(this->toFloat() * t.toFloat());
}

Bf16 Bf16::operator/(int i) {
    return Bf16(this->toFloat() / i);
}

Bf16 Bf16::operator/(Bf16 const & t) {
    return Bf16(this->toFloat() / t.toFloat());
}

void Bf16::operator+=(Bf16 const & t) {
    x = Bf16(this->toFloat() + t.toFloat()).x;
}

void Bf16::operator*=(Bf16 const & t) {
    x = Bf16(this->toFloat() * t.toFloat()).x;
}

Bf16 Bf16::operator-() {
    return Bf16(-this->toFloat());
}

Bf16::operator float() {
    return toFloat();
}

Bf16 operator*(int i, Bf16 const & t) {
    return Bf16(i * t.toFloat());
}


