//
// Created by Grady Schofield on 9/29/23.
//

#ifndef STREAMING_LLAMA_BF16_H
#define STREAMING_LLAMA_BF16_H

#include<cstdint>

class Bf16 {
    uint16_t x;
public:
    Bf16();
    Bf16(float x);
    float toFloat() const;
    Bf16 operator+(Bf16 const & t);
    Bf16 operator-(Bf16 const & t);
    Bf16 operator*(Bf16 const & t);
    Bf16 operator/(int i);
    Bf16 operator/(Bf16 const & t);
    void operator+=(Bf16 const & t);
    void operator*=(Bf16 const & t);
    Bf16 operator-();
    operator float();
};

Bf16 operator*(int i, Bf16 const & t);


#endif //STREAMING_LLAMA_BF16_H
