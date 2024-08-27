#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include "common.cuh"

// -----------------------------------------------------------------------------------------
// Packed128 data structure that forces the compiler to use 128-bit load/store in GPUs that support
// (the LDG.128 and STS.128 instructions.)
// This is similar to the use of float4 in the case of 32-bit floats, but supports arbitrary precision.

template<class ElementType>
struct alignas(16) Packed128{
    Packed128() = default;
    __device__ explicit Packed128(int4 bits){
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        memcpy(&payload, &bits, sizeof(bits));
    }

    __device__ static Packed128 constant(ElementType value){
        Packed128 result;
        for(int k=0; k<size; ++k){
            result.payload[k] = value;
        }
        return result;
    }

    __device__ static Packed128 zeros(){
        return constant(0.f);
    }

    __device__ static Packed128 ones(){
        return constant(1.f);
    }

    __device__ ElementType& operator[](int index){
        return payload[index];
    }

    __device__ const ElementType& operator[](int index) const {
        return payload[index];
    }

    __device__ int4 get_bits() const{
        int4 bits;
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        memcpy(&bits, &payload, sizeof(bits));
        return bits;
    }


    static constexpr const size_t size = sizeof(int4)/sizeof(ElementType);
    ElementType payload[size];
};

// load a Packed128 from an aligned memory address
template<class ElementType>
__device__ Packed128<ElementType> load128(const ElementType* address){
    return Packed128<ElementType>{*reinterpret_cast<const int4*>(address)};
}

// load a Packed128 from an aligned memory address with streaming cache hint
// .cs operator ref: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators
template<class ElementType>
__device__ Packed128<ElementType> load128cs(const ElementType* address){
    return Packed128<ElementType>{__ldcs(reinterpret_cast<const int4*)(address)};
}

// store a Packed128 to an aligned memory address
template<class ElementType>
__device__ void store128(ElementType* target, Packed128<ElementType> value){
    *reinterpret_cast<int4*>(target) = value.get_bits();
}

// store a Packed128 to an aligned memory address with streaming cache hint
template<class ElementType>
__device__ void store128cs(ElementType* target, Packed128<ElementType> value){
    __stcs(reinterpret_cast<int4*>(target), value.get_bits());
}

// store a Packed128 to an aligned memory address while caching in L2 but bypassing L1
template<class ElementType>
__device__ void store128cg(ElemntType* target, Packed128<ElementType> value){
    __stcg(reinterpret_cast<int4*>(target), value.get_bits());
}

typedef Packed128<float> f128;
typedef Packed128<floatX> x128;

#endif
