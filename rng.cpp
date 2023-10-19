#include "rng.h"
#include <cstdlib>
#include <iostream>

using namespace std;

// Using version from gcc 10.2.0, to improve reproducability
template <typename _IntType = int> class uniform_int_distribution_from_10_2_0 {
  static_assert(std::is_integral<_IntType>::value,
                "template argument must be an integral type");

public:
  /** The type of the range of the distribution. */
  typedef _IntType result_type;
  /** Parameter type. */
  struct param_type {
    typedef uniform_int_distribution_from_10_2_0<_IntType> distribution_type;

    param_type() : param_type(0) {}

    explicit param_type(_IntType __a,
                        _IntType __b = numeric_limits<_IntType>::max())
        : _M_a(__a), _M_b(__b) {
      __glibcxx_assert(_M_a <= _M_b);
    }

    result_type a() const { return _M_a; }

    result_type b() const { return _M_b; }

  private:
    _IntType _M_a;
    _IntType _M_b;
  };

public:
  /**
   * @brief Constructs a uniform distribution object.
   */
  uniform_int_distribution_from_10_2_0()
      : uniform_int_distribution_from_10_2_0(0) {}

  /**
   * @brief Constructs a uniform distribution object.
   */
  explicit uniform_int_distribution_from_10_2_0(
      _IntType __a, _IntType __b = numeric_limits<_IntType>::max())
      : _M_param(__a, __b) {}

  explicit uniform_int_distribution_from_10_2_0(const param_type &__p)
      : _M_param(__p) {}

  template <typename _UniformRandomNumberGenerator>
  result_type operator()(_UniformRandomNumberGenerator &__urng) {
    return this->operator()(__urng, _M_param);
  }

  template <typename _UniformRandomNumberGenerator>
  result_type operator()(_UniformRandomNumberGenerator &__urng,
                         const param_type &__p);

private:
  param_type _M_param;
};

template <typename _IntType>
template <typename _UniformRandomNumberGenerator>
typename uniform_int_distribution_from_10_2_0<_IntType>::result_type
uniform_int_distribution_from_10_2_0<_IntType>::operator()(
    _UniformRandomNumberGenerator &__urng, const param_type &__param) {
  typedef typename _UniformRandomNumberGenerator::result_type _Gresult_type;
  typedef typename std::make_unsigned<result_type>::type __utype;
  typedef typename std::common_type<_Gresult_type, __utype>::type __uctype;

  const __uctype __urngmin = __urng.min();
  const __uctype __urngmax = __urng.max();
  const __uctype __urngrange = __urngmax - __urngmin;
  const __uctype __urange = __uctype(__param.b()) - __uctype(__param.a());

  __uctype __ret;

  if (__urngrange > __urange) {
    // downscaling
    const __uctype __uerange = __urange + 1; // __urange can be zero
    const __uctype __scaling = __urngrange / __uerange;
    const __uctype __past = __uerange * __scaling;
    do
      __ret = __uctype(__urng()) - __urngmin;
    while (__ret >= __past);
    __ret /= __scaling;
  } else if (__urngrange < __urange) {
    // upscaling
    /*
      Note that every value in [0, urange]
      can be written uniquely as

      (urngrange + 1) * high + low

      where

      high in [0, urange / (urngrange + 1)]

      and

      low in [0, urngrange].
    */
    __uctype __tmp; // wraparound control
    do {
      const __uctype __uerngrange = __urngrange + 1;
      __tmp =
          (__uerngrange * operator()(__urng,
                                     param_type(0, __urange / __uerngrange)));
      __ret = __tmp + (__uctype(__urng()) - __urngmin);
    } while (__ret > __urange || __ret < __tmp);
  } else
    __ret = __uctype(__urng()) - __urngmin;

  return __ret + __param.a();
}

Rng::Rng(unsigned long seed)
	: m_rng(seed), m_seed(seed)
{
}

Rng::~Rng()
{
}

// Algorithm from gsl_ran_choose (https://github.com/ampl/gsl/blob/master/randist/shuffle.c)
void Rng::pickRandomIndices(size_t n, std::vector<size_t> &indices)
{
	if (indices.size() > n)
	{
		cerr << "Requested more numbers than available" << endl;
		exit(-1);
	}

	for (size_t i = 0, j = 0 ; i < n && j < indices.size(); i++)
	{
		uniform_int_distribution_from_10_2_0<> dist(0, n-i-1);
		if ((size_t)dist(m_rng) < indices.size()-j)
		{
			indices[j] = i;
			j++;
		}
	}
}
