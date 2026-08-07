![licence](https://img.shields.io/github/license/Zncl2222/Stochastic_UC_SGSIM)
![python](https://img.shields.io/pypi/pyversions/uc-sgsim)
[![ci](https://img.shields.io/github/actions/workflow/status/Zncl2222/uc_sgsim/github-pre-commit.yml?logo=pre-commit&label=pre-commit)](https://github.com/Zncl2222/Stochastic_UC_SGSIM/actions/workflows/github-pre-commit.yml)
[![build](https://img.shields.io/github/actions/workflow/status/Zncl2222/uc_sgsim/cmake.yml?logo=cmake&logoColor=red&label=CMake)](https://github.com/Zncl2222/Stochastic_UC_SGSIM/actions/workflows/cmake.yml)
[![pytest](https://img.shields.io/github/actions/workflow/status/Zncl2222/uc_sgsim/sonarcloud.yml?logo=pytest&label=pytest)](https://github.com/Zncl2222/Stochastic_UC_SGSIM/actions/workflows/sonarcloud.yml)
[![build](https://github.com/Zncl2222/Stochastic_UC_SGSIM/actions/workflows/codeql.yml/badge.svg)](https://github.com/Zncl2222/Stochastic_UC_SGSIM/actions/workflows/codeql.yml)
[![codecov](https://codecov.io/gh/Zncl2222/uc_sgsim/branch/main/graph/badge.svg?token=3qZt0OqDNI)](https://codecov.io/gh/Zncl2222/uc_sgsim)

<div>
 <table>
  <thead>
    <tr>
      <td colspan="5" align="center"><strong>Sonar Cloud Quality</strong></td>
    </tr>
    <tr>
      <th>Metric</th>
      <th>Python</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Quality Gate</strong></td>
      <td><a href="https://sonarcloud.io/summary/new_code?id=zncl2222_Stochastic_UC_SGSIM_py"><img src="https://sonarcloud.io/api/project_badges/measure?project=zncl2222_Stochastic_UC_SGSIM_py&metric=alert_status" alt="Quality Gate Status"></a></td>
      <td><a href="https://sonarcloud.io/summary/new_code?id=zncl2222_Stochastic_UC_SGSIM_c"><img src="https://sonarcloud.io/api/project_badges/measure?project=zncl2222_Stochastic_UC_SGSIM_c&metric=alert_status" alt="Quality Gate Status"></a></td>
    <tr>
      <td><strong>Reliability Rating</strong></td>
      <td><a href="https://sonarcloud.io/summary/new_code?id=zncl2222_Stochastic_UC_SGSIM_py"><img src="https://sonarcloud.io/api/project_badges/measure?project=zncl2222_Stochastic_UC_SGSIM_py&metric=reliability_rating" alt="Reliability Rating"></a></td>
      <td><a href="https://sonarcloud.io/summary/new_code?id=zncl2222_Stochastic_UC_SGSIM_c"><img src="https://sonarcloud.io/api/project_badges/measure?project=zncl2222_Stochastic_UC_SGSIM_c&metric=reliability_rating" alt="Reliability Rating"></a></td>
    </tr>
    <tr>
      <td><strong>Security Rating</strong></td>
      <td><a href="https://sonarcloud.io/summary/new_code?id=zncl2222_Stochastic_UC_SGSIM_py"><img src="https://sonarcloud.io/api/project_badges/measure?project=zncl2222_Stochastic_UC_SGSIM_py&metric=security_rating" alt="Security Rating"></a></td>
      <td><a href="https://sonarcloud.io/summary/new_code?id=zncl2222_Stochastic_UC_SGSIM_c"><img src="https://sonarcloud.io/api/project_badges/measure?project=zncl2222_Stochastic_UC_SGSIM_c&metric=security_rating" alt="Security Rating"></a></td>
    </tr>
    <tr>
      <td><strong>Maintainability Rating</strong></td>
      <td><a href="https://sonarcloud.io/summary/new_code?id=zncl2222_Stochastic_UC_SGSIM_py"><img src="https://sonarcloud.io/api/project_badges/measure?project=zncl2222_Stochastic_UC_SGSIM_py&metric=sqale_rating" alt="Maintainability Rating"></a></td>
      <td><a href="https://sonarcloud.io/summary/new_code?id=zncl2222_Stochastic_UC_SGSIM_c"><img src="https://sonarcloud.io/api/project_badges/measure?project=zncl2222_Stochastic_UC_SGSIM_c&metric=sqale_rating" alt="Maintainability Rating"></a></td>
    </tr>
  </tbody>
 </table>
</div>

<h3 align="center">

> __Warning__
> This project is still in the pre-dev stage, the API usuage may be subject to change

</h3>

## UnConditional Sequential Gaussian SIMulation (UCSGSIM)

<h3 align="center">An unconditional random field generation tools that are easy to use.</h3>

## Introduction to UCSGSIM
Unconditional Sequential Gaussian Simulation (SGS) samples a multi-Gaussian
random field without conditioning on observed hard data. Values simulated
earlier along the path still become conditioning data for later nodes.

Let \(A\) be the previously simulated nodes, \(K=C(A,A)\), and
\(k=C(A,x)\). For a known global mean \(m\), Simple Kriging uses

$$
\lambda = K^{-1}k,\qquad
\mu_c = m + \lambda^\mathsf{T}(z_A-m),\qquad
\sigma_c^2 = C(0)-\lambda^\mathsf{T}k.
$$

The next value is sampled from the exact conditional Gaussian distribution:

$$
Z(x)=\mu_c+\sigma_c\epsilon,\qquad \epsilon\sim\mathcal N(0,1).
$$

Repeating this operation along a path factorizes the joint Gaussian density.
With every prior node included, it is equivalent to direct multivariate-normal
sampling. A limited neighborhood is a computational approximation.

The covariance contract treats `sill` as the total point variance. With nugget
\(\tau^2\), \(C(0)=\text{sill}\), while for \(h>0\),
\(C(h)=(\text{sill}-\tau^2)\rho(h)\). See
[the scientific model and validation contract](docs/scientific-model.md) for
the model equations, numerical safeguards, tests, and current limitations.

## Installation
```bash
pip install uc-sgsim
```

## Features
* One-dimensional unconditional random-field generation using SGS
* A mathematically validated Python Simple Kriging reference implementation
* Multi-core simulation using Python multiprocessing
* A legacy C backend retained for compatibility; it is not yet scientifically
  equivalent to the Python reference

## Examples
```py
import matplotlib.pyplot as plt
import uc_sgsim as uc
from uc_sgsim.cov_model import Gaussian

if __name__ == '__main__':
    grid_size = 151
    realization_count = 10
    covariance = Gaussian(
        bandwidth_len=35,
        bandwidth_step=1,
        k_range=17.32,
        sill=1.0,
        nugget=0.0,
    )

    simulation = uc.UCSgsim(
        grid_size,
        realization_count,
        covariance,
        mean=0.0,
        max_neighbor=8,
        engine='python',
    )
    simulation.run(n_processes=1, randomseed=151)
    simulation.plot()
    plt.show()
```

The default simulation is unbounded. Supplying `min_value` or `max_value`
enables whole-realization rejection and therefore samples a bounded,
conditional distribution rather than the original Gaussian field.

<p align="center">
   <img src="https://github.com/Zncl2222/Stochastic_SGSIM/blob/main/figure/Realizations.png"  width="40%"/>
   <img src="https://github.com/Zncl2222/Stochastic_SGSIM/blob/main/figure/Mean.png"  width="40%"/>
   <img src="https://github.com/Zncl2222/Stochastic_SGSIM/blob/main/figure/Variance.png"  width="40%"/>
   <img src="https://github.com/Zncl2222/Stochastic_SGSIM/blob/main/figure/Variogram.png"  width="50%"/>
   <img src="https://github.com/Zncl2222/Stochastic_SGSIM/blob/main/figure/HIST.png"  width="40%"/>
   <img src="https://github.com/Zncl2222/Stochastic_SGSIM/blob/main/figure/CDF.png"  width="50%"/>
</p>

If you prefer to utilize pure C to execute this code, you can make modifications to the c_example.c file located in the root directory. Once you've made the necessary changes to c_example.c, you can compile and execute the code using the following commands:

On Linux
```bash
sh cmake_build.sh
```
On Windows
```bat
cmake_build.bat
```

C example file
```c
// c_example.c
# include <stdio.h>
# include <stdlib.h>

# include "./uc_sgsim/c_core/include/sgsim.h"
# include "./uc_sgsim/c_core/include/cov_model.h"
# if defined(__linux__) || defined(__unix__)
# define PAUSE printf("Press Enter key to continue..."); fgetc(stdin);//NOLINT
# elif _WIN32
# define PAUSE system("PAUSE");
# endif

int main() {
    // you can also set z_min and z_max at sgsim_t. Default value will depend on
    // sill value in cov_model_t
    sgsim_t sgsim_example = {
        .x_len = 150,
        .realization_numbers = 5,
        .randomseed = 12345,
        .kriging_method = 1,
        .if_alloc_memory = 1,  // This should be equal to 1 if you want to run by c.
    };

    // you can also set max_negibor at cov_model_t. Defualt value is 4.
    cov_model_t cov_example = {
        .bw_l = 35,
        .bw_s = 1,
        .k_range = 17.32,
        .use_cov_cache = 0,
        .sill = 1,
        .nugget = 0,
    };

    sgsim_run(&sgsim_example, &cov_example, 0);
    sgsim_t_free(&sgsim_example);
    PAUSE
    return 0;
}
```

## Future plans
* 2D unconditional randomfield generation
* GUI (pyhton)
* More covariance models
* More kriging methods (etc. Oridinary Kriging)
* Performance enhancement
* Providing more comprehensive documentation and user-friendly design improvements.

## Performance
<p align="center">
<img src="https://github.com/Zncl2222/Stochastic_SGSIM/blob/main/figure/C_Cpp_py_comparision.png"  width="70%"/>
</p>

```
Parameters:

model len = 150

number of realizations = 1000

Range scale = 17.32

Variogram model = Gaussian model

---------------------------------------------------------------------------------------

Testing platform:

CPU: AMD Ryzen 9 4900 hs

RAM: DDR4 - 3200 40GB (Dual channel 16GB)

Disk: WD SN530
```
