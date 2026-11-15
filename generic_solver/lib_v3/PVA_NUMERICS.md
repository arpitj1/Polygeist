# PVA parameter configuration and numerical contracts

Application recognition remains the ordinary imported-reference Linalg/region
match. Neither API configuration nor an error budget can authorize a match.
The original application must first match the imported reference under the
selected scalar arithmetic model.

## Compile-time API configuration

The typed C adapter accepts an optional `configuration` with scalar inputs from
the reference's `defines`, ordered scalar expressions, and required checks.
Configured arguments name a checked value and its exact C type. The shared
evaluator supports bounded arithmetic, comparisons, `sqrt`, `log`, `exp`,
`floor`, `ceil`, `abs`, and explicit `f32` rounding. It does not evaluate Python
code, inspect an application name, recognize an operation, or synthesize a
pointer. Unknown inputs, nonfinite values, failed checks and unchecked FP32
conversions reject the ABI candidate.

The Gaussian reference's horizontal and vertical shifts describe its actual
coefficients. The manifest recovers sigma from their ratios, rounds sigma to
FP32, models quantization, and checks all coefficient classes and the source
rounding offset. For the imported binomial reference, both recovered sigmas are
0.8493217825889587, with Q8 coefficients `[16,32,16;32,64,32;16,32,16]`.
The generated adapter passes those hexadecimal FP32 constants to the general
Gaussian runtime entry. No fixed sigma remains in that runtime wrapper.

This currently derives parameters for imported, compile-time specializations.
It is not a search over arbitrary application coefficient arrays or runtime
sigma values. A new reference specialization still needs an import and the
ordinary equivalence check. Library equations belong in the manifest; no
Gaussian recognizer was added to the generic matcher.

The public PVA documentation describes coefficient truncation and center
adjustment, but not every floating-point intermediate. The manifest's FP32
model is explicit and the actual backend is tested independently. The model
check alone is not a proof of a closed vendor implementation.

## Quantized backends

`target.numerical_contract` describes backend differences separately from
reference `deviations`. A bounded contract describes a normalized linear
filter with integer coefficients, rational scales, a bounded input interval,
and explicit rounding-error bounds. The shared implementation uses rational
interval arithmetic to bound the difference. The bound is conditional on the
documented coefficient, rounding, neighborhood and overflow assumptions.

The U8 OpenCV box reference computes the same rounded average as the source
fixture. PVA's documented Q8 coefficients are
`[29,28,29;28,29,28;29,28,28]/256`. Over inputs in `[0,255]`, the modeled real
coefficient error is at most `425/192`; including both rounding errors gives
`617/192`. Integer outputs therefore differ by at most 3. This bound is computed
from the contract, not fitted to measured errors. A unit test checks the source
normalization for every possible nine-pixel sum and tests all 512 extreme
neighborhoods, including an observed difference of 3.

Approximate backends are disabled by default, even with relaxed scalar algebra.
Enable a particular imported specification explicitly:

```sh
export POLYGEIST_BACKEND_ERROR_BUDGETS='{"opencv_box3x3_u8":3}'
```

The budget must cover the calculated bound. Missing, negative, nonfinite or
too-small budgets cannot authorize a replacement. The match report records the
resolved parameters, contract assumptions and accepted budget. Build-time proof
verification checks the current configuration, ABI, reference and contract;
changed policies or parameters invalidate saved evidence. Generated lowering
only marshals the selected API and checked parameters.

Bilateral's square/circular neighborhood difference and histogram equalization's
constant-image behavior are not modeled by this linear quantization contract.
Their public reference variants declare these differences and use a separate
`recorded_approximation` contract. Typed integer Box and Gaussian reference
variants use the same policy where no numerical error bound has been established.
These variants require relaxed reference semantics and explicit per-specification
acceptance, for example:

```sh
export POLYGEIST_NUMERIC_MODEL=relaxed
export POLYGEIST_BACKEND_APPROXIMATIONS='["opencv_bilateral3x3_square_u8"]'
```

This policy does not assert numerical equivalence to the vendor implementation.
The application must still match the complete imported reference. Execution
records output differences separately from input preservation, guard regions,
border handling and the selected vendor call. A numerical error budget cannot
enable an unbounded approximation. Saved proof verification includes the exact
list of accepted specifications.

Public sources:

- [Gaussian coefficient generation](https://docs.nvidia.com/pva/solutions/0.3.0/impl/operator/gaussianfilter.html)
- [Box coefficient generation](https://docs.nvidia.com/pva/solutions/0.4.0/impl/operator/boxfilter.html)

No external PVA SDK implementation or headers are distributed here.
