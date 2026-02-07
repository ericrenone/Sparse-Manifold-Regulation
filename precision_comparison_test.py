"""
SCIENTIFIC COMPARISON: Q16.16 Fixed-Point vs Full Floating-Point Precision
============================================================================
Rigorous evaluation of numerical accuracy, error propagation, and computational
characteristics across identical mathematical operations.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple
import math
import statistics

# =============================================================================
# Q16.16 FIXED-POINT IMPLEMENTATION
# =============================================================================

class Q16_16:
    """Fixed-point arithmetic with 16.16 bit allocation"""
    SHIFT = 16
    SCALE = 1 << SHIFT  # 65536
    
    @staticmethod
    def to_fixed(f: float) -> int:
        return int(f * Q16_16.SCALE)
    
    @staticmethod
    def to_float(v: int) -> float:
        return v / Q16_16.SCALE
    
    @staticmethod
    def mul(a: int, b: int) -> int:
        return (a * b) >> Q16_16.SHIFT
    
    @staticmethod
    def div(a: int, b: int) -> int:
        if b == 0: return 0
        return (a << Q16_16.SHIFT) // b

# =============================================================================
# TEST DATA STRUCTURES
# =============================================================================

@dataclass
class TestCase:
    """Single test case for comparison"""
    name: str
    value_a: float
    value_b: float
    operation: str

@dataclass
class ComparisonResult:
    """Results from comparing fixed vs float"""
    test_name: str
    operation: str
    
    # Input values
    input_a: float
    input_b: float
    
    # Fixed-point path
    fixed_result_raw: int
    fixed_result_float: float
    
    # Floating-point path
    float_result: float
    
    # Error metrics
    absolute_error: float
    relative_error: float
    ulp_error: float

@dataclass
class StatisticalSummary:
    """Aggregate statistics across all tests"""
    total_tests: int
    mean_abs_error: float
    median_abs_error: float
    max_abs_error: float
    min_abs_error: float
    std_abs_error: float
    mean_rel_error: float
    max_rel_error: float

# =============================================================================
# CORE COMPUTATION ENGINE
# =============================================================================

class PrecisionComparator:
    """Dual-path computation engine for rigorous comparison"""
    
    def __init__(self):
        self.results: List[ComparisonResult] = []
        
    def _compute_ulp_error(self, expected: float, actual: float) -> float:
        """Calculate error in Units in Last Place (ULP)"""
        if expected == 0 and actual == 0:
            return 0.0
        if expected == 0:
            return float('inf')
        # Simplified ULP calculation
        return abs((actual - expected) / (abs(expected) * 2.220446049250313e-16))
    
    def run_test(self, test: TestCase) -> ComparisonResult:
        """Execute single test in both fixed and floating-point"""
        
        # ===== FIXED-POINT PATH =====
        a_fixed = Q16_16.to_fixed(test.value_a)
        b_fixed = Q16_16.to_fixed(test.value_b)
        
        if test.operation == "multiply":
            result_fixed = Q16_16.mul(a_fixed, b_fixed)
        elif test.operation == "divide":
            result_fixed = Q16_16.div(a_fixed, b_fixed)
        elif test.operation == "add":
            result_fixed = a_fixed + b_fixed
        elif test.operation == "subtract":
            result_fixed = a_fixed - b_fixed
        else:
            raise ValueError(f"Unknown operation: {test.operation}")
        
        result_fixed_as_float = Q16_16.to_float(result_fixed)
        
        # ===== FLOATING-POINT PATH =====
        a_float = float(test.value_a)
        b_float = float(test.value_b)
        
        if test.operation == "multiply":
            result_float = a_float * b_float
        elif test.operation == "divide":
            result_float = a_float / b_float if b_float != 0 else 0.0
        elif test.operation == "add":
            result_float = a_float + b_float
        elif test.operation == "subtract":
            result_float = a_float - b_float
        else:
            raise ValueError(f"Unknown operation: {test.operation}")
        
        # ===== ERROR ANALYSIS =====
        abs_error = abs(result_float - result_fixed_as_float)
        rel_error = abs_error / abs(result_float) if result_float != 0 else 0.0
        ulp_error = self._compute_ulp_error(result_float, result_fixed_as_float)
        
        result = ComparisonResult(
            test_name=test.name,
            operation=test.operation,
            input_a=test.value_a,
            input_b=test.value_b,
            fixed_result_raw=result_fixed,
            fixed_result_float=result_fixed_as_float,
            float_result=result_float,
            absolute_error=abs_error,
            relative_error=rel_error,
            ulp_error=ulp_error
        )
        
        self.results.append(result)
        return result
    
    def compute_statistics(self) -> StatisticalSummary:
        """Compute aggregate statistics across all tests"""
        abs_errors = [r.absolute_error for r in self.results]
        rel_errors = [r.relative_error for r in self.results if not math.isinf(r.relative_error)]
        
        return StatisticalSummary(
            total_tests=len(self.results),
            mean_abs_error=statistics.mean(abs_errors),
            median_abs_error=statistics.median(abs_errors),
            max_abs_error=max(abs_errors),
            min_abs_error=min(abs_errors),
            std_abs_error=statistics.stdev(abs_errors) if len(abs_errors) > 1 else 0,
            mean_rel_error=statistics.mean(rel_errors) if rel_errors else 0,
            max_rel_error=max(rel_errors) if rel_errors else 0
        )

# =============================================================================
# ML-RELEVANT TEST: NOVELTY DETECTION (FROM ORIGINAL CODE)
# =============================================================================

class NoveltyDetectionTest:
    """Test the actual novelty detection formula in both precisions"""
    
    def __init__(self):
        self.comparator = PrecisionComparator()
    
    def compute_phi_fixed(self, kl: float, fisher: float, tau: float, 
                          alpha: float = 512.0, eps: float = 1.52587890625e-5) -> float:
        """Compute Φ(x) using Q16.16 fixed-point"""
        kl_q16 = Q16_16.to_fixed(kl)
        fisher_q16 = Q16_16.to_fixed(fisher)
        tau_q16 = Q16_16.to_fixed(tau)
        alpha_q16 = Q16_16.to_fixed(alpha)
        eps_q16 = Q16_16.to_fixed(eps)
        
        # Φ(x) = (KL * Fisher) / ((τ / α) + ε)
        numerator = Q16_16.mul(kl_q16, fisher_q16)
        denominator = Q16_16.div(tau_q16, alpha_q16) + eps_q16
        phi_q16 = Q16_16.div(numerator, denominator)
        
        return Q16_16.to_float(phi_q16)
    
    def compute_phi_float(self, kl: float, fisher: float, tau: float, 
                          alpha: float = 512.0, eps: float = 1.52587890625e-5) -> float:
        """Compute Φ(x) using full floating-point"""
        numerator = kl * fisher
        denominator = (tau / alpha) + eps
        return numerator / denominator
    
    def run_novelty_tests(self) -> List[Tuple[str, float, float, float]]:
        """Test novelty calculation with realistic ML values"""
        test_cases = [
            ("Low KL, Low Fisher", 0.5, 1.0, 128.0),
            ("High KL, Low Fisher", 5.0, 1.0, 256.0),
            ("Low KL, High Fisher", 0.5, 2.5, 128.0),
            ("High KL, High Fisher", 5.0, 2.5, 512.0),
            ("Medium All", 2.5, 1.42857, 384.0),
        ]
        
        results = []
        for name, kl, fisher, tau in test_cases:
            phi_fixed = self.compute_phi_fixed(kl, fisher, tau)
            phi_float = self.compute_phi_float(kl, fisher, tau)
            error = abs(phi_float - phi_fixed)
            results.append((name, phi_fixed, phi_float, error))
        
        return results

# =============================================================================
# COMPREHENSIVE TEST SUITE
# =============================================================================

def create_test_suite() -> List[TestCase]:
    """Generate comprehensive test cases"""
    tests = []
    
    # Basic arithmetic tests
    tests.extend([
        TestCase("Small multiply", 0.5, 0.3, "multiply"),
        TestCase("Large multiply", 123.456, 78.901, "multiply"),
        TestCase("Tiny multiply", 0.0001, 0.0002, "multiply"),
        TestCase("Mixed multiply", 1000.5, 0.001, "multiply"),
    ])
    
    tests.extend([
        TestCase("Simple divide", 10.0, 2.0, "divide"),
        TestCase("Fractional divide", 1.5, 3.0, "divide"),
        TestCase("Large/Small divide", 1000.0, 0.01, "divide"),
        TestCase("Small/Large divide", 0.01, 1000.0, "divide"),
    ])
    
    tests.extend([
        TestCase("Small add", 0.1, 0.2, "add"),
        TestCase("Large add", 12345.67, 98765.43, "add"),
        TestCase("Negative add", -123.45, 67.89, "add"),
        TestCase("Mixed sign add", 100.0, -50.5, "add"),
    ])
    
    tests.extend([
        TestCase("Simple subtract", 10.0, 3.5, "subtract"),
        TestCase("Negative result", 5.0, 12.0, "subtract"),
        TestCase("Large subtract", 99999.9, 88888.8, "subtract"),
        TestCase("Tiny subtract", 0.001, 0.0005, "subtract"),
    ])
    
    # ML-relevant values
    tests.extend([
        TestCase("KL divergence scale", 2.5, 1.42857, "multiply"),
        TestCase("Attention normalize", 384.0, 512.0, "divide"),
        TestCase("Fisher information", 1.42857, 0.5, "multiply"),
    ])
    
    return tests

# =============================================================================
# MAIN EXECUTION & REPORTING
# =============================================================================

def print_separator(char="=", length=130):
    print(char * length)

def print_header(text: str):
    print_separator()
    print(f"{text:^130}")
    print_separator()

def main():
    print_header(f"SCIENTIFIC COMPARISON: Q16.16 FIXED-POINT vs FLOATING-POINT | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ===== PART 1: BASIC ARITHMETIC TESTS =====
    print("\n" + "="*130)
    print("PART 1: BASIC ARITHMETIC OPERATIONS - DETAILED RESULTS")
    print("="*130)
    print(f"{'Test Name':<25} | {'Operation':<10} | {'Fixed Result':<18} | {'Float Result':<18} | {'Abs Error':<15} | {'Rel Error %':<12}")
    print("-" * 130)
    
    comparator = PrecisionComparator()
    tests = create_test_suite()
    
    for test in tests:
        result = comparator.run_test(test)
        rel_error_pct = result.relative_error * 100
        print(f"{result.test_name:<25} | {result.operation:<10} | {result.fixed_result_float:<18.12f} | "
              f"{result.float_result:<18.12f} | {result.absolute_error:<15.2e} | {rel_error_pct:<12.6f}")
    
    # ===== PART 2: STATISTICAL SUMMARY =====
    print("\n" + "="*130)
    print("PART 2: STATISTICAL SUMMARY OF ERRORS")
    print("="*130)
    
    stats = comparator.compute_statistics()
    
    print(f"\nTotal Tests Conducted:      {stats.total_tests}")
    print(f"\n{'ABSOLUTE ERROR METRICS':-^60}")
    print(f"  Mean Absolute Error:      {stats.mean_abs_error:.2e}")
    print(f"  Median Absolute Error:    {stats.median_abs_error:.2e}")
    print(f"  Maximum Absolute Error:   {stats.max_abs_error:.2e}")
    print(f"  Minimum Absolute Error:   {stats.min_abs_error:.2e}")
    print(f"  Std Dev Absolute Error:   {stats.std_abs_error:.2e}")
    
    print(f"\n{'RELATIVE ERROR METRICS':-^60}")
    print(f"  Mean Relative Error:      {stats.mean_rel_error*100:.6f}%")
    print(f"  Maximum Relative Error:   {stats.max_rel_error*100:.6f}%")
    
    # ===== PART 3: NOVELTY DETECTION FORMULA =====
    print("\n" + "="*130)
    print("PART 3: ML NOVELTY DETECTION - Φ(x) FORMULA COMPARISON")
    print("="*130)
    print(f"{'Scenario':<25} | {'Fixed-Point Φ':<20} | {'Float-Point Φ':<20} | {'Absolute Error':<20} | {'Error %':<12}")
    print("-" * 130)
    
    novelty_test = NoveltyDetectionTest()
    novelty_results = novelty_test.run_novelty_tests()
    
    for name, phi_fixed, phi_float, error in novelty_results:
        error_pct = (error / phi_float * 100) if phi_float != 0 else 0
        print(f"{name:<25} | {phi_fixed:<20.12f} | {phi_float:<20.12f} | {error:<20.2e} | {error_pct:<12.6f}")
    
    # ===== PART 4: PRECISION CHARACTERISTICS =====
    print("\n" + "="*130)
    print("PART 4: PRECISION CHARACTERISTICS COMPARISON")
    print("="*130)
    
    print(f"\n{'Q16.16 FIXED-POINT':-^65} | {'IEEE 754 FLOATING-POINT':-^63}")
    print(f"{'Representation:':<30} 16-bit integer + 16-bit fraction  | {'Representation:':<30} Sign + Exponent + Mantissa")
    print(f"{'Total Bits:':<30} 32 bits                              | {'Total Bits:':<30} 32 bits (float) / 64 bits (double)")
    print(f"{'Resolution:':<30} 1/65536 ≈ 1.53e-5                  | {'Resolution:':<30} Variable (7-15 decimal digits)")
    print(f"{'Range:':<30} [-32768, 32767.999985]               | {'Range:':<30} [±1.18e-38, ±3.40e+38] (float)")
    print(f"{'Precision Type:':<30} Uniform across range                | {'Precision Type:':<30} Relative (decreases with magnitude)")
    print(f"{'Overflow Behavior:':<30} Wraps or saturates                   | {'Overflow Behavior:':<30} Infinity")
    print(f"{'Rounding Errors:':<30} Accumulate linearly                   | {'Rounding Errors:':<30} Accumulate relatively")
    
    # ===== PART 5: KEY FINDINGS =====
    print("\n" + "="*130)
    print("PART 5: KEY SCIENTIFIC FINDINGS")
    print("="*130)
    
    print(f"""
1. ACCURACY:
   - Mean absolute error across all tests: {stats.mean_abs_error:.2e}
   - Mean relative error: {stats.mean_rel_error*100:.6f}%
   - Q16.16 provides ~4-5 decimal places of accuracy (1/65536 resolution)
   
2. ERROR PROPAGATION:
   - Fixed-point: Errors accumulate additively with each operation
   - Floating-point: Errors accumulate multiplicatively (relative to magnitude)
   - Complex calculations (like Φ formula) show compounding in fixed-point
   
3. RANGE LIMITATIONS:
   - Fixed-point saturates at ±32,768
   - Floating-point handles much wider dynamic range
   - Critical for ML: gradient magnitudes can vary widely
   
4. DETERMINISM:
   - Fixed-point: Perfectly reproducible across platforms
   - Floating-point: Can vary slightly due to optimization flags
   
5. PERFORMANCE (Theoretical):
   - Fixed-point: Integer ALU operations (faster on some hardware)
   - Floating-point: Dedicated FPU (faster on modern CPUs/GPUs)
   
6. USE CASE RECOMMENDATIONS:
   - Fixed-point: Embedded systems, deterministic requirements, limited range
   - Floating-point: General ML/AI, wide dynamic range, modern hardware
    """)
    
    # ===== PART 6: CRITICAL DIFFERENCES TABLE =====
    print("="*130)
    print("PART 6: SIDE-BY-SIDE CRITICAL DIFFERENCES")
    print("="*130)
    
    print(f"\n{'Characteristic':<30} | {'Q16.16 Fixed-Point':<48} | {'IEEE Floating-Point':<48}")
    print("-" * 130)
    print(f"{'Smallest positive value':<30} | {1/65536:<48.12f} | {1.401298464324817e-45:<48.2e}")
    print(f"{'Largest positive value':<30} | {32767.999985:<48.6f} | {3.4028235e+38:<48.2e}")
    print(f"{'Precision uniformity':<30} | {'Constant everywhere':<48} | {'Relative to magnitude':<48}")
    print(f"{'Division by small numbers':<30} | {'Can overflow easily':<48} | {'Gracefully scales':<48}")
    print(f"{'Multiplication accuracy':<30} | {'Loses 16 bits per operation':<48} | {'Loses ~7 bits per operation':<48}")
    print(f"{'Memory bandwidth':<30} | {'32 bits':<48} | {'32 bits (float) / 64 bits (double)':<48}")
    print(f"{'Hardware support':<30} | {'Integer units':<48} | {'Dedicated FPUs on all modern CPUs':<48}")
    
    print_separator()
    print(f"{'ANALYSIS COMPLETE':^130}")
    print_separator()
    
    print(f"\nQ16.16 Resolution Demonstration:")
    print(f"  1 unit in Q16.16 = {1/65536:.16f}")
    print(f"  This means values are quantized to multiples of {1/65536:.2e}")
    print(f"\nConclusion: For ML workloads with wide dynamic ranges (like novelty detection),")
    print(f"            floating-point provides superior accuracy and range at minimal cost")
    print(f"            on modern hardware. Fixed-point excels in constrained embedded systems.")

if __name__ == "__main__":
    main()
