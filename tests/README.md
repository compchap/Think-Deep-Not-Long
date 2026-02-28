# Test Suite for DTREngine

This test suite provides comprehensive coverage for the DTREngine implementation.

## Test Organization

### TestDTREngineUnit (12 tests)
**Fast unit tests with mocking** - Tests pure logic without loading the model.

#### JSD Calculation Tests (4 tests)
- `test_jsd_identical_distributions`: Verifies JSD(p, p) ≈ 0
- `test_jsd_symmetric`: Verifies JSD(p, q) == JSD(q, p)
- `test_jsd_bounds`: Verifies 0 <= JSD <= 1
- `test_jsd_with_zeros`: Tests numerical stability with zero values

#### Late Regime & Settling Depth Tests (2 tests)
- `test_late_regime_calculation`: Verifies late_regime_start = int(rho * L)
- `test_is_deep_threshold`: Tests boundary cases for deep token classification

#### Parameter Validation Tests (2 tests)
- `test_g_parameter_bounds`: Validates settling threshold parameter
- `test_rho_parameter_bounds`: Validates depth fraction parameter

#### DTR Calculation Tests (4 tests)
- `test_dtr_calculation`: Verifies DTR = deep_tokens / total_tokens
- `test_dtr_all_shallow`: Tests DTR = 0.0 edge case
- `test_dtr_all_deep`: Tests DTR = 1.0 edge case
- `test_dtr_incremental`: Tests DTR updates correctly over time

### TestDTREngineIntegration (10 tests)
**Integration tests with real qwen.6b model** - Tests with actual model inference.

#### Model Loading Tests (3 tests)
- `test_model_loads_successfully`: Verifies model and tokenizer load from cache
- `test_model_config`: Checks num_layers, device placement, eval mode
- `test_tokenizer_config`: Validates vocab_size, eos_token_id

#### Generate Step Tests (3 tests)
- `test_generate_step_output_shape`: Verifies output tuple structure
- `test_generate_step_settling_depth`: Checks c_t is in valid range [1, L]
- `test_hidden_states_structure`: Verifies hidden_states has length L+1

#### Generation & DTR Tests (2 tests)
- `test_generation_and_dtr_bounds`: Tests 0 <= DTR <= 1 during generation
- `test_mechanistic_distinction`: Verifies diverse token classifications

#### Edge Case Tests (2 tests)
- `test_single_token_generation`: Tests max_tokens=1
- `test_eos_termination`: Verifies EOS token handling

## Running Tests

### Run all tests
```bash
uv run python -m unittest tests.test_dtr_engine -v
```

### Run unit tests only (fast)
```bash
uv run python -m unittest tests.test_dtr_engine.TestDTREngineUnit -v
```

### Run integration tests only
```bash
uv run python -m unittest tests.test_dtr_engine.TestDTREngineIntegration -v
```

### Run specific test
```bash
uv run python -m unittest tests.test_dtr_engine.TestDTREngineUnit.test_jsd_identical_distributions -v
```

## Test Results

### Unit Tests
✅ All 12 unit tests pass in ~0.01s

### Integration Tests
⚠️ Integration tests require:
- Model downloaded to `models/qwen/`
- Network access or proper HuggingFace cache setup
- `protobuf` library installed

If the model cannot be loaded, integration tests will be gracefully skipped with a warning message.

## Dependencies

The test suite uses:
- `unittest` (Python standard library)
- `torch` (for tensor operations)
- `unittest.mock` (for mocking in unit tests)

## Test Coverage

The test suite covers:
- ✅ JSD calculation correctness and numerical stability
- ✅ Late regime and settling depth logic
- ✅ DTR calculation and bounds
- ✅ Parameter validation
- ✅ Model loading and configuration
- ✅ Token generation mechanics
- ✅ Hidden states structure
- ✅ Edge cases (EOS, single token, etc.)

## Notes

- Unit tests are fast (<0.01s) and require no model
- Integration tests are slower (~10-30s) and require the qwen.6b model
- All tests use the qwen.6b model (Qwen3-0.6B) for fastest integration testing
- Tests gracefully skip if model loading fails
