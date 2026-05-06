# Model Card: BBO Capstone Optimisation Model

## Model Overview
This project uses a Gaussian Process (GP) as a surrogate model for black-box optimisation. The GP is combined with acquisition functions such as Upper Confidence Bound (UCB) and Expected Improvement (EI) to guide the search for optimal inputs.

## Model Details
- Model type: Gaussian Process Regression
- Implementation: Custom implementation using NumPy
- Acquisition functions:
  - Upper Confidence Bound (UCB)
  - Expected Improvement (EI)

## Training Process
The model was updated iteratively as new data points were collected each week. Different strategies were applied across phases:
- Early phase: exploration using UCB with higher kappa
- Middle phase: mixed exploration and exploitation
- Final phase: strong local exploitation using EI

## Key Strategies
- Function-specific optimisation rather than a global approach
- Local sampling around best-performing points
- Reduction of search space once patterns were identified
- Filtering of observations (Top-K approach) to reduce noise

## Performance
- Strong performance on structured functions (e.g. F5, F7)
- Less consistent results on noisy or unstructured functions (e.g. F1, F6)
- Significant improvements achieved through boundary exploitation and local refinement

## Limitations
- Sensitive to small data size
- Risk of overfitting due to limited observations
- Performance depends heavily on function structure
- Acquisition function parameters require tuning

## Ethical Considerations
- No personal or sensitive data is used
- The model operates in a simulated optimisation environment

## Future Improvements
- Earlier identification of function behaviour (structured vs unstructured)
- Adaptive tuning of acquisition parameters
- Incorporation of multi-point sampling strategies
- Better handling of noisy functions
