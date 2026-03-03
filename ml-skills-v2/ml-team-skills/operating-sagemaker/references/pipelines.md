# SageMaker Pipelines

## Contents
- Pipeline structure for forecasting
- Step types
- Conditional execution
- Pipeline parameters

## Forecasting Pipeline Structure

A typical time series forecasting pipeline:

```
DataQuality → Preprocessing → Training → Evaluation → Conditional → Register/Deploy
                                              ↓ (if metrics fail)
                                         Alert + Stop
```

## Minimal Pipeline Example

```python
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.parameters import ParameterString, ParameterFloat

# Parameters — configurable at runtime
input_data = ParameterString(name="InputData", default_value="s3://bucket/data/")
mase_threshold = ParameterFloat(name="MASEThreshold", default_value=1.2)

# Step 1: Data validation
quality_step = ProcessingStep(
    name="DataQuality",
    processor=sklearn_processor,
    code="scripts/validate_data.py",
    inputs=[ProcessingInput(source=input_data, destination="/opt/ml/input")],
    outputs=[ProcessingOutput(source="/opt/ml/output", destination="s3://bucket/quality/")],
)

# Step 2: Training
train_step = TrainingStep(
    name="TrainModel",
    estimator=estimator,
    inputs={"training": training_input},
)
train_step.add_depends_on([quality_step])

# Step 3: Evaluation
eval_step = ProcessingStep(
    name="EvaluateModel",
    processor=sklearn_processor,
    code="scripts/evaluate.py",
)
eval_step.add_depends_on([train_step])

# Step 4: Conditional registration
cond_step = ConditionStep(
    name="CheckMetrics",
    conditions=[ConditionLessThanOrEqualTo(
        left=JsonGet(step_name="EvaluateModel", property_file="metrics", json_path="mase"),
        right=mase_threshold,
    )],
    if_steps=[register_step],
    else_steps=[alert_step],
)

pipeline = Pipeline(
    name="ts-forecasting-pipeline",
    parameters=[input_data, mase_threshold],
    steps=[quality_step, train_step, eval_step, cond_step],
)
```
