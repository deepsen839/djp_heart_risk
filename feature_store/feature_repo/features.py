from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

synthetic_person = Entity(
    name="synthetic_person_id",
    join_keys=["synthetic_person_id"],
    description="Synthetic entity identifier; not linked to any real person.",
)

heart_risk_source = FileSource(
    name="synthetic_heart_risk_source",
    path="/app/feature_store/data/heart_risk_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

heart_risk_features = FeatureView(
    name="synthetic_heart_risk_features",
    entities=[synthetic_person],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="age", dtype=Int64),
        Field(name="systolic_bp", dtype=Float32),
        Field(name="cholesterol", dtype=Float32),
        Field(name="resting_heart_rate", dtype=Float32),
        Field(name="activity_minutes_7d", dtype=Float32),
        Field(name="smoking_flag", dtype=Int64),
        Field(name="bmi", dtype=Float32),
    ],
    online=True,
    source=heart_risk_source,
    tags={"data_origin": "synthetic", "use": "feature_store_demo_only"},
)
