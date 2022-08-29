from google.protobuf.duration_pb2 import Duration
import os
from feast import Entity, Feature, FeatureView, ValueType
from feast.infra.offline_stores.file_source import FileSource

# We pass the FileSource path through the environment variable FEAST_ITEM_FEATURES_PATH
file_source_path = os.getenv("FEAST_ITEM_FEATURES_PATH")
if file_source_path is None:
    raise ValueError(
        "You must specify the path to the (temporary) feast feature "
        + "store with the environment variable FEAST_ITEM_FEATURES_PATH"
    )

item_features = FileSource(
    path=file_source_path,
    event_timestamp_column="datetime",
    created_timestamp_column="created",
)

item = Entity(
    name="item_id",
    value_type=ValueType.INT32,
    description="item id",
)

item_features_view = FeatureView(
    name="item_features",
    entities=["item_id"],
    ttl=Duration(seconds=86400 * 7),
    features=[
        Feature(name="item_category", dtype=ValueType.INT32),
        Feature(name="item_shop", dtype=ValueType.INT32),
        Feature(name="item_brand", dtype=ValueType.INT32),
    ],
    online=True,
    input=item_features,
    tags={},
)
