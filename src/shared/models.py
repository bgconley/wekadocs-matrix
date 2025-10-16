from pydantic import BaseModel, ConfigDict


class WekaBaseModel(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=(),  # allow fields like model_name
        arbitrary_types_allowed=True,
    )
