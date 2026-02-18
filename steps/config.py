# from zenml.steps import BaseParameters

# from zenml import BaseParameters

from pydantic import BaseModel

class ModelNameConfig(BaseModel):
    """Model configs"""

    model_name: str = "LinearRegression"

