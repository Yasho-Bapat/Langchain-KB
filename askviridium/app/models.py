from typing import List

from pydantic.v1 import BaseModel, Field


class ChemicalInfo(BaseModel):
    name: str = Field(description="Name of the chemical")
    cas_no: str = Field(description="CAS number of the chemical")
    source: str = Field(description="Source for this piece of information, will be a hyperlink")


class ChemicalComposition(BaseModel):
    product_name: str = Field(description="Name of the product specified")
    chemicals: List[ChemicalInfo] = Field(description="List of chemicals present in the product.")
    confidence: int = Field(description="Confidence score of the result")


class MaterialInfo(BaseModel):
    """Information to extract."""
    analyzed_material: str = Field(description="Name of the material that was analyzed")
    composition: str = Field(description="Chemical composition of the material")
    analysis_method: str = Field(description="How its PFAS analysis was conducted - methods, sources")
    decision: str = Field(description="Decision of whether the material is PFAS compliant or not: PFAS (Yes/No)")
    confidence: float = Field(description="Confidence score of response.")
    primary_reason: str = Field(description="Primary reasoning of response content.")
    secondary_reason: str = Field(description="Secondary reasoning of response content.")
    evidence: List[str] = Field(description="Evidence supporting the response given.")
    health_problems: List[str] = Field(description="List of health problems that could potentially be attached to the product.")
    confidence_level: str = Field(description="Confidence level of response between low, medium and high.")
    recommendation: str = Field(description="Recommendation of what to do with the material with regards to its PFAS compliance.")
    suggestion: str = Field(description="Suggestion of what to do with the material with regards to its PFAS compliance.")
    limitations_and_uncertainties: str = Field(description="Limitations and uncertainties of material and its PFAS compliance based on the data that could be looked up.")
