"""
Generator configuration and input types.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Union
from enum import Enum

from paper2slides.summary import OriginalElements, PaperContent, GeneralContent


class OutputType(str, Enum):
    """Output type for generation."""
    POSTER = "poster"
    SLIDES = "slides"


class PosterDensity(str, Enum):
    """Content density level for poster."""
    SPARSE = "sparse"   
    MEDIUM = "medium"   
    DENSE = "dense"     


class SlidesLength(str, Enum):
    """Page count level for slides."""
    SHORT = "short"      # 5-8 pages
    MEDIUM = "medium"    # 8-12 pages
    LONG = "long"        # 12-15 pages


class StyleType(str, Enum):
    """Predefined style types."""
    ACADEMIC = "academic"
    DORAEMON = "doraemon"
    CUSTOM = "custom"


DEFAULT_PRESENTATION_PROFILE = "consulting_exec_cn"
PROFILE_PROMPT_FILES: Dict[str, Dict[str, str]] = {
    DEFAULT_PRESENTATION_PROFILE: {
        "planning": "general_slides_consulting_profile.md",
        "visual": "general_slides_consulting_visual_profile.md",
    }
}


def normalize_profile(profile: Optional[str]) -> str:
    """Normalize profile name and fall back to the consulting default."""
    return str(profile or "").strip() or DEFAULT_PRESENTATION_PROFILE


def resolve_profile_prompt(profile: Optional[str], prompt_kind: str) -> str:
    """Resolve a built-in prompt file for the given profile and stage."""
    prompt_file = PROFILE_PROMPT_FILES.get(normalize_profile(profile), {}).get(prompt_kind)
    if not prompt_file:
        return ""
    prompt_path = Path(__file__).resolve().parent.parent / "prompts" / prompt_file
    if not prompt_path.exists():
        return ""
    return prompt_path.read_text(encoding="utf-8").strip()


# Page count ranges for each slides length
SLIDES_PAGE_RANGES: Dict[str, tuple[int, int]] = {
    "short": (5, 8),
    "medium": (8, 12),
    "long": (12, 15),
}


@dataclass
class GenerationConfig:
    """
    User configuration for generation.
    
    Attributes:
        output_type: Type of output (poster or slides)
        poster_density: Content density for poster (sparse/medium/dense)
        slides_length: Page count level for slides (short/medium/long)
        style: Style type (academic/doraemon/custom)
        custom_style: User's custom style description (used when style=custom)
    """
    output_type: OutputType = OutputType.POSTER
    
    # Poster specific
    poster_density: PosterDensity = PosterDensity.MEDIUM
    
    # Slides specific
    slides_length: SlidesLength = SlidesLength.MEDIUM
    
    # Style
    style: StyleType = StyleType.ACADEMIC
    custom_style: Optional[str] = None
    output_language: str = "zh-CN"
    profile: str = DEFAULT_PRESENTATION_PROFILE
    
    def get_page_range(self) -> tuple[int, int]:
        """Get page count range for slides."""
        return SLIDES_PAGE_RANGES.get(self.slides_length.value, (8, 12))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_type": self.output_type.value,
            "poster_density": self.poster_density.value,
            "slides_length": self.slides_length.value,
            "style": self.style.value,
            "custom_style": self.custom_style,
            "output_language": self.output_language,
            "profile": normalize_profile(self.profile),
        }


@dataclass
class GenerationInput:
    """
    Complete input for generation.
    
    Attributes:
        config: User generation config
        content: PaperContent or GeneralContent from summary module
        origin: Original tables and figures from source_extractor
    """
    config: GenerationConfig
    content: Union[PaperContent, GeneralContent]
    origin: OriginalElements
    
    def is_paper(self) -> bool:
        """Check if content is from a paper document."""
        return isinstance(self.content, PaperContent)
    
    def get_summary_text(self) -> str:
        """Get the full summary text."""
        if isinstance(self.content, PaperContent):
            return self.content.to_summary()
        else:
            return self.content.content
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "is_paper": self.is_paper(),
            "summary": self.get_summary_text(),
            "tables": self.origin.get_table_info(),
            "figures": self.origin.get_figure_info(),
        }
