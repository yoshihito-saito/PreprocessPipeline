"""Bounded matching primitives for separately sorted recording epochs."""

from .sparse import iter_spatial_pairs, score_template_pairs, template_centers
from .rolling import RollingTemplate, TemplateSnapshot

__all__ = ["iter_spatial_pairs", "score_template_pairs", "template_centers",
           "RollingTemplate", "TemplateSnapshot"]
