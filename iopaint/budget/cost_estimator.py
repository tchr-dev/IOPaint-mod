"""Cost estimation for API operations."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Literal


@dataclass
class PricingRule:
    """Pricing rule for a specific model.

    Attributes:
        base_cost: Base cost per image in USD
        size_multiplier: Multipliers for different image sizes
        quality_multiplier: Multipliers for different quality levels
    """

    base_cost: float
    size_multiplier: Dict[str, float] = field(default_factory=dict)
    quality_multiplier: Dict[str, float] = field(default_factory=dict)


# Default pricing rules based on OpenAI pricing (as of 2024)
# These are approximations and should be updated as pricing changes
DEFAULT_PRICING: Dict[str, PricingRule] = {
    # OpenAI gpt-image-1 (DALL-E 3)
    "gpt-image-1": PricingRule(
        base_cost=0.04,
        size_multiplier={
            "256x256": 0.25,
            "512x512": 0.5,
            "1024x1024": 1.0,
            "1792x1024": 2.0,
            "1024x1792": 2.0,
        },
        quality_multiplier={
            "standard": 1.0,
            "hd": 2.0,
            "low": 0.5,
        },
    ),
    # DALL-E 3
    "dall-e-3": PricingRule(
        base_cost=0.04,
        size_multiplier={
            "1024x1024": 1.0,
            "1792x1024": 2.0,
            "1024x1792": 2.0,
        },
        quality_multiplier={
            "standard": 1.0,
            "hd": 2.0,
        },
    ),
    # DALL-E 2 (cheaper)
    "dall-e-2": PricingRule(
        base_cost=0.02,
        size_multiplier={
            "256x256": 0.8,
            "512x512": 0.9,
            "1024x1024": 1.0,
        },
        quality_multiplier={
            "standard": 1.0,
        },
    ),
    # Refine/chat operations (much cheaper)
    "gpt-4o-mini": PricingRule(
        base_cost=0.0001,  # ~$0.0001 per call
        size_multiplier={},
        quality_multiplier={},
    ),
    "gpt-4o": PricingRule(
        base_cost=0.001,  # ~$0.001 per call
        size_multiplier={},
        quality_multiplier={},
    ),
    # Default fallback for unknown models
    "default": PricingRule(
        base_cost=0.05,
        size_multiplier={
            "default": 1.0,
            "256x256": 0.25,
            "512x512": 0.5,
            "1024x1024": 1.0,
            "1792x1024": 2.0,
            "1024x1792": 2.0,
        },
        quality_multiplier={
            "default": 1.0,
            "standard": 1.0,
            "hd": 2.0,
            "low": 0.5,
        },
    ),
}

# Cost tier thresholds
COST_TIER_LOW = 0.02  # $0.02 and below = low
COST_TIER_HIGH = 0.10  # $0.10 and above = high


class CostEstimator:
    """Estimate costs for API operations.

    Usage:
        estimator = CostEstimator()
        cost = estimator.estimate("gpt-image-1", "1024x1024", "standard")
        tier = estimator.get_tier(cost)
    """

    def __init__(
        self,
        pricing: Optional[Dict[str, PricingRule]] = None,
        tier_low: float = COST_TIER_LOW,
        tier_high: float = COST_TIER_HIGH,
    ):
        self.pricing = pricing or DEFAULT_PRICING
        self.tier_low = tier_low
        self.tier_high = tier_high

    def estimate(
        self,
        model: str,
        size: str = "1024x1024",
        quality: str = "standard",
        n: int = 1,
        operation: str = "generate",
    ) -> float:
        """Estimate cost for an operation.

        Args:
            model: Model name (e.g., "gpt-image-1", "dall-e-3")
            size: Image size (e.g., "1024x1024", "1792x1024")
            quality: Quality level (e.g., "standard", "hd")
            n: Number of images to generate
            operation: Operation type ("generate", "edit", "variation", "refine")

        Returns:
            Estimated cost in USD
        """
        # Get pricing rule for model or use default
        rule = self.pricing.get(model.lower(), self.pricing.get("default"))
        if rule is None:
            rule = DEFAULT_PRICING["default"]

        # Get multipliers
        size_mult = rule.size_multiplier.get(size, 1.0)
        if not size_mult and rule.size_multiplier:
            # Try to find closest match or use default
            size_mult = rule.size_multiplier.get("default", 1.0)

        quality_mult = rule.quality_multiplier.get(quality.lower(), 1.0)
        if not quality_mult and rule.quality_multiplier:
            quality_mult = rule.quality_multiplier.get("default", 1.0)

        # Refine operations are much cheaper
        if operation == "refine":
            # Use refine model pricing if available
            refine_rule = self.pricing.get("gpt-4o-mini", self.pricing.get("default"))
            if refine_rule:
                return refine_rule.base_cost * n

        # Calculate total cost
        cost = rule.base_cost * size_mult * quality_mult * n
        return round(cost, 4)

    def get_tier(
        self, cost: float
    ) -> Literal["low", "medium", "high"]:
        """Get cost tier for a given cost.

        Args:
            cost: Cost in USD

        Returns:
            "low", "medium", or "high"
        """
        if cost <= self.tier_low:
            return "low"
        elif cost >= self.tier_high:
            return "high"
        else:
            return "medium"

    def estimate_with_tier(
        self,
        model: str,
        size: str = "1024x1024",
        quality: str = "standard",
        n: int = 1,
        operation: str = "generate",
    ) -> tuple:
        """Estimate cost and return with tier.

        Returns:
            Tuple of (cost_usd, tier, warning_message)
        """
        cost = self.estimate(model, size, quality, n, operation)
        tier = self.get_tier(cost)

        warning = None
        if tier == "high":
            warning = f"This operation costs ${cost:.2f}. Consider using Draft preset for lower cost."

        return cost, tier, warning

    def get_preset_costs(
        self, model: str, operation: str = "generate"
    ) -> Dict[str, float]:
        """Get estimated costs for standard presets.

        Args:
            model: Model name
            operation: Operation type

        Returns:
            Dict with preset names and their estimated costs
        """
        return {
            "draft": self.estimate(model, "512x512", "standard", 1, operation),
            "standard": self.estimate(model, "1024x1024", "standard", 1, operation),
            "final": self.estimate(model, "1024x1024", "hd", 1, operation),
            "high_res": self.estimate(model, "1792x1024", "hd", 1, operation),
        }
