"""Budget-aware wrapper for OpenAI-compatible client."""

from typing import Optional, Dict, Any
from pathlib import Path

from .config import BudgetConfig
from .storage import BudgetStorage
from .guard import BudgetGuard
from .dedupe import DedupeCache, calculate_fingerprint, hash_image_bytes
from .rate_limiter import RateLimiter
from .cost_estimator import CostEstimator
from .errors import BudgetExceededError, RateLimitedError, DuplicateRequestError
from .models import LedgerEntry


class BudgetAwareOpenAIClient:
    """Wrapper that adds budget safety features to OpenAI-compatible client.

    Intercepts all expensive API calls to:
    1. Check dedupe cache (return cached result if available)
    2. Check rate limit (prevent rapid-fire requests)
    3. Estimate cost and check budget
    4. Execute request
    5. Record operation and cache result

    Usage:
        from iopaint.openai_compat.client import OpenAICompatClient
        from iopaint.budget import BudgetAwareOpenAIClient, BudgetConfig, BudgetStorage

        config = BudgetConfig()
        storage = BudgetStorage(config)
        client = OpenAICompatClient(openai_config)

        budget_client = BudgetAwareOpenAIClient(
            client=client,
            config=config,
            storage=storage,
        )

        # Use budget_client instead of client for API calls
        result = budget_client.generate_image(request, session_id="abc123")
    """

    def __init__(
        self,
        client,  # OpenAICompatClient - not typed to avoid circular import
        config: BudgetConfig,
        storage: BudgetStorage,
        cost_estimator: Optional[CostEstimator] = None,
    ):
        self._client = client
        self.config = config
        self.storage = storage

        # Initialize components
        self.guard = BudgetGuard(config, storage)
        self.dedupe = DedupeCache(config, storage)
        self.rate_limiter = RateLimiter(config, storage)
        self.cost_estimator = cost_estimator or CostEstimator()

    def generate_image(
        self,
        request,  # GenerateImageRequest
        session_id: str,
        skip_dedupe: bool = False,
    ) -> bytes:
        """Generate image with budget safety checks.

        Args:
            request: GenerateImageRequest with prompt, size, quality, etc.
            session_id: Session identifier for budget tracking
            skip_dedupe: If True, skip dedupe cache lookup

        Returns:
            Image bytes

        Raises:
            BudgetExceededError: If budget cap is exceeded
            RateLimitedError: If rate limit is hit
            DuplicateRequestError: If duplicate request with cached result (optional)
        """
        model = request.model or self._client.config.model
        operation = "generate"

        # 1. Calculate fingerprint for dedupe
        params = {
            "size": request.size,
            "quality": request.quality,
            "style": getattr(request, "style", None),
            "n": request.n,
        }
        fingerprint = calculate_fingerprint(
            model=model,
            action=operation,
            prompt=request.prompt,
            negative_prompt=getattr(request, "negative_prompt", ""),
            params=params,
        )

        # 2. Check dedupe cache
        if not skip_dedupe and self.dedupe.enabled:
            cached = self.dedupe.get_image_bytes(fingerprint)
            if cached:
                return cached

        # 3. Check rate limit
        self.rate_limiter.check_or_raise(session_id)

        # 4. Estimate cost and check budget
        estimated_cost = self.cost_estimator.estimate(
            model=model,
            size=request.size,
            quality=request.quality,
            n=request.n,
            operation=operation,
        )
        self.guard.ensure_allowed_or_raise(operation, estimated_cost, session_id)

        # 5. Record operation start
        entry_id = self.guard.record_operation_start(
            operation=operation,
            model=model,
            estimated_cost=estimated_cost,
            session_id=session_id,
            fingerprint=fingerprint,
        )

        try:
            # 6. Execute request
            result = self._client.generate_image(request)

            # 7. Record success
            self.guard.record_success(entry_id, estimated_cost)
            self.rate_limiter.record(session_id)

            # 8. Store in dedupe cache
            if self.dedupe.enabled:
                self.dedupe.store(fingerprint, result)

            return result

        except Exception as e:
            # Record failure (cost not charged)
            self.guard.record_failure(entry_id)
            raise

    def edit_image(
        self,
        request,  # EditImageRequest
        session_id: str,
        image_bytes: bytes,
        mask_bytes: Optional[bytes] = None,
        skip_dedupe: bool = False,
    ) -> bytes:
        """Edit image with budget safety checks.

        Args:
            request: EditImageRequest with prompt, size, etc.
            session_id: Session identifier
            image_bytes: Input image bytes (for fingerprint)
            mask_bytes: Mask bytes (for fingerprint)
            skip_dedupe: If True, skip dedupe cache

        Returns:
            Image bytes

        Raises:
            BudgetExceededError: If budget cap is exceeded
            RateLimitedError: If rate limit is hit
        """
        model = request.model or self._client.config.model
        operation = "edit"

        # Calculate fingerprint including input image hashes
        input_hashes = [hash_image_bytes(image_bytes)]
        if mask_bytes:
            input_hashes.append(hash_image_bytes(mask_bytes))

        params = {
            "size": getattr(request, "size", "1024x1024"),
            "n": getattr(request, "n", 1),
        }
        fingerprint = calculate_fingerprint(
            model=model,
            action=operation,
            prompt=request.prompt,
            negative_prompt="",
            params=params,
            input_hashes=input_hashes,
        )

        # Check dedupe cache
        if not skip_dedupe and self.dedupe.enabled:
            cached = self.dedupe.get_image_bytes(fingerprint)
            if cached:
                return cached

        # Check rate limit
        self.rate_limiter.check_or_raise(session_id)

        # Estimate cost and check budget
        estimated_cost = self.cost_estimator.estimate(
            model=model,
            size=getattr(request, "size", "1024x1024"),
            quality="standard",
            n=getattr(request, "n", 1),
            operation=operation,
        )
        self.guard.ensure_allowed_or_raise(operation, estimated_cost, session_id)

        # Record operation start
        entry_id = self.guard.record_operation_start(
            operation=operation,
            model=model,
            estimated_cost=estimated_cost,
            session_id=session_id,
            fingerprint=fingerprint,
        )

        try:
            # Execute request
            result = self._client.edit_image(request)

            # Record success
            self.guard.record_success(entry_id, estimated_cost)
            self.rate_limiter.record(session_id)

            # Store in dedupe cache
            if self.dedupe.enabled:
                self.dedupe.store(fingerprint, result)

            return result

        except Exception as e:
            self.guard.record_failure(entry_id)
            raise

    def refine_prompt(
        self,
        request,  # RefinePromptRequest
        session_id: str,
    ):
        """Refine prompt with budget tracking (cheaper operation).

        Refine is a cheap text operation, so we:
        - Don't check dedupe (prompts vary)
        - Don't rate limit (it's cheap)
        - Do track cost in budget
        """
        model = getattr(request, "model", None) or self._client.config.refine_model
        operation = "refine"

        # Estimate cost (much cheaper than image operations)
        estimated_cost = self.cost_estimator.estimate(
            model=model,
            operation=operation,
        )

        # Check budget (but don't rate limit)
        self.guard.ensure_allowed_or_raise(operation, estimated_cost, session_id)

        # Record operation
        entry_id = self.guard.record_operation_start(
            operation=operation,
            model=model,
            estimated_cost=estimated_cost,
            session_id=session_id,
        )

        try:
            result = self._client.refine_prompt(request)
            self.guard.record_success(entry_id, estimated_cost)
            return result
        except Exception as e:
            self.guard.record_failure(entry_id)
            raise

    def create_variation(
        self,
        request,  # CreateVariationRequest
        session_id: str,
        image_bytes: bytes,
        skip_dedupe: bool = False,
    ) -> bytes:
        """Create image variation with budget safety checks."""
        model = getattr(request, "model", None) or self._client.config.model
        operation = "variation"

        input_hashes = [hash_image_bytes(image_bytes)]
        params = {
            "size": getattr(request, "size", "1024x1024"),
            "n": getattr(request, "n", 1),
        }
        fingerprint = calculate_fingerprint(
            model=model,
            action=operation,
            prompt="",  # Variations don't use prompt
            negative_prompt="",
            params=params,
            input_hashes=input_hashes,
        )

        # Check dedupe
        if not skip_dedupe and self.dedupe.enabled:
            cached = self.dedupe.get_image_bytes(fingerprint)
            if cached:
                return cached

        # Check rate limit
        self.rate_limiter.check_or_raise(session_id)

        # Estimate and check budget
        estimated_cost = self.cost_estimator.estimate(
            model=model,
            size=getattr(request, "size", "1024x1024"),
            quality="standard",
            n=getattr(request, "n", 1),
            operation=operation,
        )
        self.guard.ensure_allowed_or_raise(operation, estimated_cost, session_id)

        entry_id = self.guard.record_operation_start(
            operation=operation,
            model=model,
            estimated_cost=estimated_cost,
            session_id=session_id,
            fingerprint=fingerprint,
        )

        try:
            result = self._client.create_variation(request)
            self.guard.record_success(entry_id, estimated_cost)
            self.rate_limiter.record(session_id)

            if self.dedupe.enabled:
                self.dedupe.store(fingerprint, result)

            return result
        except Exception as e:
            self.guard.record_failure(entry_id)
            raise

    # Passthrough methods (no cost tracking needed)

    def list_models(self):
        """List available models (free operation)."""
        return self._client.list_models()

    @property
    def config(self):
        """Access underlying client config."""
        return self._budget_config

    @config.setter
    def config(self, value):
        self._budget_config = value
