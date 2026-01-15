from iopaint.openai_compat.errors import (
    ErrorStatus,
    classify_error,
    create_network_error,
    create_timeout_error,
)


def test_classify_error_rate_limit():
    error = classify_error(429, {"error": {"message": "Too many requests"}})

    assert error.status == ErrorStatus.RATE_LIMITED
    assert error.retryable is True
    assert error.http_status == 429


def test_classify_error_quota():
    error = classify_error(403, {"error": {"message": "Billing quota exceeded"}})

    assert error.status == ErrorStatus.INSUFFICIENT_QUOTA
    assert error.retryable is False


def test_classify_error_content_policy():
    error = classify_error(
        400,
        {"error": {"type": "content_policy", "message": "Unsafe content"}},
    )

    assert error.status == ErrorStatus.CONTENT_POLICY
    assert error.retryable is False


def test_classify_error_server_error():
    error = classify_error(500, {"error": {"message": "Server error"}})

    assert error.status == ErrorStatus.SERVER_ERROR
    assert error.retryable is True


def test_create_timeout_and_network_errors():
    timeout_error = create_timeout_error(Exception("timeout"))
    network_error = create_network_error(Exception("network"))

    assert timeout_error.status == ErrorStatus.TIMEOUT
    assert timeout_error.retryable is True
    assert network_error.status == ErrorStatus.NETWORK_ERROR
    assert network_error.retryable is True
