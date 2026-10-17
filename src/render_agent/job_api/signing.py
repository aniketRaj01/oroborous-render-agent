from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Protocol

from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials, ReadOnlyCredentials
from botocore.session import get_session

LAMBDA_SERVICE = "lambda"
CONTENT_TYPE = "application/json"


class RequestSigner(Protocol):
    def headers_for(self, *, method: str, url: str, body: bytes) -> dict[str, str]:
        """Headers to send with this exact request, including any signature."""
        ...


class UnsignedSigner:
    """For a loopback Job API harness only. Config validation refuses it for any other host."""

    def headers_for(self, *, method: str, url: str, body: bytes) -> dict[str, str]:
        return {"content-type": CONTENT_TYPE}


class NoCredentialsError(Exception):
    pass


def _default_credentials() -> ReadOnlyCredentials:
    credentials = get_session().get_credentials()
    if credentials is None:
        raise NoCredentialsError("no AWS credentials found in the default credential chain")
    frozen = credentials.get_frozen_credentials()
    if not frozen.access_key or not frozen.secret_key:
        raise NoCredentialsError("AWS credentials are incomplete")
    return frozen


class SigV4Signer:
    def __init__(
        self,
        *,
        region: str,
        credentials: Callable[[], ReadOnlyCredentials] = _default_credentials,
    ) -> None:
        self._region = region
        self._credentials = credentials

    def headers_for(self, *, method: str, url: str, body: bytes) -> dict[str, str]:
        frozen = self._credentials()
        if frozen.access_key is None or frozen.secret_key is None:
            raise NoCredentialsError("AWS credentials are incomplete")
        # Lambda Function URLs do not accept unsigned payloads: the body hash must travel in
        # x-amz-content-sha256 and be covered by the signature.
        request = AWSRequest(
            method=method,
            url=url,
            data=body,
            headers={
                "content-type": CONTENT_TYPE,
                "x-amz-content-sha256": hashlib.sha256(body).hexdigest(),
            },
        )
        SigV4Auth(
            Credentials(frozen.access_key, frozen.secret_key, frozen.token),
            LAMBDA_SERVICE,
            self._region,
        ).add_auth(request)
        return {key: str(value) for key, value in request.headers.items()}
