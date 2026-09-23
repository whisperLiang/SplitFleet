from typing import Optional
from abc import ABC, abstractmethod

import numpy as np
import torch

from splitfleet.common import (
    ControlCode,
    BatchData,
    RequestType,
    RequestArgumentFormat,
)
from splitfleet.common.parameter import (
    ndarray_dict_to_bytes,
    bytes_to_ndarray_dict
)

FORBIDDEN_SERVER_MODEL_METHODS = {
    "get_parameters",
    "configure_fit",
    "configure_evaluate",
    "to_server_model",
}


class ServerModelProxy(ABC):

    def __init__(self, cid: str):
        """Initialize a server model proxy.

        Parameters
        ----------
        cid : str
            ID of the client associated with the given proxy
        """
        self.cid = cid
        self._request_argument_format: RequestArgumentFormat = RequestArgumentFormat.RAW

    def numpy(self) -> None:
        """Informs the server model proxy that the client will be sending numpy arrays
        """
        self._request_argument_format = RequestArgumentFormat.NUMPY

    def torch(self) -> None:
        """Informs the server model proxy that the client will be sending pytorch tensors
        """
        self._request_argument_format = RequestArgumentFormat.TORCH

    @abstractmethod
    def _blocking_request(
        self,
        method: str,
        batch_data: BatchData,
        _timeout_: Optional[float],
        _streams_: bool,
    ) -> BatchData:
        """Issue a blocking request to the server model. The client will wait until receiving a
        response

        Parameters
        ----------
        method : str
            Name of the server method to be invoked
        batch_data : BatchData
            Request data
        _timeout_ : Optional[float]
            Maximum time allows for the request to complete

        Returns
        -------
        BatchData
            Data returned by the server model

        Raises
        ------
        AttributeError
            If the server model does not have the requested method
        Exception
            If any other exception occurs during the request
        """

    @abstractmethod
    def _streaming_request(
        self,
        method: str,
        batch_data: BatchData,
        _timeout_: Optional[float],
        _streams_: bool,
    ) -> None:
        """Issue a non blocking request to the server model. The client will immediately continue

        Parameters
        ----------
        method : str
            Name of the server method to be invoked
        batch_data : BatchData
            Request data
        """

    @abstractmethod
    def _nonblocking_request(
        self,
        method: str,
        batch_data: BatchData,
        _timeout_: Optional[float],
        _streams_: bool,
    ):
        """Issue a request and return a future for its response."""

    @abstractmethod
    def close_stream(self):
        """Wait until the server finishes processing the data the client sent to it
        """

    def _parse_request_args(self, *args, **kwargs):
        request = None
        if self._request_argument_format == RequestArgumentFormat.TORCH:
            np_dict = {}
            for key, val in kwargs.items():
                np_dict[key] = val.detach().cpu().numpy() \
                    if isinstance(val, torch.Tensor) else [v.detach().cpu().numpy() for v in val]

            kwargs = np_dict

        if self._request_argument_format in (
            RequestArgumentFormat.NUMPY,
            RequestArgumentFormat.TORCH
        ):
            data = ndarray_dict_to_bytes(kwargs)
            request = BatchData(data=data, control_code=ControlCode.OK)

        elif len(args) == 1:
            request = args[0]
            assert isinstance(request, BatchData) and \
                self._request_argument_format == RequestArgumentFormat.RAW

        assert request is not None, \
            "You must specify type of request data using numpy(), torch()"
        return request

    def _parse_response_args(self, batch_data: BatchData):
        if self._request_argument_format in (
            RequestArgumentFormat.NUMPY,
            RequestArgumentFormat.TORCH
        ):
            out = bytes_to_ndarray_dict(batch_data.data)

            if self._request_argument_format == RequestArgumentFormat.NUMPY:
                return out

            if isinstance(out, np.ndarray):
                return torch.from_numpy(out)
            if isinstance(out, list):
                return [torch.from_numpy(v) for v in out]

            torch_out = {}
            for key, val in out.items():
                torch_out[key] = torch.from_numpy(val) if isinstance(val, np.ndarray) else \
                    [torch.from_numpy(v) for v in val]
            return torch_out

        assert self._request_argument_format == RequestArgumentFormat.RAW
        return batch_data

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        if name in FORBIDDEN_SERVER_MODEL_METHODS:
            raise Exception(f"Client should not invoke the {name} method")

        def _request(
            *args,
            _type_: RequestType = RequestType.BLOCKING,
            _timeout_: Optional[float] = None,
            _streams_: bool = True,
            **kwargs
        ) -> Optional[BatchData]:

            batch_data = self._parse_request_args(*args, **kwargs)
            if _type_ == RequestType.BLOCKING:
                res = self._blocking_request(
                    method=name,
                    batch_data=batch_data,
                    _timeout_=_timeout_,
                    _streams_=_streams_,
                )
                res = self._parse_response_args(res)
                return res

            elif _type_ == RequestType.STREAM:
                self._streaming_request(
                    method=name,
                    batch_data=batch_data,
                    _timeout_=_timeout_,
                    _streams_=_streams_,
                )

            elif _type_ == RequestType.FUTURE:
                future = self._nonblocking_request(
                    method=name,
                    batch_data=batch_data,
                    _timeout_=_timeout_,
                    _streams_=_streams_,
                )
                return future
            else:
                raise ValueError(f"Unknown request type: {_type_!r}")

        return _request
