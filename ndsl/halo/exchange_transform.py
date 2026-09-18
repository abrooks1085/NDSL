from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any, Literal

import numpy as np

from ndsl.buffer import Buffer
from ndsl.halo.data_transformer import HaloDataTransformer
from ndsl.optional_imports import cupy as cp
from ndsl.quantity import Quantity


def _array_module(array: Any) -> ModuleType:
    """Return the NumPy-like module associated with an array."""
    if isinstance(array, np.ndarray):
        return np

    try:
        return cp.get_array_module(array)
    except AttributeError as err:
        raise TypeError(
            f"Unsupported halo transport array type: {type(array)}"
        ) from err


class HaloExchangeTransform(abc.ABC):
    """Transform the transport representation of a halo exchange.

    HaloDataTransformer handles the normal conversion between Quantity storage
    and packed halo buffers. A HaloExchangeTransform can alter the
    representation
    used for communication without changing HaloUpdater's communication logic.
    """

    @abc.abstractmethod
    def get_recv_buffer(
        self,
        data_transformer: HaloDataTransformer,
    ) -> Buffer:
        """Return the buffer MPI should receive into."""
        pass

    @abc.abstractmethod
    def async_pack(
        self,
        data_transformer: HaloDataTransformer,
        quantities_x: Sequence[Quantity],
        quantities_y: Sequence[Quantity] | None = None,
    ) -> None:
        """Prepare the buffer that MPI will send."""
        pass

    @abc.abstractmethod
    def get_send_buffer(
        self,
        data_transformer: HaloDataTransformer,
    ) -> Buffer:
        """Return the prepared MPI send buffer."""
        pass

    @abc.abstractmethod
    def async_unpack(
        self,
        data_transformer: HaloDataTransformer,
        quantities_x: Sequence[Quantity],
        quantities_y: Sequence[Quantity] | None = None,
    ) -> None:
        """Consume the received transport representation."""
        pass

    def synchronize(self) -> None:
        """Synchronize transform-specific work."""
        return None

    def finalize(self) -> None:
        """Release transform-owned resources."""
        return None


class IdentityHaloExchangeTransform(HaloExchangeTransform):
    """Use the ordinary HaloDataTransformer representation unchanged."""

    def get_recv_buffer(
        self,
        data_transformer: HaloDataTransformer,
    ) -> Buffer:
        return data_transformer.get_unpack_buffer()

    def async_pack(
        self,
        data_transformer: HaloDataTransformer,
        quantities_x: Sequence[Quantity],
        quantities_y: Sequence[Quantity] | None = None,
    ) -> None:
        data_transformer.async_pack(quantities_x, quantities_y)

    def get_send_buffer(
        self,
        data_transformer: HaloDataTransformer,
    ) -> Buffer:
        return data_transformer.get_pack_buffer()

    def async_unpack(
        self,
        data_transformer: HaloDataTransformer,
        quantities_x: Sequence[Quantity],
        quantities_y: Sequence[Quantity] | None = None,
    ) -> None:
        data_transformer.async_unpack(quantities_x, quantities_y)


class Coarse2FineHaloExchangeTransform(HaloExchangeTransform):
    """Transport transform for a one-way coarse-to-fine halo update.

    A nested coarse-to-fine exchange can contain several explicit windows from
    the same Quantity for a single peer. The coarse side packs those windows and
    applies a prolongation operation before communication. The fine side scatters
    the received fine-resolution data into its configured windows.
    """

    def __init__(
        self,
        *,
        role: Literal["coarse", "fine"],
        transport_size: int,
        windows: Sequence[tuple[slice, ...]],
        prolongation: Callable[[Any, Any], None] | None = None,
    ) -> None:
        if role not in ("coarse", "fine"):
            raise ValueError(f"Invalid coarse-to-fine halo role: {role}")
        if transport_size <= 0:
            raise ValueError("transport_size must be positive")
        if len(windows) == 0:
            raise ValueError("coarse-to-fine exchange requires at least one window")
        if role == "coarse" and prolongation is None:
            raise ValueError("coarse role requires a prolongation operation")
        if role == "fine" and prolongation is not None:
            raise ValueError("fine role must not supply a prolongation operation")

        self._role = role
        self._transport_size = transport_size
        self._windows = tuple(windows)
        self._prolongation = prolongation

        self._send_buffer: Buffer | None = None
        self._recv_buffer: Buffer | None = None

    @staticmethod
    def _get_scalar_quantity(
        quantities_x: Sequence[Quantity],
        quantities_y: Sequence[Quantity] | None,
    ) -> Quantity:
        if quantities_y is not None:
            raise NotImplementedError(
                "Coarse-to-fine exchange currently supports scalar quantities only."
            )

        if len(quantities_x) != 1:
            raise ValueError(
                "Coarse-to-fine exchange expects exactly one scalar Quantity, "
                f"got {len(quantities_x)}."
            )

        return quantities_x[0]

    def _allocate_transport_buffer(
        self,
        data_transformer: HaloDataTransformer,
        *,
        receive: bool,
    ) -> Buffer:
        if receive:
            template = data_transformer.get_unpack_buffer()
        else:
            template = data_transformer.get_pack_buffer()

        np_module = _array_module(template.array)
        return Buffer.pop_from_cache(
            np_module.zeros,
            (self._transport_size,),
            template.array.dtype,
        )

    def _ensure_send_buffer(
        self,
        data_transformer: HaloDataTransformer,
    ) -> Buffer:
        if self._send_buffer is None:
            self._send_buffer = self._allocate_transport_buffer(
                data_transformer,
                receive=False,
            )
        return self._send_buffer

    def _ensure_recv_buffer(
        self,
        data_transformer: HaloDataTransformer,
    ) -> Buffer:
        if self._recv_buffer is None:
            self._recv_buffer = self._allocate_transport_buffer(
                data_transformer,
                receive=True,
            )
        return self._recv_buffer

    def get_recv_buffer(
        self,
        data_transformer: HaloDataTransformer,
    ) -> Buffer:
        if self._role != "fine":
            raise RuntimeError(
                "coarse side of coarse-to-fine exchange must not post a receive"
            )
        return self._ensure_recv_buffer(data_transformer)

    def async_pack(
        self,
        data_transformer: HaloDataTransformer,
        quantities_x: Sequence[Quantity],
        quantities_y: Sequence[Quantity] | None = None,
    ) -> None:
        if self._role != "coarse":
            raise RuntimeError(
                "fine side of coarse-to-fine exchange must not pack a send buffer"
            )

        quantity = self._get_scalar_quantity(quantities_x, quantities_y)
        source_parts = [quantity[window].reshape(-1) for window in self._windows]
        source = quantity.np.concatenate(source_parts)

        send_buffer = self._ensure_send_buffer(data_transformer)
        assert self._prolongation is not None
        self._prolongation(source, send_buffer.array)

    def get_send_buffer(
        self,
        data_transformer: HaloDataTransformer,
    ) -> Buffer:
        if self._role != "coarse":
            raise RuntimeError("fine side of coarse-to-fine exchange must not send")

        send_buffer = self._ensure_send_buffer(data_transformer)
        self.synchronize()
        return send_buffer

    def async_unpack(
        self,
        data_transformer: HaloDataTransformer,
        quantities_x: Sequence[Quantity],
        quantities_y: Sequence[Quantity] | None = None,
    ) -> None:
        if self._role != "fine":
            raise RuntimeError(
                "coarse side of coarse-to-fine exchange must not unpack received data"
            )

        quantity = self._get_scalar_quantity(quantities_x, quantities_y)

        if self._recv_buffer is None:
            raise RuntimeError("coarse-to-fine receive buffer has not been allocated")

        offset = 0
        for window in self._windows:
            destination = quantity[window]
            window_size = destination.size
            next_offset = offset + window_size

            if next_offset > self._transport_size:
                raise RuntimeError(
                    "coarse-to-fine receive buffer is smaller than "
                    "the configured fine windows"
                )

            destination[...] = quantity.np.reshape(
                self._recv_buffer.array[offset:next_offset],
                destination.shape,
                order="C",
            )
            offset = next_offset

        if offset != self._transport_size:
            raise RuntimeError(
                "coarse-to-fine receive windows consumed "
                f"{offset} values from a transport buffer containing "
                f"{self._transport_size}"
            )

    def synchronize(self) -> None:
        if self._send_buffer is not None:
            self._send_buffer.finalize_memory_transfer()
        if self._recv_buffer is not None:
            self._recv_buffer.finalize_memory_transfer()

    def finalize(self) -> None:
        self.synchronize()

        if self._send_buffer is not None:
            Buffer.push_to_cache(self._send_buffer)
            self._send_buffer = None

        if self._recv_buffer is not None:
            Buffer.push_to_cache(self._recv_buffer)
            self._recv_buffer = None


class IndexedProlongation:
    """Nearest-neighbor prolongation from packed coarse data."""

    def __init__(self, source_indices: Sequence[int]) -> None:
        if len(source_indices) == 0:
            raise ValueError("source_indices must not be empty")

        self._source_indices = tuple(source_indices)

    def __call__(self, source: Any, destination: Any) -> None:
        source = source.reshape(-1)
        destination = destination.reshape(-1)

        if destination.size != len(self._source_indices):
            raise ValueError(
                "prolongation destination size does not match index count: "
                f"{destination.size} != {len(self._source_indices)}"
            )

        min_index = min(self._source_indices)
        max_index = max(self._source_indices)
        if min_index < 0 or max_index >= source.size:
            raise ValueError(
                "prolongation source index lies outside packed coarse buffer "
                f"of size {source.size}"
            )

        np_module = _array_module(source)
        source_indices = np_module.asarray(
            self._source_indices,
            dtype=np_module.intp,
        )
        np_module.take(
            source,
            source_indices,
            out=destination,
        )
