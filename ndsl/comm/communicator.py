from __future__ import annotations

import abc
import copy
import math
from collections.abc import Callable, Mapping, Sequence
from types import ModuleType
from typing import Any, Generic, Self, TypeVar, cast

import numpy as np

import ndsl.constants as constants
from ndsl.buffer import array_buffer, device_synchronize, recv_buffer, send_buffer
from ndsl.comm.boundary import Boundary
from ndsl.comm.comm_abc import Comm as CommABC
from ndsl.comm.comm_abc import ReductionOperator
from ndsl.comm.partitioner import (
    CoarseToFineExchangePlan,
    CubedSpherePartitioner,
    NestedPartitioner,
    Partitioner,
    TilePartitioner,
)
from ndsl.halo.exchange_transform import (
    Coarse2FineHaloExchangeTransform,
    IndexedProlongation,
)
from ndsl.halo.updater import HaloUpdater, HaloUpdateRequest, VectorInterfaceHaloUpdater
from ndsl.optional_imports import cupy
from ndsl.performance.timer import NullTimer, Timer
from ndsl.quantity import Quantity, QuantityHaloSpec, QuantityMetadata


def to_numpy(array, dtype=None) -> np.ndarray:  # type: ignore[no-untyped-def]
    """
    Input array can be a numpy array or a cupy array. Returns numpy array.
    """
    try:
        output = np.asarray(array)
    except ValueError as err:
        if err.args[0] == "object __array__ method not producing an array":
            output = cupy.asnumpy(array)
        else:
            raise err
    except TypeError as err:
        if err.args[0].startswith(
            "Implicit conversion to a NumPy array is not allowed."
        ):
            output = cupy.asnumpy(array)
        else:
            raise err
    if dtype:
        output = output.astype(dtype=dtype)
    return output


P = TypeVar("P", bound=Partitioner)

class GridHierarchyCommunicator:
    def __init__(
        self,
        world_comm: CommABC,
        parent_comm: Communicator | None = None,
        nested_comms: Mapping[int, Communicator] | None = None,
    ) -> None:
        self.world_comm = world_comm
        self.parent_comm = parent_comm
        self.nested_comms = (
                dict(nested_comms)
                if nested_comms is not None
                else {}
        )

    @property
    def rank(self) -> int:
        return self.world_comm.Get_rank()

    @property
    def size(self) -> int:
        return self.world_comm.Get_size()


class Communicator(abc.ABC, Generic[P]):
    def __init__(
        self,
        comm: CommABC,
        partitioner: P,
        force_cpu: bool = False,
        timer: Timer | None = None,
    ):
        self.comm = comm
        self.partitioner: P = partitioner
        self._force_cpu = force_cpu
        self._boundaries: Mapping[int, Boundary] | None = None
        self._last_halo_tag = 0
        self.timer: Timer = timer if timer is not None else NullTimer()

    @property
    @abc.abstractmethod
    def tile(self) -> TileCommunicator:
        pass

    @classmethod
    @abc.abstractmethod
    def from_layout(
        cls,
        comm: CommABC,
        layout: tuple[int, int],
        force_cpu: bool = False,
        timer: Timer | None = None,
    ) -> Self:
        pass

    @property
    def rank(self) -> int:
        """Rank of the current process within this communicator."""
        return self.comm.Get_rank()

    @property
    def size(self) -> int:
        """Total number of ranks in this communicator."""
        return self.comm.Get_size()

    def _maybe_force_cpu(self, module: ModuleType) -> ModuleType:
        """
        Get a numpy-like module depending on configuration and
        Quantity original allocator.
        """
        if self._force_cpu:
            return np
        return module

    @staticmethod
    def _device_synchronize() -> None:
        """Wait for all work that could be in-flight to finish."""
        # this is a method so we can profile it separately from other device syncs
        device_synchronize()

    def _create_all_reduce_quantity(
        self, input_metadata: QuantityMetadata, input_data: Any
    ) -> Quantity:
        """Create a Quantity for all_reduce data and metadata."""
        all_reduce_quantity = Quantity(
            input_data,
            dims=input_metadata.dims,
            units=input_metadata.units,
            origin=input_metadata.origin,
            extent=input_metadata.extent,
            backend=input_metadata.backend,
            allow_mismatch_float_precision=False,
        )
        return all_reduce_quantity

    def all_reduce(
        self,
        input_quantity: Quantity,
        op: ReductionOperator,
        output_quantity: Quantity | None = None,
    ) -> Quantity:
        reduced_quantity_data = self.comm.allreduce(input_quantity._data, op)
        if output_quantity is None:
            return self._create_all_reduce_quantity(
                input_quantity.metadata, reduced_quantity_data
            )

        if output_quantity.shape != input_quantity.shape:
            raise TypeError("Shapes not matching")

        input_quantity.metadata.duplicate_metadata(output_quantity.metadata)

        output_quantity[:] = reduced_quantity_data[:]
        return output_quantity

    def all_reduce_per_element(
        self,
        input_quantity: Quantity,
        output_quantity: Quantity,
        op: ReductionOperator,
    ) -> None:
        self.comm.Allreduce(input_quantity._data, output_quantity._data, op)

    def all_reduce_per_element_in_place(
        self, quantity: Quantity, op: ReductionOperator
    ) -> None:
        # Note that device_synchronization is Cupy/Cuda specific
        # at the moment.
        device_synchronize()
        self.comm.Allreduce_inplace(quantity._data, op)

    def _Scatter(self, numpy_module, sendbuf, recvbuf, **kwargs):  # type: ignore[no-untyped-def]
        with send_buffer(numpy_module.zeros, sendbuf) as send:
            with recv_buffer(numpy_module.zeros, recvbuf) as recv:
                self.comm.Scatter(send, recv, **kwargs)

    def _Gather(self, numpy_module, sendbuf, recvbuf, **kwargs):  # type: ignore[no-untyped-def]
        with send_buffer(numpy_module.zeros, sendbuf) as send:
            with recv_buffer(numpy_module.zeros, recvbuf) as recv:
                self.comm.Gather(send, recv, **kwargs)

    def scatter(
        self,
        send_quantity: Quantity | None = None,
        recv_quantity: Quantity | None = None,
    ) -> Quantity:
        """Transfer subtile regions of a full-tile quantity
        from the tile root rank to all subtiles.

        Args:
            send_quantity: quantity to send, only required/used on the tile root rank
            recv_quantity: if provided, assign received data into this Quantity.
        Returns:
            recv_quantity
        """
        if self.rank == constants.ROOT_RANK and send_quantity is None:
            raise TypeError("send_quantity is a required argument on the root rank")
        if self.rank == constants.ROOT_RANK:
            send_quantity = cast(Quantity, send_quantity)
            metadata: QuantityMetadata = self.comm.bcast(
                send_quantity.metadata, root=constants.ROOT_RANK
            )  # type: ignore[assignment]
        else:
            metadata = self.comm.bcast(None, root=constants.ROOT_RANK)  # type: ignore[assignment]
        shape = self.partitioner.subtile_extent(metadata, self.rank)
        if recv_quantity is None:
            recv_quantity = self._get_scatter_recv_quantity(shape, metadata)
        if self.rank == constants.ROOT_RANK:
            send_quantity = cast(Quantity, send_quantity)
            with array_buffer(
                self._maybe_force_cpu(metadata.np).zeros,
                (self.partitioner.total_ranks,) + shape,
                dtype=metadata.dtype,
            ) as sendbuf:
                for rank in range(0, self.partitioner.total_ranks):
                    subtile_slice = self.partitioner.subtile_slice(
                        rank=rank,
                        global_dims=metadata.dims,
                        global_extent=metadata.extent,
                        overlap=True,
                    )
                    sendbuf.assign_from(
                        send_quantity.view[subtile_slice],
                        buffer_slice=np.index_exp[rank, :],
                    )
                self._Scatter(
                    metadata.np,
                    sendbuf.array,
                    recv_quantity.view[:],
                    root=constants.ROOT_RANK,
                )
        else:
            self._Scatter(
                metadata.np,
                None,
                recv_quantity.view[:],
                root=constants.ROOT_RANK,
            )
        return recv_quantity

    def _get_gather_recv_quantity(
        self, global_extent: Sequence[int], send_metadata: QuantityMetadata
    ) -> Quantity:
        """Initialize a Quantity for use when receiving global data during gather"""
        recv_quantity = Quantity(
            send_metadata.np.zeros(global_extent, dtype=send_metadata.dtype),
            dims=send_metadata.dims,
            units=send_metadata.units,
            origin=tuple([0 for dim in send_metadata.dims]),
            extent=global_extent,
            backend=send_metadata.backend,
            allow_mismatch_float_precision=True,
        )
        return recv_quantity

    def _get_scatter_recv_quantity(
        self, shape: Sequence[int], send_metadata: QuantityMetadata
    ) -> Quantity:
        """Initialize a Quantity for use when receiving subtile data during scatter"""
        recv_quantity = Quantity(
            send_metadata.np.zeros(shape, dtype=send_metadata.dtype),
            dims=send_metadata.dims,
            units=send_metadata.units,
            backend=send_metadata.backend,
            allow_mismatch_float_precision=True,
        )
        return recv_quantity

    def gather(
        self, send_quantity: Quantity, recv_quantity: Quantity | None = None
    ) -> Quantity | None:
        """Transfer subtile regions of a full-tile quantity
        from each rank to the tile root rank.

        Args:
            send_quantity: quantity to send
            recv_quantity: if provided, assign received data into this Quantity (only
                used on the tile root rank)
        Returns:
            recv_quantity: quantity if on root rank, otherwise None
        """
        result: Quantity | None
        if self.rank == constants.ROOT_RANK:
            with array_buffer(
                send_quantity.np.zeros,
                (self.partitioner.total_ranks,) + tuple(send_quantity.extent),
                dtype=send_quantity.dtype,
            ) as recvbuf:
                self._Gather(
                    send_quantity.np,
                    send_quantity.view[:],
                    recvbuf.array,
                    root=constants.ROOT_RANK,
                )
                if recv_quantity is None:
                    global_extent = self.partitioner.global_extent(
                        send_quantity.metadata
                    )
                    recv_quantity = self._get_gather_recv_quantity(
                        global_extent, send_quantity.metadata
                    )
                for rank in range(self.partitioner.total_ranks):
                    to_slice = self.partitioner.subtile_slice(
                        rank=rank,
                        global_dims=recv_quantity.dims,
                        global_extent=recv_quantity.extent,
                        overlap=True,
                    )
                    recvbuf.assign_to(
                        recv_quantity.view[to_slice], buffer_slice=np.index_exp[rank, :]
                    )
                result = recv_quantity
        else:
            self._Gather(
                send_quantity.np,
                send_quantity.view[:],
                None,
                root=constants.ROOT_RANK,
            )
            result = None
        return result

    def gather_state(self, send_state=None, recv_state=None, transfer_type=None):  # type: ignore[no-untyped-def]
        """Transfer a state dictionary from subtile ranks to the tile root rank.

        'time' is assumed to be the same on all ranks, and its value will be set
        to the value from the root rank.

        Args:
            send_state: the model state to be sent containing the subtile data
            recv_state: the pre-allocated state in which to receive the full tile
                state. Only variables which are scattered will be written to.
        Returns:
            recv_state: on the root rank, the state containing the entire tile
        """
        if self.rank == constants.ROOT_RANK and recv_state is None:
            recv_state = {}
        for name, quantity in send_state.items():
            if name == "time":
                if self.rank == constants.ROOT_RANK:
                    recv_state["time"] = send_state["time"]
            else:
                gather_value = to_numpy(quantity.view[:], dtype=transfer_type)
                gather_quantity = Quantity(
                    data=gather_value,
                    dims=quantity.dims,
                    units=quantity.units,
                    allow_mismatch_float_precision=True,
                    backend=quantity.backend,
                )
                if recv_state is not None and name in recv_state:
                    tile_quantity = self.gather(
                        gather_quantity, recv_quantity=recv_state[name]
                    )
                else:
                    tile_quantity = self.gather(gather_quantity)
                if self.rank == constants.ROOT_RANK:
                    recv_state[name] = tile_quantity
                del gather_quantity
        return recv_state

    def scatter_state(self, send_state=None, recv_state=None):  # type: ignore[no-untyped-def]
        """Transfer a state dictionary from the tile root rank to all subtiles.

        Args:
            send_state: the model state to be sent containing the entire tile,
                required only from the root rank
            recv_state: the pre-allocated state in which to receive the scattered
                state. Only variables which are scattered will be written to.
        Returns:
            rank_state: the state corresponding to this rank's subdomain
        """

        def scatter_root() -> None:
            if send_state is None:
                raise TypeError("send_state is a required argument on the root rank")
            name_list = list(send_state.keys())
            while "time" in name_list:
                name_list.remove("time")
            name_list = self.comm.bcast(name_list, root=constants.ROOT_RANK)  # type: ignore[assignment]
            array_list = [send_state[name] for name in name_list]
            for name, array in zip(name_list, array_list):
                if name in recv_state:
                    self.scatter(send_quantity=array, recv_quantity=recv_state[name])
                else:
                    recv_state[name] = self.scatter(send_quantity=array)
            recv_state["time"] = self.comm.bcast(
                send_state.get("time", None), root=constants.ROOT_RANK
            )

        def scatter_client() -> None:
            name_list = self.comm.bcast(None, root=constants.ROOT_RANK)
            for name in name_list:  # type: ignore
                if name in recv_state:
                    self.scatter(recv_quantity=recv_state[name])
                else:
                    recv_state[name] = self.scatter()
            recv_state["time"] = self.comm.bcast(None, root=constants.ROOT_RANK)

        if recv_state is None:
            recv_state = {}
        if self.rank == constants.ROOT_RANK:
            scatter_root()
        else:
            scatter_client()
        if recv_state["time"] is None:
            recv_state.pop("time")
        return recv_state

    def halo_update(self, quantity: Quantity | list[Quantity], n_points: int) -> None:
        """Perform a halo update on a quantity or quantities

        Args:
            quantity: the quantity to be updated
            n_points: how many halo points to update, starting from the interior
        """
        if isinstance(quantity, Quantity):
            quantities = [quantity]
        else:
            quantities = quantity

        halo_updater = self.start_halo_update(quantities, n_points)
        halo_updater.wait()

    def start_halo_update(
        self, quantity: Quantity | list[Quantity], n_points: int
    ) -> HaloUpdater:
        """Start an asynchronous halo update on a quantity.

        Args:
            quantity: the quantity to be updated
            n_points: how many halo points to update, starting from the interior

        Returns:
            request: an asynchronous request object with a .wait() method
        """
        if isinstance(quantity, Quantity):
            quantities = [quantity]
        else:
            quantities = quantity

        specifications = []
        for quantity in quantities:
            specification = QuantityHaloSpec(
                n_points=n_points,
                shape=quantity.shape,
                strides=quantity._data.strides,
                itemsize=quantity._data.itemsize,
                origin=quantity.origin,
                extent=quantity.extent,
                dims=quantity.dims,
                numpy_module=self._maybe_force_cpu(quantity.np),
                dtype=quantity.metadata.dtype,
            )
            specifications.append(specification)

        halo_updater = self.get_scalar_halo_updater(specifications)
        halo_updater.force_finalize_on_wait()
        halo_updater.start(quantities)
        return halo_updater

    def vector_halo_update(
        self,
        x_quantity: Quantity | list[Quantity],
        y_quantity: Quantity | list[Quantity],
        n_points: int,
    ) -> None:
        """Perform a halo update of a horizontal vector quantity or quantities.

        Assumes the x and y dimension indices are the same between the two quantities.

        Args:
            x_quantity: the x-component quantity to be halo updated
            y_quantity: the y-component quantity to be halo updated
            n_points: how many halo points to update, starting at the interior
        """
        if isinstance(x_quantity, Quantity):
            x_quantities = [x_quantity]
        else:
            x_quantities = x_quantity
        if isinstance(y_quantity, Quantity):
            y_quantities = [y_quantity]
        else:
            y_quantities = y_quantity

        halo_updater = self.start_vector_halo_update(
            x_quantities, y_quantities, n_points
        )
        halo_updater.wait()

    def start_vector_halo_update(
        self,
        x_quantity: Quantity | list[Quantity],
        y_quantity: Quantity | list[Quantity],
        n_points: int,
    ) -> HaloUpdater:
        """Start an asynchronous halo update of a horizontal vector quantity.

        Assumes the x and y dimension indices are the same between the two quantities.

        Args:
            x_quantity: the x-component quantity to be halo updated
            y_quantity: the y-component quantity to be halo updated
            n_points: how many halo points to update, starting at the interior

        Returns:
            request: an asynchronous request object with a .wait() method
        """
        if isinstance(x_quantity, Quantity):
            x_quantities = [x_quantity]
        else:
            x_quantities = x_quantity
        if isinstance(y_quantity, Quantity):
            y_quantities = [y_quantity]
        else:
            y_quantities = y_quantity

        x_specifications = []
        y_specifications = []
        for x_quantity, y_quantity in zip(x_quantities, y_quantities):
            x_specification = QuantityHaloSpec(
                n_points=n_points,
                shape=x_quantity.shape,
                strides=x_quantity._data.strides,
                itemsize=x_quantity._data.itemsize,
                origin=x_quantity.metadata.origin,
                extent=x_quantity.metadata.extent,
                dims=x_quantity.metadata.dims,
                numpy_module=self._maybe_force_cpu(x_quantity.np),
                dtype=x_quantity.metadata.dtype,
            )
            x_specifications.append(x_specification)
            y_specification = QuantityHaloSpec(
                n_points=n_points,
                shape=y_quantity.shape,
                strides=y_quantity._data.strides,
                itemsize=y_quantity._data.itemsize,
                origin=y_quantity.metadata.origin,
                extent=y_quantity.metadata.extent,
                dims=y_quantity.metadata.dims,
                numpy_module=self._maybe_force_cpu(y_quantity.np),
                dtype=y_quantity.metadata.dtype,
            )
            y_specifications.append(y_specification)

        halo_updater = self.get_vector_halo_updater(x_specifications, y_specifications)
        halo_updater.force_finalize_on_wait()
        halo_updater.start(x_quantities, y_quantities)
        return halo_updater

    def synchronize_vector_interfaces(
        self, x_quantity: Quantity, y_quantity: Quantity
    ) -> None:
        """
        Synchronize shared points at the edges of a vector interface variable.

        Sends the values on the south and west edges to overwrite the values on adjacent
        subtiles. Vector must be defined on the Arakawa C grid.

        For interface variables, the edges of the tile are computed on both ranks
        bordering that edge. This routine copies values across those shared edges
        so that both ranks have the same value for that edge. It also handles any
        rotation of vector quantities needed to move data across the edge.

        Args:
            x_quantity: the x-component quantity to be synchronized
            y_quantity: the y-component quantity to be synchronized
        """
        req = self.start_synchronize_vector_interfaces(x_quantity, y_quantity)
        req.wait()

    def start_synchronize_vector_interfaces(
        self, x_quantity: Quantity, y_quantity: Quantity
    ) -> HaloUpdateRequest:
        """
        Synchronize shared points at the edges of a vector interface variable.

        Sends the values on the south and west edges to overwrite the values on adjacent
        subtiles. Vector must be defined on the Arakawa C grid.

        For interface variables, the edges of the tile are computed on both ranks
        bordering that edge. This routine copies values across those shared edges
        so that both ranks have the same value for that edge. It also handles any
        rotation of vector quantities needed to move data across the edge.

        Args:
            x_quantity: the x-component quantity to be synchronized
            y_quantity: the y-component quantity to be synchronized

        Returns:
            request: an asynchronous request object with a .wait() method
        """
        halo_updater = VectorInterfaceHaloUpdater(
            comm=self.comm,
            boundaries=self.boundaries,
            force_cpu=self._force_cpu,
            timer=self.timer,
        )
        req = halo_updater.start_synchronize_vector_interfaces(x_quantity, y_quantity)
        return req

    def get_scalar_halo_updater(
        self, specifications: list[QuantityHaloSpec]
    ) -> HaloUpdater:
        if len(specifications) == 0:
            raise RuntimeError("Cannot create updater with specifications list")
        if specifications[0].n_points == 0:
            raise ValueError("cannot perform a halo update on zero halo points")
        return HaloUpdater.from_scalar_specifications(
            self,
            self._maybe_force_cpu(specifications[0].numpy_module),
            specifications,
            self.boundaries.values(),
            self._get_halo_tag(),
            self.timer,
        )

    def get_vector_halo_updater(
        self,
        specifications_x: list[QuantityHaloSpec],
        specifications_y: list[QuantityHaloSpec],
    ) -> HaloUpdater:
        if len(specifications_x) == 0 and len(specifications_y) == 0:
            raise RuntimeError("Cannot create updater with empty specifications list")
        if specifications_x[0].n_points == 0 and specifications_y[0].n_points == 0:
            raise ValueError("Cannot perform a halo update on zero halo points")
        return HaloUpdater.from_vector_specifications(
            self,
            self._maybe_force_cpu(specifications_x[0].numpy_module),
            specifications_x,
            specifications_y,
            self.boundaries.values(),
            self._get_halo_tag(),
            self.timer,
        )

    def _get_halo_tag(self) -> int:
        self._last_halo_tag += 1
        return self._last_halo_tag

    @property
    def boundaries(self) -> Mapping[int, Boundary]:
        """boundaries of this tile with neighboring tiles"""
        if self._boundaries is None:
            self._boundaries = {}
            for boundary_type in constants.BOUNDARY_TYPES:
                boundary = self.partitioner.boundary(boundary_type, self.rank)
                if boundary is not None:
                    self._boundaries[boundary_type] = boundary
        return self._boundaries


def bcast_metadata_list(comm: CommABC, quantity_list: list[Quantity]):  # type: ignore[no-untyped-def]
    is_root = comm.Get_rank() == constants.ROOT_RANK
    if is_root:
        metadata_list = []
        for quantity in quantity_list:
            metadata_list.append(quantity.metadata)
    else:
        metadata_list = None
    return comm.bcast(metadata_list, root=constants.ROOT_RANK)


def bcast_metadata(comm: CommABC, array: Quantity):  # type: ignore[no-untyped-def]
    return bcast_metadata_list(comm, [array])[0]


class TileCommunicator(Communicator[TilePartitioner]):
    """Performs communications within a single tile or region of a tile."""

    def __init__(
        self,
        comm: CommABC,
        partitioner: TilePartitioner,
        force_cpu: bool = False,
        timer: Timer | None = None,
    ) -> None:
        """Initialize a TileCommunicator.

        Args:
            comm: communication object behaving like mpi4py.Comm
            partitioner: tile partitioner
            force_cpu: force all communication to go through central memory
            timer: Time communication operations.
        """
        super().__init__(comm, partitioner, force_cpu, timer)

    @classmethod
    def from_layout(
        cls,
        comm: CommABC,
        layout: tuple[int, int],
        force_cpu: bool = False,
        timer: Timer | None = None,
    ) -> TileCommunicator:
        return cls(comm, TilePartitioner(layout=layout), force_cpu, timer)

    @property
    def tile(self) -> TileCommunicator:
        return self

    def start_halo_update(
        self, quantity: Quantity | list[Quantity], n_points: int
    ) -> HaloUpdater:
        """Start an asynchronous halo update on a quantity.

        Args:
            quantity: the quantity to be updated
            n_points: how many halo points to update, starting from the interior

        Returns:
            request: an asynchronous request object with a .wait() method
        """
        if self.partitioner.layout[0] < 3 or self.partitioner.layout[1] < 3:
            raise NotImplementedError(
                "implementing halo updates on smaller layouts requires "
                "refactoring our code to remove the assumption that any pair "
                "of ranks only share one boundary"
            )

        return super().start_halo_update(quantity, n_points)

    def start_vector_halo_update(
        self,
        x_quantity: Quantity | list[Quantity],
        y_quantity: Quantity | list[Quantity],
        n_points: int,
    ) -> HaloUpdater:
        """Start an asynchronous halo update of a horizontal vector quantity.

        Assumes the x and y dimension indices are the same between the two quantities.

        Args:
            x_quantity: the x-component quantity to be halo updated
            y_quantity: the y-component quantity to be halo updated
            n_points: how many halo points to update, starting at the interior

        Returns:
            request: an asynchronous request object with a .wait() method
        """
        if self.partitioner.layout[0] < 3 or self.partitioner.layout[1] < 3:
            raise NotImplementedError(
                "implementing halo updates on smaller layouts requires "
                "refactoring our code to remove the assumption that any pair "
                "of ranks only share one boundary"
            )

        return super().start_vector_halo_update(x_quantity, y_quantity, n_points)

    def start_synchronize_vector_interfaces(
        self, x_quantity: Quantity, y_quantity: Quantity
    ) -> HaloUpdateRequest:
        """
        Synchronize shared points at the edges of a vector interface variable.

        Sends the values on the south and west edges to overwrite the values on adjacent
        subtiles. Vector must be defined on the Arakawa C grid.

        For interface variables, the edges of the tile are computed on both ranks
        bordering that edge. This routine copies values across those shared edges
        so that both ranks have the same value for that edge. It also handles any
        rotation of vector quantities needed to move data across the edge.

        Args:
            x_quantity: the x-component quantity to be synchronized
            y_quantity: the y-component quantity to be synchronized

        Returns:
            request: an asynchronous request object with a .wait() method
        """
        if self.partitioner.layout[0] < 3 or self.partitioner.layout[1] < 3:
            raise NotImplementedError(
                "implementing halo updates on smaller layouts requires "
                "refactoring our code to remove the assumption that any pair "
                "of ranks only share one boundary"
            )

        return super().start_synchronize_vector_interfaces(x_quantity, y_quantity)


class CubedSphereCommunicator(Communicator[CubedSpherePartitioner]):
    """Performs communications within a cubed sphere."""

    _tile_communicator: TileCommunicator | None

    def __init__(
        self,
        comm: CommABC,
        partitioner: CubedSpherePartitioner,
        force_cpu: bool = False,
        timer: Timer | None = None,
    ):
        """Initialize a CubedSphereCommunicator.

        Args:
            comm: mpi4py.Comm object
            partitioner: cubed sphere partitioner
            force_cpu: Force all communication to go through central memory.
            timer: Time communication operations.
        """
        if not issubclass(type(comm), CommABC):
            raise TypeError(
                "Communicator needs to be instantiated with communication subsystem"
                f" derived from `comm_abc.Comm`, got {type(comm)}."
            )
        if comm.Get_size() < partitioner.total_ranks:
            raise ValueError(
                f"was given a partitioner for {partitioner.total_ranks} ranks but a "
                f"comm object with only {comm.Get_size()} ranks, are we running "
                "with mpi and the correct number of ranks?"
            )

        super().__init__(comm, partitioner, force_cpu, timer)
        self._tile_communicator = None

    @classmethod
    def from_layout(
        cls,
        comm: CommABC,
        layout: tuple[int, int],
        force_cpu: bool = False,
        timer: Timer | None = None,
    ) -> CubedSphereCommunicator:
        partitioner = CubedSpherePartitioner(tile=TilePartitioner(layout=layout))
        return cls(comm=comm, partitioner=partitioner, force_cpu=force_cpu, timer=timer)

    @property
    def tile(self) -> TileCommunicator:
        """Communicator for within a tile."""
        if self._tile_communicator is None:
            tile_comm = self.comm.Split(
                color=self.partitioner.tile_index(self.rank), key=self.rank
            )
            self._tile_communicator = TileCommunicator(tile_comm, self.partitioner.tile)

        return self._tile_communicator

    def _get_gather_recv_quantity(
        self, global_extent: Sequence[int], send_metadata: QuantityMetadata
    ) -> Quantity:
        """Initialize a Quantity for use when receiving global data during gather.

        Args:
            shape: ndarray shape, numpy-style
            send_metadata: metadata to the created Quantity
        """
        # needs to change the quantity dimensions since we add a "tile" dimension,
        # unlike for tile scatter/gather which retains the same dimensions
        recv_quantity = Quantity(
            send_metadata.np.zeros(global_extent, dtype=send_metadata.dtype),
            dims=(constants.TILE_DIM,) + send_metadata.dims,
            units=send_metadata.units,
            origin=(0,) + tuple([0 for dim in send_metadata.dims]),
            extent=global_extent,
            backend=send_metadata.backend,
            allow_mismatch_float_precision=True,
        )
        return recv_quantity

    def _get_scatter_recv_quantity(
        self, shape: Sequence[int], send_metadata: QuantityMetadata
    ) -> Quantity:
        """Initialize a Quantity for use when receiving subtile data during scatter.

        Args:
            shape: ndarray shape, numpy-style
            send_metadata: metadata to the created Quantity
        """
        # needs to change the quantity dimensions since we remove a "tile" dimension,
        # unlike for tile scatter/gather which retains the same dimensions
        recv_quantity = Quantity(
            send_metadata.np.zeros(shape, dtype=send_metadata.dtype),
            dims=send_metadata.dims[1:],
            units=send_metadata.units,
            backend=send_metadata.backend,
            allow_mismatch_float_precision=True,
        )
        return recv_quantity


class NestedGridCommunicator(Communicator[NestedPartitioner]):
    """Communicator for ranks belonging to one nested grid patch."""

    @classmethod
    def from_layout(
        cls,
        comm: CommABC,
        layout: tuple[int, int],
        force_cpu: bool = False,
        timer: Timer | None = None,
    ) -> NestedGridCommunicator:
        raise NotImplementedError(
            "NestedGridCommunicator requires an existing NestedPartitioner"
        )

    @property
    def tile(self) -> NestedGridCommunicator:
        return self


class NestedCommunicator:
    """Coordinate a parent domain and one nested fine-grid patch.

    World ranks are assigned contiguously, with parent ranks first followed by
    nested ranks.
    """

    def __init__(
        self,
        comm: CommABC,
        parent_partitioner: Partitioner,
        nested_partitioner: NestedPartitioner,
        force_cpu: bool = False,
        timer: Timer | None = None,
        parent_communicator_factory: Callable[..., Communicator] | None = None,
    ) -> None:
        if not issubclass(type(comm), CommABC):
            raise TypeError(
                "NestedCommunicator requires a CommABC communication "
                f"subsystem, got {type(comm)}."
            )

        self.world_comm = comm
        self.comm = self.world_comm

        self.parent_partitioner = parent_partitioner
        self.nested_partitioner = nested_partitioner

        self._force_cpu = force_cpu
        self.timer = timer if timer is not None else NullTimer()
        self._last_halo_tag = 0

        self.world_rank = comm.Get_rank()
        self.world_size = comm.Get_size()

        self.parent_size = parent_partitioner.total_ranks
        self.nested_size = nested_partitioner.total_ranks

        required_size = self.parent_size + self.nested_size
        if self.world_size != required_size:
            raise ValueError(
                f"NestedCommunicator requires exactly {required_size} world ranks: "
                f"{self.parent_size} parent + {self.nested_size} nested. "
                f"Got {self.world_size}."
            )

        self.is_parent_rank = self.world_rank < self.parent_size
        self.is_nested_rank = not self.is_parent_rank

        raw_role_comm = self.world_comm.Split(
            color=0 if self.is_parent_rank else 1,
            key=self.world_rank,
        )

        role_comm = copy.copy(self.world_comm)
        role_comm._comm = raw_role_comm

        self.parent_mpi_comm: CommABC | None = None
        self.nested_mpi_comm: CommABC | None = None

        self.parent_communicator: Communicator | None = None
        self.nested_communicator: NestedGridCommunicator | None = None

        if self.is_parent_rank:
            self.parent_mpi_comm = role_comm

            if parent_communicator_factory is None:
                self.parent_communicator = self._build_default_parent_communicator(
                    comm=self.parent_mpi_comm,
                    partitioner=self.parent_partitioner,
                )
            else:
                self.parent_communicator = parent_communicator_factory(
                    self.parent_mpi_comm,
                    self.parent_partitioner,
                    force_cpu,
                    self.timer,
                )
        else:
            self.nested_mpi_comm = role_comm
            self.nested_communicator = NestedGridCommunicator(
                comm=self.nested_mpi_comm,
                partitioner=self.nested_partitioner,
                force_cpu=force_cpu,
                timer=self.timer,
            )

    def _build_default_parent_communicator(
        self,
        comm: CommABC,
        partitioner: Partitioner,
    ) -> Communicator:
        if isinstance(partitioner, CubedSpherePartitioner):
            return CubedSphereCommunicator(
                comm=comm,
                partitioner=partitioner,
                force_cpu=self._force_cpu,
                timer=self.timer,
            )

        if isinstance(partitioner, TilePartitioner):
            return TileCommunicator(
                comm=comm,
                partitioner=partitioner,
                force_cpu=self._force_cpu,
                timer=self.timer,
            )

        raise TypeError(
            "No default communicator is known for parent partitioner "
            f"{type(partitioner)}. Supply parent_communicator_factory."
        )

    def _parent_tile_partitioner(self) -> TilePartitioner:
        if isinstance(self.parent_partitioner, CubedSpherePartitioner):
            return self.parent_partitioner.tile

        if isinstance(self.parent_partitioner, TilePartitioner):
            return self.parent_partitioner

        raise TypeError(
            "coarse-to-fine communication currently supports "
            "TilePartitioner or CubedSpherePartitioner parents, "
            f"got {type(self.parent_partitioner)}"
        )

    @staticmethod
    def _horizontal_axes(dims: Sequence[str]) -> tuple[int, int]:
        i_axes = [index for index, dim in enumerate(dims) if dim in constants.I_DIMS]
        j_axes = [index for index, dim in enumerate(dims) if dim in constants.J_DIMS]

        if len(i_axes) != 1 or len(j_axes) != 1:
            raise ValueError(
                "coarse-to-fine exchange requires exactly one I dimension "
                f"and one J dimension, got {tuple(dims)}"
            )

        return i_axes[0], j_axes[0]

    @property
    def parent_rank(self) -> int | None:
        if self.parent_mpi_comm is None:
            return None
        return self.parent_mpi_comm.Get_rank()

    @property
    def nested_rank(self) -> int | None:
        if self.nested_mpi_comm is None:
            return None
        return self.nested_mpi_comm.Get_rank()

    def update_nested_halo(
        self,
        fine_quantity: Quantity | None,
        n_points: int,
    ) -> None:
        """Perform same-resolution fine-to-fine halo communication."""
        if not self.is_nested_rank:
            return

        if fine_quantity is None:
            raise ValueError("fine_quantity is required on nested ranks")

        assert self.nested_communicator is not None
        self.nested_communicator.halo_update(
            fine_quantity,
            n_points=n_points,
        )

    def _parent_quantity_geometry(
        self,
        coarse_quantity: Quantity | None,
        anchor_parent_rank: int,
    ) -> tuple[tuple[str, ...], tuple[int, ...]]:
        parent_info = None

        if self.world_rank == anchor_parent_rank:
            if coarse_quantity is None:
                raise ValueError(
                    "coarse_quantity must be supplied on the "
                    "NestMapping anchor parent rank"
                )

            parent_tile = self._parent_tile_partitioner()
            parent_info = (
                tuple(coarse_quantity.dims),
                tuple(parent_tile.global_extent(coarse_quantity.metadata)),
            )

        parent_info = self.world_comm.bcast(
            parent_info,
            root=anchor_parent_rank,
        )

        if parent_info is None:
            raise RuntimeError("Failed to broadcast parent quantity geometry")

        dims = tuple(parent_info[0])
        extent = tuple(int(value) for value in parent_info[1])
        return dims, extent

    def _nested_quantity_geometry(
        self,
        fine_quantity: Quantity | None,
    ) -> tuple[tuple[str, ...], tuple[int, ...]]:
        fine_anchor_world_rank = self.parent_size
        fine_info = None

        if self.world_rank == fine_anchor_world_rank:
            if fine_quantity is None:
                raise ValueError("fine_quantity must be supplied on nested rank zero")

            fine_info = (
                tuple(fine_quantity.dims),
                tuple(self.nested_partitioner.global_extent(fine_quantity.metadata)),
            )

        fine_info = self.world_comm.bcast(
            fine_info,
            root=fine_anchor_world_rank,
        )

        if fine_info is None:
            raise RuntimeError("Failed to broadcast nested quantity geometry")

        dims = tuple(fine_info[0])
        extent = tuple(int(value) for value in fine_info[1])
        return dims, extent

    def _collect_coarse_to_fine_exchanges(
        self,
        parent_tile_extent: tuple[int, ...],
        fine_global_extent: tuple[int, ...],
        dims: tuple[str, ...],
        n_points: int,
    ) -> tuple[
        list[Boundary],
        dict[int, list[CoarseToFineExchangePlan]],
        dict[int, list[Boundary]],
    ]:
        boundaries: list[Boundary] = []
        peer_plans: dict[int, list[CoarseToFineExchangePlan]] = {}
        peer_boundaries: dict[int, list[Boundary]] = {}

        if self.is_parent_rank:
            nested_ranks_to_process = range(self.nested_size)
        else:
            nested_rank = self.nested_rank
            assert nested_rank is not None
            nested_ranks_to_process = (nested_rank,)

        for nested_rank in nested_ranks_to_process:
            nested_world_rank = self.parent_size + nested_rank
            external_boundaries = set(
                self.nested_partitioner.external_boundary_types(nested_rank)
            )

            for boundary_type in constants.BOUNDARY_TYPES:
                if boundary_type not in external_boundaries:
                    continue

                exchanges = self.nested_partitioner.coarse_to_fine_boundaries(
                    parent_partitioner=self.parent_partitioner,
                    parent_tile_extent=parent_tile_extent,
                    fine_global_extent=fine_global_extent,
                    dims=dims,
                    boundary_type=boundary_type,
                    nested_rank=nested_rank,
                    nested_world_rank=nested_world_rank,
                    n_points=n_points,
                )

                for plan, coarse_boundary, fine_boundary in exchanges:
                    if self.is_parent_rank:
                        if self.world_rank != plan.parent_rank:
                            continue

                        peer_rank = nested_world_rank
                        boundary = coarse_boundary
                    else:
                        peer_rank = plan.parent_rank
                        boundary = fine_boundary

                    boundaries.append(boundary)
                    peer_plans.setdefault(peer_rank, []).append(plan)
                    peer_boundaries.setdefault(peer_rank, []).append(boundary)

        return boundaries, peer_plans, peer_boundaries

    @staticmethod
    def _window_shape(window: tuple[slice, ...]) -> tuple[int, ...]:
        shape = []

        for entry in window:
            if entry.start is None or entry.stop is None:
                raise ValueError("coarse-to-fine windows must be bounded")

            shape.append(entry.stop - entry.start)

        return tuple(shape)

    @staticmethod
    def _ravel_index(
        indices: Sequence[int],
        shape: Sequence[int],
    ) -> int:
        flat_index = 0

        for index, extent in zip(indices, shape):
            flat_index = flat_index * extent + index

        return flat_index

    def _build_coarse_to_fine_transforms(
        self,
        specification: QuantityHaloSpec,
        peer_plans: dict[int, list[CoarseToFineExchangePlan]],
        peer_boundaries: dict[int, list[Boundary]],
    ) -> dict[int, Coarse2FineHaloExchangeTransform]:
        exchange_transforms: dict[int, Coarse2FineHaloExchangeTransform] = {}

        i_axis, j_axis = self._horizontal_axes(specification.dims)

        for peer_rank, plans in peer_plans.items():
            boundaries = peer_boundaries[peer_rank]

            if len(boundaries) != len(plans):
                raise RuntimeError("peer boundary/plan count mismatch")

            if self.is_parent_rank:
                windows = tuple(
                    boundary.send_slice(specification) for boundary in boundaries
                )

                aggregated_source_indices: list[int] = []
                coarse_offset = 0

                for plan, window in zip(plans, windows):
                    coarse_shape = self._window_shape(window)

                    if (
                        coarse_shape[i_axis] != plan.coarse_extent[0]
                        or coarse_shape[j_axis] != plan.coarse_extent[1]
                    ):
                        raise RuntimeError(
                            "planned coarse extent does not match "
                            "the coarse Quantity window"
                        )

                    horizontal_fine_size = plan.fine_extent[0] * plan.fine_extent[1]

                    if len(plan.source_indices) != horizontal_fine_size:
                        raise RuntimeError(
                            "coarse-to-fine plan has inconsistent "
                            "source-index count: "
                            f"{len(plan.source_indices)} != "
                            f"{horizontal_fine_size}"
                        )

                    fine_shape = list(coarse_shape)
                    fine_shape[i_axis] = plan.fine_extent[0]
                    fine_shape[j_axis] = plan.fine_extent[1]

                    for fine_indices in np.ndindex(*fine_shape):
                        fine_i = fine_indices[i_axis]
                        fine_j = fine_indices[j_axis]

                        horizontal_fine_index = fine_i * plan.fine_extent[1] + fine_j
                        horizontal_source_index = plan.source_indices[
                            horizontal_fine_index
                        ]

                        coarse_i = horizontal_source_index // plan.coarse_extent[1]
                        coarse_j = horizontal_source_index % plan.coarse_extent[1]

                        coarse_indices = list(fine_indices)
                        coarse_indices[i_axis] = coarse_i
                        coarse_indices[j_axis] = coarse_j

                        source_index = self._ravel_index(
                            coarse_indices,
                            coarse_shape,
                        )

                        aggregated_source_indices.append(coarse_offset + source_index)

                    coarse_offset += math.prod(coarse_shape)

                transport_size = len(aggregated_source_indices)

                exchange_transforms[peer_rank] = Coarse2FineHaloExchangeTransform(
                    role="coarse",
                    transport_size=transport_size,
                    windows=windows,
                    prolongation=IndexedProlongation(aggregated_source_indices),
                )

            else:
                windows = tuple(
                    boundary.recv_slice(specification) for boundary in boundaries
                )

                transport_size = 0

                for plan, window in zip(plans, windows):
                    fine_shape = self._window_shape(window)

                    if (
                        fine_shape[i_axis] != plan.fine_extent[0]
                        or fine_shape[j_axis] != plan.fine_extent[1]
                    ):
                        raise RuntimeError(
                            "planned fine extent does not match "
                            "the fine Quantity window"
                        )

                    transport_size += math.prod(fine_shape)

                exchange_transforms[peer_rank] = Coarse2FineHaloExchangeTransform(
                    role="fine",
                    transport_size=transport_size,
                    windows=windows,
                )

        return exchange_transforms

    def coarse_to_fine(
        self,
        coarse_quantity: Quantity | None,
        fine_quantity: Quantity | None,
        n_points: int,
    ) -> None:
        """Perform transformed parent-to-nested halo exchanges.

        Exchange geometry is derived from Quantity dimensions and parent/nested
        global extents rather than from a named grid staggering.
        """
        if n_points <= 0:
            raise ValueError("n_points must be positive")

        anchor_parent_rank = self.nested_partitioner.mapping.parent_rank
        if anchor_parent_rank >= self.parent_size:
            raise ValueError(
                f"NestMapping parent_rank={anchor_parent_rank} is outside "
                f"parent communicator of size {self.parent_size}"
            )

        parent_dims, parent_tile_extent = self._parent_quantity_geometry(
            coarse_quantity,
            anchor_parent_rank,
        )
        fine_dims, fine_global_extent = self._nested_quantity_geometry(fine_quantity)

        if parent_dims != fine_dims:
            raise ValueError(
                "parent and nested quantities must have matching dimensions: "
                f"parent={parent_dims}, nested={fine_dims}"
            )

        dims = parent_dims

        boundaries, peer_plans, peer_boundaries = (
            self._collect_coarse_to_fine_exchanges(
                parent_tile_extent=parent_tile_extent,
                fine_global_extent=fine_global_extent,
                dims=dims,
                n_points=n_points,
            )
        )

        # Advance the world-communicator tag on every rank, including parent
        # ranks which do not participate in this particular nested exchange.
        tag = self._get_halo_tag()

        if not boundaries:
            return

        if self.is_parent_rank:
            if coarse_quantity is None:
                raise ValueError(
                    "coarse_quantity is required on a parent rank "
                    "participating in a nested exchange"
                )
            quantity = coarse_quantity
        else:
            if fine_quantity is None:
                raise ValueError("fine_quantity is required on nested ranks")
            quantity = fine_quantity

        if tuple(quantity.dims) != dims:
            raise ValueError(
                "local Quantity dimensions do not match the planned exchange: "
                f"expected {dims}, got {quantity.dims}"
            )

        specification = self._quantity_halo_spec(
            quantity,
            n_points=n_points,
        )

        exchange_transforms = self._build_coarse_to_fine_transforms(
            specification=specification,
            peer_plans=peer_plans,
            peer_boundaries=peer_boundaries,
        )

        updater = HaloUpdater.from_scalar_specifications(
            comm=self,
            numpy_like_module=self._maybe_force_cpu(quantity.np),
            specifications=[specification],
            boundaries=boundaries,
            tag=tag,
            optional_timer=self.timer,
            exchange_transforms=exchange_transforms,
        )

        updater.force_finalize_on_wait()
        updater.update([quantity])

    def update_nested_boundaries(
        self,
        coarse_quantity: Quantity | None,
        fine_quantity: Quantity | None,
        n_points: int,
    ) -> None:
        """Update fine-grid internal halos and external coarse boundaries.

        Fine-to-fine halo communication is completed before coarse-to-fine
        values are written into the external nested-grid halos.
        """
        self.update_nested_halo(
            fine_quantity=fine_quantity,
            n_points=n_points,
        )

        self.coarse_to_fine(
            coarse_quantity=coarse_quantity,
            fine_quantity=fine_quantity,
            n_points=n_points,
        )

    def _get_halo_tag(self) -> int:
        self._last_halo_tag += 1
        return self._last_halo_tag

    def __getattr__(self, name: str):
        """Delegate parent-domain operations on parent ranks."""
        parent_communicator = object.__getattribute__(
            self,
            "parent_communicator",
        )

        if parent_communicator is None:
            raise AttributeError(
                f"Nested world rank {self.world_rank} cannot access "
                f"parent communicator attribute {name!r}."
            )

        return getattr(parent_communicator, name)

    def _device_synchronize(self) -> None:
        Communicator._device_synchronize()

    def _maybe_force_cpu(self, module: ModuleType) -> ModuleType:
        if self._force_cpu:
            return np
        return module

    def _quantity_halo_spec(
        self,
        quantity: Quantity,
        n_points: int,
    ) -> QuantityHaloSpec:
        data = quantity[:]

        return QuantityHaloSpec(
            n_points=n_points,
            shape=quantity.shape,
            strides=data.strides,
            itemsize=data.itemsize,
            origin=quantity.origin,
            extent=quantity.extent,
            dims=quantity.dims,
            numpy_module=self._maybe_force_cpu(quantity.np),
            dtype=quantity.metadata.dtype,
        )
