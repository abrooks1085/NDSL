"""End-to-end nested-grid communication example for NDSL.

This example constructs a cubed-sphere parent domain and a refined nested patch,
then exercises both parts of the nested boundary update:

1. same-resolution halo exchange between neighboring nested ranks;
2. coarse-to-fine exchange from parent ranks into external nested halos.

The parent and nested grids use independent NDSL sizing/QuantityFactory contexts.
A higher-resolution parent grid is used only as a physical reference for the
nested grid metrics; it is not involved in the coarse-to-fine communication.

With the configuration below the example requires 28 MPI ranks: 24 parent
ranks (six tiles with a 2 x 2 layout) and four nested ranks (a 2 x 2 layout).
Run it with ``mpirun -np 28 python test_nested_grid.py``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

import ndsl.constants as constants
from ndsl import (
    Backend,
    CubedSpherePartitioner,
    QuantityFactory,
    SubtileGridSizer,
    TilePartitioner,
)
from ndsl.comm.communicator import (
    CubedSphereCommunicator,
    NestedCommunicator,
    NestTileCommunicator,
)
from ndsl.comm.mpi import MPIComm
from ndsl.comm.partitioner import NestedPartitioner, NestMapping
from ndsl.constants import (
    I_DIM,
    I_INTERFACE_DIM,
    J_DIM,
    J_INTERFACE_DIM,
    N_HALO_DEFAULT,
)
from ndsl.grid import MetricTerms
from ndsl.quantity import Quantity

# Parent cubed-sphere configuration.
PARENT_NX = 12
PARENT_NY = 12
PARENT_LAYOUT = (2, 2)

# Nested patch geometry, expressed in parent-grid cells.
NEST_PARENT_REGIONS = {
    0: 0,
    1: 1,
}
NEST_PARENT_RANKS = {
    0: 0,
    1: 4,
}
NEST_PARENT_STARTS = {
    0: (3, 3),
    1: (3, 3),
}
NEST_PARENT_EXTENTS = {
    0: (4, 4),
    1: (4, 4),
}
REFINEMENT_RATIO = 2
NESTED_LAYOUT = (2, 2)

NEST_IDS = (0, 1)

# Use the normal NDSL metric halo width so the example exercises a realistic
# multi-point halo rather than a one-point special case.
N_HALO = N_HALO_DEFAULT
NZ = 1

MetricPayload = dict[str, Any]
ReferenceMetricData = dict[str, MetricPayload]
HorizontalDims = tuple[str, str]


@dataclass
class NestedPhysicalGrid:
    """Physical grid data local to one nested rank.

    Coordinate fields are retained as NumPy arrays while metric fields use NDSL
    Quantity objects. This keeps the example focused on nested communication
    without introducing a separate coordinate-container abstraction.
    """

    agrid_lon_lat: np.ndarray
    dgrid_lon_lat: np.ndarray
    area: Quantity
    dx: Quantity
    dy: Quantity
    dxa: Quantity
    dya: Quantity
    dxc: Quantity
    dyc: Quantity


@dataclass
class NestedGridContext:
    """NDSL objects owned by one nested rank."""

    quantity_factory: QuantityFactory
    physical_grid: NestedPhysicalGrid


def make_parent_metric_terms(
    communicator: NestedCommunicator,
    backend: Backend,
    nx_tile: int,
    ny_tile: int,
) -> MetricTerms:
    """Construct ordinary cubed-sphere MetricTerms on the parent ranks.

    MetricTerms remains unaware of nesting. Parent ranks use the existing parent
    communicator, while nested ranks later construct an independent grid context.
    """

    assert communicator.is_parent_rank
    assert communicator.parent_comm is not None

    sizer = SubtileGridSizer.from_tile_params(
        nx_tile=nx_tile,
        ny_tile=ny_tile,
        nz=NZ,
        n_halo=N_HALO,
        layout=PARENT_LAYOUT,
        tile_partitioner=communicator.parent_partitioner.tile,
        tile_rank=communicator.parent_comm.tile.rank,
        backend=backend,
    )
    quantity_factory = QuantityFactory(sizer=sizer, backend=backend)

    return MetricTerms(
        quantity_factory=quantity_factory,
        communicator=communicator.parent_comm,
        grid_type=0,
    )


def materialize_metric_terms(metrics: MetricTerms, names: tuple[str, ...]) -> None:
    """Force lazy MetricTerms fields to be generated collectively."""

    for name in names:
        _ = getattr(metrics, name).view[:]


def check_physical_refinement(
    communicator: NestedCommunicator,
    coarse_metrics: MetricTerms | None,
    fine_metrics: MetricTerms | None,
    nest_id: int,
) -> None:
    """Verify that the reference fine grid refines the selected parent region."""

    if not communicator.is_parent_rank:
        return

    assert communicator.parent_comm is not None
    assert coarse_metrics is not None
    assert fine_metrics is not None

    partitioner = communicator.nested_partitioners[nest_id]
    mapping = partitioner.mapping

    tile_comm = communicator.parent_comm.tile

    # Reconstruct complete coarse and fine-reference tiles on each tile root.
    # Only parent rank zero (the root of the selected tile) uses the gathered
    # arrays below, but every rank in the tile must participate in the gathers.
    coarse_dgrid_quantity = tile_comm.gather(coarse_metrics.dgrid_lon_lat)
    fine_dgrid_quantity = tile_comm.gather(fine_metrics.dgrid_lon_lat)
    coarse_area_quantity = tile_comm.gather(coarse_metrics.area)
    fine_area_quantity = tile_comm.gather(fine_metrics.area)
    coarse_dx_quantity = tile_comm.gather(coarse_metrics.dx)
    coarse_dy_quantity = tile_comm.gather(coarse_metrics.dy)
    fine_dx_quantity = tile_comm.gather(fine_metrics.dx)
    fine_dy_quantity = tile_comm.gather(fine_metrics.dy)

    parent_partitioner = communicator.parent_partitioner

    if not isinstance(parent_partitioner, CubedSpherePartitioner):
        raise TypeError("nested-grid example requires a CubedSpherePartitioner parent")

    parent_root_rank = parent_partitioner.tile_root_rank(mapping.parent_rank)

    if communicator.world_rank != parent_root_rank:
        return

    assert coarse_dgrid_quantity is not None
    assert fine_dgrid_quantity is not None
    assert coarse_area_quantity is not None
    assert fine_area_quantity is not None
    assert coarse_dx_quantity is not None
    assert coarse_dy_quantity is not None
    assert fine_dx_quantity is not None
    assert fine_dy_quantity is not None

    coarse_dgrid = np.asarray(coarse_dgrid_quantity.view[:])
    fine_dgrid = np.asarray(fine_dgrid_quantity.view[:])
    coarse_area = np.asarray(coarse_area_quantity.view[:])
    fine_area = np.asarray(fine_area_quantity.view[:])
    coarse_dx = np.asarray(coarse_dx_quantity.view[:])
    coarse_dy = np.asarray(coarse_dy_quantity.view[:])
    fine_dx = np.asarray(fine_dx_quantity.view[:])
    fine_dy = np.asarray(fine_dy_quantity.view[:])

    ci0, cj0 = mapping.parent_start
    ci1 = ci0 + mapping.parent_extent[0]
    cj1 = cj0 + mapping.parent_extent[1]

    refinement = mapping.refinement_ratio

    fi0 = ci0 * refinement
    fj0 = cj0 * refinement
    fi1 = ci1 * refinement
    fj1 = cj1 * refinement

    # The corners of the selected coarse region must coincide with the
    # corresponding points on the higher-resolution reference grid.
    coarse_corners = np.asarray(
        [
            coarse_dgrid[ci0, cj0, :],
            coarse_dgrid[ci1, cj0, :],
            coarse_dgrid[ci0, cj1, :],
            coarse_dgrid[ci1, cj1, :],
        ]
    )
    fine_corners = np.asarray(
        [
            fine_dgrid[fi0, fj0, :],
            fine_dgrid[fi1, fj0, :],
            fine_dgrid[fi0, fj1, :],
            fine_dgrid[fi1, fj1, :],
        ]
    )
    corner_error = float(np.max(np.abs(coarse_corners - fine_corners)))

    coarse_patch_area = float(np.sum(coarse_area[ci0:ci1, cj0:cj1]))
    fine_patch_area = float(np.sum(fine_area[fi0:fi1, fj0:fj1]))
    area_relative_error = abs(fine_patch_area - coarse_patch_area) / coarse_patch_area

    # dx uses (I_DIM, J_INTERFACE_DIM), while dy uses
    # (I_INTERFACE_DIM, J_DIM). Include the extra interface point when slicing
    # each metric so the coarse/fine averages cover the same physical region.
    coarse_dx_patch = coarse_dx[ci0:ci1, cj0 : cj1 + 1]
    fine_dx_patch = fine_dx[fi0:fi1, fj0 : fj1 + 1]
    coarse_dy_patch = coarse_dy[ci0 : ci1 + 1, cj0:cj1]
    fine_dy_patch = fine_dy[fi0 : fi1 + 1, fj0:fj1]

    mean_coarse_dx = float(np.mean(coarse_dx_patch))
    mean_fine_dx = float(np.mean(fine_dx_patch))
    mean_coarse_dy = float(np.mean(coarse_dy_patch))
    mean_fine_dy = float(np.mean(fine_dy_patch))
    dx_ratio = mean_coarse_dx / mean_fine_dx
    dy_ratio = mean_coarse_dy / mean_fine_dy

    if corner_error > 1.0e-8:
        raise AssertionError(
            "Fine reference grid does not align with the selected coarse-grid region: "
            f"maximum corner mismatch is {corner_error:.6e} rad."
        )
    if area_relative_error > 1.0e-4:
        raise AssertionError(
            "Fine reference grid does not preserve the selected coarse-grid area: "
            f"relative mismatch is {area_relative_error:.6e}."
        )

    # Cubed-sphere spacing is nonuniform, so require consistency with the
    # refinement ratio rather than exact equality at every point.
    if not np.isclose(dx_ratio, REFINEMENT_RATIO, rtol=0.10):
        raise AssertionError(
            f"dx refinement ratio is {dx_ratio}, expected approximately "
            f"{REFINEMENT_RATIO}."
        )
    if not np.isclose(dy_ratio, REFINEMENT_RATIO, rtol=0.10):
        raise AssertionError(
            f"dy refinement ratio is {dy_ratio}, expected approximately "
            f"{REFINEMENT_RATIO}."
        )

    print(
        "\nPhysical refinement check:\n"
        f"  max corner mismatch   = {corner_error:.6e} rad\n"
        f"  relative area mismatch = {area_relative_error:.6e}\n"
        f"  coarse/fine dx ratio   = {dx_ratio:.6f}\n"
        f"  coarse/fine dy ratio   = {dy_ratio:.6f}\n"
        f"  expected ratio          = {REFINEMENT_RATIO}\n"
        "  physical refinement    = PASSED",
        flush=True,
    )


def reference_quantity_payload(quantity: Quantity) -> MetricPayload:
    """Copy one gathered MetricTerms field into a serializable payload."""

    return {
        "data": np.asarray(quantity.view[:]).copy(),
        "dims": tuple(quantity.dims),
        "units": quantity.units,
    }


def get_reference_metric_data(
    communicator: NestedCommunicator,
    fine_metrics: MetricTerms | None,
    nest_id: int,
) -> ReferenceMetricData | None:
    """Gather the fine-reference tile used to construct nested grid metrics."""

    if not communicator.is_parent_rank:
        return None

    assert communicator.parent_comm is not None
    assert fine_metrics is not None

    partitioner = communicator.nested_partitioners[nest_id]
    mapping = partitioner.mapping

    tile_comm = communicator.parent_comm.tile
    names = (
        "agrid_lon_lat",
        "dgrid_lon_lat",
        "area",
        "dx",
        "dy",
        "dxa",
        "dya",
        "dxc",
        "dyc",
    )

    parent_partitioner = communicator.parent_partitioner

    if not isinstance(parent_partitioner, CubedSpherePartitioner):
        raise TypeError("nested-grid example requires a CubedSpherePartitioner parent")

    parent_root_rank = parent_partitioner.tile_root_rank(mapping.parent_rank)

    reference: ReferenceMetricData = {}
    for name in names:
        tile_quantity = tile_comm.gather(getattr(fine_metrics, name))
        if communicator.world_rank == parent_root_rank:
            assert tile_quantity is not None
            reference[name] = reference_quantity_payload(tile_quantity)

    return reference if communicator.world_rank == parent_root_rank else None


def horizontal_global_extent(
    dims: HorizontalDims,
    nx: int,
    ny: int,
) -> tuple[int, int]:
    """Return the global horizontal extent implied by NDSL staggering."""

    extent = tuple(
        size + (1 if dim in constants.INTERFACE_DIMS else 0)
        for dim, size in zip(dims, (nx, ny))
    )
    return extent[0], extent[1]


def nested_rank_global_slice(
    *,
    partitioner: NestedPartitioner,
    nested_rank: int,
    dims: HorizontalDims,
) -> tuple[slice, slice]:
    """Return one nested rank's compute-domain slice in nested-global indices."""

    fine_nx, fine_ny = partitioner.mapping.fine_extent
    rank_slice = partitioner.subtile_slice(
        rank=nested_rank,
        global_dims=dims,
        global_extent=horizontal_global_extent(dims, fine_nx, fine_ny),
        overlap=True,
    )
    i_slice, j_slice = rank_slice

    if not isinstance(i_slice, slice) or not isinstance(j_slice, slice):
        raise TypeError(f"Expected horizontal slices, got {rank_slice}")

    return i_slice, j_slice


def copy_reference_metric_to_nested_quantity(
    *,
    payload: MetricPayload,
    quantity_factory: QuantityFactory,
    partitioner: NestedPartitioner,
    nested_rank: int,
) -> Quantity:
    """Copy one 2-D reference metric field into a nested rank Quantity."""

    source = payload["data"]
    dims = tuple(payload["dims"])
    units = payload["units"]

    if len(dims) != 2:
        raise ValueError(f"Expected a 2-D metric field, got dims={dims}")

    horizontal_dims: HorizontalDims = (dims[0], dims[1])
    quantity = quantity_factory.zeros(
        dims=horizontal_dims,
        units=units,
        dtype=float,
    )

    i_slice, j_slice = nested_rank_global_slice(
        partitioner=partitioner,
        nested_rank=nested_rank,
        dims=horizontal_dims,
    )
    if i_slice.start is None or j_slice.start is None:
        raise RuntimeError("Nested rank metric slice must have bounded starts")

    rank_i0 = i_slice.start
    rank_j0 = j_slice.start
    fine_parent_i0 = (
        partitioner.mapping.parent_start[0] * partitioner.mapping.refinement_ratio
    )
    fine_parent_j0 = (
        partitioner.mapping.parent_start[1] * partitioner.mapping.refinement_ratio
    )

    origin_i, origin_j = quantity.origin
    extent_i, extent_j = quantity.extent

    # Copy the compute domain and its logical halo from the fine reference tile.
    # Negative nested-local coordinates are valid here because the selected nest
    # lies away from the parent-tile boundary in this example.
    for data_i in range(origin_i - N_HALO, origin_i + extent_i + N_HALO):
        nested_i = rank_i0 + data_i - origin_i
        source_i = fine_parent_i0 + nested_i

        for data_j in range(origin_j - N_HALO, origin_j + extent_j + N_HALO):
            nested_j = rank_j0 + data_j - origin_j
            source_j = fine_parent_j0 + nested_j
            quantity[data_i, data_j] = source[source_i, source_j]

    return quantity


def extract_reference_coordinates(
    *,
    payload: MetricPayload,
    partitioner: NestedPartitioner,
    nested_rank: int,
) -> np.ndarray:
    """Extract lon/lat coordinates for one nested rank, including its halo."""

    source = payload["data"]
    dims = tuple(payload["dims"])
    if len(dims) != 3:
        raise ValueError(
            "Expected coordinates with two horizontal dimensions plus lon/lat, "
            f"got dims={dims}"
        )

    horizontal_dims: HorizontalDims = (dims[0], dims[1])
    i_slice, j_slice = nested_rank_global_slice(
        partitioner=partitioner,
        nested_rank=nested_rank,
        dims=horizontal_dims,
    )
    if (
        i_slice.start is None
        or i_slice.stop is None
        or j_slice.start is None
        or j_slice.stop is None
    ):
        raise RuntimeError("Nested coordinate slice must be bounded")

    local_extent_i = i_slice.stop - i_slice.start
    local_extent_j = j_slice.stop - j_slice.start
    output = np.empty(
        (
            local_extent_i + 2 * N_HALO,
            local_extent_j + 2 * N_HALO,
            source.shape[2],
        ),
        dtype=source.dtype,
    )

    fine_parent_i0 = (
        partitioner.mapping.parent_start[0] * partitioner.mapping.refinement_ratio
    )
    fine_parent_j0 = (
        partitioner.mapping.parent_start[1] * partitioner.mapping.refinement_ratio
    )

    for local_i in range(-N_HALO, local_extent_i + N_HALO):
        source_i = fine_parent_i0 + i_slice.start + local_i
        output_i = local_i + N_HALO

        for local_j in range(-N_HALO, local_extent_j + N_HALO):
            source_j = fine_parent_j0 + j_slice.start + local_j
            output_j = local_j + N_HALO
            output[output_i, output_j, :] = source[source_i, source_j, :]

    return output


def make_nested_grid_context(
    communicator: NestedCommunicator,
    backend: Backend,
    reference: ReferenceMetricData,
    nest_id: int,
) -> NestedGridContext:
    """Construct an independent physical grid context on one nested rank."""

    nested_rank = communicator.nested_rank(nest_id)
    assert nested_rank is not None

    partitioner = communicator.nested_partitioners[nest_id]
    fine_nx, fine_ny = partitioner.mapping.fine_extent

    sizer = SubtileGridSizer.from_tile_params(
        nx_tile=fine_nx,
        ny_tile=fine_ny,
        nz=NZ,
        n_halo=N_HALO,
        layout=partitioner.layout,
        tile_partitioner=partitioner.tile,
        tile_rank=nested_rank,
        backend=backend,
    )
    quantity_factory = QuantityFactory(sizer=sizer, backend=backend)

    physical_grid = NestedPhysicalGrid(
        agrid_lon_lat=extract_reference_coordinates(
            payload=reference["agrid_lon_lat"],
            partitioner=partitioner,
            nested_rank=nested_rank,
        ),
        dgrid_lon_lat=extract_reference_coordinates(
            payload=reference["dgrid_lon_lat"],
            partitioner=partitioner,
            nested_rank=nested_rank,
        ),
        area=copy_reference_metric_to_nested_quantity(
            payload=reference["area"],
            quantity_factory=quantity_factory,
            partitioner=partitioner,
            nested_rank=nested_rank,
        ),
        dx=copy_reference_metric_to_nested_quantity(
            payload=reference["dx"],
            quantity_factory=quantity_factory,
            partitioner=partitioner,
            nested_rank=nested_rank,
        ),
        dy=copy_reference_metric_to_nested_quantity(
            payload=reference["dy"],
            quantity_factory=quantity_factory,
            partitioner=partitioner,
            nested_rank=nested_rank,
        ),
        dxa=copy_reference_metric_to_nested_quantity(
            payload=reference["dxa"],
            quantity_factory=quantity_factory,
            partitioner=partitioner,
            nested_rank=nested_rank,
        ),
        dya=copy_reference_metric_to_nested_quantity(
            payload=reference["dya"],
            quantity_factory=quantity_factory,
            partitioner=partitioner,
            nested_rank=nested_rank,
        ),
        dxc=copy_reference_metric_to_nested_quantity(
            payload=reference["dxc"],
            quantity_factory=quantity_factory,
            partitioner=partitioner,
            nested_rank=nested_rank,
        ),
        dyc=copy_reference_metric_to_nested_quantity(
            payload=reference["dyc"],
            quantity_factory=quantity_factory,
            partitioner=partitioner,
            nested_rank=nested_rank,
        ),
    )

    return NestedGridContext(
        quantity_factory=quantity_factory,
        physical_grid=physical_grid,
    )


def check_nested_physical_grid(
    communicator: NestedCommunicator,
    context: NestedGridContext | None,
) -> None:
    """Check that nested metrics are finite and physically meaningful."""

    if not communicator.is_nested_rank:
        return

    assert context is not None
    grid = context.physical_grid

    for name in ("area", "dx", "dy"):
        values = np.asarray(getattr(grid, name).view[:])
        if not np.all(np.isfinite(values)):
            raise AssertionError(f"Nested {name} contains non-finite values")
        if not np.all(values > 0.0):
            raise AssertionError(f"Nested {name} contains non-positive values")

    if not np.all(np.isfinite(grid.agrid_lon_lat)):
        raise AssertionError("Nested A-grid coordinates contain non-finite values")
    if not np.all(np.isfinite(grid.dgrid_lon_lat)):
        raise AssertionError("Nested D-grid coordinates contain non-finite values")


def print_nested_grid_summary(
    communicator: NestedCommunicator,
    context: NestedGridContext | None,
    nest_id: int,
) -> None:
    """Print a compact physical-grid summary from each nested rank."""

    nested_rank = communicator.nested_rank(nest_id)
    if nested_rank is None:
        return

    assert context is not None
    grid = context.physical_grid
    center_lon = grid.agrid_lon_lat[..., 0]
    center_lat = grid.agrid_lon_lat[..., 1]

    print(
        f"nest {nest_id}, nested rank {nested_rank}: "
        f"mean area={np.mean(np.asarray(grid.area.view[:])):.6e} m^2, "
        f"mean dx={np.mean(np.asarray(grid.dx.view[:])):.6e} m, "
        f"mean dy={np.mean(np.asarray(grid.dy.view[:])):.6e} m, "
        f"lon=[{np.degrees(np.min(center_lon)):.3f}, "
        f"{np.degrees(np.max(center_lon)):.3f}] deg, "
        f"lat=[{np.degrees(np.min(center_lat)):.3f}, "
        f"{np.degrees(np.max(center_lat)):.3f}] deg",
        flush=True,
    )


def make_parent_quantity(
    communicator: NestedCommunicator,
    backend: Backend,
    dims: HorizontalDims,
) -> Quantity:
    """Allocate one parent test Quantity with the requested staggering."""

    assert communicator.is_parent_rank
    assert communicator.parent_comm is not None

    sizer = SubtileGridSizer.from_tile_params(
        nx_tile=PARENT_NX,
        ny_tile=PARENT_NY,
        nz=NZ,
        n_halo=N_HALO,
        layout=PARENT_LAYOUT,
        tile_partitioner=communicator.parent_partitioner.tile,
        tile_rank=communicator.parent_comm.tile.rank,
        backend=backend,
    )
    return QuantityFactory(sizer=sizer, backend=backend).zeros(
        dims=dims,
        units="1",
        dtype=float,
    )


def make_nested_quantity(
    context: NestedGridContext,
    dims: HorizontalDims,
) -> Quantity:
    """Allocate one nested test Quantity with the requested staggering."""

    return context.quantity_factory.zeros(dims=dims, units="1", dtype=float)


def initialize_parent_quantity(
    quantity: Quantity,
    communicator: NestedCommunicator,
    nest_id: int,
) -> None:
    """Fill tile zero with values that encode global horizontal coordinates."""

    assert communicator.is_parent_rank
    parent_rank = communicator.parent_rank
    assert parent_rank is not None

    parent_partitioner = communicator.parent_partitioner
    parent_tile = parent_partitioner.tile
    tile_index = parent_partitioner.tile_index(parent_rank)

    mapping = communicator.nested_partitioners[nest_id].mapping

    # Only the selected parent tile supplies this nest. Distinct values on the
    # other tiles make accidental cross-tile communication immediately visible.
    if tile_index != mapping.parent_region:
        quantity.view[:] = -100.0 - parent_rank
        return

    dims = tuple(quantity.dims)
    if len(dims) != 2:
        raise ValueError(f"Expected a 2-D horizontal Quantity, got dims={dims}")

    tile_rank = parent_rank % parent_tile.total_ranks
    rank_slice = parent_tile.subtile_slice(
        rank=tile_rank,
        global_dims=dims,
        global_extent=horizontal_global_extent(
            (dims[0], dims[1]),
            PARENT_NX,
            PARENT_NY,
        ),
        overlap=True,
    )
    i_slice, j_slice = rank_slice

    if not isinstance(i_slice, slice) or not isinstance(j_slice, slice):
        raise TypeError(f"Expected horizontal slices, got {rank_slice}")
    if (
        i_slice.start is None
        or i_slice.stop is None
        or j_slice.start is None
        or j_slice.stop is None
    ):
        raise RuntimeError("Parent rank slice must be bounded")

    for local_i, global_i in enumerate(range(i_slice.start, i_slice.stop)):
        for local_j, global_j in enumerate(range(j_slice.start, j_slice.stop)):
            quantity.view[local_i, local_j] = 1000.0 + 100.0 * global_i + global_j


def initialize_nested_quantity(quantity: Quantity, nested_rank: int) -> None:
    """Fill each nested compute domain with a rank-identifying value."""

    quantity.view[:] = 10.0 * (nested_rank + 1)


def logical_data_with_halo(quantity: Quantity, n_points: int) -> np.ndarray:
    """Return the 2-D compute domain plus ``n_points`` of logical halo."""

    origin_i, origin_j = quantity.origin
    extent_i, extent_j = quantity.extent
    return np.asarray(
        quantity[
            origin_i - n_points : origin_i + extent_i + n_points,
            origin_j - n_points : origin_j + extent_j + n_points,
        ]
    )


def print_nested_quantity_domains(
    communicator: NestedCommunicator,
    fine_quantity: Quantity | None,
    label: str,
    nest_id: int,
) -> None:
    """Print nested compute and halo data in rank order.

    Serializing the output keeps the MPI diagnostics readable. The logical
    compute+halo view makes the two communication paths visible: internal
    halos come from neighboring nested ranks, while exterior halos come from
    the parent domain.
    """

    nested_rank = communicator.nested_rank(nest_id)
    partitioner = communicator.nested_partitioners[nest_id]

    for rank_to_print in range(communicator.nested_size(nest_id)):
        communicator.comm.Barrier()

        if nested_rank is None:
            continue

        if nested_rank != rank_to_print:
            continue

        assert fine_quantity is not None
        j, i = partitioner.subtile_index(nested_rank)
        logical_data = logical_data_with_halo(fine_quantity, N_HALO)
        compute_data = np.asarray(fine_quantity.view[:])

        print(
            "\n"
            "------------------------------------------------------------\n"
            f"nested Quantity after communication: {label}\n"
            "------------------------------------------------------------\n"
            f"world rank           = {communicator.world_rank}\n"
            f"nest id              = {nest_id}\n"
            f"nested rank          = {nested_rank}\n"
            f"nested position      = (j={j}, i={i})\n"
            f"dims                 = {tuple(fine_quantity.dims)}\n"
            f"origin               = {fine_quantity.origin}\n"
            f"compute extent       = {fine_quantity.extent}\n"
            f"logical halo shape   = {logical_data.shape}\n"
            f"compute view shape   = {compute_data.shape}\n"
            "\nLOGICAL COMPUTE + HALO DOMAIN:\n"
            f"{logical_data}\n"
            "\nCOMPUTE DOMAIN (view[:]):\n"
            f"{compute_data}\n"
            "------------------------------------------------------------",
            flush=True,
        )

    communicator.comm.Barrier()


def print_configuration(communicator: NestedCommunicator) -> None:
    """Print the nested decomposition once from world rank zero."""

    if communicator.world_rank != 0:
        return

    print(
        "\nNDSL nested-grid example\n"
        f"  parent tile size       = {PARENT_NX} x {PARENT_NY}\n"
        f"  parent layout/tile     = {PARENT_LAYOUT}\n"
        f"  halo width             = {N_HALO}\n"
        f"  parent ranks           = {communicator.parent_size}\n"
    )

    for nest_id in NEST_IDS:
        partitioner = communicator.nested_partitioners[nest_id]
        mapping = partitioner.mapping
        fine_nx, fine_ny = mapping.fine_extent

        print(
            f"\n  nest {nest_id}\n"
            f"    parent tile_rank       ="
            f"{mapping.parent_region}/{mapping.parent_rank}\n"
            f"    nest parent start      = {mapping.parent_start}\n"
            f"    nest parent extent     = {mapping.parent_extent}\n"
            f"    refinement ratio       = {mapping.refinement_ratio}\n"
            f"    nested fine extent     = {fine_nx} x {fine_ny}\n"
            f"    nested layout          = {partitioner.layout}\n"
            f"    nested ranks           = "
            f"{communicator.nested_size(nest_id)}/n"
            f"    world ranks            = "
            f"{communicator.nested_world_ranks[nest_id]}",
            flush=True,
        )


def print_rank_mapping(communicator: NestedCommunicator) -> None:
    """Print each world's rank role in the nested decomposition."""

    for rank_to_print in range(communicator.world_size):
        communicator.comm.Barrier()

        if communicator.world_rank != rank_to_print:
            continue

        if communicator.is_parent_rank:
            parent_rank = communicator.parent_rank
            assert parent_rank is not None
            tile = communicator.parent_partitioner.tile_index(parent_rank)
            print(
                f"world rank {communicator.world_rank}: "
                f"parent rank {parent_rank}, parent tile {tile}",
                flush=True,
            )
        else:
            if len(communicator.nested_comms) != 1:
                raise RuntimeError(
                    "test expects each nested rank to belong to one nest"
                )

            nest_id = next(iter(communicator.nested_comms))

            nested_rank = communicator.nested_rank(nest_id)
            assert nested_rank is not None
            partitioner = communicator.nested_partitioners[nest_id]
            j, i = partitioner.subtile_index(nested_rank)
            print(
                f"world rank {communicator.world_rank}: "
                f"nest {nest_id}, nested rank {nested_rank}, "
                f"position=(j={j}, i={i})",
                flush=True,
            )

    communicator.comm.Barrier()


def check_configuration(communicator: NestedCommunicator) -> None:
    """Check that communicator sizes and nest mapping match this example."""

    expected_parent_size = 6 * PARENT_LAYOUT[0] * PARENT_LAYOUT[1]
    expected_nested_size = NESTED_LAYOUT[0] * NESTED_LAYOUT[1]

    assert communicator.parent_size == expected_parent_size

    for nest_id in NEST_IDS:
        partitioner = communicator.nested_partitioners[nest_id]
        mapping = partitioner.mapping

        assert mapping.parent_region == NEST_PARENT_REGIONS[nest_id]
        assert mapping.parent_start == NEST_PARENT_STARTS[nest_id]
        assert mapping.parent_extent == NEST_PARENT_EXTENTS[nest_id]
        assert mapping.refinement_ratio == REFINEMENT_RATIO
        assert mapping.fine_extent == (
            NEST_PARENT_EXTENTS[nest_id][0] * REFINEMENT_RATIO,
            NEST_PARENT_EXTENTS[nest_id][1] * REFINEMENT_RATIO,
        )
        assert communicator.nested_size(nest_id) == expected_nested_size

    assert communicator.world_size == expected_parent_size + expected_nested_size * len(
        NEST_IDS
    )


def check_nested_result(
    communicator: NestedCommunicator,
    fine_quantity: Quantity | None,
    nest_id: int,
) -> None:
    """Validate internal fine halos and external parent-provided halos."""

    nested_rank = communicator.nested_rank(nest_id)

    if nested_rank is None:
        return

    assert fine_quantity is not None

    partitioner = communicator.nested_partitioners[nest_id]
    data = logical_data_with_halo(fine_quantity, N_HALO)
    expected_shape = (
        fine_quantity.extent[0] + 2 * N_HALO,
        fine_quantity.extent[1] + 2 * N_HALO,
    )
    assert data.shape == expected_shape

    own_value = 10.0 * (nested_rank + 1)
    np.testing.assert_allclose(
        data[N_HALO:-N_HALO, N_HALO:-N_HALO],
        own_value,
    )

    j, i = partitioner.subtile_index(nested_rank)
    ny, nx = partitioner.layout

    # Internal halo slabs come from neighboring nested ranks. External halo
    # slabs are overwritten by coarse-to-fine communication. Parent tile zero
    # was initialized with values >= 1000, making the two cases easy to tell
    # apart without duplicating the prolongation algorithm in this checker.
    if i > 0:
        west_value = 10.0 * nested_rank
        np.testing.assert_allclose(
            data[:N_HALO, N_HALO:-N_HALO],
            west_value,
        )
    else:
        assert np.all(data[:N_HALO, N_HALO:-N_HALO] >= 1000.0)

    if i < nx - 1:
        east_value = 10.0 * (nested_rank + 2)
        np.testing.assert_allclose(
            data[-N_HALO:, N_HALO:-N_HALO],
            east_value,
        )
    else:
        assert np.all(data[-N_HALO:, N_HALO:-N_HALO] >= 1000.0)

    if j > 0:
        south_rank = nested_rank - nx
        south_value = 10.0 * (south_rank + 1)
        np.testing.assert_allclose(
            data[N_HALO:-N_HALO, :N_HALO],
            south_value,
        )
    else:
        assert np.all(data[N_HALO:-N_HALO, :N_HALO] >= 1000.0)

    if j < ny - 1:
        north_rank = nested_rank + nx
        north_value = 10.0 * (north_rank + 1)
        np.testing.assert_allclose(
            data[N_HALO:-N_HALO, -N_HALO:],
            north_value,
        )
    else:
        assert np.all(data[N_HALO:-N_HALO, -N_HALO:] >= 1000.0)


def run_communication_case(
    *,
    communicator: NestedCommunicator,
    backend: Backend,
    nested_context: NestedGridContext | None,
    nest_id: int,
    label: str,
    dims: HorizontalDims,
) -> None:
    """Run one staggered nested-boundary communication case."""

    coarse_quantity: Quantity | None = None
    fine_quantity: Quantity | None = None

    if communicator.is_parent_rank:
        coarse_quantity = make_parent_quantity(
            communicator=communicator,
            backend=backend,
            dims=dims,
        )
        initialize_parent_quantity(coarse_quantity, communicator, nest_id)
    else:
        nested_rank = communicator.nested_rank(nest_id)

        if nested_rank is not None:
            assert nested_context is not None
            fine_quantity = make_nested_quantity(nested_context, dims)
            initialize_nested_quantity(fine_quantity, nested_rank)

    communicator.comm.Barrier()
    if communicator.world_rank == 0:
        print(f"\nTesting nested communication: {label}, dims={dims}", flush=True)

    # update_nested_boundaries performs the two required stages in order:
    # first fine-to-fine communication across internal nested boundaries, then
    # parent-to-fine communication across the exterior of the nested patch.
    communicator.update_nested_boundaries(
        nest_id=nest_id,
        coarse_quantity=coarse_quantity,
        fine_quantity=fine_quantity,
        n_points=N_HALO,
    )
    check_nested_result(communicator, fine_quantity, nest_id)
    print_nested_quantity_domains(
        communicator=communicator,
        fine_quantity=fine_quantity,
        label=label,
        nest_id=nest_id,
    )

    nested_rank = communicator.nested_rank(nest_id)

    if nested_rank is not None:
        assert fine_quantity is not None
        print(
            f"nested {nest_id}, nested rank  {nested_rank}: "
            f"{label} PASSED, extent={fine_quantity.extent}",
            flush=True,
        )

    communicator.comm.Barrier()


def main() -> None:
    world_comm = MPIComm()
    world_rank = world_comm.Get_rank()
    backend = Backend("st:numpy:cpu:IJK")

    parent_partitioner = CubedSpherePartitioner(
        tile=TilePartitioner(layout=PARENT_LAYOUT)
    )
    nested_partitioners = {
        0: NestedPartitioner(
            layout=NESTED_LAYOUT,
            mapping=NestMapping(
                parent_rank=NEST_PARENT_RANKS[0],
                parent_region=NEST_PARENT_REGIONS[0],
                parent_start=NEST_PARENT_STARTS[0],
                parent_extent=NEST_PARENT_EXTENTS[0],
                refinement_ratio=REFINEMENT_RATIO,
            ),
        ),
        1: NestedPartitioner(
            layout=NESTED_LAYOUT,
            mapping=NestMapping(
                parent_rank=NEST_PARENT_RANKS[1],
                parent_region=NEST_PARENT_REGIONS[1],
                parent_start=NEST_PARENT_STARTS[1],
                parent_extent=NEST_PARENT_EXTENTS[1],
                refinement_ratio=REFINEMENT_RATIO,
            ),
        ),
    }

    parent_size = parent_partitioner.total_ranks

    nested_world_ranks = {
        0: tuple(range(24, 28)),
        1: tuple(range(28, 32)),
    }

    world_rank = world_comm.Get_rank()
    is_parent_rank = world_rank < parent_size

    if world_rank < parent_size:
        color = 0
        local_nest_id: int | None = None
    elif world_rank in nested_world_ranks[0]:
        color = 1
        local_nest_id = 0
    elif world_rank in nested_world_ranks[1]:
        color = 2
        local_nest_id = 1

    raw_role_comm = world_comm.Split(color=color, key=world_rank)

    role_comm = copy.copy(world_comm)
    role_comm._comm = raw_role_comm

    parent_comm = None
    nested_comms: dict[int, NestTileCommunicator] = {}

    if is_parent_rank:
        parent_comm = CubedSphereCommunicator(
            comm=role_comm,
            partitioner=parent_partitioner,
        )
    else:
        assert local_nest_id is not None
        nested_comms[local_nest_id] = NestTileCommunicator(
            comm=role_comm,
            partitioner=nested_partitioners[local_nest_id],
        )

    communicator = NestedCommunicator(
        comm=world_comm,
        parent_partitioner=parent_partitioner,
        nested_partitioners=nested_partitioners,
        nested_world_ranks=nested_world_ranks,
        parent_comm=parent_comm,
        nested_comms=nested_comms,
    )

    check_configuration(communicator)
    print_configuration(communicator)
    print_rank_mapping(communicator)

    coarse_metric_terms: MetricTerms | None = None
    fine_reference_metric_terms: MetricTerms | None = None

    if communicator.is_parent_rank:
        coarse_metric_terms = make_parent_metric_terms(
            communicator=communicator,
            backend=backend,
            nx_tile=PARENT_NX,
            ny_tile=PARENT_NY,
        )
        fine_reference_metric_terms = make_parent_metric_terms(
            communicator=communicator,
            backend=backend,
            nx_tile=PARENT_NX * REFINEMENT_RATIO,
            ny_tile=PARENT_NY * REFINEMENT_RATIO,
        )

        # MetricTerms creates many fields lazily. Materialize the fields used by
        # this example while all parent ranks are available for their collectives.
        materialize_metric_terms(
            coarse_metric_terms,
            ("dgrid_lon_lat", "area", "dx", "dy"),
        )
        materialize_metric_terms(
            fine_reference_metric_terms,
            (
                "agrid_lon_lat",
                "dgrid_lon_lat",
                "area",
                "dx",
                "dy",
                "dxa",
                "dya",
                "dxc",
                "dyc",
            ),
        )

    for nest_id in NEST_IDS:
        check_physical_refinement(
            communicator=communicator,
            coarse_metrics=coarse_metric_terms,
            fine_metrics=fine_reference_metric_terms,
            nest_id=nest_id,
        )

    # Gather a higher-resolution physical reference tile and make it available
    # to the nested ranks. This broadcast is example setup, not part of the
    # nested boundary-exchange API being tested below.
    reference_data_by_nest = {}

    for nest_id in NEST_IDS:
        reference_data_by_nest[nest_id] = get_reference_metric_data(
            communicator=communicator,
            fine_metrics=fine_reference_metric_terms,
            nest_id=nest_id,
        )
        reference_data_by_nest[nest_id] = communicator.comm.bcast(
            reference_data_by_nest[nest_id],
            root=NEST_PARENT_RANKS[nest_id],
        )

    nested_context: NestedGridContext | None = None
    if communicator.is_nested_rank:
        assert local_nest_id is not None
        reference = reference_data_by_nest[local_nest_id]
        assert reference is not None

        nested_context = make_nested_grid_context(
            communicator=communicator,
            backend=backend,
            reference=reference,
            nest_id=local_nest_id,
        )

    check_nested_physical_grid(communicator, nested_context)

    if communicator.is_nested_rank:
        assert local_nest_id is not None
        print_nested_grid_summary(communicator, nested_context, local_nest_id)

    # A-grid cell centers plus the two horizontal interface staggerings cover
    # the scalar storage patterns needed by A-, C-, and D-grid fields without
    # hard-coding a named grid type into the nested communication API.
    staggerings: tuple[tuple[str, HorizontalDims], ...] = (
        ("A-grid", (I_DIM, J_DIM)),
        ("I-interface", (I_INTERFACE_DIM, J_DIM)),
        ("J-interface", (I_DIM, J_INTERFACE_DIM)),
    )
    for nest_id in NEST_IDS:
        for label, dims in staggerings:
            run_communication_case(
                communicator=communicator,
                backend=backend,
                nested_context=(nested_context if local_nest_id == nest_id else None),
                nest_id=nest_id,
                label=label,
                dims=dims,
            )

    if communicator.world_rank == 0:
        print("\nA/C/D nested-grid communication PASSED", flush=True)


if __name__ == "__main__":
    main()
