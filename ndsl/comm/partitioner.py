from __future__ import annotations

import abc
import copy
import functools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Self, TypeVar, cast

import f90nml
import numpy as np

import ndsl.constants as constants
from ndsl.comm import boundary as bd
from ndsl.comm._boundary_utils import boundary_at_start_of_dim
from ndsl.constants import (
    EAST,
    NORTH,
    NORTHEAST,
    NORTHWEST,
    SOUTH,
    SOUTHEAST,
    SOUTHWEST,
    WEST,
)
from ndsl.quantity import QuantityMetadata
from ndsl.utils import list_by_dims

# we're caching slice objects which are pretty small, and the number we
# generate depends on the number of different array shapes/sizes which
# should not be that many
DEFAULT_CACHE_SIZE = None

__all__ = [
    "TilePartitioner",
    "CubedSpherePartitioner",
    "NestedPartitioner",
    "NestMapping",
    "CoarseToFineExchangePlan",
    "get_tile_index",
]


def get_tile_index(rank: int, total_ranks: int) -> int:
    """
    Returns the zero-indexed tile number, given a rank and total number of ranks.
    """
    if total_ranks % 6 != 0:
        raise ValueError(f"total_ranks {total_ranks} is not evenly divisible by 6")
    ranks_per_tile = total_ranks // 6
    return rank // ranks_per_tile


@dataclass(frozen=True)
class NestMapping:
    """Describe the relationship between a nested patch and its parent domain.

    parent_rank
        Parent-communicator rank containing the nest anchor.

    parent_region
        Optional identifier for the parent region. For a cubed sphere this can
        be the parent tile index. NestedPartitioner does not interpret it.

    parent_start
        (i, j) index of the lower-left coarse cell covered by the nest.

    parent_extent
        Number of coarse cells covered by the nest in (i, j).

    refinement_ratio
        Integer fine/coarse refinement ratio.
    """

    parent_rank: int
    parent_start: tuple[int, int]
    parent_extent: tuple[int, int]
    refinement_ratio: int
    parent_region: int | None = None

    def __post_init__(self) -> None:
        if self.parent_rank < 0:
            raise ValueError("parent_rank must be non-negative")
        if self.parent_start[0] < 0 or self.parent_start[1] < 0:
            raise ValueError("parent_start must be non-negative")
        if self.parent_extent[0] <= 0 or self.parent_extent[1] <= 0:
            raise ValueError("parent_extent must be positive")
        if self.refinement_ratio <= 1:
            raise ValueError("refinement_ratio must be greater than one")

    @property
    def fine_extent(self) -> tuple[int, int]:
        """Nested compute-domain size in fine cells."""
        return (
            self.parent_extent[0] * self.refinement_ratio,
            self.parent_extent[1] * self.refinement_ratio,
        )


@dataclass(frozen=True)
class CoarseToFineExchangePlan:
    """Describe one parent-rank contribution to a nested boundary."""

    parent_rank: int
    nested_rank: int
    boundary_type: int

    # Relative to the parent-rank compute origin.
    coarse_start: tuple[int, int]
    coarse_extent: tuple[int, int]

    # Relative to the nested-rank compute origin. May reference halo storage.
    fine_start: tuple[int, int]
    fine_extent: tuple[int, int]

    # Nearest-neighbor source indices into the flattened coarse window.
    source_indices: tuple[int, ...]


class Partitioner(abc.ABC):
    tile: TilePartitioner
    layout: tuple[int, int]

    def __init__(
        self, tile: TilePartitioner, layout: tuple[int, int] | list[int]
    ) -> None:
        self.tile = tile
        if len(layout) != 2:
            raise ValueError(
                f"Expected layout to be a tuple/list of two integers. Got {layout} instead."
            )
        self.layout = tuple(layout)  # type: ignore[assignment]

    @abc.abstractmethod
    def boundary(self, boundary_type: int, rank: int) -> bd.SimpleBoundary | None: ...

    @abc.abstractmethod
    def tile_index(self, rank: int) -> int:
        pass

    @abc.abstractmethod
    def global_extent(self, rank_metadata: QuantityMetadata) -> tuple[int, ...]:
        """Return the shape of a full tile representation for the given dimensions.

        Args:
            rank_metadata: quantity metadata

        Returns:
            extent: shape of full tile representation
        """
        pass

    @abc.abstractmethod
    def subtile_slice(
        self,
        rank: int,
        global_dims: Sequence[str],
        global_extent: Sequence[int],
        overlap: bool = False,
    ) -> tuple[int | slice, ...]:
        """Return the subtile slice of a given rank on an array.

        Global refers to the domain being partitioned. For example, for a partitioning
        of a tile, the tile would be the "global" domain.

        Args:
            rank: the rank of the process
            global_dims: dimensions of the global quantity being partitioned
            global_extent: extent of the global quantity being partitioned
            overlap (optional): if True, for interface variables include the part
                of the array shared by adjacent ranks in both ranks. If False, ensure
                only one of those ranks (the greater rank) is assigned the overlapping
                section. Default is False.

        Returns:
            subtile_slice: the slice of the global compute domain corresponding
                to the subtile compute domain
        """
        pass

    @abc.abstractmethod
    def subtile_extent(self, metadata: QuantityMetadata, rank: int) -> tuple[int, ...]:
        """Return the shape of a single rank representation for the given dimensions.

        Args:
            metadata: quantity metadata.
            rank: rank of the process.

        Returns:
            extent: shape of a single rank representation for the given dimensions.
        """
        pass

    @property
    @abc.abstractmethod
    def total_ranks(self) -> int:
        pass


class TilePartitioner(Partitioner):
    def __init__(
        self,
        layout: tuple[int, int] | list[int],
        edge_interior_ratio: float = 1.0,
    ):
        """Create an object for fv3gfs tile decomposition."""
        self.edge_interior_ratio = edge_interior_ratio
        super().__init__(self, layout)

    def tile_index(self, rank: int) -> int:
        return 0

    @classmethod
    def from_namelist(cls, namelist: f90nml.Namelist) -> Self:
        """Initialize a TilePartitioner from a Fortran namelist.

        Args:
            namelist (dict): the Fortran namelist
        """
        return cls(layout=namelist["fv_core_nml"]["layout"])

    def subtile_index(self, rank: int) -> tuple[int, int]:
        """
        Return the (y, x) subtile position of a given rank
        as an integer number of subtiles.
        """
        return subtile_index(rank, self.total_ranks, self.layout)

    @property
    def total_ranks(self) -> int:
        return self.layout[0] * self.layout[1]

    def global_extent(self, rank_metadata: QuantityMetadata) -> tuple[int, ...]:
        """Return the shape of a full tile representation for the given dimensions.

        Args:
            rank_metadata: quantity metadata

        Returns:
            extent: shape of full tile representation
        """
        return tile_extent_from_rank_metadata(
            rank_metadata.dims, rank_metadata.extent, self.layout
        )

    def subtile_extent(self, metadata: QuantityMetadata, rank: int) -> tuple[int, ...]:
        """Return the shape of a single rank representation for the given dimensions.

        Args:
            metadata: quantity metadata.
            rank: rank of the process.

        Returns:
            extent: shape of a single rank representation for the given dimensions.
        """
        rank_slice = rank_slice_from_tile_metadata(
            metadata.dims,
            extent=metadata.extent,
            layout=self.layout,
            subtile_index=self.subtile_index(rank),
            edge_interior_ratio=self.edge_interior_ratio,
            overlap=True,
        )
        return tuple(item.stop - item.start for item in rank_slice)

    def subtile_slice(
        self,
        rank: int,
        global_dims: Sequence[str],
        global_extent: Sequence[int],
        overlap: bool = False,
    ) -> tuple[slice, ...]:
        """Return the subtile slice of a given rank on an array.

        Global refers to the domain being partitioned. For example, for a partitioning
        of a tile, the tile would be the "global" domain.

        Args:
            rank: the rank of the process
            global_dims: dimensions of the global quantity being partitioned
            global_extent: extent of the global quantity being partitioned
            overlap (optional): if True, for interface variables include the part
                of the array shared by adjacent ranks in both ranks. If False, ensure
                only one of those ranks (the greater rank) is assigned the overlapping
                section. Default is False.

        Returns:
            subtile_slice: the slice of the global compute domain corresponding
                to the subtile compute domain
        """
        return subtile_slice(
            dims=global_dims,
            global_extent=global_extent,
            layout=self.layout,
            subtile_index=self.subtile_index(rank),
            edge_interior_ratio=self.edge_interior_ratio,
            overlap=overlap,
        )

    def on_tile_top(self, rank: int) -> bool:
        return on_tile_top(self.subtile_index(rank), self.layout)

    def on_tile_bottom(self, rank: int) -> bool:
        return on_tile_bottom(self.subtile_index(rank))

    def on_tile_left(self, rank: int) -> bool:
        return on_tile_left(self.subtile_index(rank))

    def on_tile_right(self, rank: int) -> bool:
        return on_tile_right(self.subtile_index(rank), self.layout)

    def boundary(self, boundary_type: int, rank: int) -> bd.SimpleBoundary | None:
        """Returns a boundary of the requested type for a given rank.

        Target ranks will be on the same tile as the given rank, wrapping around as
        in a doubly-periodic boundary condition.

        Args:
            boundary_type: the type of boundary
            rank: the processor rank

        Returns:
            boundary
        """
        boundary = copy.copy(self._cached_boundary(boundary_type, rank))
        return boundary

    @functools.lru_cache(maxsize=DEFAULT_CACHE_SIZE)
    def _cached_boundary(
        self, boundary_type: int, rank: int
    ) -> bd.SimpleBoundary | None:
        boundary = {
            WEST: self._left_edge,
            EAST: self._right_edge,
            NORTH: self._top_edge,
            SOUTH: self._bottom_edge,
            NORTHWEST: self._top_left_corner,
            NORTHEAST: self._top_right_corner,
            SOUTHWEST: self._bottom_left_corner,
            SOUTHEAST: self._bottom_right_corner,
        }[boundary_type](rank)
        return boundary

    def _left_edge(self, rank: int) -> bd.SimpleBoundary:
        if self.on_tile_left(rank):
            to_rank = rank + self.layout[1] - 1
        else:
            to_rank = rank - 1
        return bd.SimpleBoundary(
            boundary_type=constants.WEST,
            from_rank=rank,
            to_rank=to_rank,
            n_clockwise_rotations=0,
        )

    def _right_edge(self, rank: int) -> bd.SimpleBoundary:
        if self.on_tile_right(rank):
            to_rank = rank - self.layout[1] + 1
        else:
            to_rank = rank + 1
        return bd.SimpleBoundary(
            boundary_type=constants.EAST,
            from_rank=rank,
            to_rank=to_rank,
            n_clockwise_rotations=0,
        )

    def _top_edge(self, rank: int) -> bd.SimpleBoundary:
        if self.on_tile_top(rank):
            to_rank = rank - (self.layout[0] - 1) * self.layout[1]
        else:
            to_rank = rank + self.layout[1]
        return bd.SimpleBoundary(
            boundary_type=constants.NORTH,
            from_rank=rank,
            to_rank=to_rank,
            n_clockwise_rotations=0,
        )

    def _bottom_edge(self, rank: int) -> bd.SimpleBoundary:
        if self.on_tile_bottom(rank):
            to_rank = rank + (self.layout[0] - 1) * self.layout[1]
        else:
            to_rank = rank - self.layout[1]
        return bd.SimpleBoundary(
            boundary_type=constants.SOUTH,
            from_rank=rank,
            to_rank=to_rank,
            n_clockwise_rotations=0,
        )

    def _top_left_corner(self, rank: int) -> bd.SimpleBoundary | None:
        return _get_corner(constants.NORTHWEST, rank, self._left_edge, self._top_edge)

    def _top_right_corner(self, rank: int) -> bd.SimpleBoundary | None:
        return _get_corner(constants.NORTHEAST, rank, self._right_edge, self._top_edge)

    def _bottom_left_corner(self, rank: int) -> bd.SimpleBoundary | None:
        return _get_corner(
            constants.SOUTHWEST, rank, self._left_edge, self._bottom_edge
        )

    def _bottom_right_corner(self, rank: int) -> bd.SimpleBoundary | None:
        return _get_corner(
            constants.SOUTHEAST, rank, self._right_edge, self._bottom_edge
        )

    def fliplr_rank(self, rank: int) -> int:
        return fliplr_subtile_rank(rank, self.layout)

    def rotate_rank(self, rank: int, n_clockwise_rotations: int) -> int:
        return rotate_subtile_rank(rank, self.layout, n_clockwise_rotations)


def _get_corner(
    boundary_type: int,
    rank: int,
    edge_func_1: Callable[[int], bd.Boundary],
    edge_func_2: Callable[[int], bd.Boundary],
) -> bd.SimpleBoundary:
    edge_1 = edge_func_1(rank)
    edge_2 = edge_func_2(edge_1.to_rank)
    rotations = edge_1.n_clockwise_rotations + edge_2.n_clockwise_rotations
    return bd.SimpleBoundary(
        boundary_type=boundary_type,
        from_rank=rank,
        to_rank=edge_2.to_rank,
        n_clockwise_rotations=rotations,
    )


class CubedSpherePartitioner(Partitioner):
    def __init__(self, tile: TilePartitioner):
        """Create an object for fv3gfs cubed-sphere domain decomposition.

        Args:
            tile: partitioner for the cube faces
        """
        if not isinstance(tile, TilePartitioner):
            raise TypeError("tile must be a TilePartitioner")
        super().__init__(tile, tile.layout)

    @classmethod
    def from_namelist(cls, namelist: f90nml.Namelist) -> Self:
        """Initialize a CubedSpherePartitioner from a Fortran namelist.

        Args:
            namelist (dict): the Fortran namelist
        """
        return cls(TilePartitioner.from_namelist(namelist))

    def _ensure_square_layout(self) -> None:
        if not self.tile.layout[0] == self.tile.layout[1]:
            raise NotImplementedError("currently only square layouts are supported")

    def tile_index(self, rank: int) -> int:
        """Returns the tile index of a given rank"""
        return get_tile_index(rank, self.total_ranks)

    def tile_root_rank(self, rank: int) -> int:
        """Returns the lowest rank on the same tile as a given rank."""
        return self.tile.total_ranks * (rank // self.tile.total_ranks)

    @property
    def total_ranks(self) -> int:
        """the number of ranks on the cubed sphere"""
        return 6 * self.tile.total_ranks

    def boundary(self, boundary_type: int, rank: int) -> bd.SimpleBoundary | None:
        """Returns a boundary of the requested type for a given rank, or None.

        On tile corners, the boundary across that corner does not exist.

        Args:
            boundary_type: the type of boundary
            rank: the processor rank

        Returns:
            boundary
        """
        boundary = copy.copy(self._cached_boundary(boundary_type, rank))
        return boundary

    @functools.lru_cache(maxsize=DEFAULT_CACHE_SIZE)
    def _cached_boundary(
        self, boundary_type: int, rank: int
    ) -> bd.SimpleBoundary | None:
        boundary = {
            WEST: self._left_edge,
            EAST: self._right_edge,
            NORTH: self._top_edge,
            SOUTH: self._bottom_edge,
            NORTHWEST: self._top_left_corner,
            NORTHEAST: self._top_right_corner,
            SOUTHWEST: self._bottom_left_corner,
            SOUTHEAST: self._bottom_right_corner,
        }[boundary_type](rank)
        if boundary is not None:
            boundary.to_rank = boundary.to_rank % self.total_ranks
        return boundary

    def _left_edge(self, rank: int) -> bd.SimpleBoundary:
        self._ensure_square_layout()
        if self.tile.on_tile_left(rank):
            if is_even(self.tile_index(rank)):
                to_root_rank = self.tile_root_rank(rank - 2 * self.tile.total_ranks)
                tile_rank = rank % self.tile.total_ranks
                to_tile_rank = self.tile.fliplr_rank(
                    self.tile.rotate_rank(tile_rank, 1)
                )
                to_rank = to_root_rank + to_tile_rank
                rotations = 1
                boundary = bd.SimpleBoundary(
                    boundary_type=constants.WEST,
                    from_rank=rank,
                    to_rank=to_rank,
                    n_clockwise_rotations=rotations,
                )
            else:
                boundary = cast(bd.SimpleBoundary, self.tile.boundary(WEST, rank=rank))
                boundary.to_rank -= self.tile.total_ranks
        else:
            boundary = cast(bd.SimpleBoundary, self.tile.boundary(WEST, rank=rank))
        return boundary

    def _right_edge(self, rank: int) -> bd.SimpleBoundary:
        self._ensure_square_layout()
        if self.tile.on_tile_right(rank):
            if not is_even(self.tile_index(rank)):
                to_root_rank = self.tile_root_rank(rank + 2 * self.tile.total_ranks)
                tile_rank = rank % self.tile.total_ranks
                to_tile_rank = self.tile.fliplr_rank(
                    self.tile.rotate_rank(tile_rank, 1)
                )
                boundary = bd.SimpleBoundary(
                    boundary_type=constants.EAST,
                    from_rank=rank,
                    to_rank=to_root_rank + to_tile_rank,
                    n_clockwise_rotations=1,
                )
            else:
                boundary = cast(bd.SimpleBoundary, self.tile.boundary(EAST, rank=rank))
                boundary.to_rank += self.tile.total_ranks
        else:
            boundary = cast(bd.SimpleBoundary, self.tile.boundary(EAST, rank=rank))
        return boundary

    def _top_edge(self, rank: int) -> bd.SimpleBoundary:
        self._ensure_square_layout()
        if self.tile.on_tile_top(rank):
            if is_even(self.tile_index(rank)):
                to_root_rank = (self.tile_index(rank) + 2) * self.tile.total_ranks
                tile_rank = rank % self.tile.total_ranks
                to_tile_rank = self.tile.fliplr_rank(
                    self.tile.rotate_rank(tile_rank, 1)
                )
                boundary = bd.SimpleBoundary(
                    boundary_type=constants.NORTH,
                    from_rank=rank,
                    to_rank=to_root_rank + to_tile_rank,
                    n_clockwise_rotations=3,
                )
            else:
                boundary = cast(bd.SimpleBoundary, self.tile.boundary(NORTH, rank))
                boundary.to_rank += self.tile.total_ranks
        else:
            boundary = cast(bd.SimpleBoundary, self.tile.boundary(NORTH, rank=rank))
        return boundary

    def _bottom_edge(self, rank: int) -> bd.SimpleBoundary:
        self._ensure_square_layout()
        if self.tile.on_tile_bottom(rank) and not is_even(self.tile_index(rank)):
            to_root_rank = (self.tile_index(rank) - 2) * self.tile.total_ranks
            tile_rank = rank % self.tile.total_ranks
            to_tile_rank = self.tile.fliplr_rank(self.tile.rotate_rank(tile_rank, 1))
            boundary = bd.SimpleBoundary(
                boundary_type=constants.SOUTH,
                from_rank=rank,
                to_rank=to_root_rank + to_tile_rank,
                n_clockwise_rotations=3,
            )
        else:
            boundary = cast(bd.SimpleBoundary, self.tile.boundary(SOUTH, rank=rank))
            if self.tile.on_tile_bottom(rank):
                boundary.to_rank -= self.tile.total_ranks
        return boundary

    def _top_left_corner(self, rank: int) -> bd.SimpleBoundary | None:
        if self.tile.on_tile_top(rank) and self.tile.on_tile_left(rank):
            corner = None
        else:
            if is_even(self.tile_index(rank)) and on_tile_left(
                self.tile.subtile_index(rank)
            ):
                second_edge = self._left_edge
            else:
                second_edge = self._top_edge
            corner = self._get_corner(
                constants.NORTHWEST, rank, self._left_edge, second_edge
            )
        return corner

    def _top_right_corner(self, rank: int) -> bd.SimpleBoundary | None:
        if on_tile_top(self.tile.subtile_index(rank), self.layout) and on_tile_right(
            self.tile.subtile_index(rank), self.layout
        ):
            corner = None
        else:
            if is_even(self.tile_index(rank)) and on_tile_top(
                self.tile.subtile_index(rank), self.layout
            ):
                second_edge = self._bottom_edge
            else:
                second_edge = self._right_edge
            corner = self._get_corner(
                constants.NORTHEAST, rank, self._top_edge, second_edge
            )
        return corner

    def _bottom_left_corner(self, rank: int) -> bd.SimpleBoundary | None:
        if on_tile_bottom(self.tile.subtile_index(rank)) and on_tile_left(
            self.tile.subtile_index(rank)
        ):
            corner = None
        else:
            if not is_even(self.tile_index(rank)) and on_tile_bottom(
                self.tile.subtile_index(rank)
            ):
                second_edge = self._top_edge
            else:
                second_edge = self._left_edge
            corner = self._get_corner(
                constants.SOUTHWEST, rank, self._bottom_edge, second_edge
            )
        return corner

    def _bottom_right_corner(self, rank: int) -> bd.SimpleBoundary | None:
        if on_tile_bottom(self.tile.subtile_index(rank)) and on_tile_right(
            self.tile.subtile_index(rank), self.layout
        ):
            corner = None
        else:
            if not is_even(self.tile_index(rank)) and on_tile_bottom(
                self.tile.subtile_index(rank)
            ):
                second_edge = self._bottom_edge
            else:
                second_edge = self._right_edge
            corner = self._get_corner(
                constants.SOUTHEAST, rank, self._bottom_edge, second_edge
            )
        return corner

    def _get_corner(
        self,
        boundary_type: int,
        rank: int,
        edge_func_1: Callable[[int], bd.Boundary],
        edge_func_2: Callable[[int], bd.Boundary],
    ) -> bd.SimpleBoundary:
        edge_1 = edge_func_1(rank)
        edge_2 = edge_func_2(edge_1.to_rank)
        rotations = edge_1.n_clockwise_rotations + edge_2.n_clockwise_rotations
        return bd.SimpleBoundary(
            boundary_type=boundary_type,
            from_rank=rank,
            to_rank=edge_2.to_rank,
            n_clockwise_rotations=rotations,
        )

    def global_extent(self, rank_metadata: QuantityMetadata) -> tuple[int, ...]:
        """Return the shape of a full cube representation for the given dimensions.

        Args:
            rank_metadata: quantity metadata

        Returns:
            extent: shape of full cube representation
        """
        return (6,) + tile_extent_from_rank_metadata(
            rank_metadata.dims, rank_metadata.extent, self.layout
        )

    def subtile_extent(self, metadata: QuantityMetadata, rank: int) -> tuple[int, ...]:
        """Return the shape of a single rank representation for the given dimensions.

        Args:
            metadata: quantity metadata.
            rank: rank of the process.

        Returns:
            extent: shape of a single rank representation for the given dimensions.
        """

        return self.tile.subtile_extent(metadata, rank)

    def subtile_slice(
        self,
        rank: int,
        global_dims: Sequence[str],
        global_extent: Sequence[int],
        overlap: bool = False,
    ) -> tuple[int | slice, ...]:
        """Return the subtile slice of a given rank on an array.

        Global refers to the domain being partitioned. For example, for a partitioning
        of a tile, the tile would be the "global" domain.

        Args:
            rank: the rank of the process
            global_dims: dimensions of the global quantity being partitioned
            global_extent: extent of the global quantity being partitioned
            overlap (optional): if True, for interface variables include the part
                of the array shared by adjacent ranks in both ranks. If False, ensure
                only one of those ranks (the greater rank) is assigned the overlapping
                section. Default is False.

        Returns:
            subtile_slice: the tuple slice of the global compute domain corresponding
                to the subtile compute domain
        """
        if global_dims[0] != constants.TILE_DIM:
            raise NotImplementedError(
                "currently only supports tile dimension {constants.TILE_DIM} as the "
                "first dimension, got dims {cube_metadata.dims}"
            )
        i_tile = self.tile_index(rank)
        return (i_tile,) + self.tile.subtile_slice(
            rank=rank,
            global_dims=global_dims[1:],
            global_extent=global_extent[1:],
            overlap=overlap,
        )


class NestedPartitioner(Partitioner):
    """Partition a single non-periodic nested grid region.

    Internal boundaries connect nested ranks. Boundaries on the exterior of
    the nested region return None and are supplied by coarse-to-fine updates.
    """

    def __init__(
        self,
        layout: tuple[int, int],
        mapping: NestMapping,
    ) -> None:
        self.mapping = mapping

        # Reuse TilePartitioner decomposition without its periodic topology.
        super().__init__(
            tile=TilePartitioner(layout),
            layout=layout,
        )

    @property
    def total_ranks(self) -> int:
        return self.layout[0] * self.layout[1]

    @property
    def fine_extent(self) -> tuple[int, int]:
        return self.mapping.fine_extent

    def tile_index(self, rank: int) -> int:
        """Return the logical region index for a nested rank."""
        if rank < 0 or rank >= self.total_ranks:
            raise ValueError(
                f"rank {rank} is outside nested communicator "
                f"of size {self.total_ranks}"
            )
        return 0

    def subtile_index(self, rank: int) -> tuple[int, int]:
        return self.tile.subtile_index(rank)

    def global_extent(
        self,
        rank_metadata: QuantityMetadata,
    ) -> tuple[int, ...]:
        return self.tile.global_extent(rank_metadata)

    def subtile_extent(
        self,
        metadata: QuantityMetadata,
        rank: int,
    ) -> tuple[int, ...]:
        return self.tile.subtile_extent(metadata, rank)

    def subtile_slice(
        self,
        rank: int,
        global_dims: Sequence[str],
        global_extent: Sequence[int],
        overlap: bool = False,
    ) -> tuple[int | slice, ...]:
        return self.tile.subtile_slice(
            rank=rank,
            global_dims=global_dims,
            global_extent=global_extent,
            overlap=overlap,
        )

    def boundary(
        self,
        boundary_type: int,
        rank: int,
    ) -> bd.SimpleBoundary | None:
        """Return a fine-to-fine boundary or None at the nest exterior."""
        j, i = self.subtile_index(rank)
        ny, nx = self.layout

        offsets = {
            WEST: (-1, 0),
            EAST: (1, 0),
            SOUTH: (0, -1),
            NORTH: (0, 1),
            SOUTHWEST: (-1, -1),
            SOUTHEAST: (1, -1),
            NORTHWEST: (-1, 1),
            NORTHEAST: (1, 1),
        }
        di, dj = offsets[boundary_type]

        neighbor_i = i + di
        neighbor_j = j + dj

        if neighbor_i < 0 or neighbor_i >= nx or neighbor_j < 0 or neighbor_j >= ny:
            return None

        return bd.SimpleBoundary(
            boundary_type=boundary_type,
            from_rank=rank,
            to_rank=neighbor_j * nx + neighbor_i,
            n_clockwise_rotations=0,
        )

    def is_external_boundary(
        self,
        boundary_type: int,
        rank: int,
    ) -> bool:
        return self.boundary(boundary_type, rank) is None

    def external_boundary_types(
        self,
        rank: int,
    ) -> tuple[int, ...]:
        return tuple(
            boundary_type
            for boundary_type in constants.BOUNDARY_TYPES
            if self.is_external_boundary(boundary_type, rank)
        )

    @staticmethod
    def _horizontal_axes(
        dims: Sequence[str],
    ) -> tuple[int, int]:
        i_axes = [index for index, dim in enumerate(dims) if dim in constants.I_DIMS]
        j_axes = [index for index, dim in enumerate(dims) if dim in constants.J_DIMS]

        if len(i_axes) != 1 or len(j_axes) != 1:
            raise ValueError(
                "coarse-to-fine exchange requires exactly one I dimension "
                f"and one J dimension, got {tuple(dims)}"
            )

        return i_axes[0], j_axes[0]

    def coarse_to_fine_exchange_plans(
        self,
        parent_partitioner: Partitioner,
        parent_tile_extent: tuple[int, ...],
        fine_global_extent: tuple[int, ...],
        dims: tuple[str, ...],
        boundary_type: int,
        rank: int,
        n_points: int,
    ) -> tuple[CoarseToFineExchangePlan, ...]:
        """Describe parent-rank contributions to one external nested boundary.

        Grid staggering is determined from the Quantity dimensions and supplied
        extents rather than from a named A-, C-, or D-grid type.
        """
        if not self.is_external_boundary(boundary_type, rank):
            raise ValueError(
                f"boundary type {boundary_type} on nested rank {rank} "
                "is not external"
            )

        if n_points <= 0:
            raise ValueError("n_points must be positive")

        i_axis, j_axis = self._horizontal_axes(dims)

        refinement = self.mapping.refinement_ratio
        parent_i0, parent_j0 = self.mapping.parent_start

        # overlap=True preserves shared interface points in the actual
        # fine-grid compute domain.
        fine_slice = self.subtile_slice(
            rank=rank,
            global_dims=dims,
            global_extent=fine_global_extent,
            overlap=True,
        )

        fine_i_slice = fine_slice[i_axis]
        fine_j_slice = fine_slice[j_axis]

        if not isinstance(fine_i_slice, slice):
            raise TypeError(f"expected horizontal slice, got {fine_i_slice}")
        if not isinstance(fine_j_slice, slice):
            raise TypeError(f"expected horizontal slice, got {fine_j_slice}")

        fi0 = fine_i_slice.start
        fi1 = fine_i_slice.stop
        fj0 = fine_j_slice.start
        fj1 = fine_j_slice.stop

        if None in (fi0, fi1, fj0, fj1):
            raise ValueError(f"bounded fine slices required, got {fine_slice}")

        assert fi0 is not None
        assert fi1 is not None
        assert fj0 is not None
        assert fj1 is not None

        def target_indices(
            start: int,
            stop: int,
            at_start: bool | None,
        ) -> list[int]:
            if at_start is True:
                return list(range(start - n_points, start))
            if at_start is False:
                return list(range(stop, stop + n_points))
            return list(range(start, stop))

        fine_i = target_indices(
            fi0,
            fi1,
            boundary_at_start_of_dim(
                boundary_type,
                dims[i_axis],
            ),
        )
        fine_j = target_indices(
            fj0,
            fj1,
            boundary_at_start_of_dim(
                boundary_type,
                dims[j_axis],
            ),
        )

        # Map each fine-grid coordinate to its order-zero parent coordinate.
        def parent_index(
            fine_index: int,
            dim: str,
            parent_start: int,
        ) -> int:
            offset = 0.0 if dim in constants.INTERFACE_DIMS else 0.5
            return parent_start + math.floor((fine_index + offset) / refinement)

        coarse_i = [
            parent_index(
                fine_index=i,
                dim=dims[i_axis],
                parent_start=parent_i0,
            )
            for i in fine_i
        ]
        coarse_j = [
            parent_index(
                fine_index=j,
                dim=dims[j_axis],
                parent_start=parent_j0,
            )
            for j in fine_j
        ]

        if isinstance(parent_partitioner, CubedSpherePartitioner):
            parent_tile = parent_partitioner.tile
            tile_root_rank = parent_partitioner.tile_root_rank(self.mapping.parent_rank)
        elif isinstance(parent_partitioner, TilePartitioner):
            parent_tile = parent_partitioner
            tile_root_rank = 0
        else:
            raise TypeError(
                "coarse-to-fine overlap planning currently supports "
                "TilePartitioner or CubedSpherePartitioner parents, got "
                f"{type(parent_partitioner)}"
            )

        plans: list[CoarseToFineExchangePlan] = []
        covered_fine_points = 0

        # overlap=False assigns each shared parent interface point to one
        # communication owner.
        for parent_tile_rank in range(parent_tile.total_ranks):
            parent_slice = parent_tile.subtile_slice(
                rank=parent_tile_rank,
                global_dims=dims,
                global_extent=parent_tile_extent,
                overlap=False,
            )

            parent_i_slice = parent_slice[i_axis]
            parent_j_slice = parent_slice[j_axis]

            assert isinstance(parent_i_slice, slice)
            assert isinstance(parent_j_slice, slice)
            assert parent_i_slice.start is not None
            assert parent_i_slice.stop is not None
            assert parent_j_slice.start is not None
            assert parent_j_slice.stop is not None

            i_positions = [
                k
                for k, coarse_index in enumerate(coarse_i)
                if parent_i_slice.start <= coarse_index < parent_i_slice.stop
            ]
            j_positions = [
                k
                for k, coarse_index in enumerate(coarse_j)
                if parent_j_slice.start <= coarse_index < parent_j_slice.stop
            ]

            if not i_positions or not j_positions:
                continue

            expected_i = list(range(i_positions[0], i_positions[-1] + 1))
            expected_j = list(range(j_positions[0], j_positions[-1] + 1))

            if i_positions != expected_i or j_positions != expected_j:
                raise RuntimeError(
                    "parent overlap is not a contiguous rectangular window"
                )

            ip0 = i_positions[0]
            ip1 = i_positions[-1] + 1
            jp0 = j_positions[0]
            jp1 = j_positions[-1] + 1

            selected_fine_i = fine_i[ip0:ip1]
            selected_fine_j = fine_j[jp0:jp1]
            selected_coarse_i = coarse_i[ip0:ip1]
            selected_coarse_j = coarse_j[jp0:jp1]

            coarse_global_i0 = min(selected_coarse_i)
            coarse_global_i1 = max(selected_coarse_i) + 1
            coarse_global_j0 = min(selected_coarse_j)
            coarse_global_j1 = max(selected_coarse_j) + 1

            coarse_start = (
                coarse_global_i0 - parent_i_slice.start,
                coarse_global_j0 - parent_j_slice.start,
            )
            coarse_extent = (
                coarse_global_i1 - coarse_global_i0,
                coarse_global_j1 - coarse_global_j0,
            )

            fine_start = (
                selected_fine_i[0] - fi0,
                selected_fine_j[0] - fj0,
            )
            fine_extent = (
                len(selected_fine_i),
                len(selected_fine_j),
            )

            coarse_nj = coarse_extent[1]
            source_indices: list[int] = []

            for coarse_index_i in selected_coarse_i:
                local_i = coarse_index_i - coarse_global_i0

                for coarse_index_j in selected_coarse_j:
                    local_j = coarse_index_j - coarse_global_j0
                    source_indices.append(local_i * coarse_nj + local_j)

            plans.append(
                CoarseToFineExchangePlan(
                    parent_rank=(tile_root_rank + parent_tile_rank),
                    nested_rank=rank,
                    boundary_type=boundary_type,
                    coarse_start=coarse_start,
                    coarse_extent=coarse_extent,
                    fine_start=fine_start,
                    fine_extent=fine_extent,
                    source_indices=tuple(source_indices),
                )
            )

            covered_fine_points += fine_extent[0] * fine_extent[1]

        expected_fine_points = len(fine_i) * len(fine_j)

        if covered_fine_points != expected_fine_points:
            raise ValueError(
                "nested halo is not completely covered by "
                "the selected parent tile: "
                f"covered {covered_fine_points} of "
                f"{expected_fine_points} points"
            )

        return tuple(plans)

    def coarse_to_fine_boundaries(
        self,
        parent_partitioner: Partitioner,
        parent_tile_extent: tuple[int, ...],
        fine_global_extent: tuple[int, ...],
        dims: tuple[str, ...],
        boundary_type: int,
        nested_rank: int,
        nested_world_rank: int,
        n_points: int,
    ) -> tuple[
        tuple[
            CoarseToFineExchangePlan,
            bd.NestedBoundary,
            bd.NestedBoundary,
        ],
        ...,
    ]:
        plans = self.coarse_to_fine_exchange_plans(
            parent_partitioner=parent_partitioner,
            parent_tile_extent=parent_tile_extent,
            fine_global_extent=fine_global_extent,
            dims=dims,
            boundary_type=boundary_type,
            rank=nested_rank,
            n_points=n_points,
        )

        result: list[
            tuple[
                CoarseToFineExchangePlan,
                bd.NestedBoundary,
                bd.NestedBoundary,
            ]
        ] = []

        for plan in plans:
            coarse_boundary = bd.NestedBoundary(
                from_rank=plan.parent_rank,
                to_rank=nested_world_rank,
                n_clockwise_rotations=0,
                window_start=plan.coarse_start,
                window_extent=plan.coarse_extent,
                comm_type=bd.CommType.SEND_ONLY,
            )
            fine_boundary = bd.NestedBoundary(
                from_rank=nested_world_rank,
                to_rank=plan.parent_rank,
                n_clockwise_rotations=0,
                window_start=plan.fine_start,
                window_extent=plan.fine_extent,
                comm_type=bd.CommType.RECV_ONLY,
            )

            result.append((plan, coarse_boundary, fine_boundary))

        return tuple(result)

    def on_tile_bottom(self, rank: int) -> bool:
        return self.tile.on_tile_bottom(rank)

    def on_tile_top(self, rank: int) -> bool:
        return self.tile.on_tile_top(rank)

    def on_tile_left(self, rank: int) -> bool:
        return self.tile.on_tile_left(rank)

    def on_tile_right(self, rank: int) -> bool:
        return self.tile.on_tile_right(rank)


def on_tile_left(subtile_index: tuple[int, int]) -> bool:
    return subtile_index[1] == 0


def on_tile_right(subtile_index: tuple[int, int], layout: tuple[int, int]) -> bool:
    return subtile_index[1] == layout[1] - 1


def on_tile_top(subtile_index: tuple[int, int], layout: tuple[int, int]) -> bool:
    return subtile_index[0] == layout[0] - 1


def on_tile_bottom(subtile_index: tuple[int, int]) -> bool:
    return subtile_index[0] == 0


def rotate_subtile_rank(
    rank: int, layout: tuple[int, int], n_clockwise_rotations: int
) -> int:
    """Returns the rank position where this rank would be if you rotated the
    tile n_clockwise_rotations times.
    """
    if n_clockwise_rotations == 0:
        to_tile_rank = rank
    elif n_clockwise_rotations == 1:
        total_ranks = layout[0] * layout[1]
        rank_array = np.arange(total_ranks).reshape(layout)
        rotated_rank_array = np.rot90(rank_array)
        to_tile_rank = rank_array[np.where(rotated_rank_array == rank)][0]
    else:
        raise NotImplementedError()
    return to_tile_rank


def transpose_subtile_rank(rank: int, layout: tuple[int, int]) -> int:
    """Returns the rank position where this rank would be if you transposed
    the tile.
    """
    return transform_subtile_rank(np.transpose, rank, layout)


def fliplr_subtile_rank(rank: int, layout: tuple[int, int]) -> int:
    """Returns the rank position where this rank would be if you flipped the
    tile along a vertical axis
    """
    return transform_subtile_rank(np.fliplr, rank, layout)


def flipud_subtile_rank(rank: int, layout: tuple[int, int]) -> int:
    """Returns the rank position where this rank would be if you flipped the
    tile along a horizontal axis
    """
    return transform_subtile_rank(np.flipud, rank, layout)


def transform_subtile_rank(
    transform_func: Callable[[np.ndarray], np.ndarray],
    rank: int,
    layout: tuple[int, int],
) -> int:
    """Returns the rank position where this rank would be if you performed
    a transformation on the tile which strictly moves ranks.
    """
    total_ranks = layout[0] * layout[1]
    rank_array = np.arange(total_ranks).reshape(layout)
    transformed_rank_array = transform_func(rank_array)
    return rank_array[np.where(transformed_rank_array == rank)][0]


def subtile_index(
    rank: int, ranks_per_tile: int, layout: tuple[int, int]
) -> tuple[int, int]:
    within_tile_rank = rank % ranks_per_tile
    j = within_tile_rank // layout[1]
    i = within_tile_rank % layout[1]
    return j, i


def is_even(value: int | float) -> bool:
    return value % 2 == 0


def tile_extent_from_rank_metadata(
    dims: Sequence[str],
    rank_extent: Sequence[int],
    layout: tuple[int, int],
    edge_interior_ratio: float = 1.0,
) -> tuple[int, ...]:
    """
    Returns the extent of a tile given data about a single rank, and the tile
    layout.

    Args:
        dims: dimension names
        rank_extent: the extent of one rank
        layout: the (y, x) number of ranks along each tile axis
        edge_interior_ratio: target value for the relative 1-dimensional
            extent of the compute domains of ranks on tile edges and corners compared
            to ranks on the tile interior. In all cases, the closest valid value will
            be used, which enables some previously invalid configurations
            (e.g. C128 on a 3 by 3 layout will use the closest valid
            edge_interior_ratio to 1.0).

    Returns:
        tile_extent: the extent of one tile
    """
    if edge_interior_ratio != 1.0:
        raise NotImplementedError(
            "Only equal sized subdomains are supported, was given "
            f"an edge_interior_ratio of {edge_interior_ratio}"
        )
    layout_factors = np.asarray(list_by_dims(dims, layout, non_horizontal_value=1))
    return extent_from_metadata(dims, rank_extent, layout_factors)


def rank_slice_from_tile_metadata(
    dims: Sequence[str],
    *,
    extent: Sequence[int],
    layout: tuple[int, int],
    subtile_index: tuple[int, int],
    edge_interior_ratio: float,
    overlap: bool,
) -> tuple[slice, ...]:
    return _rank_slice_from_tile_metadata_cached(
        dims=tuple(dims),
        extent=tuple(extent),
        layout=tuple(layout),
        subtile_index=tuple(subtile_index),
        edge_interior_ratio=edge_interior_ratio,
        overlap=overlap,
    )


@functools.lru_cache(maxsize=DEFAULT_CACHE_SIZE)
def _rank_slice_from_tile_metadata_cached(
    dims: tuple[str, ...],
    *,
    extent: tuple[int, ...],
    layout: tuple[int, int],
    subtile_index: tuple[int, int],
    edge_interior_ratio: float,
    overlap: bool,
) -> tuple[slice, ...]:
    # detect if one of the given dims is the tile dimension and ignore it
    cartesian_dims = discard_dimension(dims, constants.TILE_DIM, data=dims)
    cartesian_extent = discard_dimension(dims, constants.TILE_DIM, data=extent)

    interior_extents, edge_extents = _subtile_extents_from_tile_metadata(
        cartesian_dims, cartesian_extent, layout, edge_interior_ratio
    )
    return_slice = []

    for dim, dim_interior_extent, dim_edge_extent in zip(
        cartesian_dims, interior_extents, edge_extents
    ):
        if dim in constants.HORIZONTAL_DIMS:
            if dim in constants.J_DIMS:
                index = subtile_index[0]
                n_ranks = layout[0]
            else:
                index = subtile_index[1]
                n_ranks = layout[1]
            start, end = 0, 0
            for i in range(index + 1):
                if i == 0:
                    end += dim_edge_extent
                elif i == n_ranks - 1:
                    start = end
                    end += dim_edge_extent
                else:
                    start = end
                    end += dim_interior_extent
            if dim in constants.INTERFACE_DIMS and (overlap or (index == n_ranks - 1)):
                end += 1
        else:
            start, end = 0, dim_interior_extent
            if dim in constants.INTERFACE_DIMS:
                end += 1
        return_slice.append(slice(start, end))
    return tuple(return_slice)


T = TypeVar("T")


def discard_dimension(
    dims: tuple[str, ...], dim_name: str, data: Sequence[T]
) -> list[T]:
    return [item for (item, dim) in zip(data, dims) if dim != dim_name]


def _subtile_extents_from_tile_metadata(
    dims: Sequence[str],
    tile_extent: Sequence[int],
    layout: tuple[int, int],
    edge_interior_ratio: float = 1.0,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Returns the extent of a given rank given data about a tile, and the tile
    layout.

    Args:
        dims: dimension names
        tile_extent: the extent of a tile
        layout: the (y, x) number of ranks along each tile axis
        edge_interior_ratio: target value for the relative 1-dimensional
            extent of the compute domains of ranks on tile edges and corners compared
            to ranks on the tile interior. In all cases, the closest valid value will
            be used, which enables some previously invalid configurations
            (e.g. C128 on a 3 by 3 layout will use the closest valid
            edge_interior_ratio to 1.0).

    Returns:
        subtile_extents: the extents of first all interior tiles,
            then all edge tiles along all dimensions.
    """

    def _valid_edge_tile_sizes(
        dim_extent: int, subtile_count: int, start: int
    ) -> Sequence[int]:
        """
        Returns a list of valid edge tile sizes, counting down from the
        starting edge size to the smallest possible one
        that lets the interior tile sizes still be an integer.
        After that, it counts up from the starting edge size.
        """
        bottom = 1
        top = int((dim_extent - subtile_count + 2) / 2) + 1
        unsorted_valid_sizes = range(bottom, top)
        valid_sizes = []

        index = start
        offset = 0
        factor = -1

        # steps through all valid sizes to sort them:
        # [start, counting down to 1, counting up from start]
        for _i in range(len(unsorted_valid_sizes) + 1):
            index = start + factor * offset
            if index in unsorted_valid_sizes and index not in valid_sizes:
                valid_sizes.append(index)
            offset = offset + 1
            if index == 1:
                offset = 0
                factor = 1
        return valid_sizes

    layout_factors = np.asarray(list_by_dims(dims, layout, non_horizontal_value=1))

    return_extents = []
    edge_extents = []
    # for each dimension, find a valid edge:interior decomposition
    # that has a ratio close to the desired edge_interior_ratio
    for dim, subtile_count, dim_extent in zip(dims, layout_factors, tile_extent):
        dim_edge_interior_ratio = edge_interior_ratio
        if dim in constants.INTERFACE_DIMS:
            dim_extent = dim_extent - 1
        if (not subtile_count % 2) and dim_extent % 2:
            raise ValueError(
                f"Cannot find valid decomposition for odd ({dim_extent}) "
                f"gridpoints along an even count ({subtile_count}) of ranks."
            )

        # only do shrinked edges in x,y and if there is interior
        if subtile_count >= 3 and dim in constants.HORIZONTAL_DIMS:
            # starting edge subtile size, rounded to an integer
            edge_subtile_size = round(
                dim_extent / (2 + (subtile_count - 2) / dim_edge_interior_ratio)
            )

            # searching of a valid integer pair for edge and interior tile sizes
            # that add up to the entire dimension extent.
            found = False
            for edge_size in _valid_edge_tile_sizes(
                dim_extent, subtile_count, edge_subtile_size
            ):
                dim_edge_interior_ratio = edge_size / (
                    (dim_extent - 2 * edge_size) / (subtile_count - 2)
                )
                # validation that the integer pair
                # (edge_subtile_size, int(edge_size / dim_edge_interior_ratio))
                # multiplied by their respective subtile counts together
                # add up to the entire dimension's extent
                if (
                    edge_size * 2
                    + (subtile_count - 2) * int(edge_size / dim_edge_interior_ratio)
                    == dim_extent
                ):
                    found = True
                    break
            if not found:
                raise ValueError(
                    f"No valid subdomain assignment found for dimension {dim} "
                    f"with {dim_extent} gridpoints along {subtile_count} ranks."
                )
            return_extents.append(int(edge_size / dim_edge_interior_ratio))
            edge_extents.append(int(edge_size))
        else:
            # trivial case of no special handling
            subtile_size = int(dim_extent / subtile_count)
            return_extents.append(subtile_size)
            edge_extents.append(subtile_size)

    return tuple(return_extents), tuple(edge_extents)


def extent_from_metadata(
    dims: Sequence[str], extent: Sequence[int], layout_factors: np.ndarray
) -> tuple[int, ...]:
    return_extents = []
    for dim, rank_extent, layout_factor in zip(dims, extent, layout_factors):
        if dim in constants.INTERFACE_DIMS:
            add_extent = -1
        else:
            add_extent = 0
        tile_extent = (rank_extent + add_extent) * layout_factor - add_extent
        return_extents.append(int(tile_extent))  # layout_factor is float, need to cast
    return tuple(return_extents)


def subtile_slice(
    dims: Sequence[str],
    global_extent: Sequence[int],
    layout: tuple[int, int],
    subtile_index: tuple[int, int],
    edge_interior_ratio: float = 1.0,
    overlap: bool = False,
) -> tuple[slice, ...]:
    """
    Returns the slice of data within a tile's computational domain belonging
    to a single rank.

    Args:
        dims: dimension names for each axis
        global_extent: size of the tile or cube's computational domain
        layout: the (y, x) number of ranks along each tile axis
        subtile_index: the (y, x) position of the rank on the tile
        edge_interior_ratio: target value for the relative 1-dimensional
            extent of the compute domains of ranks on tile edges and corners compared
            to ranks on the tile interior. In all cases, the closest valid value will
            be used, which enables some previously invalid configurations
            (e.g. C128 on a 3 by 3 layout will use the closest valid
            edge_interior_ratio to 1.0).
        overlap (optional): if True, for interface variables include the part
            of the array shared by adjacent ranks in both ranks. If False, ensure
            only one of those ranks (the greater rank) is assigned the overlapping
            section. Default is False.
    """
    return rank_slice_from_tile_metadata(
        dims=dims,
        extent=global_extent,
        layout=layout,
        subtile_index=subtile_index,
        edge_interior_ratio=edge_interior_ratio,
        overlap=overlap,
    )
