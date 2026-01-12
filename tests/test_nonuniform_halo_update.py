import os

import numpy as np
from mpi4py import MPI

import pyfms

from ndsl import (
    CubedSphereCommunicator,
    CubedSpherePartitioner,
    QuantityFactory,
    TilePartitioner,
    SubtileGridSizer,
)

from ndsl.constants import (
    X_DIM,
    X_INTERFACE_DIM,
    Y_DIM,
    Y_INTERFACE_DIM,
)
from ndsl.grid import MetricTerms

import argparse

### Imports for Class TestPartitioner ##############################################################

import abc
import copy
import functools
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar, Union, cast

import ndsl.constants as constants
from ndsl.comm import boundary as bd
from ndsl.constants import (
    EAST,
    NORTH,
    NORTHEAST,
    NORTHWEST,
    SOUTH,
    SOUTHEAST,
    SOUTHWEST,
    WEST,
    BOUNDARY_TYPES,
)
from ndsl.quantity import Quantity, QuantityMetadata
from ndsl.utils import list_by_dims
from ndsl.comm.partitioner import Partitioner, tile_extent_from_rank_metadata, subtile_slice, get_tile_index

####################################################################################################

class TestPartitioner(Partitioner):
    def __init__(
        self,
        tile: TilePartitioner,
        boundaries: dict,
    ):
        """Create an object for fv3gfs tile decomposition."""
        if not isinstance(tile, TilePartitioner):
            raise TypeError("tile must be a TilePartitioner")
        self.tile = tile
        self.boundaries = boundaries

    def tile_index(self, rank: int) -> int:
        """Returns the tile index of a given rank"""
        print("tile_index used")
        return get_tile_index(rank, self.total_ranks)

    @property
    def layout(self) -> Tuple[int, int]:
        print("layout used")
        return self.tile.layout

    @property
    def total_ranks(self) -> int:
        print("total_rank used")
        return 6 * self.layout[0] * self.layout[1]

    def global_extent(self, rank_metadata: QuantityMetadata) -> Tuple[int, ...]:
        """Return the shape of a full cube representation for the given dimensions.

        Args:
            metadata: quantity metadata

        Returns:
            extent: shape of full cube representation
        """
        print("global_extent used")
        return (6,) + tile_extent_from_rank_metadata(
            rank_metadata.dims, rank_metadata.extent, self.layout
        )

    def subtile_slice(
        self,
        rank: int,
        global_dims: Sequence[str],
        global_extent: Sequence[int],
        overlap: bool = False,
    ) -> Tuple[Union[int, slice], ...]:
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
        #if global_dims[0] != constants.TILE_DIM:
        #    raise NotImplementedError(
        #        "currently only supports tile dimension {constants.TILE_DIM} as the "
        #        "first dimension, got dims {cube_metadata.dims}"
        #    )
        i_tile = self.tile_index(rank)
        return (i_tile,) + self.tile.subtile_slice(
            rank=rank,
            global_dims=global_dims[1:],
            global_extent=global_extent[1:],
            overlap=overlap,
        )

    def subtile_extent(
        self,
        cube_metadata: QuantityMetadata,
        rank: int,
    ) -> Tuple[int, ...]:
        """Return the shape of a single rank representation for the given dimensions.

        Args:
            global_metadata: quantity metadata.
            rank: rank of the process.

        Returns:
            extent: shape of a single rank representation for the given dimensions.
        """
        return self.tile.subtile_extent(cube_metadata, rank)

    def boundary(self, boundary_type: int, rank: int) -> Optional[bd.SimpleBoundary]:
        """Returns a boundary of the requested type for a given rank.

        Args:
            boundary_type: the type of boundary
            rank: the processor rank

        Returns:
            boundary
        """
        entry = self.boundaries.get(boundary_type)
        if entry is None:
            return None

        to_ranks, rotations = entry

        if not to_ranks:
            return None

        boundaries = []
        for to_rank in to_ranks:
            boundaries.append(
                    bd.SimpleBoundary(
                        boundary_type = boundary_type,
                        from_rank = rank,
                        to_rank = to_rank,
                        n_clockwise_rotations = rotations,
                    )
            )

        print("SimpleBoundaries:", boundaries)
            
        #if self.boundaries[boundary_type] is not None:
        #    to_rank = self.boundaries[boundary_type][0]
        #    rotations = self.boundaries[boundary_type][1]
        #    boundary = bd.SimpleBoundary(
        #                boundary_type=boundary_type,
        #                from_rank=rank,
        #                to_rank=to_rank,
        #                n_clockwise_rotations=rotations,
        #            )
        #else:
        #    boundary = None
        # boundary_type not in boundary?
        return boundaries

    def get_cubed_sphere_boundaries(self, pe, tile_layout):
        pe_local_index = self._get_local_pe_index(pe, tile_layout)
        return pe_local_index

    def _get_local_pe_index(self, pe, tile_layout):
        nx, ny = tile_layout
        tile_size = nx * ny
        tile = pe // tile_size
        local_index = pe % tile_size
        j, i = divmod(local_index, nx)
        return (tile, i, j)

def get_boundary_spec(data, nhalo):
    dnx = data.shape[0]
    dny = data.shape[1]
    cnx = dnx - 2 * nhalo
    cny = dny - 2 * nhalo

    # Corner
    SW = data[:nhalo,:nhalo]
    NW = data[:nhalo,dny-nhalo:]
    SE = data[dnx-nhalo:,:nhalo]
    NE = data[dnx-nhalo:,dny-nhalo:]

    # Edge
    W = data[:nhalo,nhalo:nhalo+cny]
    E = data[nhalo+cnx:,nhalo:nhalo+cny]
    N = data[nhalo:nhalo+cnx,nhalo+cny:]
    S = data[nhalo:nhalo+cnx,:nhalo]

    # Create boundary dict
    boundaries = {
            EAST: E,
            WEST: W,
            NORTH: N,
            SOUTH: S,
            SOUTHWEST: SW,
            NORTHWEST: NW,
            SOUTHEAST: SE,
            NORTHEAST: NE
    }

    boundary_spec = {}
    for direction in BOUNDARY_TYPES:
        section = boundaries[direction]
        pe_layout = abs(section).astype(int) % 100
        if np.all(section == 0.0):
            boundary_spec[direction] = None
        else:
            rotation  = get_rotation(section)
            to_ranks  = tuple(np.unique(pe_layout, return_counts=False))
            boundary_spec[direction] = (to_ranks, rotation)
    return boundary_spec

def get_rotation(data):
    x_diff = abs(data[1,0] - data[0,0])
    y_diff = abs(data[0,1] - data[0,0])

    if (x_diff < 0.1) and (y_diff < 0.001 ):
        rotation = 0
    else:
        rotation = 1
    if data[0,0] < 0.0:
        rotation = 3
    return rotation

def get_layout():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    try:
        if rank == 0:
            npes = comm.Get_size()

            pes_per_tile = npes // 6
            layout_dim = int((pes_per_tile) ** 0.5)

            uneven_pes_per_tile = (npes % 6 != 0)
            nonsquare_layout = (pes_per_tile != layout_dim*layout_dim)

            #if uneven_pes_per_tile or nonsquare_layout:
                #raise ValueError("Number of processes must be divisible by 6 and form a square layout per tile.")

            layout = (layout_dim, layout_dim)
        else:
            layout = None
            
        layout = comm.bcast(layout, root=0)

    except Exception as e:
        print(f"[Rank {rank}] Error: {e}")
        comm.Abort(1)

    return (1,2) #layout

def init_data(nx, ny, halo, grid_type, layout):
    # Decompose local grid size
    mx = int(nx / layout[0])
    my = int(ny / layout[1])

    pe = MPI.COMM_WORLD.Get_rank()

    # A-grid shapes
    x_shape = (mx + 2*halo, my + 2*halo)
    y_shape = (mx + 2*halo, my + 2*halo)

    # Modify grid shape according to grid type
    if grid_type == 'C':
        x_shape = (x_shape[0]+1,x_shape[1]  )
        y_shape = (y_shape[0]  ,y_shape[1]+1)
    if grid_type == 'D':
        x_shape = (x_shape[0]  ,x_shape[1]+1)
        y_shape = (y_shape[0]+1,y_shape[1]  )

    x_data = np.zeros(x_shape)
    y_data = np.zeros(y_shape)

    # Compute Domain Shape
    xc_shape = (x_shape[0]-2*halo, x_shape[1]-2*halo)
    yc_shape = (y_shape[0]-2*halo, y_shape[1]-2*halo)

    # Initialize Compute Domain
    for i in range(xc_shape[0]):
        for j in range(xc_shape[1]):
            x_data[i+halo, j+halo] = 100 + pe + (i + 1) * 1e-2 + (j + 1) * 1e-4

    for i in range(yc_shape[0]):
        for j in range(yc_shape[1]):
            y_data[i+halo, j+halo] = 200 + pe + (i + 1) * 1e-2 + (j + 1) * 1e-4

    return x_data, y_data

def print_formatted_data(data, skip_first_column):
    transposed_data = list(zip(*data))

    for column in reversed(transposed_data):
        if skip_first_column:
            skip_first_column = False  # Skip this column and unset the flag
            continue

        formatted_column = []
        for value in column:
            if value == 0.0:
                formatted_column.append(f" 00{value:.4f}")
            elif value > 0.0:
                formatted_column.append(f" {value:.4f}")
            else:
                formatted_column.append(f"{value:.4f}")

        print(" ".join(formatted_column))

def define_cubic_mosaic(nx, ny, halo, layout):
    domain_id = pyfms.mpp_domains.define_cubic_mosaic(
        ni=[nx, nx, nx, nx, nx, nx],
        nj=[ny, ny, ny, ny, ny, ny],
        global_indices=[0, nx - 1, 0, ny - 1],
        layout=layout,
        ntiles=6,
        halo=halo,
        use_memsize=False,
    )
    return domain_id

def vector_update_domains(x_data, y_data, domain_id, gridtype, halo):
    pyfms.mpp_domains.vector_update_domains(
        fieldx=x_data,
        fieldy=y_data,
        domain_id=domain_id,
        gridtype=gridtype,
        whalo=halo,
        ehalo=halo,
        shalo=halo,
        nhalo=halo,
    )

def get_pyFMS_boundaries(partitioner, pe):
    boundaries = {}
    for boundary_type in BOUNDARY_TYPES:
        boundary = partitioner.boundary(boundary_type, pe)
        if boundary is not None:
            to_rank = boundary.to_rank
            rotations = boundary.n_clockwise_rotations
            boundaries[boundary_type] = (to_rank, rotations)
        else:
            boundaries[boundary_type] = None
    return boundaries

def vhu_pyFMS(localcomm, x_init, y_init, grid_data, grid_type):
    layout = get_layout()
    x_data, y_data = x_init, y_init
    boundary_spec = {}

    pyfms.fms.init(localcomm=localcomm)

    if grid_type == 'A':
        gridtype = pyfms.mpp_domains.AGRID
    elif grid_type == 'C':
        gridtype = pyfms.mpp_domains.CGRID_NE
    else: 
        gridtype = pyfms.mpp_domains.DGRID_NE

    domain_id = define_cubic_mosaic(nx=grid_data[0], ny=grid_data[1], halo=grid_data[2], layout=layout)
    vector_update_domains(x_data=x_data, y_data=y_data, domain_id=domain_id, gridtype=gridtype, halo=grid_data[2])
    pe = pyfms.mpp.pe()
    compute_domain = pyfms.mpp_domains.get_compute_domain(
        domain_id=domain_id, whalo=3, shalo=3
    )
    data_domain = pyfms.mpp_domains.get_data_domain(
        domain_id=domain_id, whalo=3, shalo=3
    )
    boundary_spec = get_boundary_spec(x_data, grid_data[2])
    print("boundary_spec:", boundary_spec)
    #pyfms.fms.end()

    return x_data, y_data, boundary_spec

def vhu_NDSL(x_init, y_init, grid_data, grid_type, boundary_spec):
    layout = get_layout()
    mpi_comm = MPI.COMM_WORLD
    nx = grid_data[0]
    ny = grid_data[1]
    nhalo = grid_data[2]
    xc_shape = (x_init.shape[0]-2*nhalo, x_init.shape[1]-2*nhalo)
    yc_shape = (y_init.shape[0]-2*nhalo, y_init.shape[1]-2*nhalo)
    pe = mpi_comm.Get_rank()

    partitioner = CubedSpherePartitioner(TilePartitioner(layout))
    partitioner = TestPartitioner(TilePartitioner(layout), boundary_spec)
    communicator = CubedSphereCommunicator(mpi_comm, partitioner)
    sizer = SubtileGridSizer.from_tile_params(
        nx_tile=nx,
        ny_tile=ny,
        nz=1,
        n_halo=nhalo,
        extra_dim_lengths={},
        layout=layout,
        tile_partitioner=partitioner.tile,
        tile_rank=communicator.tile.rank,
    )
    quantity_factory = QuantityFactory.from_backend(sizer=sizer, backend="numpy")
    #metric_terms = MetricTerms(quantity_factory=quantity_factory, communicator=communicator)

    if   grid_type == "C":
        ux_dim = X_INTERFACE_DIM  ; uy_dim = Y_DIM
        vx_dim = X_DIM            ; vy_dim = Y_INTERFACE_DIM
    elif grid_type == "D":
        ux_dim = X_DIM            ; uy_dim = Y_INTERFACE_DIM
        vx_dim = X_INTERFACE_DIM  ; vy_dim = Y_DIM
    else:
        ux_dim = X_DIM            ; uy_dim = Y_DIM
        vx_dim = X_DIM            ; vy_dim = Y_DIM

    x_vel = quantity_factory.zeros(dims=(ux_dim, uy_dim), units="m/s", dtype="float")
    y_vel = quantity_factory.zeros(dims=(vx_dim, vy_dim), units="m/s", dtype="float")
    x_vel_update = quantity_factory.zeros(dims=(ux_dim, uy_dim), units="m/s", dtype="float")
    y_vel_update = quantity_factory.zeros(dims=(vx_dim, vy_dim), units="m/s", dtype="float")

    # print("pe:", pe, "x_vel.shape:", x_vel.data.shape)  #, "y_vel.shape:", y_vel.shape())

    # Initialize Compute Domain
    for i in range(xc_shape[0]):
        for j in range(xc_shape[1]):
            x_vel.data[i+nhalo,j+nhalo] = x_init[i+nhalo,j+nhalo]

    for i in range(yc_shape[0]):
        for j in range(yc_shape[1]):
            y_vel.data[i+nhalo,j+nhalo] = y_init[i+nhalo,j+nhalo]

    x_vel_update.view[:] = x_vel.view[:]
    y_vel_update.view[:] = y_vel.view[:]

    #y_slice, x_slice = partitioner.subtile_slice(rank=pe, global_dims=[Y_DIM, X_DIM], global_extent=[nx,ny], overlap=False)

    #print("x_slice:", x_slice)
    #print("y_slice:", y_slice)

    print("shape:", y_vel_update.data.shape)
    print("strides:", y_vel_update.data.strides)
    print("itemsize:", y_vel_update.data.itemsize)
    print("origin:", y_vel_update.metadata.origin)
    print("extent:", y_vel_update.metadata.extent)
    print("dims:", y_vel_update.metadata.dims)

    # Conducting vector halo update operation
    communicator.vector_halo_update(x_vel_update, y_vel_update, n_points=nhalo)

    return x_vel_update.data, y_vel_update.data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vhu test with specified grid type.")
    parser.add_argument("--grid_type", type=str, default="A", choices=["A", "C", "D"],
                        help="Grid type: A, C, or D")
    args = parser.parse_args()

    grid_type = grid_type = args.grid_type
    nx = 6
    ny = 3
    halo = 2
    boundary_spec = {}

    fcomm = MPI.COMM_WORLD.py2f()
    pe = MPI.COMM_WORLD.Get_rank()
    layout = get_layout()

    nx = nx*layout[0]
    ny = ny*layout[1]

    # Generate Initial Data
    u_init, v_init = init_data(nx=nx, ny=ny, halo=halo, grid_type=grid_type, layout=layout)

    # Perform VHU with pyFMS and NDSL using init data
    u_pyFMS, v_pyFMS, boundary_spec = vhu_pyFMS(localcomm = fcomm, x_init=u_init, y_init=v_init, grid_data=(nx,ny,halo), grid_type=grid_type)
    print("pyFMS:")
    print_formatted_data(u_pyFMS, False)
    u_ndsl, v_ndsl = vhu_NDSL(x_init=u_init, y_init=v_init, grid_data=(nx,ny,halo), grid_type=grid_type, boundary_spec=boundary_spec)

    pyfms.fms.end()

    print("pyFMS:")
    print_formatted_data(u_pyFMS, False)
    print("NDSL:")
    print_formatted_data(u_ndsl, False)
    print("pe:", pe, "boundary_spec:", get_boundary_spec(u_pyFMS, halo))

    # Compute ndsl offsets necessary to compare results between ndsl and pyFMS
    if   grid_type == "C":
        x_off = None ; y_off = -1
    elif grid_type == "D":
        x_off = -1   ; y_off = None
    else:
        x_off = -1   ; y_off = -1

    assert np.allclose(u_ndsl[:x_off, :y_off], u_pyFMS, atol=1e-8)
    assert np.allclose(v_ndsl[:y_off, :x_off], v_pyFMS, atol=1e-8)
