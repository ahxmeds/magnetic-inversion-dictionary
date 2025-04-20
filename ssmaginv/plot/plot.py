import torch
import numpy as np
import pyvista as pv


def plot_model_2d(M, n_slices=None, cmap="rainbow", figsize=(12, 6)):
    """
    Plot an array of 2D slices for a given 3D tensor in a single grid-style plot.

    Parameters:
    - M (torch.Tensor): 3D tensor representing the model (e.g., [x, y, z]).
    - n_slices (int, optional): Number of slices to plot. Defaults to all slices.
    - cmap (str): Colormap for the plot.
    - figsize (tuple): Size of the figure.

    Returns:
    - matplotlib.figure.Figure: The created figure.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Dimensions of the 3D tensor
    nx, ny, nz = M.shape[-3], M.shape[-2], M.shape[-1]

    # Determine the number of slices
    if n_slices is None:
        n_slices = nz
    else:
        n_slices = min(n_slices, nz)

    # Determine the grid shape (rows and columns)
    ncols = int(np.ceil(np.sqrt(n_slices)))
    nrows = int(np.ceil(n_slices / ncols))

    # Create a blank canvas for the grid plot
    grid = torch.zeros((nrows * nx, ncols * ny), dtype=M.dtype, device=M.device)

    # Fill the grid with slices
    slice_idx = 0
    for i in range(nrows):
        for j in range(ncols):
            if slice_idx < n_slices:
                grid[i * nx : (i + 1) * nx, j * ny : (j + 1) * ny] = M[..., slice_idx]
                slice_idx += 1

    # Plot the grid
    fig = plt.figure(figsize=figsize)
    plt.imshow(grid.cpu().detach().numpy(), cmap=cmap)
    plt.colorbar()
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_model_3d(M, plotter=None, threshold_value=0.05):
    """
    Plot the magnetization model with an outline showing the overall bounds.

    Parameters:
        M (torch.Tensor): Magnetization model tensor of shape [B, C, X, Y, Z] or [X, Y, Z].
        threshold_value (float): Threshold value for magnetization to visualize.
    """

    if plotter is None:
        plotter = pv.Plotter()

    # Remove unnecessary dimensions if present
    M = torch.squeeze(M)

    # Reverse the Z-axis to match the PyVista coordinate system
    M = torch.flip(M, dims=[2])

    # Convert the PyTorch tensor to a NumPy array
    m_plot = M.detach().cpu().numpy()

    # Define grid parameters
    spacing = (1.0, 1.0, 1.0)
    origin = (0.0, 0.0, 0.0)

    # Create a PyVista Uniform Grid (ImageData)
    grid = pv.ImageData()

    # Set grid dimensions (number of points = cells + 1)
    grid.dimensions = np.array(m_plot.shape) + 1
    grid.spacing = spacing
    grid.origin = origin

    # Assign magnetization data to cell data
    grid.cell_data["M"] = m_plot.flatten(order="F")

    # Apply threshold to isolate regions with M > threshold_value
    thresholded = grid.threshold(value=threshold_value, scalars="M")

    # Create an outline of the entire grid
    outline = grid.outline()

    # Add the thresholded mesh
    plotter.add_mesh(
        thresholded,
        cmap="rainbow",
        opacity=0.7,
        show_edges=True,
        label="Magnetization > {:.2f}".format(threshold_value),
    )

    # Add the outline mesh
    plotter.add_mesh(outline, color="black", line_width=2, label="Model Bounds")

    # Optionally, add axes and a legend for better context
    plotter.add_axes()
    plotter.add_legend()

    # Set camera position for better visualization (optional)
    plotter.view_isometric()

    return plotter


def plot_model_with_forward(
    mag_data,
    forward_data,
    spacing=(1.0, 1.0, 1.0),
    height=0,
    plotter=None,
    n_isovals=10,
    clim_mag=None,
    isovals=None,
    clim_forward=None,
):
    if plotter is None:
        plotter = pv.Plotter()

    # Remove unnecessary dimensions if present
    if isinstance(mag_data, torch.Tensor):
        M = mag_data.squeeze().cpu().numpy()
    else:
        M = mag_data

    if isinstance(forward_data, torch.Tensor):
        D = forward_data.squeeze().cpu().numpy()
    else:
        D = forward_data

    # Define grid parameters XYZ spacing
    origin = (0.0, 0.0, 0.0)

    # Create a PyVista Uniform Grid (ImageData) for the 3D volume
    grid = pv.ImageData()

    # Set grid dimensions
    grid.dimensions = np.array(M.shape[-3:])
    grid.spacing = spacing
    grid.origin = origin

    # Assign magnetization data to cell data
    grid.point_data["M"] = M.flatten(order="F")

    # Apply contour with specified isovals
    if isovals is not None:
        thresholded = grid.contour(isosurfaces=isovals, scalars="M")
    else:
        thresholded = grid.contour(isosurfaces=n_isovals, scalars="M")

    outline = grid.outline()

    # Add the thresholded mesh with specified clim
    plotter.add_mesh(
        thresholded,
        cmap="rainbow",
        opacity=0.7,
        show_edges=False,
        clim=clim_mag,
        show_scalar_bar=False,
    )
    plotter.add_scalar_bar(
        title="Magnetic Susceptibility",
        title_font_size=18,
        fmt="%.2f",
        n_labels=5,
        shadow=True,
        position_x=0.25,
    )

    # Add the outline mesh
    plotter.add_mesh(outline, color="black", line_width=2, label="Model Bounds")

    nz = M.shape[-1]
    h = height - 0.2 * nz * spacing[-1]
    surface_mesh = forward_data_to_grid(D, spacing=spacing, height=h)

    # Add the 2D surface with specified clim
    plotter.add_mesh(
        surface_mesh,
        cmap="coolwarm",
        show_edges=False,
        opacity=0.95,
        label="Forward Problem Data",
        show_scalar_bar=False,
        clim=clim_forward,
    )
    plotter.add_scalar_bar(
        title="Forward Data",
        title_font_size=18,
        fmt="%.2f",
        n_labels=5,
        shadow=True,
        position_x=0.25,
        position_y=0.17,
    )

    # Add axes and a legend
    # plotter.add_axes()
    plotter.view_isometric(negative=True)
    plotter.view_vector(vector=[1, 1, -0.5], viewup=[1, 1, -1])

    # Get the current camera position
    camera_pos = plotter.camera_position

    # Shift the camera position vertically (adjust the z-coordinate)
    new_camera_pos = [
        camera_pos[0],  # Camera position remains unchanged
        (
            camera_pos[1][0],
            camera_pos[1][1],
            camera_pos[1][2] + 10,
        ),  # Lower focal point
        camera_pos[2],  # View up direction remains unchanged
    ]

    # Apply the new camera position
    plotter.camera_position = new_camera_pos

    # # Adjust the zoom to fit data in window
    # plotter.camera.zoom(zoom)
    return plotter


def forward_data_to_grid(forward_data, spacing=(1.0, 1.0, 1.0), height=0):
    # Remove unnecessary dimensions if present in torch tensor
    if forward_data is torch.Tensor:
        D = forward_data.squeeze().cpu().numpy()
    else:
        D = forward_data

    # Create a structured grid for the 2D surface data
    nx, ny = forward_data.shape[-2:]
    x = np.linspace(0, nx * spacing[0], nx)
    y = np.linspace(0, ny * spacing[1], ny)
    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, height)  # Position the surface above the volume

    # Create a PyVista mesh for the 2D surface
    surface_mesh = pv.StructuredGrid(X, Y, Z)
    surface_mesh.point_data["Forward Data"] = D.flatten()

    return surface_mesh


def plot_mixed(
    mag_data_true,
    mag_data_model,
    forward_data_true,
    forward_data_model,
    spacing=(1.0, 1.0, 1.0),
    height=0,
    n_isovals=10,
    n_slices=None,
):

    # Plot the model with forward in one panel and the 2d slices in top right, other data in bottom right
    pv.global_theme.multi_rendering_splitting_position = 0.6

    shape = (2, 4)
    row_weights = [1, 1]
    col_wights = [1, 1, 0.5, 0.5]
    groups = [
        ([0, 1], 0),
        ([0, 1], 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
    ]

    p = pv.Plotter(
        shape=shape,
        row_weights=row_weights,
        col_weights=col_wights,
        groups=groups,
        window_size=(1400, 600),
    )

    # Compute common clims and isovalues for magnetization data
    scalar_min_mag = min(mag_data_true.min(), mag_data_model.min()).item()
    scalar_max_mag = max(mag_data_true.max(), mag_data_model.max()).item()
    clim_mag = (scalar_min_mag, scalar_max_mag)

    # Compute common clims for forward data
    scalar_min_forward = min(forward_data_true.min(), forward_data_model.min()).item()
    scalar_max_forward = max(forward_data_true.max(), forward_data_model.max()).item()
    clim_forward = (scalar_min_forward, scalar_max_forward)

    p.subplot(0, 0)
    plot_model_with_forward(
        mag_data_true,
        forward_data_true,
        spacing,
        height,
        plotter=p,
        clim_mag=clim_mag,
        clim_forward=clim_forward,
    )

    p.subplot(0, 1)
    plot_model_with_forward(
        mag_data_model,
        forward_data_model,
        spacing,
        height,
        plotter=p,
        clim_mag=clim_mag,
        clim_forward=clim_forward,
    )

    p.subplot(0, 2)
    plot_model_2d_pyvista(
        mag_data_true, cmap="rainbow", clim=clim_mag, n_slices=n_slices, plotter=p
    )
    p.add_title("True Model", font_size=6)

    p.subplot(0, 3)
    plot_model_2d_pyvista(
        mag_data_model, cmap="rainbow", clim=clim_mag, n_slices=n_slices, plotter=p
    )
    p.add_title("Inverted Model", font_size=6)

    p.subplot(1, 2)
    surface_mesh = forward_data_to_grid(
        forward_data_true, spacing=spacing, height=height
    )
    p.add_mesh(
        surface_mesh,
        cmap="coolwarm",
        clim=clim_forward,
        show_edges=False,
        label="Forward Problem Data",
    )
    p.view_yx()
    p.view_vector(vector=[0, 0, 1], viewup=[-1, 0, 0])
    p.add_title("True Forward Data", font_size=6)
    p.add_scalar_bar(fmt="%.2f", n_labels=3, shadow=True)

    p.subplot(1, 3)
    surface_mesh = forward_data_to_grid(
        forward_data_model, spacing=spacing, height=height
    )
    p.add_mesh(
        surface_mesh,
        cmap="coolwarm",
        clim=clim_forward,
        show_edges=False,
        label="Forward Problem Data",
    )
    p.view_yx()
    p.view_vector(vector=[0, 0, 1], viewup=[-1, 0, 0])
    p.add_title("Inverted Forward Data", font_size=6)

    p.link_views((0, 1))
    p.link_views((2, 3))
    p.link_views((4, 5))

    return p


def plot_model_2d_pyvista(
    M, n_slices=None, cmap="rainbow", clim=None, spacing=(1.0, 1.0), plotter=None
):
    if plotter is None:
        plotter = pv.Plotter()

    # Convert torch tensor to numpy array for PyVista compatibility
    M_np = M.squeeze().cpu().detach().numpy()

    # Dimensions of the 3D tensor
    nx, ny, nz = M_np.shape[-3], M_np.shape[-2], M_np.shape[-1]

    # Determine the number of slices
    if n_slices is None:
        n_slices = nz
    else:
        n_slices = min(n_slices, nz)

    # Determine the grid shape (rows and columns)
    ncols = int(np.ceil(np.sqrt(n_slices)))
    nrows = int(np.ceil(n_slices / ncols))

    # Create a 2D array for the data to be plotted
    data = np.zeros((nrows * nx, ncols * ny))

    # Fill the grid with slices
    slice_idx = 0
    for i in range(nrows):
        for j in range(ncols):
            if slice_idx < n_slices:
                data[i * nx : (i + 1) * nx, j * ny : (j + 1) * ny] = M_np[
                    :, :, slice_idx
                ]
                slice_idx += 1

    # Create a UniformGrid
    grid = pv.ImageData(dimensions=(data.shape[1] + 1, data.shape[0] + 1, 1))
    # Add the data values
    grid.cell_data["Model Data"] = data.flatten(order="F")  # Fortran order flattening

    # Add the grid to the plotter
    plotter.add_mesh(
        grid,
        cmap=cmap,
        clim=clim,
        show_edges=False,
        show_scalar_bar=False,
    )

    # Manage scalarbar
    plotter.add_scalar_bar(title="Model Data", fmt="%.2f", n_labels=3, shadow=True)

    plotter.view_yx()
    plotter.view_vector(vector=[0, 0, 1], viewup=[-1, 0, 0])

    return plotter


def plot_triplet(
    mag_data_true,
    mag_data_rl,
    mag_data_var,
    forward_data_true,
    forward_data_rl,
    forward_data_var,
    spacing=(1.0, 1.0, 1.0),
    height=0,
    n_isovals=10,
    n_slices=None,
):

    # Plot the model with forward in one panel and the 2d slices in top right, other data in bottom right
    pv.global_theme.multi_rendering_splitting_position = 0.6

    shape = (2, 6)
    row_weights = [1, 1]
    col_wights = [1, 1, 1, 2 / 3, 2 / 3, 2 / 3]
    groups = [
        ([0, 1], 0),
        ([0, 1], 1),
        ([0, 1], 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (1, 3),
        (1, 4),
        (1, 5),
    ]

    p = pv.Plotter(
        shape=shape,
        row_weights=row_weights,
        col_weights=col_wights,
        groups=groups,
        window_size=(1800, 600),
    )

    # Compute common clims and isovalues for magnetization data
    scalar_min_mag = min(
        mag_data_true.min(), mag_data_rl.min(), mag_data_var.min()
    ).item()
    scalar_max_mag = max(
        mag_data_true.max(), mag_data_rl.max(), mag_data_var.max()
    ).item()
    clim_mag = (scalar_min_mag, scalar_max_mag)

    # Compute common clims for forward data
    scalar_min_forward = min(
        forward_data_true.min(), forward_data_rl.min(), forward_data_var.min()
    ).item()
    scalar_max_forward = max(
        forward_data_true.max(), forward_data_rl.max(), forward_data_var.max()
    ).item()
    clim_forward = (scalar_min_forward, scalar_max_forward)

    p.subplot(0, 0)
    plot_model_with_forward(
        mag_data_true,
        forward_data_true,
        spacing,
        height,
        plotter=p,
        clim_mag=clim_mag,
        clim_forward=clim_forward,
    )

    p.subplot(0, 1)
    plot_model_with_forward(
        mag_data_rl,
        forward_data_rl,
        spacing,
        height,
        plotter=p,
        clim_mag=clim_mag,
        clim_forward=clim_forward,
    )

    p.subplot(0, 2)
    plot_model_with_forward(
        mag_data_var,
        forward_data_var,
        spacing,
        height,
        plotter=p,
        clim_mag=clim_mag,
        clim_forward=clim_forward,
    )

    p.subplot(0, 3)
    plot_model_2d_pyvista(
        mag_data_true, cmap="rainbow", clim=clim_mag, n_slices=n_slices, plotter=p
    )
    p.add_title("True Model", font_size=6)

    p.subplot(0, 4)
    plot_model_2d_pyvista(
        mag_data_rl, cmap="rainbow", clim=clim_mag, n_slices=n_slices, plotter=p
    )
    p.add_title("Scale Space Model", font_size=6)

    p.subplot(0, 5)
    plot_model_2d_pyvista(
        mag_data_var, cmap="rainbow", clim=clim_mag, n_slices=n_slices, plotter=p
    )
    p.add_title("Variational Model", font_size=6)

    p.subplot(1, 3)
    surface_mesh = forward_data_to_grid(
        forward_data_true, spacing=spacing, height=height
    )
    p.add_mesh(
        surface_mesh,
        cmap="coolwarm",
        clim=clim_forward,
        show_edges=False,
        label="Forward Problem Data",
    )
    p.view_yx()
    p.view_vector(vector=[0, 0, 1], viewup=[-1, 0, 0])
    p.add_title("True Forward Data", font_size=6)
    p.add_scalar_bar(fmt="%.2f", n_labels=3, shadow=True)

    p.subplot(1, 4)
    surface_mesh = forward_data_to_grid(forward_data_rl, spacing=spacing, height=height)
    p.add_mesh(
        surface_mesh,
        cmap="coolwarm",
        clim=clim_forward,
        show_edges=False,
        label="Forward Problem Data",
    )
    p.view_yx()
    p.view_vector(vector=[0, 0, 1], viewup=[-1, 0, 0])
    p.add_title("Scale Space Forward Data", font_size=6)

    p.subplot(1, 5)
    surface_mesh = forward_data_to_grid(
        forward_data_var, spacing=spacing, height=height
    )
    p.add_mesh(
        surface_mesh,
        cmap="coolwarm",
        clim=clim_forward,
        show_edges=False,
        label="Forward Problem Data",
    )
    p.view_yx()
    p.view_vector(vector=[0, 0, 1], viewup=[-1, 0, 0])
    p.add_title("Variational Forward Data", font_size=6)

    p.link_views((0, 1, 2))
    p.link_views((3, 4, 5))
    p.link_views((6, 7, 8))

    return p
