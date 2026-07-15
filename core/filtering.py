def mask_inductive_points(dataset) -> None: #remove inductive tail
    """Mask points with a positive imaginary impedance (inductive artifacts), in place."""
    Z = dataset.data.get_impedances(masked=None)  # all points, incl. masked
    dataset.data.set_mask({i: bool(z.imag > 0) for i, z in enumerate(Z)})


def clear_mask(dataset) -> None: #re-add the inductive points
    """Unmask all points of an EISDataset (in place)."""
    dataset.data.set_mask({})

#def mask_diffusion_points(dataset):
# this is a placeholder for mathematically removing the diffusion points