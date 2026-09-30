"""Compare grain elongation and direction on a fixed set of 64 seeds."""
import argparse
import json
from pathlib import Path

import numpy as np

from applications.polycrystal.model import generate_seeds, voxel_grains, create_mesh
from applications.polycrystal.render import render_comparison


def run_example(output, divisions=192, check_refinement=False, write_mesh=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    bounds = np.array([[-.15, .15]]*3)  # mm
    seeds = generate_seeds(bounds, 64, seed=42)
    cases = [(1, 0), (2, 0), (4, 0), (4, 45), (4, 90)]
    fields, results = {}, []
    for stretch, angle in cases:
        print(f'Generating r={stretch}, theta={angle}, {divisions}^3 cells', flush=True)
        theta = np.deg2rad(angle)
        direction = np.array([np.sin(theta), 0., np.cos(theta)])
        labels = voxel_grains(bounds, (divisions,)*3, seeds, stretch, direction)
        fields[stretch, angle] = labels
        fractions = np.bincount(labels.ravel(), minlength=len(seeds))/labels.size
        result = {'stretch': stretch, 'angle_deg': angle, 'divisions': divisions,
                  'elements': labels.size, 'spacing_mm': .3/divisions,
                  'represented_grains': int(np.count_nonzero(fractions)),
                  'min_cells_per_grain': int(np.bincount(labels.ravel(), minlength=len(seeds)).min())}
        if check_refinement:
            refined_n = 4*divisions//3
            finer = voxel_grains(bounds, (refined_n,)*3, seeds, stretch, direction)
            fine_fractions = np.bincount(finer.ravel(), minlength=len(seeds))/finer.size
            difference = fractions - fine_fractions
            result.update(refined_divisions=refined_n,
                          grain_volume_total_variation=float(.5*np.abs(difference).sum()),
                          max_absolute_volume_fraction_change=float(np.abs(difference).max()))
            del finer
        np.savez_compressed(output / f'r{stretch}_theta{angle}.npz', bounds=bounds,
                            divisions=np.array([divisions]*3), seeds=seeds,
                            grain_ids=labels, stretch=stretch, direction=direction)
        results.append(result)
    print('Rendering comparisons', flush=True)
    render_comparison([fields[r, 0] for r in (1, 2, 4)],
                      [f'r = {r}' for r in (1, 2, 4)], output / 'stretch_ratios', len(seeds))
    render_comparison([fields[4, angle] for angle in (0, 45, 90)],
                      [f'{angle}\N{DEGREE SIGN}' for angle in (0, 45, 90)], output / 'directions', len(seeds))
    if write_mesh:
        import meshio
        print('Writing HEX8 meshes', flush=True)
        points, cells = create_mesh(bounds, (divisions,)*3)
        for (stretch, angle), labels in fields.items():
            meshio.write(output / f'r{stretch}_theta{angle}.vtu', meshio.Mesh(
                points, [('hexahedron', cells)], cell_data={'grain_id': [labels.ravel()]}))
    (output / 'validation.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(results, indent=2), flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'output')
    parser.add_argument('--divisions', type=int, default=192, help='cells per edge; 48 for a quick preview')
    parser.add_argument('--check-refinement', action='store_true', help='compare grain volumes on a finer grid')
    parser.add_argument('--write-mesh', action='store_true', help='also export the full HEX8 grids as VTU')
    args = parser.parse_args()
    run_example(args.output, args.divisions, args.check_refinement, args.write_mesh)
