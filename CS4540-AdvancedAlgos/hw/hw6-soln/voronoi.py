import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite Voronoi regions in a 2D diagram to finite regions.
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # Finite region
            new_regions.append(vertices)
            continue

        # Reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # Finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge
            t = vor.points[p2] - vor.points[p1]  # Tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # Normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # Sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # Finish the region
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

points = np.array([[0, -0.3], [0.75, 0.75], [1.5, -0.3], [2, 1]])
vor = Voronoi(points)
regions, vertices = voronoi_finite_polygons_2d(vor, 1000)  # Large radius to extend far

fig, ax = plt.subplots()

# Plot and slightly shade circles around the points
for point in points:
    circle = plt.Circle((point[0], point[1]), 1, color='blue', \
        fill=True, alpha=0.1, linestyle='--', linewidth=1.5)
    ax.add_artist(circle)

# Set equal aspect ratio
ax.set_aspect('equal', 'box')

# find the average of the points and then set the limits of the plot
center = points.mean(axis=0)
ax.set_xlim(center[0] - 2.2, center[0] + 2.2)
ax.set_ylim(center[1] - 2.2, center[1] + 2.2)

# Plot Voronoi
for region in regions:
    polygon = vertices[region]
    ax.fill(*zip(*polygon), alpha=0.4)

ax.plot(points[:, 0], points[:, 1], 'ko')

plt.savefig('voronoi.pdf', bbox_inches='tight')