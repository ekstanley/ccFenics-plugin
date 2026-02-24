# Gmsh Patterns Reference

## Common Geometry Patterns for DOLFINx

### 2D Rectangle with Hole

```python
import gmsh
gmsh.initialize()
gmsh.model.add("plate_with_hole")

# Outer rectangle
rect = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
# Inner circle (hole)
circle = gmsh.model.occ.addDisk(0.5, 0.5, 0, 0.15, 0.15)
# Boolean difference
gmsh.model.occ.cut([(2, rect)], [(2, circle)])
gmsh.model.occ.synchronize()

# Physical groups for BCs
gmsh.model.addPhysicalGroup(1, [left_lines], tag=1, name="left")
gmsh.model.addPhysicalGroup(1, [right_lines], tag=2, name="right")
gmsh.model.addPhysicalGroup(1, [hole_lines], tag=3, name="hole")
gmsh.model.addPhysicalGroup(2, [surface], tag=100, name="domain")

gmsh.model.mesh.generate(2)
gmsh.write("plate_hole.msh")
gmsh.finalize()
```

### 3D Cylinder

```python
import gmsh
gmsh.initialize()
gmsh.model.add("cylinder")

cyl = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, 1, 0.5)
gmsh.model.occ.synchronize()

# Tag inlet (z=0), outlet (z=1), wall
# ... assign physical groups
gmsh.model.mesh.generate(3)
gmsh.write("cylinder.msh")
gmsh.finalize()
```

### L-Shaped Domain (Re-entrant Corner)

```python
import gmsh
gmsh.initialize()
gmsh.model.add("L_shape")

r1 = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
r2 = gmsh.model.occ.addRectangle(0, 0, 0, 0.5, 0.5)
gmsh.model.occ.cut([(2, r1)], [(2, r2)])
gmsh.model.occ.synchronize()

# Refine near re-entrant corner at (0.5, 0.5)
gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "PointsList", [corner_point])
gmsh.model.mesh.field.add("Threshold", 2)
gmsh.model.mesh.field.setNumber(2, "InField", 1)
gmsh.model.mesh.field.setNumber(2, "SizeMin", 0.01)
gmsh.model.mesh.field.setNumber(2, "SizeMax", 0.1)
gmsh.model.mesh.field.setNumber(2, "DistMin", 0.05)
gmsh.model.mesh.field.setNumber(2, "DistMax", 0.3)
gmsh.model.mesh.field.setAsBackgroundMesh(2)

gmsh.model.mesh.generate(2)
gmsh.write("L_shape.msh")
gmsh.finalize()
```

## Mesh Size Control

### Uniform Size

```python
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.05)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.05)
```

### Graded Mesh (Distance Field)

```python
# Fine near a point, coarse far away
gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "PointsList", [point_tag])

gmsh.model.mesh.field.add("Threshold", 2)
gmsh.model.mesh.field.setNumber(2, "InField", 1)
gmsh.model.mesh.field.setNumber(2, "SizeMin", h_min)
gmsh.model.mesh.field.setNumber(2, "SizeMax", h_max)
gmsh.model.mesh.field.setNumber(2, "DistMin", r_min)
gmsh.model.mesh.field.setNumber(2, "DistMax", r_max)

gmsh.model.mesh.field.setAsBackgroundMesh(2)
```

### Boundary Layer Mesh

```python
# Structured layers near a surface
gmsh.model.mesh.field.add("BoundaryLayer", 1)
gmsh.model.mesh.field.setNumbers(1, "CurvesList", [boundary_curves])
gmsh.model.mesh.field.setNumber(1, "Size", first_layer_thickness)
gmsh.model.mesh.field.setNumber(1, "Ratio", growth_ratio)
gmsh.model.mesh.field.setNumber(1, "NbLayers", num_layers)
```

## Element Type Control

```python
# Force triangles (default)
gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay

# Force quadrilaterals
gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.option.setNumber("Mesh.Algorithm", 8)  # Frontal-Delaunay for Quads

# 3D: tetrahedra (default)
gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay
```

## DOLFINx Import Notes

After importing with `create_custom_mesh`:

- Cell tags become available as `{mesh_name}_cell_tags`
- Facet tags become available as `{mesh_name}_facet_tags`
- Use facet tags directly with `apply_boundary_condition(boundary_tag=N)`
- Use cell tags with `create_submesh` for multi-material problems
