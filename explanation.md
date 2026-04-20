# How the Fake Bevel Baker Works

A short explanation of the CG concepts this tool relies on, and how each one is used inside the pipeline.

## The problem

A normal map that visually rounds off the hard edges of a low-poly mesh is usually produced by modelling a high-poly version and baking the difference between the two surface directions into a texture. This tool skips the high-poly step. It evaluates Blender Cycles' `svm_bevel` shader node per texel, treating the low-poly mesh as both the sampling target and the shading surface. The output is a tangent-space normal map that, when applied to the same low-poly mesh in any PBR viewer, makes hard edges look like they were physically chamfered.

## Two meshes, not one

Most of the pipeline operates on two separate representations of the same geometry, kept in lockstep:

- The **split mesh** is used for ray tracing. At every vertex, faces that meet at a sharp dihedral angle become separate vertex copies, each carrying its own angle-weighted smooth normal (the rule Blender calls "shade smooth"). Hard edges are encoded as normal discontinuities between duplicated vertices at the same position.
- The **tangent mesh** is the vertex layout as the engine sees it — one vertex per unique `(position, uv, normal)` triple from the source file. The UV unwrap and the authored vertex normals come straight from here. MikkTSpace tangents are computed on this mesh, not the split mesh.

The two meshes describe the same triangles in the same order, so a triangle index means the same thing on both sides. That equivalence is what lets the tool sample on the split mesh and project onto the tangent mesh without any extra remapping.

## UV rasterization: from texel to world point

Each texel of the output image needs a 3D shading point on the surface. The rasterizer walks every UV triangle's 2D bounding box in texel space and tests each candidate texel.

Inclusion is **conservative**: a texel is claimed by a triangle if any part of the triangle overlaps the texel's 1×1 square. A cheaper center-only test — "is the texel's midpoint inside the triangle?" — drops a thin strip of edge texels, which looks like the UV island is inset by one pixel from its real boundary. This matters because the edge-padding step that runs later can then only start filling *outside* that inset, not at the true island edge.

For each valid texel the tool stores the triangle index and the barycentric weights `(u, v, w)` of the texel's center relative to that triangle. The shading point `P` is reconstructed by weighting the split mesh's three triangle vertices by those barycentrics. The geometric normal `Ng` is the triangle's face normal. These two values drive everything that comes next.

## The disk-sample probe

At each texel we have `P` and `Ng`. The algorithm looks at nearby geometry within a fixed world-space radius `R` (the "bevel radius") and blends its normals into `Ng`. The sampling strategy is lifted from Cycles directly.

Per sample:

1. A local orthonormal frame is built around `Ng`: two tangent vectors `T` and `B` perpendicular to `Ng`.
2. One of the three axes (`Ng`, `T`, `B`) is picked at random with probabilities 0.5, 0.25, 0.25 respectively. Whichever axis is chosen becomes the probe direction for this sample.
3. A disk of radius `R` is placed perpendicular to the chosen axis. A point on the disk is sampled using the cubic BSSRDF inverse-CDF — the same falloff profile Cycles uses for subsurface scattering — so samples concentrate near `P` and fall off smoothly out to `R`. The cubic CDF's inverse requires solving a quintic, which is done with ten Newton-Raphson iterations.
4. The sampled disk point gives a radius `r` and a height `h = sqrt(R² − r²)`. The probe ray starts at `P + axis·h + disk_offset` and travels in direction `−axis` for a distance of `2h`. Geometrically this ray enters a sphere of radius `R` centered at `P` from one side and exits out the other.

This is a 3-axis MIS strategy: running probes along `Ng` alone would miss surfaces that are roughly perpendicular to `Ng` (exactly where the bevel is), so the tangent and bitangent axes fill in those directions.

## Multi-hit ray casting

A single probe ray can pass through more than one triangle of the mesh. The tool needs every hit, not just the first, because each nearby surface contributes a normal to the blend.

The ray tracer is Intel Embree via the `embreex` Python binding. The binding exposes only single-hit scene queries, so multi-hit is built on top: fire the ray, record the hit, advance the ray origin by `t_hit + ε` along the direction, fire again, repeat. The loop caps at 4 hits per probe, matching Cycles' `LOCAL_MAX_HITS`. Rays that miss, run out of budget, or hit their max count are dropped from the active set between iterations, so each Embree call only sees rays still in flight.

Every recorded hit carries: the triangle it hit, the hit's barycentric coordinates on that triangle, the triangle's geometric normal, and the cumulative distance from `P`.

## Weighting and accumulating normals

A hit's contribution is not simply its normal. Each hit is weighted by how "likely" it would have been found by each of the three possible probe axes, combined via multiple importance sampling.

For each hit:

- `pdf_X = pick_pdf_X · |dot(disk_X, hit_Ng)|` for each of the three local-frame axes `X ∈ {N, T, B}`. The cosine term is a geometric probability: a surface nearly aligned with an axis is cheap to hit along that axis, so its pdf is high.
- The MIS weight is `pdf_N / (pdf_N² + pdf_T² + pdf_B²)` — the power heuristic, combined across axes.
- That weight is scaled by `pdf_cubic(r_real) / pdf_cubic(disk_r)`, where `r_real` is the actual distance from `P` to the hit point and `disk_r` is the disk radius the sampler drew. This reweights for the fact that the sample landed on a different spot than it was asked to.

The smoothed normal at the hit — barycentrically interpolated from the split mesh's per-vertex normals — is multiplied by this scalar weight and added to an accumulator.

After all `num_samples` samples are processed, the accumulator is normalized. If it's numerically zero (can happen in texels where no sample ever found any nearby geometry) the output falls back to `Ng`.

## Tangent space: MikkTSpace

The baked image lives in the engine's tangent space. When the engine renders, it reads a tangent-space normal from the image, interpolates the vertex `T`, `B`, `N` basis across the triangle, and combines the two to recover a world-space normal. For the output to decode to exactly the world-space normal the baker computed, the basis used at bake time has to match the basis the engine will use at render time.

This tool uses **MikkTSpace** — the open-standard tangent basis used by Blender, Substance, Marmoset, Unity, and Unreal. The reference C implementation is compiled to a shared library and called via ctypes. It runs on the tangent mesh (the engine-authored vertex layout), producing one tangent vector and one handedness sign per face corner. The bitangent is reconstructed in the shader as `sign · cross(N, T)`.

At each texel, the tangent and bitangent are obtained by barycentrically interpolating the face-corner values of the texel's triangle — not by looking up a per-vertex array. MikkT deliberately does not produce one tangent per vertex; corners of the same vertex can have different tangents if the UV unwrap requires it, and the engine reconstructs the basis from corner data the same way. The world-space smoothed normal is then dotted against `T`, `B`, and the vertex normal, giving the tangent-space coordinates. These coordinates are encoded into the `[0, 1]` range (`value = x · 0.5 + 0.5`) and written to the image channel. Flat areas land at `(0.5, 0.5, 1.0)`.

## Edge padding

The valid texels form UV islands surrounded by unbaked texture space. A mipmap chain and bilinear sampling both pull color across island borders; without padding the unbaked pixels bleed in as black, which looks like a dark outline around every island.

After the main bake, a single pass computes the distance from every invalid texel to the nearest valid one (and the coordinates of that nearest valid texel). Every invalid texel within `N` texels of a valid one inherits the value of its nearest valid neighbor. The default padding ring is 16 texels, enough to survive three levels of mipmap bilinear downsampling without bleed.

## Sampling pattern

Monte Carlo variance shows up as visible noise, which at high resolution looks like slight pixel-scale blockiness. The sampler used here is designed to minimize that at the low-to-medium sample counts a user will realistically pick (16 to 256).

- The sample-index dimension uses an **Owen-scrambled Sobol** sequence. Sobol has low-discrepancy behavior across the full count; Owen scrambling randomizes it without disturbing that property, so every texel sees a well-spread set of samples.
- The texel dimension uses **Cranley-Patterson rotation**: every texel is assigned a random 2D offset at bake start, and that offset is added (modulo 1) to each Sobol sample before use. Adjacent texels therefore explore different rotations of the same sequence, so correlated structure in the Sobol points does not show up as correlated noise across the image.

The random state is seeded from a single integer, so the same inputs always produce the same output on the CPU path.

## Determinism and fallback

The bake is deterministic for a fixed seed: same mesh, same parameters, same seed → bit-identical image. When `sum_N` is numerically zero at a texel (no sample ever reached any geometry), the output falls back to the geometric normal `Ng` projected into tangent space — i.e., the flat shading value for that surface, which is `(0.5, 0.5, 1.0)` for a well-oriented texel.

## Summary of the data flow

1. Load mesh, preserving per-corner UV and normal data.
2. Build the split mesh by duplicating vertices across hard edges.
3. MikkTSpace tangents on the tangent mesh, per face corner.
4. Rasterize UVs conservatively to get a `(triangle, bary)` record per valid texel.
5. For each valid texel, reconstruct world-space `P` and `Ng`.
6. Per sample: pick an axis, sample a disk point, fire a multi-hit probe ray, weight and accumulate every hit's smoothed normal.
7. Normalize and project the accumulated world-space normal into MikkTSpace tangent space.
8. Encode to `[0, 1]`, pad island borders by dilation, write PNG or EXR.
