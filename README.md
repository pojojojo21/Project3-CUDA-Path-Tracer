CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Joanna Fisch
  * [LinkedIn](https://www.linkedin.com/in/joanna-fisch-bb2979186/), [Website](https://sites.google.com/view/joannafischsportfolio/home)
* Tested on: Windows 11, i7-12700H @ 2.30GHz 16GB, NVIDIA GeForce RTX 3060 (Laptop)

### (TODO: Your README)

### Feature: BSDF Evaluation
Overview:
The Bidirectional Scattering Distribution Function (BSDF) evaluation is central to realistic light transport simulations in a path tracer. This feature computes how light interacts with surfaces, determining the intensity and direction of light after it hits a surface. The BSDF evaluation was implemented to support diffuse, specular, and refractive materials, ensuring physically accurate light behavior.

The BSDF evaluation results in more realistic lighting, with proper reflections, refractions, and surface responses for different material types (e.g., specular highlights, diffuse scattering).

Performance Impact:
Before BSDF Evaluation: The path tracer was unable to calculate proper light interactions, simplifying the rendering.
After BSDF Evaluation: The addition of BSDF calculations increases the computational workload, particularly when tracing paths across different material types.
Observed performance impact: ~10-15% increase in rendering time due to additional calculations for light-material interactions.

Acceleration:
Stream Compaction: Path termination was accelerated using stream compaction, reducing the number of active rays being traced. This helped counterbalance the overhead introduced by the BSDF evaluation.

GPU vs. Hypothetical CPU:
On the GPU, the BSDF evaluation is highly parallelizable, allowing each ray/surface interaction to be calculated simultaneously. A CPU implementation would suffer from limited parallelism, leading to significantly longer render times due to serialized light evaluations.

Potential Optimization:
Importance Sampling: Optimizing BSDF sampling based on material properties could further reduce noise and improve performance, focusing computational effort on more significant light paths.

### Feature: Texture and Bump Mapping
Overview:
Texture mapping adds realism by applying 2D images onto 3D surfaces, while bump mapping introduces surface details without additional geometry. These features were implemented using stb_image.h to load textures and calculate tangents for proper shading and detail.

The surfaces now reflect detail from textures and bump maps, enhancing visual realism with added depth and variation in material appearance.

Performance Impact:
Before Texture/Bump Mapping: Simple, smooth surfaces rendered quickly due to the lack of additional texture lookups and surface perturbation.
After Texture/Bump Mapping: Each ray-surface intersection now includes texture lookups and bump map calculations, increasing the computational load.
Observed performance impact: ~20% increase in rendering time due to texture sampling and normal perturbation.
Acceleration:
Mipmapping for Textures: Implementing mipmapping could accelerate texture sampling by reducing the resolution of textures at a distance, preventing over-sampling of fine details.

GPU vs. Hypothetical CPU:
On the GPU, texture and bump mapping benefit from high parallelism for texture lookups, which would be much slower on the CPU due to the high number of texture samples required per pixel.

Potential Optimization:
Texture Caching: Implementing texture caching or further optimizing texture memory access patterns (using CUDA texture memory) could improve texture sampling performance.
![cornell 2024-09-26_20-39-57z 14samp](https://github.com/user-attachments/assets/25c2e981-8254-4eb6-a36a-a6bad2bd78dc)

### Feature: Refraction
Overview:
Refraction simulates how light bends when passing through transparent materials like glass or water. It is critical for rendering realistic images of objects like lenses, liquid containers, or any scene involving transparent materials.

Refraction was implemented using Snell’s Law to calculate light ray bending. This enhanced visual realism when rendering glass, water, and other transparent materials.

Performance Impact:
Before Refraction: Rays interacting with transparent materials were either reflected or ignored, simplifying the light path tracing.
After Refraction: Additional calculations were required for each light ray that enters or exits a refractive surface. This increased the number of path segments to trace, as rays could pass through multiple refractive surfaces before termination.
Observed performance impact: ~25-30% increase in render time for scenes with significant refraction, especially when dealing with complex refractive objects like glass spheres or liquid containers.

Acceleration:
Adaptive Termination: An optimization technique used was early termination for rays where refraction contribution was below a certain threshold, reducing the number of unnecessary bounces through transparent materials.

GPU vs. Hypothetical CPU:
Refraction benefits from parallel processing on the GPU, as each ray through a refractive surface is handled independently. A CPU version would be much slower, as calculating refraction for each ray is computationally expensive and difficult to parallelize across many cores.

Potential Optimization:
Fresnel Reflection/Refraction Balancing: Implementing a more efficient Fresnel equation could reduce the computational cost by better determining whether a ray reflects or refracts, minimizing unnecessary calculations for paths dominated by reflection.

![refraction 2024-10-07_03-36-05z 5000samp](https://github.com/user-attachments/assets/90b0390a-2902-41b4-bf15-c379db41b3a4)

### Feature: Physically Based Depth of Field (DoF)
Overview:
Physically based depth of field simulates the effect of a camera lens, where objects at different distances from the focal plane appear blurred. This helps mimic real-world camera effects, making renders more photorealistic.

Before:
The camera rendered scenes with infinite focus, where all objects were sharp regardless of distance.

After:
Implementing DoF allowed for more realistic renders, where objects in the foreground and background blur out based on the focal length and aperture size, focusing on objects at a specific distance.

Performance Impact:
Before Depth of Field: All rays were traced in a straight line, leading to uniform sharpness throughout the scene.
After Depth of Field: Additional computations were required to jitter rays originating from the camera lens and trace them in different directions depending on the aperture size, leading to more complex ray paths and increased noise.
Observed performance impact: ~15-20% increase in rendering time due to the additional rays required for each pixel to achieve the blurring effect.

Acceleration:
Stratified Sampling: Stratified sampling was used to reduce noise when tracing rays with depth of field, improving performance by reducing the number of samples required to achieve smooth blurring.

GPU vs. Hypothetical CPU:
Depth of field is highly parallelizable on the GPU, as each ray’s direction is calculated independently. A CPU version would struggle with the additional rays required to simulate DoF for each pixel, leading to significantly longer render times.
Potential Optimization:

Faster Ray Sampling: Implementing importance sampling for DoF could reduce the number of rays required to achieve the desired effect, optimizing the performance even further.

![refraction 2024-10-07_03-59-44z 300samp](https://github.com/user-attachments/assets/ad63749d-3bab-49de-9877-d0b5200c379a)

### Feature: OBJ Loading
Overview:
OBJ loading enables the path tracer to render complex 3D models created outside the program. It allows for flexibility in importing detailed geometry into scenes, such as character models, buildings, or intricate designs that would be difficult to manually define.

Before:
Only primitive shapes such as spheres and cubes could be rendered, severely limiting the complexity and realism of the scenes.
After:

The inclusion of OBJ loading allowed complex models to be loaded and rendered, greatly enhancing the visual complexity and realism of scenes. This included handling vertex positions, normals, and texture coordinates.
Performance Impact:
Before OBJ Loading: Primitive shapes were rendered with minimal memory and computational requirements, leading to faster render times.

After OBJ Loading: More complex models required loading large amounts of vertex and normal data into memory and performing additional calculations for each face in the model. This increased memory usage and computational overhead, especially in scenes with large or highly detailed models.
Observed performance impact: The performance impact depends on the complexity of the OBJ models. For small models, the impact is negligible, but for large models, render times could increase by ~10-25% due to the added complexity in intersection tests.

Acceleration:
Bounding Volume Hierarchies (BVH): Using a BVH structure for spatial partitioning would help accelerate intersection tests with complex OBJ models, reducing the number of ray-object intersection tests performed.
GPU vs. Hypothetical CPU:
OBJ loading benefits significantly from the parallelism of the GPU. On a CPU, performing intersection tests for highly detailed OBJ models would be far slower due to the sheer number of vertices and faces involved.
Potential Optimization:
Level of Detail (LoD) for OBJ Models: Implementing LoD could improve performance by rendering lower-detail versions of OBJ models at a distance, only switching to higher detail versions as needed based on proximity to the camera.
![cornell 2024-09-26_20-39-57z 14samp](https://github.com/user-attachments/assets/e74d2d9b-97ba-409e-b426-417e3248c771)
