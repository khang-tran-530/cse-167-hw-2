
[shader("compute")]
[numthreads(64, 1, 1)]
void clear_buffers(
    uint tid : SV_DispatchThreadID,
    uniform uint num_pixels,
    RWStructuredBuffer<uint> depth_buffer,
    RWStructuredBuffer<uint> triangle_id_buffer)
{
    if (tid >= num_pixels)
        return;

    depth_buffer[tid] = 0xffffffff;
    triangle_id_buffer[tid] = 0xffffffff;
}

// Transform all 3D vertices into projected screen coordinates.
[shader("compute")]
[numthreads(64, 1, 1)]
void transform_vertices(
    uint tid : SV_DispatchThreadID,
    uniform uint num_vertices,
    uniform uint vertex_offset,
    uniform float4x4 MVP,
    StructuredBuffer<float3> vertices,
    RWStructuredBuffer<float3> projected_vertices)
{
    if (tid >= num_vertices)
        return;

    uint i = vertex_offset + tid;
    float4 p = mul(MVP, float4(vertices[i], 1.0));
    projected_vertices[i] = p.xyz / p.w;
}

// Compute the signed area of a 2D parallelogram.
float edge_function(float2 a, float2 b, float2 p)
{
    float2 e = b - a;
    float2 q = p - a;
    return e.x * q.y - e.y * q.x;
}

// Calculate barycentric coordinates of a screen-space point.
float3 barycentric(
    float2 p,
    float2 p0,
    float2 p1,
    float2 p2,
    float area)
{
    float b0 = edge_function(p1, p2, p) / area;
    float b1 = edge_function(p2, p0, p) / area;
    float b2 = edge_function(p0, p1, p) / area;

    return float3(b0, b1, b2);
}

// Find the nearest depth for every covered pixel.
[shader("compute")]
[numthreads(64, 1, 1)]
void rasterize_depth(
    uint tid : SV_DispatchThreadID,
    uniform uint num_triangles,
    uniform uint width,
    uniform uint height,
    uniform float s,
    uniform float a,
    StructuredBuffer<uint3> faces,
    StructuredBuffer<float3> projected_vertices,
    RWStructuredBuffer<uint> depth_buffer)
{
    if (tid >= num_triangles)
        return;

    uint3 face = faces[tid];

    float3 v0 = projected_vertices[face.x];
    float3 v1 = projected_vertices[face.y];
    float3 v2 = projected_vertices[face.z];

    float2 p0 = v0.xy;
    float2 p1 = v1.xy;
    float2 p2 = v2.xy;

    // Skip degenerate triangles.
    float area = edge_function(p0, p1, p2);
    if (abs(area) < 1e-8)
        return;

    // Compute screen-space bounding box.
    int min_x = max(0, (int)floor(min(p0.x, min(p1.x, p2.x))));
    int min_y = max(0, (int)floor(min(p0.y, min(p1.y, p2.y))));

    int max_x = min(int(width) - 1,
                    (int)ceil(max(p0.x, max(p1.x, p2.x))));
    int max_y = min(int(height) - 1,
                    (int)ceil(max(p0.y, max(p1.y, p2.y))));

    // Visit pixels covered by this triangle's bounding box.
    for (int y = min_y; y <= max_y; y++) {
        for (int x = min_x; x <= max_x; x++) {

            float2 p = float2(x + 0.5, y + 0.5);

            float3 b = barycentric(p, p0, p1, p2, area);

            // Skip pixels outside the triangle.
            if (b.x < 0.0 || b.y < 0.0 || b.z < 0.0)
                continue;

            // Perspective-correct inverse depth.
            float inv_depth =
                b.x * v0.z +
                b.y * v1.z +
                b.z * v2.z;

            if (inv_depth <= 0.0)
                continue;

            float depth = 1.0 / inv_depth;

            uint index = uint(y) * width + uint(x);

            // Convert the positive float to its bit pattern.
            uint depth_bits = asuint(depth);

            // Keep the nearest depth safely across GPU threads.
            InterlockedMin(depth_buffer[index], depth_bits);
        }
    }
}

// Identify which triangle is closest at each pixel.
[shader("compute")]
[numthreads(64, 1, 1)]
void rasterize_id(
    uint tid : SV_DispatchThreadID,
    uniform uint num_triangles,
    uniform uint width,
    uniform uint height,
    uniform float s,
    uniform float a,
    StructuredBuffer<uint3> faces,
    StructuredBuffer<float3> projected_vertices,
    RWStructuredBuffer<uint> depth_buffer,
    RWStructuredBuffer<uint> triangle_id_buffer)
{
    if (tid >= num_triangles)
        return;

    uint3 face = faces[tid];

    float3 v0 = projected_vertices[face.x];
    float3 v1 = projected_vertices[face.y];
    float3 v2 = projected_vertices[face.z];

    float2 p0 = v0.xy;
    float2 p1 = v1.xy;
    float2 p2 = v2.xy;

    float area = edge_function(p0, p1, p2);
    if (abs(area) < 1e-8)
        return;

    int min_x = max(0, (int)floor(min(p0.x, min(p1.x, p2.x))));
    int min_y = max(0, (int)floor(min(p0.y, min(p1.y, p2.y))));

    int max_x = min(int(width) - 1,
                    (int)ceil(max(p0.x, max(p1.x, p2.x))));
    int max_y = min(int(height) - 1,
                    (int)ceil(max(p0.y, max(p1.y, p2.y))));

    for (int y = min_y; y <= max_y; y++) {
        for (int x = min_x; x <= max_x; x++) {

            float2 p = float2(x + 0.5, y + 0.5);

            float3 b = barycentric(p, p0, p1, p2, area);

            if (b.x < 0.0 || b.y < 0.0 || b.z < 0.0)
                continue;

            float inv_depth =
                b.x * v0.z +
                b.y * v1.z +
                b.z * v2.z;

            if (inv_depth <= 0.0)
                continue;

            float depth = 1.0 / inv_depth;
            uint index = uint(y) * width + uint(x);

            // Check whether this triangle has the nearest depth.
            if (asuint(depth) == depth_buffer[index]) {
                // If depths tie, choose the smallest triangle ID.
                InterlockedMin(triangle_id_buffer[index], tid);
            }
        }
    }
}

// Compute the visible pixel colors.
[shader("compute")]
[numthreads(8, 8, 1)]
void shade(
    uint2 tid : SV_DispatchThreadID,
    uniform uint width,
    uniform uint height,
    uniform float s,
    uniform float a,
    StructuredBuffer<uint3> faces,
    StructuredBuffer<float3> vertex_colors,
    StructuredBuffer<float3> projected_vertices,
    StructuredBuffer<uint> triangle_id_buffer,
    RWTexture2D<float4> image)
{
    if (tid.x >= width || tid.y >= height)
        return;

    uint index = tid.y * width + tid.x;

    uint triangle_id = triangle_id_buffer[index];

    // Leave background unchanged if no triangle is visible.
    if (triangle_id == 0xffffffff)
        return;

    uint3 face = faces[triangle_id];

    float3 v0 = projected_vertices[face.x];
    float3 v1 = projected_vertices[face.y];
    float3 v2 = projected_vertices[face.z];

    float2 p = float2(tid) + float2(0.5, 0.5);

    float area = edge_function(v0.xy, v1.xy, v2.xy);

    if (abs(area) < 1e-8)
        return;

    float3 b = barycentric(
        p, v0.xy, v1.xy, v2.xy, area
    );

    // Interpolate inverse depth.
    float inv_depth =
        b.x * v0.z +
        b.y * v1.z +
        b.z * v2.z;

    if (inv_depth <= 0.0)
        return;

    // Perspective-correct barycentric weights.
    float3 w = float3(
        b.x * v0.z,
        b.y * v1.z,
        b.z * v2.z
    ) / inv_depth;

    // Retrieve vertex colors.
    float3 c0 = vertex_colors[face.x];
    float3 c1 = vertex_colors[face.y];
    float3 c2 = vertex_colors[face.z];

    // Interpolate final color.
    float3 color =
        w.x * c0 +
        w.y * c1 +
        w.z * c2;

    image[tid] = float4(color, 1.0);
}
