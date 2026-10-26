
[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float2 sp0,
                  uniform float2 sp1,
                  uniform float2 sp2,
                  uniform float inv_depth0,
                  uniform float inv_depth1,
                  uniform float inv_depth2,
                  uniform int2 aabb_min,
                  uniform float3 c0,
                  uniform float3 c1,
                  uniform float3 c2,
                  uniform float s,
                  uniform float a,
                  RWTexture2D<float4> color_image,
                  RWTexture2D<float> depth_image) {

    // Find the current pixel.
    int2 ipixel = int2(tid) + aabb_min;

    uint width, height;
    color_image.GetDimensions(width, height);

    // Ignore pixels outside the image.
    if (ipixel.x < 0 || ipixel.y < 0 ||
        ipixel.x >= width || ipixel.y >= height) {
        return;
    }

    // Step 1: Find the pixel center.
    float2 p = float2(ipixel) + float2(0.5, 0.5);

    // Step 2: Compute triangle edge functions.
    float2 e0 = sp1 - sp0;
    float2 e1 = sp2 - sp1;
    float2 e2 = sp0 - sp2;

    float E0 = e0.x * (p.y - sp0.y) -
               e0.y * (p.x - sp0.x);

    float E1 = e1.x * (p.y - sp1.y) -
               e1.y * (p.x - sp1.x);

    float E2 = e2.x * (p.y - sp2.y) -
               e2.y * (p.x - sp2.x);

    bool inside =
        (E0 >= 0.0 && E1 >= 0.0 && E2 >= 0.0) ||
        (E0 <= 0.0 && E1 <= 0.0 && E2 <= 0.0);

    if (!inside) {
        return;
    }

    // Step 3: Calculate barycentric coordinates.
    float area = E0 + E1 + E2;

    // Avoid division by zero for degenerate triangles.
    if (abs(area) < 1e-8) {
        return;
    }

    float b0 = E1 / area;
    float b1 = E2 / area;
    float b2 = E0 / area;

    // Step 4: Interpolate inverse depth.
    float inv_depth =
        b0 * inv_depth0 +
        b1 * inv_depth1 +
        b2 * inv_depth2;

    if (inv_depth <= 0.0) {
        return;
    }

    float depth = 1.0 / inv_depth;

    // Step 5: Test whether this triangle is closer.
    if (depth >= depth_image[ipixel]) {
        return;
    }

    // Update the depth buffer.
    depth_image[ipixel] = depth;

    // Step 6: Perspective-correct barycentric coordinates.
    float w0 = b0 * inv_depth0 / inv_depth;
    float w1 = b1 * inv_depth1 / inv_depth;
    float w2 = b2 * inv_depth2 / inv_depth;

    // Step 7: Interpolate the vertex colors.
    float3 pixel_color =
        w0 * c0 +
        w1 * c1 +
        w2 * c2;

    // Step 8: Write the final pixel color.
    color_image[ipixel] = float4(pixel_color, 1.0);
}
