
[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float2 sp0,
                  uniform float2 sp1,
                  uniform float2 sp2,
                  uniform int2 aabb_min,
                  uniform float3 color,
                  RWTexture2D<float4> image) {

    int2 ipixel = int2(tid) + int2(aabb_min);

    uint width, height;
    image.GetDimensions(width, height);

    if (ipixel.x < 0 || ipixel.y < 0 ||
        ipixel.x >= width || ipixel.y >= height) {
        return;
    }

    // Step 1: Calculate the center of the current pixel.
    float2 p = float2(ipixel) + float2(0.5, 0.5);

    // Step 2: Calculate the triangle's edge vectors.
    float2 e0 = sp1 - sp0;
    float2 e1 = sp2 - sp1;
    float2 e2 = sp0 - sp2;

    // Step 3: Calculate edge equations using 2D cross products.
    float c0 = e0.x * (p.y - sp0.y) -
               e0.y * (p.x - sp0.x);

    float c1 = e1.x * (p.y - sp1.y) -
               e1.y * (p.x - sp1.x);

    float c2 = e2.x * (p.y - sp2.y) -
               e2.y * (p.x - sp2.x);

    // Step 4: Check whether the pixel is inside.
    bool inside =
        (c0 >= 0.0 && c1 >= 0.0 && c2 >= 0.0) ||
        (c0 <= 0.0 && c1 <= 0.0 && c2 <= 0.0);

    // Step 5: Color only pixels inside the triangle.
    if (inside) {
        image[ipixel] = float4(color, 1.0);
    }
}
