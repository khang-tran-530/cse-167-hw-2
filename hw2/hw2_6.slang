
StructuredBuffer<float3> vertices;
StructuredBuffer<float3> vertex_colors;
uniform float4x4 MVP;

struct VertexOutput {
    float4 position : SV_Position;
    float3 color : COLOR0;
};

[shader("vertex")]
VertexOutput vertex_main(uint vid : SV_VertexID) {
    VertexOutput output;

    // Retrieve the 3D vertex position.
    float3 position = vertices[vid];

    // Transform the vertex into clip space.
    output.position = mul(MVP, float4(position, 1.0));

    // Pass the vertex color to the fragment shader.
    output.color = vertex_colors[vid];

    return output;
}

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_Target {

    // Use the interpolated color from the vertex shader.
    return float4(input.color, 1.0);
}
