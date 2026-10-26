
StructuredBuffer<float3> vertices;
StructuredBuffer<float3> vertex_normals;
StructuredBuffer<float2> vertex_uvs;

uniform float4x4 MVP;
uniform float4x4 model_matrix;
uniform float3x3 normal_matrix;

uniform float3 light_dir;
uniform float3 light_color;
uniform float3 camera_position;

uniform float3 Ks;
uniform float exponent;

Texture2D<float4> Kd_texture;
SamplerState Kd_sampler;

struct VertexOutput {
    float4 position : SV_Position;
    float3 world_position : TEXCOORD0;
    float3 world_normal : TEXCOORD1;
    float2 uv : TEXCOORD2;
};

[shader("vertex")]
VertexOutput vertex_main(uint vid : SV_VertexID) {
    VertexOutput output;

    // 1. Retrieve the vertex attributes.
    float3 position = vertices[vid];
    float3 normal = vertex_normals[vid];

    // 2. Transform the vertex to clip space.
    output.position = mul(MVP, float4(position, 1.0));

    // 3. Transform the vertex to world space.
    float4 world_pos = mul(model_matrix, float4(position, 1.0));
    output.world_position = world_pos.xyz / world_pos.w;

    // 4. Transform the normal using inverse transpose.
    output.world_normal = mul(normal_matrix, normal);

    // 5. Pass UV coordinates for texture sampling.
    output.uv = vertex_uvs[vid];

    return output;
}

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_Target {

    // 1. Sample the diffuse texture.
    float3 Kd = Kd_texture.Sample(Kd_sampler, input.uv).rgb;

    // 2. Normalize the surface normal.
    float3 n = normalize(input.world_normal);

    // 3. Direction toward the light.
    float3 l = normalize(light_dir);

    // 4. Direction from the surface toward the camera.
    float3 v = normalize(
        camera_position - input.world_position
    );

    // 5. Ambient lighting.
    float3 ambient = 0.2 * Kd;

    // 6. Diffuse lighting.
    float ndotl = max(0.0, dot(n, l));
    float3 diffuse = Kd * light_color * ndotl;

    // 7. Specular lighting using Phong reflection.
    float3 specular = float3(0.0);

    if (dot(n, l) > 0.0) {

        // Reflect incoming light about the normal.
        float3 r = reflect(-l, n);

        // Calculate the shiny highlight.
        float spec = pow(
            max(0.0, dot(r, v)),
            exponent
        );

        specular = Ks * light_color * spec;
    }

    // 8. Combine all lighting contributions.
    float3 final_color = ambient + diffuse + specular;

    return float4(final_color, 1.0);
}
