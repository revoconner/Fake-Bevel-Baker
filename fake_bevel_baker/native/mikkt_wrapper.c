/* Thin ctypes-friendly wrapper around Morten Mikkelsen's MikkTSpace.
 * No Python callbacks: all mesh data lives in flat arrays supplied by
 * the caller; the SMikkTSpace callbacks simply read from those buffers.
 * Output is per-face-corner (xyz, sign) = 4 floats per corner.
 */

#include "mikktspace.h"

typedef struct {
    const float *positions;      /* V * 3 */
    const float *normals;        /* V * 3 */
    const float *uvs;            /* V * 2 */
    const int   *faces;          /* F * 3 */
    int          num_faces;
    float       *out_tangents;   /* F * 3 * 4 (xyz, sign) per corner */
} WrapperData;

static int wrap_getNumFaces(const SMikkTSpaceContext *ctx) {
    return ((WrapperData *)ctx->m_pUserData)->num_faces;
}

static int wrap_getNumVerticesOfFace(const SMikkTSpaceContext *ctx, const int iFace) {
    (void)ctx; (void)iFace;
    return 3;
}

static void wrap_getPosition(const SMikkTSpaceContext *ctx, float out[], const int f, const int v) {
    WrapperData *d = (WrapperData *)ctx->m_pUserData;
    int idx = d->faces[f * 3 + v];
    out[0] = d->positions[idx * 3 + 0];
    out[1] = d->positions[idx * 3 + 1];
    out[2] = d->positions[idx * 3 + 2];
}

static void wrap_getNormal(const SMikkTSpaceContext *ctx, float out[], const int f, const int v) {
    WrapperData *d = (WrapperData *)ctx->m_pUserData;
    int idx = d->faces[f * 3 + v];
    out[0] = d->normals[idx * 3 + 0];
    out[1] = d->normals[idx * 3 + 1];
    out[2] = d->normals[idx * 3 + 2];
}

static void wrap_getTexCoord(const SMikkTSpaceContext *ctx, float out[], const int f, const int v) {
    WrapperData *d = (WrapperData *)ctx->m_pUserData;
    int idx = d->faces[f * 3 + v];
    out[0] = d->uvs[idx * 2 + 0];
    out[1] = d->uvs[idx * 2 + 1];
}

static void wrap_setTSpaceBasic(
    const SMikkTSpaceContext *ctx,
    const float tangent[],
    const float sign,
    const int f,
    const int v
) {
    WrapperData *d = (WrapperData *)ctx->m_pUserData;
    int o = (f * 3 + v) * 4;
    d->out_tangents[o + 0] = tangent[0];
    d->out_tangents[o + 1] = tangent[1];
    d->out_tangents[o + 2] = tangent[2];
    d->out_tangents[o + 3] = sign;
}

#ifdef _WIN32
#  define BEVEL_EXPORT __declspec(dllexport)
#else
#  define BEVEL_EXPORT __attribute__((visibility("default")))
#endif

BEVEL_EXPORT int fbb_compute_mikkt_tangents(
    const float *positions,
    const float *normals,
    const float *uvs,
    const int   *faces,
    int          num_faces,
    float       *out_tangents
) {
    WrapperData data;
    data.positions = positions;
    data.normals = normals;
    data.uvs = uvs;
    data.faces = faces;
    data.num_faces = num_faces;
    data.out_tangents = out_tangents;

    SMikkTSpaceInterface iface;
    iface.m_getNumFaces = wrap_getNumFaces;
    iface.m_getNumVerticesOfFace = wrap_getNumVerticesOfFace;
    iface.m_getPosition = wrap_getPosition;
    iface.m_getNormal = wrap_getNormal;
    iface.m_getTexCoord = wrap_getTexCoord;
    iface.m_setTSpaceBasic = wrap_setTSpaceBasic;
    iface.m_setTSpace = 0;

    SMikkTSpaceContext ctx;
    ctx.m_pInterface = &iface;
    ctx.m_pUserData = &data;

    return genTangSpaceDefault(&ctx);
}
