/*
 * ps3recomp - D3D12 backend for the shared register-file draw engine
 *
 * Included once, at the end of rsx_d3d12_backend.c's _WIN32 section, so it
 * shares the window, device, queue and swap chain that file creates (and its
 * GL->D3D12 state tables) without exporting its state. It implements
 * rsx_draw_backend (rsx_draw_engine.h), the interface the Vulkan and Metal
 * backends implement: the engine decodes the guest's NV4097 register file
 * (rsx_dispatch.c) and drives this through semantic calls.
 *
 * Selected with the engine's own documented switch,
 * PS3RECOMP_RSX_ENGINE=dispatch; without it the rsx_state recorder above
 * stays the renderer and nothing here runs.
 *
 * Unlike the Vulkan backend, a draw is not submitted and waited on by itself:
 * commands accumulate in one list until the engine's submit_and_wait (or a
 * readback, or a full ring), which is the synchronisation the interface
 * actually requires. Everything the GPU reads -- vertices, indices,
 * constants, texture data -- is copied into one upload ring when the call
 * arrives; a full ring submits and waits, then starts over.
 *
 * Root signature, mirroring the registers the engine's HLSL declares:
 *   0  CBV b0  VPConst      (all stages)
 *   1  CBV b1  PSConstants  (all stages)
 *   2  table   t0..t15      pixel textures
 *   3  table   s0..s15      pixel samplers
 *   4  table   t16..t19     vertex textures
 *   5  table   s0..s3       vertex samplers
 *
 * The constant blend factor arrives per draw through set_blend_color.
 *
 * D3D12ENG_LOG=1      counters every ~5 s
 * D3D12ENG_DUMP=<dir> write the presented surface as BMP every
 *                     D3D12ENG_DUMP_EVERY presents (default 300)
 */
#ifndef RSX_D3D12_ENGINE_IMPL_H
#define RSX_D3D12_ENGINE_IMPL_H

#include "rsx_draw_engine.h"

#define DE_MAX_OBJ        2048       /* handle = index, 0 = none */
#define DE_MAX_PIPE       2048
#define DE_RTV_COUNT      512
#define DE_DSV_COUNT      64
#define DE_SRV_CPU_COUNT  4096       /* one SRV per texture/target/view   */
#define DE_SRV_GPU_COUNT  65536      /* shader-visible ring (20 per draw) */
#define DE_SAMP_CPU_COUNT 512        /* sampler cache                     */
#define DE_SAMP_GPU_COUNT 2048       /* shader-visible sampler tables     */
#define DE_RING_BYTES     (256u << 20)
#define DE_PS_UNITS       16
#define DE_VS_UNITS       4
#define DE_VPCONST_BYTES  ((512u + 2u) * 16u)
#define DE_PSCONST_MIN    4096u

enum { DE_NONE = 0, DE_TEXTURE, DE_COLOR, DE_DEPTH, DE_VIEW };

/* RSX comparison functions (0x200 NEVER .. 0x207 ALWAYS) and stencil ops
 * (GL enums) as D3D12 values. */
static u32 gl_func_idx(u32 f) { return (f >= 0x200u && f <= 0x207u) ? f - 0x200u : 7u; }
static D3D12_COMPARISON_FUNC zfunc_d3d(u32 fi)
{
    static const D3D12_COMPARISON_FUNC m[8] = {
        D3D12_COMPARISON_FUNC_NEVER, D3D12_COMPARISON_FUNC_LESS, D3D12_COMPARISON_FUNC_EQUAL,
        D3D12_COMPARISON_FUNC_LESS_EQUAL, D3D12_COMPARISON_FUNC_GREATER,
        D3D12_COMPARISON_FUNC_NOT_EQUAL, D3D12_COMPARISON_FUNC_GREATER_EQUAL,
        D3D12_COMPARISON_FUNC_ALWAYS };
    return m[fi & 7u];
}
/* Index from 0, so D3D12_STENCIL_OP = index + 1. */
static u32 gl_stencil_op_idx(u32 op)
{
    switch (op & 0xFFFFu) {
    case 0x0000: return D3D12_STENCIL_OP_ZERO     - 1;
    case 0x1E01: return D3D12_STENCIL_OP_REPLACE  - 1;
    case 0x1E02: return D3D12_STENCIL_OP_INCR_SAT - 1;
    case 0x1E03: return D3D12_STENCIL_OP_DECR_SAT - 1;
    case 0x150A: return D3D12_STENCIL_OP_INVERT   - 1;
    case 0x8507: return D3D12_STENCIL_OP_INCR     - 1;
    case 0x8508: return D3D12_STENCIL_OP_DECR     - 1;
    default:     return D3D12_STENCIL_OP_KEEP     - 1;   /* 0x1E00 KEEP */
    }
}

typedef struct {
    int                   kind;
    ID3D12Resource*       res;       /* NULL for a view (uses the parent's) */
    DXGI_FORMAT           fmt;       /* resource / view format              */
    u32                   w, h, bpp, block, mips, faces;
    D3D12_RESOURCE_STATES state;
    int                   rtv, dsv, srv;   /* descriptor indices, -1 = none   */
    u32                   parent;    /* DE_VIEW: the colour target          */
    u32                   remap, rsx_fmt;
    u32                   conv;      /* DE_DEPTH: depth_as_color target     */
    int                   dsrv;      /* DE_DEPTH: R24_UNORM_X8 SRV, -1 none  */
    int                   ssrv;      /* DE_DEPTH: X24_G8_UINT stencil SRV    */
} de_obj;

typedef struct {
    int                   used;
    ID3DBlob*             vs;
    ID3DBlob*             ps;
    rsx_be_render_state   rs;
    D3D12_INPUT_ELEMENT_DESC il[RSX_DSP_NUM_VERTEX_ATTR];
    u32                   nil, stride, rt_count;
    DXGI_FORMAT           rt_fmt;
    u32                   cube_mask;
    ID3D12PipelineState*  pso[3];    /* point, line, triangle */
} de_pipe;

typedef struct { u64 key; int idx; } de_samp_entry;
typedef struct { u64 key; u32 base; } de_samp_table;

static struct {
    int                        up;
    ID3D12CommandAllocator*    alloc;          /* the current frame slot's */
    ID3D12CommandAllocator*    allocs[2];      /* one per frame slot */
    u32                        slot;           /* frame slot being recorded */
    u64                        slot_fence[2];  /* fence of each slot's last frame */
    u64                        ring_base, ring_end;   /* the slot's half of the upload ring */
    u32                        srv_base, srv_end;     /* ...and of the shader-visible SRV ring */
    ID3D12GraphicsCommandList* list;
    int                        open;
    ID3D12Fence*               fence;
    HANDLE                     event;
    u64                        fence_value;
    ID3D12RootSignature*       root;
    ID3D12DescriptorHeap      *rtv_heap, *dsv_heap, *srv_cpu, *srv_gpu, *samp_cpu, *samp_gpu;
    u32                        rtv_inc, dsv_inc, srv_inc, samp_inc;
    u32                        rtv_next, dsv_next, srv_cpu_next;
    int                        rtv_free[DE_RTV_COUNT], dsv_free[DE_DSV_COUNT], srv_free[DE_SRV_CPU_COUNT];
    u32                        n_rtv_free, n_dsv_free, n_srv_free;
    u32                        srv_gpu_next;
    de_samp_entry              samp[DE_SAMP_CPU_COUNT]; u32 n_samp;
    de_samp_table              samp_tables[DE_SAMP_GPU_COUNT / (DE_PS_UNITS + DE_VS_UNITS)]; u32 n_samp_tables;
    ID3D12Resource*            ring; u8* ring_map; D3D12_GPU_VIRTUAL_ADDRESS ring_va; u64 ring_off;
    ID3D12Resource*            readback; u64 readback_bytes;
    de_obj                     obj[DE_MAX_OBJ];
    de_pipe                    pipe[DE_MAX_PIPE];
    u32                        dummy2d, dummycube;
    u32                        scratch_depth;
    /* bound state */
    u32                        rt[RSX_BE_MAX_COLOR_TARGETS], nrt, depth;
    u32                        pipeline;
    const void*                vsc; u32 vsc_bytes;
    const void*                psc; u32 psc_bytes;
    u8                         vs_copy[DE_VPCONST_BYTES];
    u8                         ps_copy[64 * 1024];
    u32                        tex[DE_PS_UNITS]; rsx_be_sampler_desc samp_desc[DE_PS_UNITS]; u32 tex_mask;
    u32                        vtex[DE_VS_UNITS]; rsx_be_sampler_desc vsamp_desc[DE_VS_UNITS]; u32 vtex_mask;
    float                      vp[4]; int have_vp;
    u32                        sc[4]; int have_sc;
    u32                        stencil_ref;
    float                      blend_color[4];
    u32                        warned;
    /* counters (D3D12ENG_LOG) */
    u64 n_present, n_clear_c, n_clear_z, n_targets, n_depths, n_draw, n_draw_skipped,
        n_pipe_ok, n_pipe_fail, n_tex, n_submit, n_ring_wrap, n_writes, n_depth_reads, n_depth_conv, n_copies;
} s_de;

static int de_log_on(void)
{
    static int v = -1;
    if (v < 0) v = getenv("D3D12ENG_LOG") ? 1 : 0;
    return v;
}
#define DE_WARN_ONCE(bit, ...) do { if (!(s_de.warned & (bit))) { s_de.warned |= (bit); \
    fprintf(stderr, "[D3D12-ENG] " __VA_ARGS__); } } while (0)

static DXGI_FORMAT de_dxgi(rsx_be_format f, u32* bpp, u32* block)
{
    u32 b = 4, bl = 0; DXGI_FORMAT d = DXGI_FORMAT_R8G8B8A8_UNORM;
    switch (f) {
    case RSX_BE_FMT_R8:            d = DXGI_FORMAT_R8_UNORM;           b = 1;  break;
    case RSX_BE_FMT_R8G8:          d = DXGI_FORMAT_R8G8_UNORM;         b = 2;  break;
    case RSX_BE_FMT_BC1:           d = DXGI_FORMAT_BC1_UNORM;  b = 0; bl = 8;  break;
    case RSX_BE_FMT_BC2:           d = DXGI_FORMAT_BC2_UNORM;  b = 0; bl = 16; break;
    case RSX_BE_FMT_BC3:           d = DXGI_FORMAT_BC3_UNORM;  b = 0; bl = 16; break;
    case RSX_BE_FMT_R16:           d = DXGI_FORMAT_R16_UNORM;          b = 2;  break;
    case RSX_BE_FMT_R16G16:        d = DXGI_FORMAT_R16G16_UNORM;       b = 4;  break;
    case RSX_BE_FMT_R16G16F:       d = DXGI_FORMAT_R16G16_FLOAT;       b = 4;  break;
    case RSX_BE_FMT_R16G16B16A16F: d = DXGI_FORMAT_R16G16B16A16_FLOAT; b = 8;  break;
    case RSX_BE_FMT_R32F:          d = DXGI_FORMAT_R32_FLOAT;          b = 4;  break;
    case RSX_BE_FMT_R32G32B32A32F: d = DXGI_FORMAT_R32G32B32A32_FLOAT; b = 16; break;
    case RSX_BE_FMT_NONE:          d = DXGI_FORMAT_UNKNOWN;            b = 0;  break;
    default: break;   /* R8G8B8A8 (decoded colour is always R,G,B,A) */
    }
    if (bpp) *bpp = b;
    if (block) *block = bl;
    return d;
}

/* ---- descriptors ------------------------------------------------------- */

static D3D12_CPU_DESCRIPTOR_HANDLE de_cpu(ID3D12DescriptorHeap* h, u32 inc, u32 i)
{
    D3D12_CPU_DESCRIPTOR_HANDLE c;
    h->lpVtbl->GetCPUDescriptorHandleForHeapStart(h, &c);
    c.ptr += (u64)i * inc;
    return c;
}
static D3D12_GPU_DESCRIPTOR_HANDLE de_gpu(ID3D12DescriptorHeap* h, u32 inc, u32 i)
{
    D3D12_GPU_DESCRIPTOR_HANDLE g;
    h->lpVtbl->GetGPUDescriptorHandleForHeapStart(h, &g);
    g.ptr += (u64)i * inc;
    return g;
}
static int de_take(u32* next, u32 cap, int* freelist, u32* nfree)
{
    if (*nfree) return freelist[--*nfree];
    if (*next < cap) return (int)(*next)++;
    return -1;
}
static void de_give(int i, int* freelist, u32* nfree) { if (i >= 0) freelist[(*nfree)++] = i; }

/* ---- command list ------------------------------------------------------ */

static void de_bind_heaps(void);
static ID3D12GraphicsCommandList* de_cl(void)
{
    if (!s_de.open) {
        s_de.alloc->lpVtbl->Reset(s_de.alloc);
        s_de.list->lpVtbl->Reset(s_de.list, s_de.alloc, NULL);
        s_de.open = 1;
        de_bind_heaps();
    }
    return s_de.list;
}

static long long s_de_wait_q, s_de_present_q;   /* QPC ticks: GPU-fence wait, Present call */
static void de_submit(void)
{
    if (!s_de.open) return;
    s_de.list->lpVtbl->Close(s_de.list);
    ID3D12CommandList* l[1] = { (ID3D12CommandList*)s_de.list };
    s_d3d.cmd_queue->lpVtbl->ExecuteCommandLists(s_d3d.cmd_queue, 1, l);
    const u64 v = ++s_de.fence_value;
    s_d3d.cmd_queue->lpVtbl->Signal(s_d3d.cmd_queue, s_de.fence, v);
    if (s_de.fence->lpVtbl->GetCompletedValue(s_de.fence) < v) {
        LARGE_INTEGER a, b; QueryPerformanceCounter(&a);
        s_de.fence->lpVtbl->SetEventOnCompletion(s_de.fence, v, s_de.event);
        WaitForSingleObject(s_de.event, 10000);
        QueryPerformanceCounter(&b); s_de_wait_q += b.QuadPart - a.QuadPart;   /* D3D12ENG_LOG */
    }
    s_de.open = 0;
    s_de.n_submit++;
    /* Nothing in flight any more: the slot's rings may start over (and the
     * other slot's frame is finished too). */
    s_de.ring_off = s_de.ring_base;
    s_de.srv_gpu_next = s_de.srv_base;
}

/* Frame slots. The rings are split in two halves and there are two command
 * allocators, so the GPU can draw frame N out of one half while the CPU
 * records frame N+1 into the other. */
static void de_use_slot(u32 s)
{
    s_de.slot = s;
    s_de.alloc = s_de.allocs[s];
    s_de.ring_base = (u64)s * (DE_RING_BYTES / 2);  s_de.ring_end = s_de.ring_base + DE_RING_BYTES / 2;
    s_de.srv_base  = s * (DE_SRV_GPU_COUNT / 2);    s_de.srv_end  = s_de.srv_base + DE_SRV_GPU_COUNT / 2;
    s_de.ring_off = s_de.ring_base;
    s_de.srv_gpu_next = s_de.srv_base;
}

/* End of frame: hand the frame to the GPU WITHOUT waiting for it, and move
 * to the other slot -- waiting only if the GPU is still on the frame that
 * slot held, i.e. the one before this. Every other submit (readbacks,
 * releases, a full ring) keeps de_submit's full wait. */
static void de_submit_frame(void)
{
    if (!s_de.open) return;
    /* D3D12ENG_SYNC_SUBMIT=1: the old full wait at the end of every frame,
     * to tell a rendering difference caused by the overlap from one that is
     * not. */
    { static int sync = -1;
      if (sync < 0) { const char* e = getenv("D3D12ENG_SYNC_SUBMIT"); sync = (e && atoi(e) > 0) ? 1 : 0; }
      if (sync) { de_submit(); return; } }
    s_de.list->lpVtbl->Close(s_de.list);
    ID3D12CommandList* l[1] = { (ID3D12CommandList*)s_de.list };
    s_d3d.cmd_queue->lpVtbl->ExecuteCommandLists(s_d3d.cmd_queue, 1, l);
    const u64 v = ++s_de.fence_value;
    s_d3d.cmd_queue->lpVtbl->Signal(s_d3d.cmd_queue, s_de.fence, v);
    s_de.slot_fence[s_de.slot] = v;
    s_de.open = 0;
    s_de.n_submit++;
    const u32 next = s_de.slot ^ 1u;
    const u64 need = s_de.slot_fence[next];
    if (need && s_de.fence->lpVtbl->GetCompletedValue(s_de.fence) < need) {
        LARGE_INTEGER a, b; QueryPerformanceCounter(&a);
        s_de.fence->lpVtbl->SetEventOnCompletion(s_de.fence, need, s_de.event);
        WaitForSingleObject(s_de.event, 10000);
        QueryPerformanceCounter(&b); s_de_wait_q += b.QuadPart - a.QuadPart;
    }
    de_use_slot(next);
}

static void de_bind_heaps(void)
{
    ID3D12DescriptorHeap* h[2] = { s_de.srv_gpu, s_de.samp_gpu };
    s_de.list->lpVtbl->SetDescriptorHeaps(s_de.list, 2, h);
    s_de.list->lpVtbl->SetGraphicsRootSignature(s_de.list, s_de.root);
}

static void de_transition(de_obj* o, D3D12_RESOURCE_STATES to)
{
    if (!o->res || o->state == to) return;
    D3D12_RESOURCE_BARRIER b = {0};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = o->res;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = o->state;
    b.Transition.StateAfter = to;
    de_cl()->lpVtbl->ResourceBarrier(de_cl(), 1, &b);
    o->state = to;
}
static void de_transition_raw(ID3D12Resource* r, D3D12_RESOURCE_STATES* cur, D3D12_RESOURCE_STATES to)
{
    if (*cur == to) return;
    D3D12_RESOURCE_BARRIER b = {0};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = r;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = *cur;
    b.Transition.StateAfter = to;
    de_cl()->lpVtbl->ResourceBarrier(de_cl(), 1, &b);
    *cur = to;
}

/* ---- upload ring ------------------------------------------------------- */

/* `bytes` of ring space at `align`; NULL only when bytes exceed the ring. */
static u8* de_ring(u64 bytes, u64 align, D3D12_GPU_VIRTUAL_ADDRESS* va, u64* off)
{
    if (bytes > DE_RING_BYTES / 2) return NULL;
    u64 o = (s_de.ring_off + align - 1) & ~(align - 1);
    if (o + bytes > s_de.ring_end) {
        de_submit();
        s_de.n_ring_wrap++;
        o = s_de.ring_off;
    }
    s_de.ring_off = o + bytes;
    if (va)  *va  = s_de.ring_va + o;
    if (off) *off = o;
    return s_de.ring_map + o;
}

static ID3D12Resource* de_buffer(D3D12_HEAP_TYPE type, u64 bytes, D3D12_RESOURCE_STATES st)
{
    D3D12_HEAP_PROPERTIES hp = {0}; hp.Type = type;
    D3D12_RESOURCE_DESC d = {0};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; d.Width = bytes; d.Height = 1;
    d.DepthOrArraySize = 1; d.MipLevels = 1; d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* r = NULL;
    if (FAILED(s_d3d.device->lpVtbl->CreateCommittedResource(s_d3d.device, &hp, D3D12_HEAP_FLAG_NONE,
            &d, st, NULL, &IID_ID3D12Resource, (void**)&r))) return NULL;
    return r;
}

/* Copy rows into one subresource of `o` through the ring (no wait). */
static void de_upload_sub(de_obj* o, u32 sub, const void* rows, u32 row_bytes, u32 nrows_src)
{
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp; UINT nrows; UINT64 rowsz, total;
    D3D12_RESOURCE_DESC rd; o->res->lpVtbl->GetDesc(o->res, &rd);
    s_d3d.device->lpVtbl->GetCopyableFootprints(s_d3d.device, &rd, sub, 1, 0, &fp, &nrows, &rowsz, &total);
    u64 off = 0;
    u8* map = de_ring(total, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT, NULL, &off);
    if (!map) return;
    const u32 copy = (row_bytes < (u32)rowsz) ? row_bytes : (u32)rowsz;
    for (u32 y = 0; y < nrows; y++) {
        u8* d = map + (u64)y * fp.Footprint.RowPitch;
        if (y < nrows_src) {
            memcpy(d, (const u8*)rows + (u64)y * row_bytes, copy);
            if (copy < rowsz) memset(d + copy, 0, (size_t)(rowsz - copy));
        } else memset(d, 0, (size_t)rowsz);
    }
    fp.Offset = off;
    const D3D12_RESOURCE_STATES was = o->state;
    de_transition(o, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_TEXTURE_COPY_LOCATION dst = {0}, src = {0};
    dst.pResource = o->res; dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; dst.SubresourceIndex = sub;
    src.pResource = s_de.ring; src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; src.PlacedFootprint = fp;
    de_cl()->lpVtbl->CopyTextureRegion(de_cl(), &dst, 0, 0, 0, &src, NULL);
    de_transition(o, was);
}

/* ---- objects ------------------------------------------------------------ */

static u32 de_alloc_obj(void)
{
    for (u32 i = 1; i < DE_MAX_OBJ; i++) if (s_de.obj[i].kind == DE_NONE) {
        memset(&s_de.obj[i], 0, sizeof(de_obj));
        s_de.obj[i].rtv = s_de.obj[i].dsv = s_de.obj[i].srv = s_de.obj[i].dsrv = s_de.obj[i].ssrv = -1;
        return i;
    }
    return 0;
}
static de_obj* de_get(u32 h, int kind)
{
    if (!h || h >= DE_MAX_OBJ || s_de.obj[h].kind == DE_NONE) return NULL;
    if (kind && s_de.obj[h].kind != kind) return NULL;
    return &s_de.obj[h];
}

/* D3D12 Shader4ComponentMapping from the unit's crossbar, the same mapping the
 * Vulkan backend builds (rsx_texture_component_remap: sel[] = A,R,G,B
 * sources; 0..3 = R,G,B,A of the decoded texel, RSX_REMAP_ZERO / _ONE). */
static UINT de_remap(u32 remap, u32 rsx_fmt)
{
    u8 sel[4];
    rsx_texture_component_remap(remap, rsx_fmt & 0x9Fu, sel);
    UINT c[4];
    const u8 order[4] = { sel[1], sel[2], sel[3], sel[0] };   /* R, G, B, A */
    for (int i = 0; i < 4; i++)
        c[i] = order[i] <= 3 ? order[i]
             : order[i] == RSX_REMAP_ONE ? D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_1
                                         : D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0;
    return D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(c[0], c[1], c[2], c[3]);
}

static int de_make_srv(ID3D12Resource* res, DXGI_FORMAT fmt, u32 mips, int cube, UINT mapping)
{
    const int i = de_take(&s_de.srv_cpu_next, DE_SRV_CPU_COUNT, s_de.srv_free, &s_de.n_srv_free);
    if (i < 0) return -1;
    D3D12_SHADER_RESOURCE_VIEW_DESC sd = {0};
    sd.Format = fmt;
    sd.Shader4ComponentMapping = mapping;
    if (cube) { sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE; sd.TextureCube.MipLevels = mips ? mips : 1; }
    else      { sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;   sd.Texture2D.MipLevels = mips ? mips : 1; }
    s_d3d.device->lpVtbl->CreateShaderResourceView(s_d3d.device, res, &sd,
        de_cpu(s_de.srv_cpu, s_de.srv_inc, (u32)i));
    return i;
}

static ID3D12Resource* de_tex2d(DXGI_FORMAT fmt, u32 w, u32 h, u32 mips, u32 layers,
                                D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES st,
                                const D3D12_CLEAR_VALUE* cv)
{
    D3D12_HEAP_PROPERTIES hp = {0}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC d = {0};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; d.Width = w; d.Height = h;
    d.DepthOrArraySize = (UINT16)layers; d.MipLevels = (UINT16)(mips ? mips : 1);
    d.Format = fmt; d.SampleDesc.Count = 1; d.Flags = flags;
    ID3D12Resource* r = NULL;
    if (FAILED(s_d3d.device->lpVtbl->CreateCommittedResource(s_d3d.device, &hp, D3D12_HEAP_FLAG_NONE, &d,
            st, cv, &IID_ID3D12Resource, (void**)&r))) return NULL;
    return r;
}

static void de_release_obj(u32 h)
{
    de_obj* o = de_get(h, 0);
    if (!o) return;
    de_submit();
    if (o->kind == DE_COLOR)       /* its views go with it */
        for (u32 i = 1; i < DE_MAX_OBJ; i++)
            if (s_de.obj[i].kind == DE_VIEW && s_de.obj[i].parent == h) de_release_obj(i);
    if (o->kind == DE_DEPTH && o->conv) { const u32 c = o->conv; o->conv = 0; de_release_obj(c); }
    de_give(o->dsrv, s_de.srv_free, &s_de.n_srv_free);
    de_give(o->ssrv, s_de.srv_free, &s_de.n_srv_free);
    if (o->res) o->res->lpVtbl->Release(o->res);
    de_give(o->rtv, s_de.rtv_free, &s_de.n_rtv_free);
    de_give(o->dsv, s_de.dsv_free, &s_de.n_dsv_free);
    de_give(o->srv, s_de.srv_free, &s_de.n_srv_free);
    memset(o, 0, sizeof *o);
}

static ID3DBlob* de_compile(const char* src, const char* target);
static D3D12_GPU_DESCRIPTOR_HANDLE de_srv_table(const int* srv, u32 n);

/* ---- lifecycle ---------------------------------------------------------- */

static u32 de_texture_create(void* user, rsx_be_format fmt, u32 w, u32 h,
                             u32 mips, u32 faces, u32 remap, u32 rsx_fmt);
static void de_texture_upload(void* user, u32 texture, u32 face, u32 mip, u32 w, u32 h,
                              const void* src, u32 row_bytes, u32 rows);

static int de_make_root(void)
{
    D3D12_DESCRIPTOR_RANGE r[4] = {0};
    r[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;     r[0].NumDescriptors = DE_PS_UNITS; r[0].BaseShaderRegister = 0;
    r[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER; r[1].NumDescriptors = DE_PS_UNITS; r[1].BaseShaderRegister = 0;
    r[2].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;     r[2].NumDescriptors = DE_VS_UNITS; r[2].BaseShaderRegister = 16;
    r[3].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER; r[3].NumDescriptors = DE_VS_UNITS; r[3].BaseShaderRegister = 0;
    D3D12_ROOT_PARAMETER p[6] = {0};
    p[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV; p[0].Descriptor.ShaderRegister = 0;
    p[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    p[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV; p[1].Descriptor.ShaderRegister = 1;
    p[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    for (int i = 0; i < 4; i++) {
        p[2 + i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        p[2 + i].DescriptorTable.NumDescriptorRanges = 1;
        p[2 + i].DescriptorTable.pDescriptorRanges = &r[i];
        p[2 + i].ShaderVisibility = i < 2 ? D3D12_SHADER_VISIBILITY_PIXEL : D3D12_SHADER_VISIBILITY_VERTEX;
    }
    D3D12_ROOT_SIGNATURE_DESC rd = {0};
    rd.NumParameters = 6; rd.pParameters = p;
    rd.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    ID3DBlob* blob = NULL; ID3DBlob* err = NULL;
    if (FAILED(D3D12SerializeRootSignature(&rd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err))) {
        if (err) { fprintf(stderr, "[D3D12-ENG] root signature: %s\n", (const char*)err->lpVtbl->GetBufferPointer(err));
                   err->lpVtbl->Release(err); }
        return -1;
    }
    const HRESULT hr = s_d3d.device->lpVtbl->CreateRootSignature(s_d3d.device, 0,
        blob->lpVtbl->GetBufferPointer(blob), blob->lpVtbl->GetBufferSize(blob),
        &IID_ID3D12RootSignature, (void**)&s_de.root);
    blob->lpVtbl->Release(blob);
    return FAILED(hr) ? -1 : 0;
}

static int de_heap(D3D12_DESCRIPTOR_HEAP_TYPE t, u32 n, int visible, ID3D12DescriptorHeap** out)
{
    D3D12_DESCRIPTOR_HEAP_DESC hd = {0};
    hd.Type = t; hd.NumDescriptors = n;
    hd.Flags = visible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    return FAILED(s_d3d.device->lpVtbl->CreateDescriptorHeap(s_d3d.device, &hd,
        &IID_ID3D12DescriptorHeap, (void**)out)) ? -1 : 0;
}

static int de_init(void* user, u32 width, u32 height)
{
    (void)user; (void)width; (void)height;
    ID3D12Device* dev = s_d3d.device;
    if (!dev) return -1;
    if (FAILED(dev->lpVtbl->CreateCommandAllocator(dev, D3D12_COMMAND_LIST_TYPE_DIRECT,
            &IID_ID3D12CommandAllocator, (void**)&s_de.allocs[0]))) return -1;
    if (FAILED(dev->lpVtbl->CreateCommandAllocator(dev, D3D12_COMMAND_LIST_TYPE_DIRECT,
            &IID_ID3D12CommandAllocator, (void**)&s_de.allocs[1]))) return -1;
    de_use_slot(0);
    if (FAILED(dev->lpVtbl->CreateCommandList(dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, s_de.alloc, NULL,
            &IID_ID3D12GraphicsCommandList, (void**)&s_de.list))) return -1;
    s_de.list->lpVtbl->Close(s_de.list);
    if (FAILED(dev->lpVtbl->CreateFence(dev, 0, D3D12_FENCE_FLAG_NONE, &IID_ID3D12Fence,
            (void**)&s_de.fence))) return -1;
    s_de.event = CreateEventA(NULL, FALSE, FALSE, NULL);
    if (de_heap(D3D12_DESCRIPTOR_HEAP_TYPE_RTV, DE_RTV_COUNT, 0, &s_de.rtv_heap) ||
        de_heap(D3D12_DESCRIPTOR_HEAP_TYPE_DSV, DE_DSV_COUNT, 0, &s_de.dsv_heap) ||
        de_heap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, DE_SRV_CPU_COUNT, 0, &s_de.srv_cpu) ||
        de_heap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, DE_SRV_GPU_COUNT, 1, &s_de.srv_gpu) ||
        de_heap(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, DE_SAMP_CPU_COUNT, 0, &s_de.samp_cpu) ||
        de_heap(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, DE_SAMP_GPU_COUNT, 1, &s_de.samp_gpu))
        return -1;
    s_de.rtv_inc  = dev->lpVtbl->GetDescriptorHandleIncrementSize(dev, D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    s_de.dsv_inc  = dev->lpVtbl->GetDescriptorHandleIncrementSize(dev, D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
    s_de.srv_inc  = dev->lpVtbl->GetDescriptorHandleIncrementSize(dev, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    s_de.samp_inc = dev->lpVtbl->GetDescriptorHandleIncrementSize(dev, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    s_de.ring = de_buffer(D3D12_HEAP_TYPE_UPLOAD, DE_RING_BYTES, D3D12_RESOURCE_STATE_GENERIC_READ);
    if (!s_de.ring || FAILED(s_de.ring->lpVtbl->Map(s_de.ring, 0, NULL, (void**)&s_de.ring_map))) return -1;
    s_de.ring_va = s_de.ring->lpVtbl->GetGPUVirtualAddress(s_de.ring);
    if (de_make_root()) return -1;
    /* What an unbound sampled unit reads: magenta, as on the Vulkan backend. */
    { static const u8 magenta[4] = { 255, 0, 255, 255 };
      s_de.dummy2d = de_texture_create(NULL, RSX_BE_FMT_R8G8B8A8, 1, 1, 1, 1, 0xE4u, 0x85u);
      de_texture_upload(NULL, s_de.dummy2d, 0, 0, 1, 1, magenta, 4, 1);
      s_de.dummycube = de_texture_create(NULL, RSX_BE_FMT_R8G8B8A8, 1, 1, 1, 6, 0xE4u, 0x85u);
      for (u32 f = 0; f < 6; f++) de_texture_upload(NULL, s_de.dummycube, f, 0, 1, 1, magenta, 4, 1); }
    s_de.up = 1;
    fprintf(stderr, "[D3D12-ENG] draw engine backend up (stage ii: pipelines, draws, textures)\n");
    return 0;
}

static void de_shutdown(void* user)
{
    (void)user;
    de_submit();
    for (u32 i = 0; i < DE_MAX_PIPE; i++) {
        de_pipe* P = &s_de.pipe[i];
        if (!P->used) continue;
        for (int t = 0; t < 3; t++) if (P->pso[t]) P->pso[t]->lpVtbl->Release(P->pso[t]);
        if (P->vs) P->vs->lpVtbl->Release(P->vs);
        if (P->ps) P->ps->lpVtbl->Release(P->ps);
    }
    for (u32 i = 1; i < DE_MAX_OBJ; i++) if (s_de.obj[i].res) s_de.obj[i].res->lpVtbl->Release(s_de.obj[i].res);
    ID3D12DescriptorHeap* hs[6] = { s_de.rtv_heap, s_de.dsv_heap, s_de.srv_cpu, s_de.srv_gpu, s_de.samp_cpu, s_de.samp_gpu };
    for (int i = 0; i < 6; i++) if (hs[i]) hs[i]->lpVtbl->Release(hs[i]);
    if (s_de.ring)     { s_de.ring->lpVtbl->Unmap(s_de.ring, 0, NULL); s_de.ring->lpVtbl->Release(s_de.ring); }
    if (s_de.readback) s_de.readback->lpVtbl->Release(s_de.readback);
    if (s_de.root)     s_de.root->lpVtbl->Release(s_de.root);
    if (s_de.list)     s_de.list->lpVtbl->Release(s_de.list);
    for (int a = 0; a < 2; a++) if (s_de.allocs[a]) s_de.allocs[a]->lpVtbl->Release(s_de.allocs[a]);
    s_de.alloc = NULL;
    if (s_de.fence)    s_de.fence->lpVtbl->Release(s_de.fence);
    if (s_de.event)    CloseHandle(s_de.event);
    memset(&s_de, 0, sizeof s_de);
}

static void de_submit_and_wait(void* user, u32 reason) { (void)user; (void)reason; de_submit(); }

/* ---- textures ----------------------------------------------------------- */

static u32 de_texture_create(void* user, rsx_be_format fmt, u32 w, u32 h,
                             u32 mips, u32 faces, u32 remap, u32 rsx_fmt)
{
    (void)user;
    u32 bpp, block;
    const DXGI_FORMAT df = de_dxgi(fmt, &bpp, &block);
    const int cube = (faces == 6);
    if (df == DXGI_FORMAT_UNKNOWN || !w || !h || w > 16384 || h > 16384) return 0;
    if ((faces != 1 && !cube) || (cube && w != h)) return 0;
    if (!mips) mips = 1;
    const u32 hd = de_alloc_obj();
    if (!hd) return 0;
    de_obj* o = &s_de.obj[hd];
    o->res = de_tex2d(df, w, h, mips, cube ? 6 : 1, D3D12_RESOURCE_FLAG_NONE,
                      D3D12_RESOURCE_STATE_COPY_DEST, NULL);
    if (!o->res) { memset(o, 0, sizeof *o); return 0; }
    o->kind = DE_TEXTURE; o->fmt = df; o->w = w; o->h = h; o->bpp = bpp; o->block = block;
    o->mips = mips; o->faces = cube ? 6 : 1; o->state = D3D12_RESOURCE_STATE_COPY_DEST;
    o->remap = remap; o->rsx_fmt = rsx_fmt;
    o->srv = de_make_srv(o->res, df, mips, cube, de_remap(remap, rsx_fmt));
    s_de.n_tex++;
    return hd;
}

static void de_texture_upload(void* user, u32 texture, u32 face, u32 mip, u32 w, u32 h,
                              const void* src, u32 row_bytes, u32 rows)
{
    (void)user; (void)w; (void)h;
    de_obj* o = de_get(texture, DE_TEXTURE);
    if (!o || !src || !row_bytes || !rows || face >= o->faces || mip >= o->mips) return;
    de_upload_sub(o, mip + face * o->mips, src, row_bytes, rows);
}

static void de_release(void* user, u32 handle) { (void)user; de_release_obj(handle); }

/* ---- targets ------------------------------------------------------------ */

static u32 de_color_target_create(void* user, rsx_be_format fmt, u32 w, u32 h,
                                  const void* seed, u32 seed_row_bytes)
{
    (void)user;
    u32 bpp, block;
    const DXGI_FORMAT df = de_dxgi(fmt, &bpp, &block);
    if (df == DXGI_FORMAT_UNKNOWN || block || !w || !h) return 0;
    const u32 hd = de_alloc_obj();
    if (!hd) return 0;
    de_obj* o = &s_de.obj[hd];
    D3D12_CLEAR_VALUE cv = {0}; cv.Format = df;
    o->res = de_tex2d(df, w, h, 1, 1, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                      D3D12_RESOURCE_STATE_RENDER_TARGET, &cv);
    if (!o->res) { memset(o, 0, sizeof *o); return 0; }
    o->kind = DE_COLOR; o->fmt = df; o->w = w; o->h = h; o->bpp = bpp; o->mips = 1; o->faces = 1;
    o->state = D3D12_RESOURCE_STATE_RENDER_TARGET;
    o->rtv = de_take(&s_de.rtv_next, DE_RTV_COUNT, s_de.rtv_free, &s_de.n_rtv_free);
    if (o->rtv < 0) { o->res->lpVtbl->Release(o->res); memset(o, 0, sizeof *o); return 0; }
    s_d3d.device->lpVtbl->CreateRenderTargetView(s_d3d.device, o->res, NULL,
        de_cpu(s_de.rtv_heap, s_de.rtv_inc, (u32)o->rtv));
    o->srv = de_make_srv(o->res, df, 1, 0, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING);
    if (seed) de_upload_sub(o, 0, seed, seed_row_bytes, h);
    else {
        const float z[4] = {0, 0, 0, 0};
        de_cl()->lpVtbl->ClearRenderTargetView(de_cl(), de_cpu(s_de.rtv_heap, s_de.rtv_inc, (u32)o->rtv), z, 0, NULL);
    }
    s_de.n_targets++;
    return hd;
}

/* Pixels the guest wrote into a target's memory outside any draw (an NV3089
 * transfer -- ICO's MLAA round trip brings the frame back this way): R,G,B,A
 * rows into a rectangle, through the ring. Without this the engine leaves the
 * target as it is, and the pass that samples it reads a cleared surface. */
static void de_color_target_write(void* user, u32 surface, u32 x, u32 y, u32 w, u32 h,
                                  const void* rows, u32 row_bytes)
{
    (void)user;
    de_obj* t = de_get(surface, DE_COLOR);
    if (!t || !rows || !w || !h || x >= t->w || y >= t->h || t->bpp == 0) return;
    if (x + w > t->w) w = t->w - x;
    if (y + h > t->h) h = t->h - y;
    const u32 pitch = (w * t->bpp + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1) & ~(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1);
    u64 off = 0;
    u8* map = de_ring((u64)pitch * h, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT, NULL, &off);
    if (!map) return;
    const u32 copy = w * t->bpp < row_bytes ? w * t->bpp : row_bytes;
    for (u32 r = 0; r < h; r++) memcpy(map + (u64)r * pitch, (const u8*)rows + (u64)r * row_bytes, copy);
    const D3D12_RESOURCE_STATES was = t->state;
    de_transition(t, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_TEXTURE_COPY_LOCATION dst = {0}, src = {0};
    dst.pResource = t->res; dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; dst.SubresourceIndex = 0;
    src.pResource = s_de.ring; src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint.Offset = off;
    src.PlacedFootprint.Footprint.Format = t->fmt;
    src.PlacedFootprint.Footprint.Width = w; src.PlacedFootprint.Footprint.Height = h;
    src.PlacedFootprint.Footprint.Depth = 1; src.PlacedFootprint.Footprint.RowPitch = pitch;
    de_cl()->lpVtbl->CopyTextureRegion(de_cl(), &dst, x, y, 0, &src, NULL);
    de_transition(t, was);
    s_de.n_writes++;
}

/* A rectangle between two colour targets of one format, on the GPU and in
 * order (rsx_draw_backend.copy_region; the engine's main-memory shadows). */
static void de_copy_region(void* user, u32 src_surface, u32 sx, u32 sy,
                           u32 dst_surface, u32 dx, u32 dy, u32 w, u32 h)
{
    (void)user;
    de_obj* s = de_get(src_surface, DE_COLOR);
    de_obj* d = de_get(dst_surface, DE_COLOR);
    if (!s || !d || s == d || s->fmt != d->fmt || !w || !h) return;
    if (sx >= s->w || sy >= s->h || dx >= d->w || dy >= d->h) return;
    if (sx + w > s->w) w = s->w - sx;
    if (sy + h > s->h) h = s->h - sy;
    if (dx + w > d->w) w = d->w - dx;
    if (dy + h > d->h) h = d->h - dy;
    de_transition(s, D3D12_RESOURCE_STATE_COPY_SOURCE);
    de_transition(d, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_TEXTURE_COPY_LOCATION dl = {0}, sl = {0};
    sl.pResource = s->res; sl.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dl.pResource = d->res; dl.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_BOX box = { sx, sy, 0, sx + w, sy + h, 1 };
    de_cl()->lpVtbl->CopyTextureRegion(de_cl(), &dl, dx, dy, 0, &sl, &box);
    s_de.n_copies++;
}

static u32 de_surface_view(void* user, u32 surface, u32 remap, u32 rsx_format)
{
    (void)user;
    de_obj* t = de_get(surface, DE_COLOR);
    if (!t) return 0;
    for (u32 i = 1; i < DE_MAX_OBJ; i++)
        if (s_de.obj[i].kind == DE_VIEW && s_de.obj[i].parent == surface &&
            s_de.obj[i].remap == remap && s_de.obj[i].rsx_fmt == rsx_format) return i;
    const u32 hd = de_alloc_obj();
    if (!hd) return 0;
    de_obj* o = &s_de.obj[hd];
    o->kind = DE_VIEW; o->parent = surface; o->remap = remap; o->rsx_fmt = rsx_format;
    o->fmt = t->fmt; o->w = t->w; o->h = t->h; o->mips = 1; o->faces = 1;
    o->srv = de_make_srv(t->res, t->fmt, 1, 0, de_remap(remap, rsx_format));
    if (o->srv < 0) { memset(o, 0, sizeof *o); return 0; }
    return hd;
}

static u32 de_depth_target_create(void* user, u32 w, u32 h)
{
    (void)user;
    if (!w || !h) return 0;
    const u32 hd = de_alloc_obj();
    if (!hd) return 0;
    de_obj* o = &s_de.obj[hd];
    D3D12_CLEAR_VALUE cv = {0}; cv.Format = DXGI_FORMAT_D24_UNORM_S8_UINT; cv.DepthStencil.Depth = 1.0f;
    /* typeless: sampleable later (depth_snapshot, stage iii) */
    o->res = de_tex2d(DXGI_FORMAT_R24G8_TYPELESS, w, h, 1, 1, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                      D3D12_RESOURCE_STATE_DEPTH_WRITE, &cv);
    if (!o->res) { memset(o, 0, sizeof *o); return 0; }
    o->kind = DE_DEPTH; o->fmt = DXGI_FORMAT_D24_UNORM_S8_UINT; o->w = w; o->h = h; o->bpp = 4;
    o->state = D3D12_RESOURCE_STATE_DEPTH_WRITE;
    o->dsv = de_take(&s_de.dsv_next, DE_DSV_COUNT, s_de.dsv_free, &s_de.n_dsv_free);
    if (o->dsv < 0) { o->res->lpVtbl->Release(o->res); memset(o, 0, sizeof *o); return 0; }
    D3D12_DEPTH_STENCIL_VIEW_DESC dv = {0};
    dv.Format = DXGI_FORMAT_D24_UNORM_S8_UINT; dv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    s_d3d.device->lpVtbl->CreateDepthStencilView(s_d3d.device, o->res, &dv,
        de_cpu(s_de.dsv_heap, s_de.dsv_inc, (u32)o->dsv));
    de_cl()->lpVtbl->ClearDepthStencilView(de_cl(), de_cpu(s_de.dsv_heap, s_de.dsv_inc, (u32)o->dsv),
        D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 1.0f, 0, 0, NULL);
    s_de.n_depths++;
    return hd;
}

static u32 de_depth_snapshot(void* user, u32 depth, u32 w, u32 h)
{ (void)user; (void)depth; (void)w; (void)h; return 0; }   /* stage iii */

/* A Z24S8 depth target as the A8R8G8B8 bytes guest memory would hold
 * (rsx_draw_backend.depth_as_color): a full-screen pass loads each depth texel
 * through an R24_UNORM_X8 view and writes A = depth[23:16], R = [15:8],
 * G = [7:0], B = 0 into an RGBA8 target of w x h, viewed through the unit's
 * crossbar. Same bytes as the depth_readback route, no CPU round trip. */
static u32 de_depth_as_color(void* user, u32 depth, u32 w, u32 h, u32 remap, u32 rsx_fmt)
{
    (void)user;
    de_obj* z = de_get(depth, DE_DEPTH);
    if (!z || !w || !h) return 0;
    if (w > z->w) w = z->w;
    if (h > z->h) h = z->h;
    static ID3D12PipelineState* pso;
    if (!pso) {
        ID3DBlob* vs = de_compile("float4 main(uint id : SV_VertexID) : SV_Position {\n"
                                  "  float2 p = float2((id << 1) & 2, id & 2);\n"
                                  "  return float4(p * float2(2, -2) + float2(-1, 1), 0, 1); }\n", "vs_5_0");
        ID3DBlob* ps = de_compile("Texture2D<float> dz : register(t0);\n"
                                  "float4 main(float4 pos : SV_Position) : SV_Target {\n"
                                  "  uint v = (uint)(saturate(dz.Load(int3(pos.xy, 0))) * 16777215.0 + 0.5);\n"
                                  "  return float4(((v >> 8) & 255) / 255.0, (v & 255) / 255.0, 0.0,\n"
                                  "                ((v >> 16) & 255) / 255.0); }\n", "ps_5_0");
        if (!vs || !ps) return 0;
        D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = {0};
        pd.pRootSignature = s_de.root;
        pd.VS.pShaderBytecode = vs->lpVtbl->GetBufferPointer(vs); pd.VS.BytecodeLength = vs->lpVtbl->GetBufferSize(vs);
        pd.PS.pShaderBytecode = ps->lpVtbl->GetBufferPointer(ps); pd.PS.BytecodeLength = ps->lpVtbl->GetBufferSize(ps);
        pd.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        pd.SampleMask = 0xFFFFFFFFu;
        pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pd.NumRenderTargets = 1; pd.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM; pd.SampleDesc.Count = 1;
        if (FAILED(s_d3d.device->lpVtbl->CreateGraphicsPipelineState(s_d3d.device, &pd, &IID_ID3D12PipelineState,
                (void**)&pso))) { pso = NULL; return 0; }
    }
    if (z->dsrv < 0) {
        z->dsrv = de_take(&s_de.srv_cpu_next, DE_SRV_CPU_COUNT, s_de.srv_free, &s_de.n_srv_free);
        if (z->dsrv < 0) return 0;
        D3D12_SHADER_RESOURCE_VIEW_DESC sd = {0};
        sd.Format = DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
        sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D; sd.Texture2D.MipLevels = 1;
        s_d3d.device->lpVtbl->CreateShaderResourceView(s_d3d.device, z->res, &sd,
            de_cpu(s_de.srv_cpu, s_de.srv_inc, (u32)z->dsrv));
    }
    de_obj* c = de_get(z->conv, DE_COLOR);
    if (!c || c->w != w || c->h != h) {
        if (z->conv) { const u32 old = z->conv; z->conv = 0; de_release_obj(old); }
        z = de_get(depth, DE_DEPTH);   /* the table may not move, but stay safe */
        z->conv = de_color_target_create(NULL, RSX_BE_FMT_R8G8B8A8, w, h, NULL, 0);
        c = de_get(z->conv, DE_COLOR);
        if (!c) return 0;
    }
    if (s_de.srv_gpu_next + DE_PS_UNITS > s_de.srv_end) { de_submit(); s_de.n_ring_wrap++; }
    int srv[DE_PS_UNITS];
    de_obj* d2 = de_get(s_de.dummy2d, DE_TEXTURE);
    for (u32 i = 0; i < DE_PS_UNITS; i++) srv[i] = d2 ? d2->srv : -1;
    srv[0] = z->dsrv;
    if (d2) de_transition(d2, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    const D3D12_GPU_DESCRIPTOR_HANDLE table = de_srv_table(srv, DE_PS_UNITS);
    de_transition(z, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    de_transition(c, D3D12_RESOURCE_STATE_RENDER_TARGET);
    ID3D12GraphicsCommandList* cl = de_cl();
    const D3D12_CPU_DESCRIPTOR_HANDLE rh = de_cpu(s_de.rtv_heap, s_de.rtv_inc, (u32)c->rtv);
    cl->lpVtbl->OMSetRenderTargets(cl, 1, &rh, FALSE, NULL);
    cl->lpVtbl->SetPipelineState(cl, pso);
    cl->lpVtbl->SetGraphicsRootDescriptorTable(cl, 2, table);
    D3D12_VIEWPORT vp = { 0, 0, (float)w, (float)h, 0, 1 };
    D3D12_RECT sc = { 0, 0, (LONG)w, (LONG)h };
    cl->lpVtbl->RSSetViewports(cl, 1, &vp);
    cl->lpVtbl->RSSetScissorRects(cl, 1, &sc);
    cl->lpVtbl->IASetPrimitiveTopology(cl, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    cl->lpVtbl->DrawInstanced(cl, 3, 1, 0, 0);
    de_transition(z, D3D12_RESOURCE_STATE_DEPTH_WRITE);
    s_de.n_depth_conv++;
    return de_surface_view(NULL, z->conv, remap, rsx_fmt);
}

/* rsx_draw_backend.depth_blit: an NV3089 blit of one Z24S8 zeta into another,
 * point-sampled (ICO halves its scene depth+stencil this way for the stencil
 * shadow-volume pass). Full-screen triangles over the destination rectangle:
 * one writes depth through SV_Depth and zeroes the stencil, then one per
 * stencil bit sets that bit (REPLACE, write mask = the bit) where the source
 * texel has it -- no SV_StencilRef needed. */
static int de_depth_blit(void* user, u32 src_depth, u32 u0, u32 v0, u32 ds_dx, u32 dt_dy,
                         u32 dst_depth, u32 dx, u32 dy, u32 w, u32 h)
{
    (void)user;
    de_obj* s = de_get(src_depth, DE_DEPTH);
    de_obj* d = de_get(dst_depth, DE_DEPTH);
    if (!s || !d || s == d || !w || !h || dx + w > d->w || dy + h > d->h) return 0;
    static ID3D12PipelineState* pso[9];   /* [0] depth, [1 + b] stencil bit b */
    static int failed;
    if (failed) return 0;
    if (!pso[0]) {
        static const char* vsrc =
            "float4 main(uint id : SV_VertexID) : SV_Position {\n"
            "  float2 p = float2((id << 1) & 2, id & 2);\n"
            "  return float4(p * float2(2, -2) + float2(-1, 1), 0, 1); }\n";
        /* a = (u0, v0, ds, dt) in source texels; b = (dx, dy, bit, 0). */
        static const char* zsrc =
            "cbuffer C : register(b1) { float4 a; float4 b; };\n"
            "Texture2D<float> dz : register(t0);\n"
            "float main(float4 pos : SV_Position) : SV_Depth {\n"
            "  int2 p = int2(a.xy + (pos.xy - b.xy) * a.zw);\n"
            "  return dz.Load(int3(p, 0)); }\n";
        static const char* ssrc =
            "cbuffer C : register(b1) { float4 a; float4 b; };\n"
            "Texture2D<uint2> st : register(t1);\n"
            "void main(float4 pos : SV_Position) {\n"
            "  int2 p = int2(a.xy + (pos.xy - b.xy) * a.zw);\n"
            "  if ((st.Load(int3(p, 0)).y & (uint)b.z) == 0) discard; }\n";
        ID3DBlob* vs = de_compile(vsrc, "vs_5_0");
        ID3DBlob* zps = de_compile(zsrc, "ps_5_0");
        ID3DBlob* sps = de_compile(ssrc, "ps_5_0");
        if (!vs || !zps || !sps) { failed = 1; return 0; }
        for (u32 i = 0; i < 9; i++) {
            ID3DBlob* ps = i ? sps : zps;
            D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = {0};
            pd.pRootSignature = s_de.root;
            pd.VS.pShaderBytecode = vs->lpVtbl->GetBufferPointer(vs); pd.VS.BytecodeLength = vs->lpVtbl->GetBufferSize(vs);
            pd.PS.pShaderBytecode = ps->lpVtbl->GetBufferPointer(ps); pd.PS.BytecodeLength = ps->lpVtbl->GetBufferSize(ps);
            pd.SampleMask = 0xFFFFFFFFu;
            pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
            pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
            pd.RasterizerState.DepthClipEnable = FALSE;
            pd.DepthStencilState.DepthEnable    = i ? FALSE : TRUE;
            pd.DepthStencilState.DepthWriteMask = i ? D3D12_DEPTH_WRITE_MASK_ZERO : D3D12_DEPTH_WRITE_MASK_ALL;
            pd.DepthStencilState.DepthFunc      = D3D12_COMPARISON_FUNC_ALWAYS;
            pd.DepthStencilState.StencilEnable  = TRUE;
            pd.DepthStencilState.StencilReadMask  = 0xFF;
            pd.DepthStencilState.StencilWriteMask = i ? (UINT8)(1u << (i - 1)) : 0xFF;
            D3D12_DEPTH_STENCILOP_DESC op = { D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP,
                                              D3D12_STENCIL_OP_REPLACE, D3D12_COMPARISON_FUNC_ALWAYS };
            pd.DepthStencilState.FrontFace = op;
            pd.DepthStencilState.BackFace  = op;
            pd.DSVFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;
            pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            pd.NumRenderTargets = 0; pd.SampleDesc.Count = 1;
            if (FAILED(s_d3d.device->lpVtbl->CreateGraphicsPipelineState(s_d3d.device, &pd, &IID_ID3D12PipelineState,
                    (void**)&pso[i]))) {
                fprintf(stderr, "[D3D12-ENG] depth_blit PSO %u failed\n", i);
                failed = 1; return 0;
            }
        }
    }
    if (s->dsrv < 0 || s->ssrv < 0) {
        for (int plane = 0; plane < 2; plane++) {
            int* idx = plane ? &s->ssrv : &s->dsrv;
            if (*idx >= 0) continue;
            *idx = de_take(&s_de.srv_cpu_next, DE_SRV_CPU_COUNT, s_de.srv_free, &s_de.n_srv_free);
            if (*idx < 0) return 0;
            D3D12_SHADER_RESOURCE_VIEW_DESC sd = {0};
            sd.Format = plane ? DXGI_FORMAT_X24_TYPELESS_G8_UINT : DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
            sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D; sd.Texture2D.MipLevels = 1;
            sd.Texture2D.PlaneSlice = (UINT)plane;
            s_d3d.device->lpVtbl->CreateShaderResourceView(s_d3d.device, s->res, &sd,
                de_cpu(s_de.srv_cpu, s_de.srv_inc, (u32)*idx));
        }
    }
    if (s_de.srv_gpu_next + DE_PS_UNITS > s_de.srv_end) { de_submit(); s_de.n_ring_wrap++; }
    int srv[DE_PS_UNITS];
    de_obj* d2 = de_get(s_de.dummy2d, DE_TEXTURE);
    for (u32 i = 0; i < DE_PS_UNITS; i++) srv[i] = d2 ? d2->srv : -1;
    srv[0] = s->dsrv; srv[1] = s->ssrv;
    if (d2) de_transition(d2, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    const D3D12_GPU_DESCRIPTOR_HANDLE table = de_srv_table(srv, DE_PS_UNITS);
    de_transition(s, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    de_transition(d, D3D12_RESOURCE_STATE_DEPTH_WRITE);
    ID3D12GraphicsCommandList* cl = de_cl();
    const D3D12_CPU_DESCRIPTOR_HANDLE dh = de_cpu(s_de.dsv_heap, s_de.dsv_inc, (u32)d->dsv);
    cl->lpVtbl->OMSetRenderTargets(cl, 0, NULL, FALSE, &dh);
    cl->lpVtbl->SetGraphicsRootDescriptorTable(cl, 2, table);
    D3D12_VIEWPORT vp = { (float)dx, (float)dy, (float)w, (float)h, 0, 1 };
    D3D12_RECT sc = { (LONG)dx, (LONG)dy, (LONG)(dx + w), (LONG)(dy + h) };
    cl->lpVtbl->RSSetViewports(cl, 1, &vp);
    cl->lpVtbl->RSSetScissorRects(cl, 1, &sc);
    cl->lpVtbl->IASetPrimitiveTopology(cl, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    for (u32 i = 0; i < 9; i++) {
        D3D12_GPU_VIRTUAL_ADDRESS va;
        float* c = (float*)de_ring(256, 256, &va, NULL);
        if (!c) break;
        c[0] = (float)u0 / 16.0f;  c[1] = (float)v0 / 16.0f;            /* 12.4 */
        c[2] = (float)ds_dx / 1048576.0f; c[3] = (float)dt_dy / 1048576.0f;  /* 20.12 */
        c[4] = (float)dx; c[5] = (float)dy; c[6] = i ? (float)(1u << (i - 1)) : 0.0f; c[7] = 0.0f;
        cl->lpVtbl->SetPipelineState(cl, pso[i]);
        cl->lpVtbl->SetGraphicsRootConstantBufferView(cl, 1, va);
        cl->lpVtbl->OMSetStencilRef(cl, i ? 0xFFu : 0u);
        cl->lpVtbl->DrawInstanced(cl, 3, 1, 0, 0);
    }
    de_transition(s, D3D12_RESOURCE_STATE_DEPTH_WRITE);
    s_de.n_depth_conv++;
    return 1;
}

/* Depth as floats in [0, 1] for a rectangle of a depth target. The engine
 * stores them into guest memory in the RSX layout when a texture reads the
 * zeta's bytes in a format of its own -- ICO's fog pass samples its D24S8
 * scene depth as A8R8G8B8 and looks the fog up from it; without this the
 * pass read zeroed memory and fogged the whole screen evenly. Copies the
 * depth plane (subresource 0: 24-bit UNORM depth + 8 bits of padding). */
static int de_depth_readback(void* user, u32 depth, u32 x, u32 y, u32 w, u32 h,
                             float* out, u32 out_pitch)
{
    (void)user;
    de_obj* z = de_get(depth, DE_DEPTH);
    if (!z || !out || !w || !h || x >= z->w || y >= z->h) return -1;
    if (x + w > z->w) w = z->w - x;
    if (y + h > z->h) h = z->h - y;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp; UINT nrows; UINT64 rowsz, total;
    D3D12_RESOURCE_DESC rd; z->res->lpVtbl->GetDesc(z->res, &rd);
    s_d3d.device->lpVtbl->GetCopyableFootprints(s_d3d.device, &rd, 0, 1, 0, &fp, &nrows, &rowsz, &total);
    if (total > s_de.readback_bytes) {
        de_submit();
        if (s_de.readback) s_de.readback->lpVtbl->Release(s_de.readback);
        s_de.readback = de_buffer(D3D12_HEAP_TYPE_READBACK, total, D3D12_RESOURCE_STATE_COPY_DEST);
        s_de.readback_bytes = s_de.readback ? total : 0;
        if (!s_de.readback) return -1;
    }
    const D3D12_RESOURCE_STATES was = z->state;
    de_transition(z, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION dst = {0}, src = {0};
    src.pResource = z->res; src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; src.SubresourceIndex = 0;
    dst.pResource = s_de.readback; dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; dst.PlacedFootprint = fp;
    de_cl()->lpVtbl->CopyTextureRegion(de_cl(), &dst, 0, 0, 0, &src, NULL);   /* whole plane */
    de_transition(z, was);
    de_submit();
    u8* map = NULL; D3D12_RANGE rr = { 0, (SIZE_T)total };
    if (FAILED(s_de.readback->lpVtbl->Map(s_de.readback, 0, &rr, (void**)&map)) || !map) return -1;
    for (u32 r = 0; r < h; r++) {
        const u32* row = (const u32*)(map + fp.Offset + (u64)(y + r) * fp.Footprint.RowPitch) + x;
        float* o = (float*)((u8*)out + (u64)r * out_pitch);
        for (u32 c = 0; c < w; c++) o[c] = (float)(row[c] & 0xFFFFFFu) / 16777215.0f;
    }
    D3D12_RANGE wr = { 0, 0 };
    s_de.readback->lpVtbl->Unmap(s_de.readback, 0, &wr);
    s_de.n_depth_reads++;
    return 0;
}

static void de_clear_color(void* user, u32 surface, const float rgba[4])
{
    (void)user;
    de_obj* t = de_get(surface, DE_COLOR);
    if (!t) return;
    de_transition(t, D3D12_RESOURCE_STATE_RENDER_TARGET);
    de_cl()->lpVtbl->ClearRenderTargetView(de_cl(), de_cpu(s_de.rtv_heap, s_de.rtv_inc, (u32)t->rtv), rgba, 0, NULL);
    s_de.n_clear_c++;
}

/* A partial-channel clear: a full-screen triangle whose PSO writes only the
 * named channels (the recorder's masked_clear_pso, on this root signature:
 * the colour comes in through b1). One PSO per (channels, format). */
static ID3D12PipelineState* de_masked_clear_pso(u32 channels, DXGI_FORMAT fmt)
{
    static struct { u32 ch; DXGI_FORMAT fmt; ID3D12PipelineState* pso; } cache[32];
    static u32 n;
    static ID3DBlob *vs, *ps;
    for (u32 i = 0; i < n; i++) if (cache[i].ch == channels && cache[i].fmt == fmt) return cache[i].pso;
    if (n == 32) return NULL;
    if (!vs) {
        vs = de_compile("float4 main(uint id : SV_VertexID) : SV_Position {\n"
                        "  float2 p = float2((id << 1) & 2, id & 2);\n"
                        "  return float4(p * float2(2, -2) + float2(-1, 1), 0, 1); }\n", "vs_5_0");
        ps = de_compile("cbuffer C : register(b1) { float4 col; };\n"
                        "float4 main() : SV_Target { return col; }\n", "ps_5_0");
    }
    if (!vs || !ps) return NULL;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = {0};
    pd.pRootSignature = s_de.root;
    pd.VS.pShaderBytecode = vs->lpVtbl->GetBufferPointer(vs); pd.VS.BytecodeLength = vs->lpVtbl->GetBufferSize(vs);
    pd.PS.pShaderBytecode = ps->lpVtbl->GetBufferPointer(ps); pd.PS.BytecodeLength = ps->lpVtbl->GetBufferSize(ps);
    pd.BlendState.RenderTarget[0].RenderTargetWriteMask = (UINT8)(channels & 0xFu);  /* R1 G2 B4 A8 */
    pd.SampleMask = 0xFFFFFFFFu;
    pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pd.NumRenderTargets = 1; pd.RTVFormats[0] = fmt; pd.SampleDesc.Count = 1;
    ID3D12PipelineState* pso = NULL;
    if (FAILED(s_d3d.device->lpVtbl->CreateGraphicsPipelineState(s_d3d.device, &pd, &IID_ID3D12PipelineState,
            (void**)&pso))) return NULL;
    cache[n].ch = channels; cache[n].fmt = fmt; cache[n].pso = pso; n++;
    return pso;
}

static void de_clear_color_masked(void* user, u32 surface, const float rgba[4], u32 channels)
{
    if ((channels & 0xFu) == 0xFu) { de_clear_color(user, surface, rgba); return; }
    de_obj* t = de_get(surface, DE_COLOR);
    if (!t || !(channels & 0xFu)) return;
    ID3D12PipelineState* pso = de_masked_clear_pso(channels, t->fmt);
    if (!pso) { DE_WARN_ONCE(16u, "masked clear PSO unavailable: channels left as they are\n"); return; }
    D3D12_GPU_VIRTUAL_ADDRESS va;
    float* c = (float*)de_ring(256, 256, &va, NULL);
    if (!c) return;
    c[0] = rgba[0]; c[1] = rgba[1]; c[2] = rgba[2]; c[3] = rgba[3];
    de_transition(t, D3D12_RESOURCE_STATE_RENDER_TARGET);
    ID3D12GraphicsCommandList* cl = de_cl();
    const D3D12_CPU_DESCRIPTOR_HANDLE rh = de_cpu(s_de.rtv_heap, s_de.rtv_inc, (u32)t->rtv);
    cl->lpVtbl->OMSetRenderTargets(cl, 1, &rh, FALSE, NULL);
    cl->lpVtbl->SetPipelineState(cl, pso);
    cl->lpVtbl->SetGraphicsRootConstantBufferView(cl, 1, va);
    D3D12_VIEWPORT vp = { 0, 0, (float)t->w, (float)t->h, 0, 1 };
    D3D12_RECT sc = { 0, 0, (LONG)t->w, (LONG)t->h };
    cl->lpVtbl->RSSetViewports(cl, 1, &vp);
    cl->lpVtbl->RSSetScissorRects(cl, 1, &sc);
    cl->lpVtbl->IASetPrimitiveTopology(cl, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    cl->lpVtbl->DrawInstanced(cl, 3, 1, 0, 0);
    s_de.n_clear_c++;
}

static void de_clear_depth_stencil(void* user, u32 depth, u32 flags, float depth_value, u8 stencil)
{
    (void)user;
    de_obj* t = de_get(depth, DE_DEPTH);
    if (!t) return;
    D3D12_CLEAR_FLAGS f = 0;
    if (flags & RSX_BE_CLEAR_DEPTH)   f |= D3D12_CLEAR_FLAG_DEPTH;
    if (flags & RSX_BE_CLEAR_STENCIL) f |= D3D12_CLEAR_FLAG_STENCIL;
    if (!f) return;
    de_transition(t, D3D12_RESOURCE_STATE_DEPTH_WRITE);
    de_cl()->lpVtbl->ClearDepthStencilView(de_cl(), de_cpu(s_de.dsv_heap, s_de.dsv_inc, (u32)t->dsv),
        f, depth_value, stencil, 0, NULL);
    s_de.n_clear_z++;
}

static void de_readback(void* user, u32 surface, u32 x, u32 y, u32 w, u32 h, void* out, u32 out_pitch)
{
    (void)user;
    de_obj* t = de_get(surface, DE_COLOR);
    if (!t || !w || !h || !out || x >= t->w || y >= t->h) return;
    if (x + w > t->w) w = t->w - x;
    if (y + h > t->h) h = t->h - y;
    const u32 pitch = (w * t->bpp + 255u) & ~255u;
    const u64 bytes = (u64)pitch * h;
    if (bytes > s_de.readback_bytes) {
        de_submit();
        if (s_de.readback) s_de.readback->lpVtbl->Release(s_de.readback);
        s_de.readback = de_buffer(D3D12_HEAP_TYPE_READBACK, bytes, D3D12_RESOURCE_STATE_COPY_DEST);
        s_de.readback_bytes = s_de.readback ? bytes : 0;
        if (!s_de.readback) return;
    }
    const D3D12_RESOURCE_STATES was = t->state;
    de_transition(t, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION dst = {0}, src = {0};
    src.pResource = t->res; src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; src.SubresourceIndex = 0;
    dst.pResource = s_de.readback; dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint.Footprint.Format = t->fmt;
    dst.PlacedFootprint.Footprint.Width = w; dst.PlacedFootprint.Footprint.Height = h;
    dst.PlacedFootprint.Footprint.Depth = 1; dst.PlacedFootprint.Footprint.RowPitch = pitch;
    D3D12_BOX box = { x, y, 0, x + w, y + h, 1 };
    de_cl()->lpVtbl->CopyTextureRegion(de_cl(), &dst, 0, 0, 0, &src, &box);
    de_transition(t, was);
    de_submit();
    u8* map = NULL; D3D12_RANGE rr = { 0, (SIZE_T)bytes };
    if (FAILED(s_de.readback->lpVtbl->Map(s_de.readback, 0, &rr, (void**)&map)) || !map) return;
    for (u32 r = 0; r < h; r++)
        memcpy((u8*)out + (u64)r * out_pitch, map + (u64)r * pitch, (size_t)w * t->bpp);
    D3D12_RANGE wr = { 0, 0 };
    s_de.readback->lpVtbl->Unmap(s_de.readback, 0, &wr);
}

/* ---- pipelines ---------------------------------------------------------- */

static ID3DBlob* de_compile(const char* src, const char* target)
{
    ID3DBlob* blob = NULL; ID3DBlob* err = NULL;
    const HRESULT hr = D3DCompile(src, strlen(src), target, NULL, NULL, "main", target,
                                  D3DCOMPILE_OPTIMIZATION_LEVEL1, 0, &blob, &err);
    if (FAILED(hr)) {
        static int n = 0;
        if (n++ < 6)
            fprintf(stderr, "[D3D12-ENG] %s compile failed: %.400s\n", target,
                    err ? (const char*)err->lpVtbl->GetBufferPointer(err) : "?");
        if (err) err->lpVtbl->Release(err);
        if (blob) blob->lpVtbl->Release(blob);
        return NULL;
    }
    if (err) err->lpVtbl->Release(err);
    return blob;
}

static u32 de_cube_units(const char* ps)
{
    u32 mask = 0; char decl[40];
    for (u32 u = 0; u < DE_PS_UNITS; u++) {
        snprintf(decl, sizeof decl, "TextureCube rsx_tex%u ", u);
        if (strstr(ps, decl)) mask |= 1u << u;
    }
    return mask;
}

static u32 de_pipeline_create(void* user, const char* vs_hlsl, const char* ps_hlsl,
                              const rsx_be_render_state* rs, const rsx_vertex_layout_plan* layout,
                              u32 vertex_stride, rsx_be_format rt_fmt, u32 rt_count)
{
    (void)user;
    if (!vs_hlsl || !ps_hlsl || !rs || !layout) { s_de.n_pipe_fail++; return 0; }
    u32 bpp, block;
    const DXGI_FORMAT cf = de_dxgi(rt_fmt, &bpp, &block);
    if (cf == DXGI_FORMAT_UNKNOWN || block) { s_de.n_pipe_fail++; return 0; }
    u32 slot = DE_MAX_PIPE;
    for (u32 i = 0; i < DE_MAX_PIPE; i++) if (!s_de.pipe[i].used) { slot = i; break; }
    if (slot == DE_MAX_PIPE) { DE_WARN_ONCE(1u, "pipeline table full\n"); s_de.n_pipe_fail++; return 0; }
    de_pipe* P = &s_de.pipe[slot];
    memset(P, 0, sizeof *P);
    P->vs = de_compile(vs_hlsl, "vs_5_0");
    P->ps = P->vs ? de_compile(ps_hlsl, "ps_5_0") : NULL;
    if (!P->vs || !P->ps) {
        if (P->vs) P->vs->lpVtbl->Release(P->vs);
        memset(P, 0, sizeof *P);
        s_de.n_pipe_fail++;
        return 0;
    }
    P->rs = *rs;
    P->stride = vertex_stride;
    P->rt_count = rt_count ? (rt_count > RSX_BE_MAX_COLOR_TARGETS ? RSX_BE_MAX_COLOR_TARGETS : rt_count) : 1;
    P->rt_fmt = cf;
    P->cube_mask = de_cube_units(ps_hlsl);
    /* Every slot is a float4 at slot*16 carrying its register's ATTRn
     * semantic. A program that declares an ATTRn the layout does not carry
     * (the engine's fixed-function VS declares all sixteen) gets it aliased
     * to slot 0: D3D12 rejects a pipeline with an unmatched VS input. */
    u32 have = 0;
    for (u32 s = 0; s < layout->count && s < RSX_DSP_NUM_VERTEX_ATTR; s++) {
        D3D12_INPUT_ELEMENT_DESC* e = &P->il[P->nil++];
        e->SemanticName = "ATTR"; e->SemanticIndex = layout->attrs[s];
        e->Format = DXGI_FORMAT_R32G32B32A32_FLOAT; e->InputSlot = 0; e->AlignedByteOffset = s * 16u;
        e->InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA;
        have |= 1u << layout->attrs[s];
    }
    if (P->nil) for (u32 a = 0; a < RSX_DSP_NUM_VERTEX_ATTR; a++) {
        char sem[16]; snprintf(sem, sizeof sem, ":ATTR%u;", a);
        if ((have >> a) & 1u || !strstr(vs_hlsl, sem)) continue;
        D3D12_INPUT_ELEMENT_DESC* e = &P->il[P->nil++];
        e->SemanticName = "ATTR"; e->SemanticIndex = a;
        e->Format = DXGI_FORMAT_R32G32B32A32_FLOAT; e->InputSlot = 0; e->AlignedByteOffset = 0;
        e->InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA;
    }
    P->used = 1;
    s_de.n_pipe_ok++;
    return slot + 1u;
}

static void de_pipeline_release(void* user, u32 pipeline)
{
    (void)user;
    if (!pipeline || pipeline > DE_MAX_PIPE || !s_de.pipe[pipeline - 1].used) return;
    de_submit();
    de_pipe* P = &s_de.pipe[pipeline - 1];
    for (int t = 0; t < 3; t++) if (P->pso[t]) P->pso[t]->lpVtbl->Release(P->pso[t]);
    if (P->vs) P->vs->lpVtbl->Release(P->vs);
    if (P->ps) P->ps->lpVtbl->Release(P->ps);
    memset(P, 0, sizeof *P);
}

/* The PSO for one primitive class (0 point, 1 line, 2 triangle). */
static ID3D12PipelineState* de_pso(de_pipe* P, int cls)
{
    if (P->pso[cls]) return P->pso[cls];
    const rsx_be_render_state* rs = &P->rs;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = {0};
    pd.pRootSignature = s_de.root;
    pd.VS.pShaderBytecode = P->vs->lpVtbl->GetBufferPointer(P->vs);
    pd.VS.BytecodeLength  = P->vs->lpVtbl->GetBufferSize(P->vs);
    pd.PS.pShaderBytecode = P->ps->lpVtbl->GetBufferPointer(P->ps);
    pd.PS.BytecodeLength  = P->ps->lpVtbl->GetBufferSize(P->ps);
    pd.InputLayout.pInputElementDescs = P->nil ? P->il : NULL;
    pd.InputLayout.NumElements = P->nil;
    /* nv40 COLOR_MASK: B [0:7], G [8:15], R [16:23], A [24:31] */
    UINT8 wm = 0;
    if ((rs->color_mask >> 16) & 0xFFu) wm |= D3D12_COLOR_WRITE_ENABLE_RED;
    if ((rs->color_mask >>  8) & 0xFFu) wm |= D3D12_COLOR_WRITE_ENABLE_GREEN;
    if ((rs->color_mask >>  0) & 0xFFu) wm |= D3D12_COLOR_WRITE_ENABLE_BLUE;
    if ((rs->color_mask >> 24) & 0xFFu) wm |= D3D12_COLOR_WRITE_ENABLE_ALPHA;
    for (u32 r = 0; r < P->rt_count; r++) {
        D3D12_RENDER_TARGET_BLEND_DESC* b = &pd.BlendState.RenderTarget[r];
        b->BlendEnable    = rs->blend_enable ? TRUE : FALSE;
        b->SrcBlend       = (D3D12_BLEND)gl_blend_factor_d3d(rs->sf_rgb, 0);
        b->DestBlend      = (D3D12_BLEND)gl_blend_factor_d3d(rs->df_rgb, 0);
        b->BlendOp        = (D3D12_BLEND_OP)gl_blend_op_d3d(rs->eq_rgb);
        b->SrcBlendAlpha  = (D3D12_BLEND)gl_blend_factor_d3d(rs->sf_a, 1);
        b->DestBlendAlpha = (D3D12_BLEND)gl_blend_factor_d3d(rs->df_a, 1);
        b->BlendOpAlpha   = (D3D12_BLEND_OP)gl_blend_op_d3d(rs->eq_a);
        b->LogicOp        = D3D12_LOGIC_OP_NOOP;
        b->RenderTargetWriteMask = wm;
        pd.RTVFormats[r] = P->rt_fmt;
    }
    pd.BlendState.IndependentBlendEnable = FALSE;
    pd.SampleMask = 0xFFFFFFFFu;
    pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    /* CULL_FACE FRONT 0x0404 / BACK 0x0405 / FRONT_AND_BACK 0x0408 (front);
     * FRONT_FACE CCW 0x0901 -- the engine's vertex stage already outputs
     * D3D clip space, so the winding is judged on the same screen. */
    pd.RasterizerState.CullMode = !rs->cull_enable ? D3D12_CULL_MODE_NONE
        : (rs->cull_face == 0x0404u || rs->cull_face == 0x0408u) ? D3D12_CULL_MODE_FRONT : D3D12_CULL_MODE_BACK;
    pd.RasterizerState.FrontCounterClockwise = (rs->front_face == 0x0901u) ? TRUE : FALSE;
    pd.RasterizerState.DepthClipEnable = TRUE;
    pd.DepthStencilState.DepthEnable    = rs->depth_test ? TRUE : FALSE;
    pd.DepthStencilState.DepthWriteMask = (rs->depth_test && rs->depth_write) ? D3D12_DEPTH_WRITE_MASK_ALL
                                                                              : D3D12_DEPTH_WRITE_MASK_ZERO;
    pd.DepthStencilState.DepthFunc      = zfunc_d3d(gl_func_idx(rs->depth_func));
    if (rs->stencil_enable) {
        D3D12_DEPTH_STENCILOP_DESC f, b;
        f.StencilFunc        = zfunc_d3d(gl_func_idx(rs->s_func));
        f.StencilFailOp      = (D3D12_STENCIL_OP)(gl_stencil_op_idx(rs->s_fail) + 1u);
        f.StencilDepthFailOp = (D3D12_STENCIL_OP)(gl_stencil_op_idx(rs->s_zfail) + 1u);
        f.StencilPassOp      = (D3D12_STENCIL_OP)(gl_stencil_op_idx(rs->s_zpass) + 1u);
        b = f;
        if (rs->stencil_two_sided) {
            b.StencilFunc        = zfunc_d3d(gl_func_idx(rs->bs_func));
            b.StencilFailOp      = (D3D12_STENCIL_OP)(gl_stencil_op_idx(rs->bs_fail) + 1u);
            b.StencilDepthFailOp = (D3D12_STENCIL_OP)(gl_stencil_op_idx(rs->bs_zfail) + 1u);
            b.StencilPassOp      = (D3D12_STENCIL_OP)(gl_stencil_op_idx(rs->bs_zpass) + 1u);
        }
        pd.DepthStencilState.StencilEnable    = TRUE;
        pd.DepthStencilState.StencilReadMask  = (UINT8)rs->s_func_mask;
        pd.DepthStencilState.StencilWriteMask = (UINT8)rs->s_write_mask;
        pd.DepthStencilState.FrontFace = f;
        pd.DepthStencilState.BackFace  = b;
    }
    pd.DSVFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;
    pd.PrimitiveTopologyType = cls == 0 ? D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT
                             : cls == 1 ? D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE
                                        : D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pd.NumRenderTargets = P->rt_count;
    pd.SampleDesc.Count = 1;
    const HRESULT hr = s_d3d.device->lpVtbl->CreateGraphicsPipelineState(s_d3d.device, &pd,
        &IID_ID3D12PipelineState, (void**)&P->pso[cls]);
    if (FAILED(hr)) {
        static int n = 0;
        if (n++ < 6) fprintf(stderr, "[D3D12-ENG] CreateGraphicsPipelineState failed 0x%08lX (class %d)\n",
                             (unsigned long)hr, cls);
        P->pso[cls] = NULL;
    }
    return P->pso[cls];
}

/* ---- binding ------------------------------------------------------------ */

static void de_bind_targets(void* user, const u32* surfaces, u32 count, u32 depth)
{
    (void)user;
    if (count > RSX_BE_MAX_COLOR_TARGETS) count = RSX_BE_MAX_COLOR_TARGETS;
    s_de.nrt = surfaces ? count : 0;
    for (u32 i = 0; i < s_de.nrt; i++) s_de.rt[i] = surfaces[i];
    s_de.depth = depth;
}
static void de_bind_pipeline(void* user, u32 pipeline) { (void)user; s_de.pipeline = pipeline; }
static void de_bind_vs_constants(void* user, const void* data, u32 bytes)
{
    (void)user;
    if (bytes > sizeof s_de.vs_copy) bytes = sizeof s_de.vs_copy;
    if (data && bytes) memcpy(s_de.vs_copy, data, bytes);
    s_de.vsc_bytes = data ? bytes : 0;
}
static void de_bind_ps_constants(void* user, const void* data, u32 bytes)
{
    (void)user;
    if (bytes > sizeof s_de.ps_copy) bytes = sizeof s_de.ps_copy;
    if (data && bytes) memcpy(s_de.ps_copy, data, bytes);
    s_de.psc_bytes = data ? bytes : 0;
}
static void de_bind_textures(void* user, const u32* textures, const rsx_be_sampler_desc* samplers, u32 mask)
{
    (void)user;
    for (u32 u = 0; u < DE_PS_UNITS; u++) {
        const int on = (mask >> u) & 1u;
        s_de.tex[u] = (on && textures) ? textures[u] : 0;
        if (on && samplers) s_de.samp_desc[u] = samplers[u];
    }
    s_de.tex_mask = mask;
}
static void de_bind_vertex_textures(void* user, const u32* textures, const rsx_be_sampler_desc* samplers, u32 mask)
{
    (void)user;
    for (u32 u = 0; u < DE_VS_UNITS; u++) {
        const int on = (mask >> u) & 1u;
        s_de.vtex[u] = (on && textures) ? textures[u] : 0;
        if (on && samplers) s_de.vsamp_desc[u] = samplers[u];
    }
    s_de.vtex_mask = mask;
}
static void de_set_viewport(void* user, float x, float y, float w, float h)
{ (void)user; s_de.vp[0] = x; s_de.vp[1] = y; s_de.vp[2] = w; s_de.vp[3] = h; s_de.have_vp = 1; }
static void de_set_scissor(void* user, u32 x, u32 y, u32 w, u32 h)
{ (void)user; s_de.sc[0] = x; s_de.sc[1] = y; s_de.sc[2] = w; s_de.sc[3] = h; s_de.have_sc = 1; }
static void de_set_stencil_ref(void* user, u32 ref) { (void)user; s_de.stencil_ref = ref; }
static void de_set_blend_color(void* user, const float rgba[4])
{ (void)user; memcpy(s_de.blend_color, rgba, sizeof s_de.blend_color); }

/* GL/RSX wrap codes 1..8 -> D3D12 address mode (same table as the Vulkan
 * backend: WRAP, MIRROR, CLAMP_TO_EDGE, BORDER, CLAMP, MIRROR_ONCE_*). */
static D3D12_TEXTURE_ADDRESS_MODE de_wrap(u8 w)
{
    switch (w) {
    case 1: return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    case 2: return D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
    case 3: return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    case 4: return D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    case 5: return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    case 6: case 7: case 8: return D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE;
    default: return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    }
}

static int de_sampler(const rsx_be_sampler_desc* d)
{
    u32 lo, hi;
    memcpy(&lo, &d->min_lod, 4); memcpy(&hi, &d->max_lod, 4);
    const u64 key = (u64)d->min_linear | ((u64)d->mag_linear << 1) | ((u64)d->mip_linear << 2)
                  | ((u64)d->mip_present << 3) | ((u64)(d->wrap_s & 0xF) << 4)
                  | ((u64)(d->wrap_t & 0xF) << 8) | ((u64)(d->wrap_r & 0xF) << 12)
                  | ((u64)(lo >> 16) << 16) | ((u64)(hi >> 16) << 32);
    for (u32 i = 0; i < s_de.n_samp; i++) if (s_de.samp[i].key == key) return s_de.samp[i].idx;
    if (s_de.n_samp >= DE_SAMP_CPU_COUNT) return 0;
    D3D12_SAMPLER_DESC sd = {0};
    const int mn = d->min_linear ? 1 : 0, mg = d->mag_linear ? 1 : 0, mp = d->mip_linear ? 1 : 0;
    sd.Filter = (D3D12_FILTER)((mn ? 0x10 : 0) | (mg ? 0x4 : 0) | (mp ? 0x1 : 0));
    sd.AddressU = de_wrap(d->wrap_s); sd.AddressV = de_wrap(d->wrap_t); sd.AddressW = de_wrap(d->wrap_r);
    sd.MinLOD = d->min_lod;
    sd.MaxLOD = d->mip_present ? d->max_lod : d->min_lod;
    if (sd.MaxLOD < sd.MinLOD) sd.MaxLOD = sd.MinLOD;
    sd.MaxAnisotropy = 1; sd.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
    const int idx = (int)s_de.n_samp;
    s_d3d.device->lpVtbl->CreateSampler(s_d3d.device, &sd, de_cpu(s_de.samp_cpu, s_de.samp_inc, (u32)idx));
    s_de.samp[s_de.n_samp].key = key; s_de.samp[s_de.n_samp].idx = idx; s_de.n_samp++;
    return idx;
}

/* A shader-visible table of n samplers, cached by the combination. */
static D3D12_GPU_DESCRIPTOR_HANDLE de_sampler_table(const int* idx, u32 n)
{
    u64 key = 1469598103934665603ull ^ n;
    for (u32 i = 0; i < n; i++) { key ^= (u64)(idx[i] + 1); key *= 1099511628211ull; }
    const u32 per = DE_PS_UNITS;     /* both tables use a 16-wide slot */
    for (u32 i = 0; i < s_de.n_samp_tables; i++)
        if (s_de.samp_tables[i].key == key) return de_gpu(s_de.samp_gpu, s_de.samp_inc, s_de.samp_tables[i].base);
    if ((s_de.n_samp_tables + 1) * per > DE_SAMP_GPU_COUNT)
        return de_gpu(s_de.samp_gpu, s_de.samp_inc, 0);   /* de_draw checked room first */
    const u32 base = s_de.n_samp_tables * per;
    for (u32 i = 0; i < n; i++)
        s_d3d.device->lpVtbl->CopyDescriptorsSimple(s_d3d.device, 1,
            de_cpu(s_de.samp_gpu, s_de.samp_inc, base + i), de_cpu(s_de.samp_cpu, s_de.samp_inc, (u32)idx[i]),
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    s_de.samp_tables[s_de.n_samp_tables].key = key;
    s_de.samp_tables[s_de.n_samp_tables].base = base;
    s_de.n_samp_tables++;
    return de_gpu(s_de.samp_gpu, s_de.samp_inc, base);
}

/* The SRV a bound handle samples through, or the dummy of the unit's kind.
 * A colour target that is also bound as a render target of this draw cannot
 * be read in the same pass; it samples the dummy (the engine's
 * color_snapshot route is not implemented yet). */
static int de_srv_for(u32 handle, int cube)
{
    de_obj* o = de_get(handle, 0);
    if (o && (o->kind == DE_TEXTURE || o->kind == DE_COLOR || o->kind == DE_VIEW) &&
        ((o->faces == 6) == (cube != 0)) && o->srv >= 0) {
        const u32 owner = o->kind == DE_VIEW ? o->parent : handle;
        for (u32 i = 0; i < s_de.nrt; i++)
            if (s_de.rt[i] == owner) { DE_WARN_ONCE(2u, "a draw samples its own render target: dummy bound\n"); goto dummy; }
        de_obj* res = de_get(owner, 0);
        if (res && res->kind == DE_COLOR)
            de_transition(res, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        else if (res && res->kind == DE_TEXTURE)
            de_transition(res, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        return o->srv;
    }
dummy:;
    de_obj* d = de_get(cube ? s_de.dummycube : s_de.dummy2d, DE_TEXTURE);
    if (d) de_transition(d, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    return d ? d->srv : -1;
}

static D3D12_GPU_DESCRIPTOR_HANDLE de_srv_table(const int* srv, u32 n)
{
    if (s_de.srv_gpu_next + n > s_de.srv_end) { de_submit(); s_de.n_ring_wrap++; }
    const u32 base = s_de.srv_gpu_next;
    s_de.srv_gpu_next += n;
    for (u32 i = 0; i < n; i++)
        if (srv[i] >= 0)
            s_d3d.device->lpVtbl->CopyDescriptorsSimple(s_d3d.device, 1,
                de_cpu(s_de.srv_gpu, s_de.srv_inc, base + i), de_cpu(s_de.srv_cpu, s_de.srv_inc, (u32)srv[i]),
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    return de_gpu(s_de.srv_gpu, s_de.srv_inc, base);
}

/* ---- draw --------------------------------------------------------------- */

static u32 de_scratch_depth(u32 w, u32 h)
{
    de_obj* z = de_get(s_de.scratch_depth, DE_DEPTH);
    if (z && z->w >= w && z->h >= h) return s_de.scratch_depth;
    if (z) de_release_obj(s_de.scratch_depth);
    s_de.scratch_depth = 0;
    s_de.scratch_depth = de_depth_target_create(NULL, w, h);
    return s_de.scratch_depth;
}

static void de_draw(void* user, rsx_topology topology, const void* vertices, u32 vertex_count,
                    u32 stride, const u32* indices, u32 index_count)
{
    (void)user;
    if (!s_de.pipeline || s_de.pipeline > DE_MAX_PIPE || !vertices || !vertex_count || !stride ||
        topology < RSX_TOPOLOGY_POINTS || topology > RSX_TOPOLOGY_TRIANGLE_STRIP) { s_de.n_draw_skipped++; return; }
    de_pipe* P = &s_de.pipe[s_de.pipeline - 1];
    if (!P->used || stride != P->stride) { s_de.n_draw_skipped++; return; }
    const u32 n = s_de.nrt;
    if (n < 1 || n != P->rt_count) {
        DE_WARN_ONCE(4u, "draw with %u colour targets for a %u-target pipeline skipped\n", n, P->rt_count);
        s_de.n_draw_skipped++; return;
    }
    de_obj* rts[RSX_BE_MAX_COLOR_TARGETS];
    for (u32 i = 0; i < n; i++) {
        rts[i] = de_get(s_de.rt[i], DE_COLOR);
        if (!rts[i]) { s_de.n_draw_skipped++; return; }
    }
    if (rts[0]->fmt != P->rt_fmt) {
        DE_WARN_ONCE(8u, "draw into DXGI format %d with a pipeline built for %d skipped\n", (int)rts[0]->fmt, (int)P->rt_fmt);
        s_de.n_draw_skipped++; return;
    }
    const u32 w = rts[0]->w, h = rts[0]->h;
    u32 zh = s_de.depth;
    de_obj* z = de_get(zh, DE_DEPTH);
    if (!z || z->w < w || z->h < h) { zh = de_scratch_depth(w, h); z = de_get(zh, DE_DEPTH); }
    if (!z) { s_de.n_draw_skipped++; return; }
    const int cls = topology == RSX_TOPOLOGY_POINTS ? 0
                  : (topology == RSX_TOPOLOGY_LINES || topology == RSX_TOPOLOGY_LINE_STRIP) ? 1 : 2;
    ID3D12PipelineState* pso = de_pso(P, cls);
    if (!pso) { s_de.n_draw_skipped++; return; }

    /* Data first: a ring wrap submits, which must not split this draw. */
    const u64 vbytes = (u64)vertex_count * stride;
    const u64 ibytes = (indices && index_count) ? (u64)index_count * 4u : 0;
    const u64 need = vbytes + ibytes + DE_VPCONST_BYTES + DE_PSCONST_MIN + (s_de.psc_bytes > DE_PSCONST_MIN ? s_de.psc_bytes : 0) + 4096;
    /* Room for everything this draw takes, checked up front: a submit in the
     * middle would restart the rings under data this draw already wrote. */
    if (s_de.ring_off + need > s_de.ring_end ||
        s_de.srv_gpu_next + DE_PS_UNITS + DE_VS_UNITS > s_de.srv_end) { de_submit(); s_de.n_ring_wrap++; }
    if ((s_de.n_samp_tables + 2) * DE_PS_UNITS > DE_SAMP_GPU_COUNT) {
        de_submit();                 /* nothing in flight references a table now */
        s_de.n_samp_tables = 0;
    }

    D3D12_GPU_VIRTUAL_ADDRESS vva, iva = 0, cva, pva;
    u8* vm = de_ring(vbytes, 16, &vva, NULL);
    if (!vm) { s_de.n_draw_skipped++; return; }
    memcpy(vm, vertices, (size_t)vbytes);
    if (ibytes) { u8* im = de_ring(ibytes, 16, &iva, NULL); memcpy(im, indices, (size_t)ibytes); }
    u8* cm = de_ring(DE_VPCONST_BYTES, 256, &cva, NULL);
    if (s_de.vsc_bytes) memcpy(cm, s_de.vs_copy, s_de.vsc_bytes);
    if (s_de.vsc_bytes < DE_VPCONST_BYTES) memset(cm + s_de.vsc_bytes, 0, DE_VPCONST_BYTES - s_de.vsc_bytes);
    const u32 pbytes = s_de.psc_bytes > DE_PSCONST_MIN ? ((s_de.psc_bytes + 255u) & ~255u) : DE_PSCONST_MIN;
    /* D3D12ENG_PSC_DBG=<pipeline>: the PS constants that pipeline receives. */
    { static int want = -2, n = 0;
      if (want == -2) { const char* e = getenv("D3D12ENG_PSC_DBG"); want = e ? atoi(e) : -1; }
      if (want == (int)s_de.pipeline && n < 6) {
          n++;
          const float* f = (const float*)s_de.ps_copy;
          fprintf(stderr, "[D3D12-ENG] pipe %u psc %u bytes: c0=(%g %g %g %g) c1=(%g %g %g %g) c2=(%g %g %g %g)%c",
                  s_de.pipeline, s_de.psc_bytes, f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7],
                  f[8], f[9], f[10], f[11], 10);
          const u32 th = (s_de.tex_mask & 1u) ? s_de.tex[0] : 0;
          de_obj* to = de_get(th, 0);
          const u32 owner = to && to->kind == DE_VIEW ? to->parent : th;
          de_obj* oo = de_get(owner, 0);
          u8 px[4] = {0, 0, 0, 0};
          if (oo && oo->kind == DE_COLOR) de_readback(NULL, owner, oo->w / 2, oo->h / 2, 1, 1, px, 4);
          fprintf(stderr, "[D3D12-ENG]   t0 handle %u kind %d owner %u kind %d %ux%u srv %d center=(%u %u %u %u) samp wrap %u/%u lin %u/%u%c",
                  th, to ? to->kind : -1, owner, oo ? oo->kind : -1, oo ? oo->w : 0, oo ? oo->h : 0,
                  to ? to->srv : -9, px[0], px[1], px[2], px[3], s_de.samp_desc[0].wrap_s, s_de.samp_desc[0].wrap_t,
                  s_de.samp_desc[0].min_linear, s_de.samp_desc[0].mag_linear, 10);
          const float* vc = (const float*)s_de.vs_copy;
          const float* vt = (const float*)vertices;
          fprintf(stderr, "[D3D12-ENG]   c32=(%g %g %g %g) c36=(%g %g %g %g) verts=%u stride=%u idx=%u "
                  "v0=(%g %g %g %g | %g %g %g %g) v1=(%g %g %g %g | %g %g %g %g) vp=%g,%g %gx%g%c",
                  vc[128], vc[129], vc[130], vc[131], vc[144], vc[145], vc[146], vc[147],
                  vertex_count, stride, index_count,
                  vt[0], vt[1], vt[2], vt[3], vt[4], vt[5], vt[6], vt[7],
                  vt[stride / 4], vt[stride / 4 + 1], vt[stride / 4 + 2], vt[stride / 4 + 3],
                  vt[stride / 4 + 4], vt[stride / 4 + 5], vt[stride / 4 + 6], vt[stride / 4 + 7],
                  s_de.vp[0], s_de.vp[1], s_de.vp[2], s_de.vp[3], 10);
      } }
    u8* pm = de_ring(pbytes, 256, &pva, NULL);
    if (s_de.psc_bytes) memcpy(pm, s_de.ps_copy, s_de.psc_bytes);
    if (s_de.psc_bytes < pbytes) memset(pm + s_de.psc_bytes, 0, pbytes - s_de.psc_bytes);

    /* Textures and samplers (transitions recorded here, before the draw). */
    int srv[DE_PS_UNITS + DE_VS_UNITS], smp[DE_PS_UNITS + DE_VS_UNITS];
    for (u32 u = 0; u < DE_PS_UNITS; u++) {
        const int cube = (P->cube_mask >> u) & 1u;
        srv[u] = de_srv_for(((s_de.tex_mask >> u) & 1u) ? s_de.tex[u] : 0, cube);
        smp[u] = ((s_de.tex_mask >> u) & 1u) ? de_sampler(&s_de.samp_desc[u]) : 0;
    }
    for (u32 u = 0; u < DE_VS_UNITS; u++) {
        srv[DE_PS_UNITS + u] = de_srv_for(((s_de.vtex_mask >> u) & 1u) ? s_de.vtex[u] : 0, 0);
        smp[DE_PS_UNITS + u] = ((s_de.vtex_mask >> u) & 1u) ? de_sampler(&s_de.vsamp_desc[u]) : 0;
    }
    if (s_de.n_samp == 0) { rsx_be_sampler_desc d0 = {0}; d0.wrap_s = d0.wrap_t = d0.wrap_r = 1; de_sampler(&d0); }
    const D3D12_GPU_DESCRIPTOR_HANDLE ps_srv  = de_srv_table(srv, DE_PS_UNITS);
    const D3D12_GPU_DESCRIPTOR_HANDLE vs_srv  = de_srv_table(srv + DE_PS_UNITS, DE_VS_UNITS);
    const D3D12_GPU_DESCRIPTOR_HANDLE ps_samp = de_sampler_table(smp, DE_PS_UNITS);
    const D3D12_GPU_DESCRIPTOR_HANDLE vs_samp = de_sampler_table(smp + DE_PS_UNITS, DE_VS_UNITS);

    for (u32 i = 0; i < n; i++) de_transition(rts[i], D3D12_RESOURCE_STATE_RENDER_TARGET);
    de_transition(z, D3D12_RESOURCE_STATE_DEPTH_WRITE);

    ID3D12GraphicsCommandList* cl = de_cl();
    D3D12_CPU_DESCRIPTOR_HANDLE rh[RSX_BE_MAX_COLOR_TARGETS];
    for (u32 i = 0; i < n; i++) rh[i] = de_cpu(s_de.rtv_heap, s_de.rtv_inc, (u32)rts[i]->rtv);
    const D3D12_CPU_DESCRIPTOR_HANDLE dh = de_cpu(s_de.dsv_heap, s_de.dsv_inc, (u32)z->dsv);
    cl->lpVtbl->OMSetRenderTargets(cl, n, rh, FALSE, &dh);
    cl->lpVtbl->SetPipelineState(cl, pso);
    cl->lpVtbl->SetGraphicsRootConstantBufferView(cl, 0, cva);
    cl->lpVtbl->SetGraphicsRootConstantBufferView(cl, 1, pva);
    cl->lpVtbl->SetGraphicsRootDescriptorTable(cl, 2, ps_srv);
    cl->lpVtbl->SetGraphicsRootDescriptorTable(cl, 3, ps_samp);
    cl->lpVtbl->SetGraphicsRootDescriptorTable(cl, 4, vs_srv);
    cl->lpVtbl->SetGraphicsRootDescriptorTable(cl, 5, vs_samp);
    D3D12_VIEWPORT vp = { 0.0f, 0.0f, (float)w, (float)h, 0.0f, 1.0f };
    if (s_de.have_vp && s_de.vp[2] > 0.0f && s_de.vp[3] > 0.0f) {
        vp.TopLeftX = s_de.vp[0]; vp.TopLeftY = s_de.vp[1]; vp.Width = s_de.vp[2]; vp.Height = s_de.vp[3];
    }
    D3D12_RECT sc = { 0, 0, (LONG)w, (LONG)h };
    if (s_de.have_sc) {
        u32 x0 = s_de.sc[0] < w ? s_de.sc[0] : w, y0 = s_de.sc[1] < h ? s_de.sc[1] : h;
        u32 x1 = s_de.sc[0] + s_de.sc[2], y1 = s_de.sc[1] + s_de.sc[3];
        if (x1 > w) x1 = w;
        if (y1 > h) y1 = h;
        sc.left = (LONG)x0; sc.top = (LONG)y0; sc.right = (LONG)(x1 > x0 ? x1 : x0); sc.bottom = (LONG)(y1 > y0 ? y1 : y0);
    }
    cl->lpVtbl->RSSetViewports(cl, 1, &vp);
    cl->lpVtbl->RSSetScissorRects(cl, 1, &sc);
    cl->lpVtbl->OMSetStencilRef(cl, s_de.stencil_ref & 0xFFu);
    cl->lpVtbl->OMSetBlendFactor(cl, s_de.blend_color);
    static const D3D12_PRIMITIVE_TOPOLOGY topo[6] = {
        D3D_PRIMITIVE_TOPOLOGY_UNDEFINED, D3D_PRIMITIVE_TOPOLOGY_POINTLIST, D3D_PRIMITIVE_TOPOLOGY_LINELIST,
        D3D_PRIMITIVE_TOPOLOGY_LINESTRIP, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP };
    cl->lpVtbl->IASetPrimitiveTopology(cl, topo[topology]);
    D3D12_VERTEX_BUFFER_VIEW vbv = { vva, (UINT)vbytes, stride };
    cl->lpVtbl->IASetVertexBuffers(cl, 0, 1, &vbv);
    if (ibytes) {
        D3D12_INDEX_BUFFER_VIEW ibv = { iva, (UINT)ibytes, DXGI_FORMAT_R32_UINT };
        cl->lpVtbl->IASetIndexBuffer(cl, &ibv);
        cl->lpVtbl->DrawIndexedInstanced(cl, index_count, 1, 0, 0, 0);
    } else {
        cl->lpVtbl->DrawInstanced(cl, vertex_count, 1, 0, 0);
    }
    s_de.n_draw++;
}

/* ---- present ------------------------------------------------------------ */

/* D3D12ENG_DUMP: the presented surface as a 24-bit BMP (RGBA8 targets only). */
static void de_dump(u32 surface, u64 n)
{
    const char* dir = getenv("D3D12ENG_DUMP");
    if (!dir || !*dir) return;
    const de_obj* t = de_get(surface, DE_COLOR);
    if (!t || t->fmt != DXGI_FORMAT_R8G8B8A8_UNORM) return;
    u8* px = (u8*)malloc((size_t)t->w * t->h * 4);
    if (!px) return;
    de_readback(NULL, surface, 0, 0, t->w, t->h, px, t->w * 4);
    char path[512]; snprintf(path, sizeof path, "%s/present_%06llu.bmp", dir, (unsigned long long)n);
    FILE* f = fopen(path, "wb");
    if (f) {
        const u32 rowb = (t->w * 3 + 3) & ~3u, datasz = rowb * t->h;
        u8 hdr[54] = {0};
        hdr[0] = 'B'; hdr[1] = 'M';
        *(u32*)(hdr + 2) = 54 + datasz; *(u32*)(hdr + 10) = 54; *(u32*)(hdr + 14) = 40;
        *(int*)(hdr + 18) = (int)t->w; *(int*)(hdr + 22) = (int)t->h;
        *(u16*)(hdr + 26) = 1; *(u16*)(hdr + 28) = 24; *(u32*)(hdr + 34) = datasz;
        fwrite(hdr, 1, 54, f);
        u8* line = (u8*)calloc(1, rowb);
        if (line) for (int y = (int)t->h - 1; y >= 0; y--) {
            const u8* s = px + (size_t)y * t->w * 4;
            for (u32 x = 0; x < t->w; x++) { line[x*3] = s[x*4+2]; line[x*3+1] = s[x*4+1]; line[x*3+2] = s[x*4]; }
            fwrite(line, 1, rowb, f);
        }
        free(line);
        fclose(f);
    }
    free(px);
}

static void de_present(void* user, u32 surface)
{
    (void)user;
    const u32 bi = s_d3d.swap_chain->lpVtbl->GetCurrentBackBufferIndex(s_d3d.swap_chain);
    ID3D12Resource* bb = s_d3d.render_targets[bi];
    D3D12_RESOURCE_STATES bst = D3D12_RESOURCE_STATE_PRESENT;
    de_transition_raw(bb, &bst, D3D12_RESOURCE_STATE_RENDER_TARGET);
    D3D12_CPU_DESCRIPTOR_HANDLE bbv;
    s_d3d.rtv_heap->lpVtbl->GetCPUDescriptorHandleForHeapStart(s_d3d.rtv_heap, &bbv);
    bbv.ptr += (u64)bi * s_d3d.rtv_descriptor_size;
    const float black[4] = {0, 0, 0, 1};
    de_cl()->lpVtbl->ClearRenderTargetView(de_cl(), bbv, black, 0, NULL);
    de_obj* t = de_get(surface, DE_COLOR);
    if (t && t->fmt == DXGI_FORMAT_R8G8B8A8_UNORM) {
        const D3D12_RESOURCE_STATES was = t->state;
        de_transition(t, D3D12_RESOURCE_STATE_COPY_SOURCE);
        de_transition_raw(bb, &bst, D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION dst = {0}, src = {0};
        src.pResource = t->res; src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dst.pResource = bb;     dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        const u32 w = t->w < s_d3d.width ? t->w : s_d3d.width;
        const u32 h = t->h < s_d3d.height ? t->h : s_d3d.height;
        D3D12_BOX box = { 0, 0, 0, w, h, 1 };
        de_cl()->lpVtbl->CopyTextureRegion(de_cl(), &dst, 0, 0, 0, &src, &box);
        de_transition(t, was);
    }
    de_transition_raw(bb, &bst, D3D12_RESOURCE_STATE_PRESENT);
    de_submit_frame();
    /* Sync interval 0 by default. Present runs on the thread that also walks
     * and translates the guest's FIFO, and with interval 1 it waited for the
     * monitor's vblank there -- ~8.5 ms a frame in ICO, a third of that
     * thread's time, all of it time the next frame's translation could not
     * use. The guest is already paced by the emulated vblank, and in a window
     * a flip-model swap chain hands the compositor the latest frame without
     * tearing. D3D12ENG_VSYNC=1 restores the old wait. */
    { static int vs = -1;
      if (vs < 0) { const char* e = getenv("D3D12ENG_VSYNC"); vs = (e && atoi(e) > 0) ? 1 : 0; }
      LARGE_INTEGER a, b; QueryPerformanceCounter(&a);
      s_d3d.swap_chain->lpVtbl->Present(s_d3d.swap_chain, (UINT)vs, 0);
      QueryPerformanceCounter(&b); s_de_present_q += b.QuadPart - a.QuadPart; }
    s_de.n_present++;

    /* Titlebar stats, refreshed once a second (the recorder's are on a path
     * the engine never takes): the title's own frame rate -- flips per
     * second, the number that matters, since the host may present a frame
     * twice -- then presents per second and draws per guest frame. */
    {
        static ULONGLONG t0;
        static u64 p0, d0;
        static unsigned f0;
        extern unsigned cellGcm_flip_request_count(void);
        const ULONGLONG now = GetTickCount64();
        const unsigned flips = cellGcm_flip_request_count();
        if (!t0) { t0 = now; p0 = s_de.n_present; d0 = s_de.n_draw; f0 = flips; }
        else if (now - t0 >= 1000 && s_d3d.hwnd) {
            extern char g_rsx_title_base[128];
            const double secs = (double)(now - t0) / 1000.0;
            const unsigned df = flips - f0;
            char tb[256];
            snprintf(tb, sizeof tb, "%s | FPS: %.1f | presents: %.1f/s | draws/frame: %llu | %ux%u (D3D12 engine)",
                     g_rsx_title_base, df / secs, (double)(s_de.n_present - p0) / secs,
                     df ? (unsigned long long)((s_de.n_draw - d0) / df) : 0ull, s_d3d.width, s_d3d.height);
            SetWindowTextA(s_d3d.hwnd, tb);
            t0 = now; p0 = s_de.n_present; d0 = s_de.n_draw; f0 = flips;
        }
    }

    { static int every = -1;
      if (every < 0) { const char* e = getenv("D3D12ENG_DUMP_EVERY"); every = e ? atoi(e) : 300;
                       if (every <= 0) every = 300; }
      if (surface && (s_de.n_present % (u64)every) == 1) de_dump(surface, s_de.n_present); }

    /* With the debug layer on (D3D12_DBG), print what it stored -- the
     * engine path never reaches the recorder's InfoQueue dump. */
    if (de_log_on()) {
        static int shown = 0;
        ID3D12InfoQueue* iq = NULL;
        if (shown < 64 && SUCCEEDED(s_d3d.device->lpVtbl->QueryInterface(s_d3d.device, &IID_ID3D12InfoQueue,
                                                                        (void**)&iq)) && iq) {
            const UINT64 n = iq->lpVtbl->GetNumStoredMessages(iq);
            for (UINT64 mi = 0; mi < n && shown < 64; mi++) {
                SIZE_T len = 0;
                iq->lpVtbl->GetMessage(iq, mi, NULL, &len);
                D3D12_MESSAGE* m = (D3D12_MESSAGE*)malloc(len);
                if (m && SUCCEEDED(iq->lpVtbl->GetMessage(iq, mi, m, &len)) &&
                    m->Severity <= D3D12_MESSAGE_SEVERITY_WARNING) {
                    fprintf(stderr, "[D3D12-ENG] debug layer: %.300s%c", m->pDescription, 10);
                    shown++;
                }
                free(m);
            }
            iq->lpVtbl->ClearStoredMessages(iq);
            iq->lpVtbl->Release(iq);
        }
    }
    if (de_log_on()) {
        static ULONGLONG t0;
        const ULONGLONG now = GetTickCount64();
        if (!t0) t0 = now;
        if (now - t0 >= 5000) {
            t0 = now;
            fprintf(stderr, "[D3D12-ENG] presents=%llu clears(c/z)=%llu/%llu targets=%llu depths=%llu "
                    "textures=%llu pipelines ok/fail=%llu/%llu draws=%llu skipped=%llu submits=%llu wraps=%llu writes=%llu depth_reads=%llu depth_conv=%llu copies=%llu%c",
                    (unsigned long long)s_de.n_present, (unsigned long long)s_de.n_clear_c,
                    (unsigned long long)s_de.n_clear_z, (unsigned long long)s_de.n_targets,
                    (unsigned long long)s_de.n_depths, (unsigned long long)s_de.n_tex,
                    (unsigned long long)s_de.n_pipe_ok, (unsigned long long)s_de.n_pipe_fail,
                    (unsigned long long)s_de.n_draw, (unsigned long long)s_de.n_draw_skipped,
                    (unsigned long long)s_de.n_submit, (unsigned long long)s_de.n_ring_wrap,
                    (unsigned long long)s_de.n_writes, (unsigned long long)s_de.n_depth_reads,
                    (unsigned long long)s_de.n_depth_conv, (unsigned long long)s_de.n_copies, 10);
            { static unsigned long long p0; LARGE_INTEGER f; QueryPerformanceFrequency(&f);
              const unsigned long long np = s_de.n_present - p0; p0 = s_de.n_present;
              fprintf(stderr, "[D3D12-ENG] per present: gpu-fence wait %.2f ms, Present() %.2f ms (%llu presents)%c",
                      np ? (double)s_de_wait_q * 1000.0 / f.QuadPart / np : 0.0,
                      np ? (double)s_de_present_q * 1000.0 / f.QuadPart / np : 0.0, np, 10);
              s_de_wait_q = s_de_present_q = 0; }
        }
    }
}

static const rsx_draw_backend s_d3d12_engine_backend = {
    .user                 = NULL,
    .init                 = de_init,
    .shutdown             = de_shutdown,
    .submit_and_wait      = de_submit_and_wait,
    .texture_create       = de_texture_create,
    .texture_upload       = de_texture_upload,
    .texture_release      = de_release,
    .color_target_create  = de_color_target_create,
    .color_target_release = de_release,
    .color_target_write   = de_color_target_write,
    .surface_view         = de_surface_view,
    .depth_target_create  = de_depth_target_create,
    .depth_target_release = de_release,
    .depth_snapshot       = de_depth_snapshot,
    .pipeline_create      = de_pipeline_create,
    .pipeline_release     = de_pipeline_release,
    .bind_targets         = de_bind_targets,
    .bind_pipeline        = de_bind_pipeline,
    .bind_vs_constants    = de_bind_vs_constants,
    .bind_ps_constants    = de_bind_ps_constants,
    .bind_textures        = de_bind_textures,
    .bind_vertex_textures = de_bind_vertex_textures,
    .set_viewport         = de_set_viewport,
    .set_scissor          = de_set_scissor,
    .set_stencil_ref      = de_set_stencil_ref,
    .set_depth_bounds     = NULL,
    .draw                 = de_draw,
    .clear_color          = de_clear_color,
    .clear_depth_stencil  = de_clear_depth_stencil,
    .present              = de_present,
    .readback             = de_readback,
    .depth_readback       = de_depth_readback,
    .color_snapshot       = NULL,
    .clear_color_masked   = de_clear_color_masked,
    .set_blend_color      = de_set_blend_color,
    .depth_as_color       = de_depth_as_color,
    .copy_region          = de_copy_region,
    .depth_blit           = de_depth_blit,
};

#endif /* RSX_D3D12_ENGINE_IMPL_H */
