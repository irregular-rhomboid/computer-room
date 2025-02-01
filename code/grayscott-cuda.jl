using CUDA

function grayscott_reaction_kernel!(du, dv, u, v, f, k)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    if i <= size(du, 1) && j <= size(du, 2)
        uv = u[i,j] * v[i,j]^2
        @inbounds du[i,j] += -uv + f * (1.0 - u[i,j])
        @inbounds dv[i,j] += uv - (f + k[i,j]) * v[i,j]
    end
    return nothing
end

function grayscott_reaction!(du, dv, u, v, f, k)
    threads = (16, 16)
    blocks = ceil.(Int, size(du) ./ threads)
    @cuda blocks=blocks threads=threads grayscott_reaction_kernel!(du, dv, u, v, f, k)
end

function diffuse_kernel!(du, u, D)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    if 2 <= i < size(du, 1) && 2 <= j < size(du, 2)
        tmp = 0.0f0
        @inbounds begin
            tmp += 0.05f0*u[i-1,j-1] + 0.2f0*u[i,j-1] + 0.05f0*u[i+1,j-1]
            tmp += 0.2f0*u[i-1,j] - 1.0f0*u[i,j] + 0.2f0*u[i+1,j]
            tmp += 0.05f0*u[i-1,j+1] + 0.2f0*u[i,j+1] + 0.05f0*u[i+1,j+1]
            du[i,j] += D * tmp
        end
    end
    return nothing
end

function diffuse!(du, u, D)
    threads = (16, 16)
    blocks = ceil.(Int, size(du) ./ threads)
    @cuda blocks=blocks threads=threads diffuse_kernel!(du, u, D)
end

function clamp01_kernel!(x)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    if i <= size(x, 1) && j <= size(x, 2)
        @inbounds x[i,j] = clamp(x[i,j], 0.0f0, 1.0f0)
    end
    return nothing
end

function clamp01!(x)
    threads = (16, 16)
    blocks = ceil.(Int, size(x) ./ threads)
    @cuda blocks=blocks threads=threads clamp01_kernel!(x)
end

function forced_grayscott(img; f=0.0347f0, k=0.057f0, D_u=0.4f0, D_v=0.06f0, tmax=10, h=0.05f0)
    # Convert input to CuArray and ensure Float32
    vals = Float32.(img)
    u = CuArray(copy(vals))
    v = CuArray(1f0 .- vals)
    #k = fill(k, size(vals)) |> CuArray  # Convert k to matrix if it's not already
    
    du = CUDA.zeros(Float32, size(vals))
    dv = CUDA.zeros(Float32, size(vals))
    
    for t in 1:tmax
        CUDA.fill!(du, 0f0)
        CUDA.fill!(dv, 0f0)
        
        grayscott_reaction!(du, dv, u, v, f, k)
        diffuse!(du, u, D_u)
        diffuse!(dv, v, D_v)
        
        u .+= h .* du
        v .+= h .* dv
        
        clamp01!(u)
        clamp01!(v)
    end
    
    return Array(u), Array(v)  # Convert back to CPU arrays
end

img_turing = Gray.(load("Turing.jpg"))
vals = Float64.(img_turing)

begin
	#D_scale = 0.01
	#D_ratio = 8.0
	D_u = 0.38 #D_ratio * D_scale
	D_v = 0.0999#D_scale / D_ratio
	
	#k_min = 0.0045
	k_min = 0.015
	#k_max = 0.075
	k_max = 0.105
	# k_ = 0.08 .* sqrt.(vals)
	#k = lerp.((vals), k_min, k_max)
	k = max.(vals .* k_max, k_min)
end

u, v = forced_grayscott(img_turing_long, D_u=D_u, D_v=D_v, k=k, tmax=1000, h=0.5)


