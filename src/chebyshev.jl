
function chebyshev(x)
    (lo, hi) = extrema(x)
    Chebyshev(lo..hi)
end

function chebyshev(x::AbstractVector{T}, m::Int) where {T}
    S = chebyshev(x)
    return chebyshev(S, x, m)
end


function chebyshev(S::Space, x::AbstractVector{T}, m::Int) where {T}
    n = length(x)
    V = Array{T}(undef,n,m)
    for k = 1:m
        V[:,k] = Fun(S,[zeros(T,k-1);1]).(x)
    end
    return V
end
