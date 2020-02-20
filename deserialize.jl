const typecodes = Dict{Int32, DataType}([
    0 => UInt8,
    1 => Float32,
    2 => Float64,
    3 => Int32
])

using SparseArrays

function deserialize(fname::AbstractString)
    file = open(fname, "r")
    typecode = read(file, Int32)
    if !(typecode in keys(typecodes))
        throw(ErrorException("Unknown typecode"))
    end

    dtype = typecodes[typecode]

    I = UInt64[]
    J = UInt64[]
    V = dtype[]

    nrows = read(file, UInt64)
    row_lens = resize!(UInt64[], nrows)
    read!(file, row_lens)

    nentries = zero(UInt64)
    for i = 1:nrows
        nentries += row_lens[i]
        append!(I, (i for j = 1:row_lens[i]))
    end

    resize!(J, nentries)
    resize!(V, nentries)
    read!(file, J)
    read!(file, V)

    sparse(I, J .+= 1, V)
end

