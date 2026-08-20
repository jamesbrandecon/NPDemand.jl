"""
    add_constraint(constraint_list::Vector{Tuple{Int,Int}}, ind_1, ind_neg1)

Records one constraint row (a `+1` at `ind_1`, a `-1` at `ind_neg1`) by appending
to `constraint_list`. `make_constraint` accumulates all rows this way and builds
the dense constraint matrix once at the end, rather than growing a matrix one
`vcat` at a time (which re-copies the whole matrix on every call and is O(n^2)
in the number of constraints -- catastrophic once J or the sieve order pushes
the matrix width into the tens of thousands of columns).
"""
function add_constraint(constraint_list::Vector{Tuple{Int,Int}}, ind_1::Int, ind_neg1::Int)
    push!(constraint_list, (ind_1, ind_neg1))
    return constraint_list
end
