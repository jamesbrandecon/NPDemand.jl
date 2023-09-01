function add_constraint(constraint_matrix, ind_1, ind_neg1)
    new_con = zeros(1, size(constraint_matrix,2));
    new_con[ind_1] = 1;
    new_con[ind_neg1] = -1;
    return vcat(constraint_matrix, new_con)
end