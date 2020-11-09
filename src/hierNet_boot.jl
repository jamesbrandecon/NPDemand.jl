function hierNet_boot(df; nfolds::Integer = 5, nlam::Integer = 10, strong::Bool = false, nboot::Integer = 5)
# --------------------------------------------------------------------------
# Options:
# nfolds - number of folds for cross-validation
# nlam - number of values of regularization parameter to try
# strong -  whether strong hierarchy is imposed or not
# nboot - number of bootstrapped samples on which lasso procedure is run
# --------------------------------------------------------------------------

# Unpack DataFrame df
s_all = convert(Array{Float64,2}, df[:, r"s"]);
p_all = convert(Array{Float64,2}, df[:, r"p"]);
x_all = convert(Array{Float64,2}, df[:, r"x"]);
iv_all = convert(Array{Float64,2}, df[:, r"z"]);

J = size(s_all,2);

@rput s_all
@rput p_all
@rput iv_all
@rput strong
@rput nlam
@rput nfolds
@rput nboot

R"
library(hierNet)
J <-  ncol(s_all)

rownames(s_all) <- seq(length=nrow(s_all))
rownames(p_all) <- seq(length=nrow(p_all))
rownames(iv_all) <- seq(length=nrow(iv_all))

for (b in 1:nboot) {
  boot_sample <-sample(1:nrow(s_all), nrow(s_all), replace=TRUE, prob=NULL)
  s <- s_all[boot_sample,]
  p <- p_all[boot_sample,]
  iv <- iv_all[boot_sample,]

  # First stage IV for market shares
  s_hat <- matrix(nrow = nrow(s),ncol = ncol(s))
  one_mat <- matrix(1,nrow(s), 1)
  iv <- as.matrix(iv)
  for (j in 1:J) {
    iv_temp <- as.matrix(cbind(matrix(1,nrow(s),1),iv))
    beta<- solve(t(iv_temp)%*%iv_temp) %*% t(iv_temp) %*% s[,j]
    s_hat[,j] <-iv_temp%*% beta
  }

  # Estimate Hierarchical Lasso
  for (j in 1:J) {
     # Estimate path of regularization parameters
    structured_fit <- hierNet.path(as.matrix(s_hat), as.vector(p[,j]),
                                  lamlist = NULL, delta=1e-8, minlam = NULL, maxlam = NULL, nlam=nlam, flmin=.05,
                                  diagonal = TRUE, strong = strong, aa = NULL, zz = NULL,
                                  stand.main = TRUE, stand.int = TRUE,
                                  rho = nrow(s), niter = 100, sym.eps = 0.001,
                                  step = 1, maxiter = 2000, backtrack = 0.2, tol = 1e-05, trace = 0)
    if (j==1) {
      matsize <- nrow(structured_fit$bp)
      coefs <- matrix(nrow = 0 ,ncol = matsize)
    }

    # Select parameter by CV, append to matrix
    a<- hierNet.cv(structured_fit, as.matrix(s_hat), as.vector(p[,j]), nfolds=nfolds,folds=NULL,trace=0)
    lambdahat <- a['lamhat']
    # lamhat lamhat.1se

    # Final Estimates
    structured_fit<- hierNet(as.matrix(s_hat), as.vector(p[,j]), as.numeric(lambdahat), delta=1e-8, strong=strong, diagonal=TRUE, aa=NULL,
                               zz=NULL,center=TRUE, stand.main=TRUE, stand.int=TRUE, rho=nrow(s), niter=100,
                               sym.eps=1e-3,step=1, maxiter=5000, backtrack=0.2, tol=1e-5, trace=0)
    coefs <- rbind(coefs,t(structured_fit$bp - structured_fit$bn))

  }
  if (b==1){
    included <- (abs(coefs)>0)
  }

  if (b>1) {
    temp <- (abs(coefs)>0)
    included <- 1*as.matrix((temp==1 & included==1))
  }

  for (j in 1:J) {
    included[j,j] = 1
  }
}
included_symmetric <- included
for (j1 in 1:J) {
  for (j2 in 1:J) {
    if (included_symmetric[j1,j2] != included_symmetric[j2,j1]){
      included_symmetric[j1,j2] = 1
      included_symmetric[j2,j1] = 1
    }
  }
}
"
@rget included
@rget included_symmetric
included = convert.(Integer, included)
included_symmetric = convert.(Integer, included_symmetric)
return included, included_symmetric
end
